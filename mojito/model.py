"""A small neural network that learns inverse kinematics for a family of legs.

Input : target foot position (3) + that leg's link lengths (3)
Output: joint angles (3), squashed so they always lie inside the joint limits

This module is the NumPy backend: hand-written backpropagation, no dependency
beyond NumPy. mojito/torch_model.py is the PyTorch backend. Both share
IKNetBase and the same weight files; mojito/backend.py picks between them.
"""
from __future__ import annotations

import json

import numpy as np

from . import leg


class IKNetBase:
    """What every backend shares: configuration, mirroring, and the weight file format.

    A subclass provides predict(p, lengths), train_step(p, lengths, lr),
    state() and _set_weights(W, b). Weights travel as NumPy arrays with W[i]
    shaped (inputs, outputs), whatever the backend stores internally.
    """

    backend = None

    def __init__(self, cfg: leg.LegConfig | None = None, hidden=(128, 128, 128)):
        self.cfg = cfg or leg.LegConfig()
        self.hidden = tuple(hidden)
        lim = self.cfg.limits
        self.q_mid = lim.mean(axis=1)
        self.q_half = 0.5 * (lim[:, 1] - lim[:, 0])

    def predict_leg(self, p, lengths, right: bool = False):
        """As `predict`, for either side of the robot."""
        if not right:
            return self.predict(p, lengths)
        return leg.mirror_angles(self.predict(leg.mirror_target(p), lengths))

    def weights(self):
        """(list of W, list of b) as NumPy arrays."""
        d = self.state()
        n = sum(1 for k in d if k.startswith("W"))
        return [d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)]

    def _config_state(self) -> dict:
        return dict(nominal=self.cfg.nominal, limits=self.cfg.limits,
                    length_tolerance=np.array(self.cfg.length_tolerance))

    def save(self, path):
        np.savez(path, **self.state())

    @classmethod
    def load(cls, path, **kwargs):
        d = np.load(path)
        n_layers = sum(1 for k in d.files if k.startswith("W"))
        cfg = leg.LegConfig(d["nominal"], d["limits"], float(d["length_tolerance"]))
        hidden = tuple(d[f"W{i}"].shape[1] for i in range(n_layers - 1))
        net = cls(cfg, hidden, **kwargs)
        net._set_weights([d[f"W{i}"] for i in range(n_layers)], [d[f"b{i}"] for i in range(n_layers)])
        return net

    def to_json(self, digits: int = 6) -> str:
        """Compact export for running the network outside Python."""
        r = lambda a: np.round(np.asarray(a, dtype=float), digits).tolist()
        W, b = self.weights()
        return json.dumps(
            {
                "W": [r(w) for w in W],
                "b": [r(x) for x in b],
                "nominal": r(self.cfg.nominal),
                "limits": r(self.cfg.limits),
                "tolerance": self.cfg.length_tolerance,
            },
            separators=(",", ":"),
        )


class IKNet(IKNetBase):
    """NumPy implementation with hand-written backpropagation."""

    backend = "numpy"

    def __init__(self, cfg: leg.LegConfig | None = None, hidden=(128, 128, 128), seed=0):
        super().__init__(cfg, hidden)
        rng = np.random.default_rng(seed)
        sizes = (6,) + self.hidden + (3,)
        self.W = [
            rng.normal(0.0, np.sqrt(1.0 / a), size=(a, b)) for a, b in zip(sizes[:-1], sizes[1:])
        ]
        self.b = [np.zeros(b) for b in sizes[1:]]
        self.W[-1] *= 0.1  # start near the middle of the joint range
        self._opt = None

    def _set_weights(self, W, b):
        self.W = [np.array(w, dtype=float) for w in W]
        self.b = [np.array(x, dtype=float) for x in b]
        self._opt = None

    # ---- parameters as one flat list, convenient for the optimiser ----
    @property
    def params(self):
        return self.W + self.b

    def features(self, p, lengths):
        """Dimensionless inputs, each roughly in [-1, 1]."""
        p = np.asarray(p, dtype=float) / self.cfg.reach
        dev = (np.asarray(lengths, dtype=float) / self.cfg.nominal - 1.0) / self.cfg.length_tolerance
        return np.concatenate([p, np.broadcast_to(dev, p.shape)], axis=-1)

    def _forward(self, x):
        acts = [x]
        for W, b in zip(self.W[:-1], self.b[:-1]):
            acts.append(np.tanh(acts[-1] @ W + b))
        t = np.tanh(acts[-1] @ self.W[-1] + self.b[-1])
        return self.q_mid + self.q_half * t, acts, t

    def predict(self, p, lengths):
        """Joint angles for canonical (left-leg) targets. Shapes (..., 3)."""
        return self._forward(self.features(p, lengths))[0]

    def loss_and_grads(self, p, lengths):
        """Mean squared foot-position error (in units of leg reach) and its gradient.

        The loss compares where the predicted angles put the foot with where it
        was asked to go. It never looks at the angles that generated the target.
        """
        n = p.shape[0]
        q, acts, t = self._forward(self.features(p, lengths))
        err = (leg.forward(q, lengths) - p) / self.cfg.reach
        loss = float(np.mean(np.sum(err * err, axis=1)))

        # Back through forward kinematics: dL/dq = J^T dL/dp.
        J = leg.jacobian(q, lengths)
        dq = np.einsum("nij,ni->nj", J, err) * (2.0 / (n * self.cfg.reach))
        delta = dq * self.q_half * (1.0 - t * t)

        gW, gb = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            gW[i] = acts[i].T @ delta
            gb[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.W[i].T) * (1.0 - acts[i] ** 2)
        return loss, gW + gb

    def train_step(self, p, lengths, lr: float) -> float:
        """One Adam update on a batch. Returns the loss before the update."""
        if self._opt is None:
            self._opt = Adam(self.params, lr=lr)
        loss, grads = self.loss_and_grads(p, lengths)
        self._opt.step(grads, lr)
        return loss

    def state(self) -> dict:
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        d.update(self._config_state())
        return d


class Adam:
    def __init__(self, params, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
        self.params, self.lr, self.b1, self.b2, self.eps = params, lr, b1, b2, eps
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads, lr=None):
        lr = self.lr if lr is None else lr
        self.t += 1
        c1, c2 = 1 - self.b1**self.t, 1 - self.b2**self.t
        for p, g, m, v in zip(self.params, grads, self.m, self.v):
            m *= self.b1
            m += (1 - self.b1) * g
            v *= self.b2
            v += (1 - self.b2) * g * g
            p -= lr * (m / c1) / (np.sqrt(v / c2) + self.eps)
