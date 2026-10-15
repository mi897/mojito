"""A small neural network that learns inverse kinematics for a family of legs.

Input : target tip position (3) + that limb's link lengths (n_params)
Output: joint values (dof), squashed so they always lie inside the joint limits

The sizes come from the limb description: a `spec.LimbSpec` built from a URDF,
or a `leg.LegConfig` for the original 3-joint leg. Saved weights carry the
LimbSpec they were trained for, and loading them against a URDF checks that
the two still describe the same limb.

This module is the NumPy backend: hand-written backpropagation, no dependency
beyond NumPy. mojito/learning/torch_model.py is the PyTorch backend. Both share
IKNetBase and the same weight files; mojito/learning/backend.py picks between them.
"""
from __future__ import annotations

import json

import numpy as np

from ..robot import leg
from ..geometry.spec import LimbSpec, SpecMismatch


def reconcile(file_cfg, spec: LimbSpec, where: str = "weights") -> LimbSpec:
    """Check that weights trained for `file_cfg` fit the URDF-derived `spec`, and return the config to use.

    The weights stay authoritative for input scaling (nominal lengths,
    tolerance, reach); the URDF has to agree on structure.
    """
    if isinstance(file_cfg, LimbSpec):
        diffs = file_cfg.describe_mismatch(spec)
        if diffs:
            raise SpecMismatch(f"{where} do not match the URDF limb {spec.name!r}:\n  " + "\n  ".join(diffs))
        return file_cfg
    # Legacy weights: a LegConfig. The URDF limb must be the same 3-joint leg.
    diffs = []
    if (spec.dof, spec.n_params) != (3, 3):
        diffs.append(f"weights are for 3 joints and 3 lengths, the URDF limb has {spec.dof} and {spec.n_params}")
    else:
        if not np.allclose(spec.limits, file_cfg.limits, atol=1e-9):
            diffs.append(f"joint limits: {file_cfg.limits.tolist()} in weights vs {spec.limits.tolist()} in URDF")
        rng = np.random.default_rng(0)
        joint_angles = rng.uniform(file_cfg.limits[:, 0], file_cfg.limits[:, 1], size=(32, 3))
        L = file_cfg.nominal * (1 + rng.uniform(-0.1, 0.1, size=(32, 3)))
        if not np.allclose(spec.forward(joint_angles, L), leg.forward(joint_angles, L), atol=1e-9):
            diffs.append("forward kinematics of the URDF limb differs from the leg model the weights were trained on")
    if diffs:
        raise SpecMismatch(f"{where} do not match the URDF limb {spec.name!r}:\n  " + "\n  ".join(diffs))
    return spec.with_normalisation(file_cfg.nominal, file_cfg.length_tolerance, file_cfg.reach)


class IKNetBase:
    """What every backend shares: configuration, mirroring, and the weight file format.

    A subclass provides predict(target_position, lengths), train_step(target_position, lengths, learning_rate),
    state() and _set_weights(W, b). Weights travel as NumPy arrays with W[i]
    shaped (inputs, outputs), whatever the backend stores internally.
    """

    backend = None

    def __init__(self, config=None, hidden=(128, 128, 128)):
        self.config = config or leg.LegConfig()
        self.hidden = tuple(hidden)
        self.in_dim = 3 + self.config.n_params
        self.out_dim = self.config.dof
        limits = self.config.limits
        # Output squashing: joint_angles = joint_mid + joint_half_range * tanh(z) always lies inside the limits.
        self.joint_mid = limits.mean(axis=1)  # centre of each joint's range
        self.joint_half_range = 0.5 * (limits[:, 1] - limits[:, 0])  # half-width of each joint's range

    def predict_leg(self, target_position, lengths, right: bool = False, mirror=None):
        """As `predict`, for either side of the robot.

        `mirror` is the spec.Mirror relating the limb to this network's canonical
        limb; by default the left/right leg mirror.
        """
        if not right:
            return self.predict(target_position, lengths)
        mirror = mirror or self.config.mirror
        return mirror.angles(self.predict(mirror.target(target_position), lengths))

    def weights(self):
        """(list of W, list of b) as NumPy arrays."""
        d = self.state()
        n = sum(1 for k in d if k.startswith("W"))
        return [d[f"W{i}"] for i in range(n)], [d[f"b{i}"] for i in range(n)]

    def _config_state(self) -> dict:
        d = dict(nominal=self.config.nominal, limits=self.config.limits,
                 length_tolerance=np.array(self.config.length_tolerance))
        if isinstance(self.config, LimbSpec):
            d["spec"] = np.array(json.dumps(self.config.to_dict()))
        return d

    def save(self, path):
        np.savez(path, **self.state())

    @classmethod
    def load(cls, path, spec: LimbSpec | None = None, **kwargs):
        """Load weights. Pass the URDF-derived `spec` to verify they belong to that limb.

        Raises spec.SpecMismatch, listing every difference, if they do not.
        """
        d = np.load(path)
        n_layers = sum(1 for k in d.files if k.startswith("W"))
        if "spec" in d.files:
            config = LimbSpec.from_dict(json.loads(str(d["spec"])))
        else:  # weights from before limbs were described by URDFs
            config = leg.LegConfig(d["nominal"], d["limits"], float(d["length_tolerance"]))
        if spec is not None:
            config = reconcile(config, spec, str(path))
        hidden = tuple(d[f"W{i}"].shape[1] for i in range(n_layers - 1))
        net = cls(config, hidden, **kwargs)
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
                "nominal": r(self.config.nominal),
                "limits": r(self.config.limits),
                "tolerance": self.config.length_tolerance,
                "reach": self.config.reach,
                **({"spec": self.config.to_dict()} if isinstance(self.config, LimbSpec) else {}),
            },
            separators=(",", ":"),
        )


class IKNet(IKNetBase):
    """NumPy implementation with hand-written backpropagation."""

    backend = "numpy"

    def __init__(self, config=None, hidden=(128, 128, 128), seed=0):
        super().__init__(config, hidden)
        rng = np.random.default_rng(seed)
        sizes = (self.in_dim,) + self.hidden + (self.out_dim,)
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

    def features(self, target_position, lengths):
        """Dimensionless inputs, each roughly in [-1, 1].

        position / reach, and each length's fractional deviation from nominal
        divided by the tolerance: (length / nominal - 1) / tolerance.
        """
        position = np.asarray(target_position, dtype=float) / self.config.reach
        deviation = (np.asarray(lengths, dtype=float) / self.config.nominal - 1.0) / self.config.length_tolerance
        deviation = np.broadcast_to(deviation, position.shape[:-1] + deviation.shape[-1:])
        return np.concatenate([position, deviation], axis=-1)

    def _forward(self, x):
        """Run the network. Returns (joint values, activations of each layer input, output tanh)."""
        activations = [x]
        for weight, bias in zip(self.W[:-1], self.b[:-1]):
            activations.append(np.tanh(activations[-1] @ weight + bias))
        squashed = np.tanh(activations[-1] @ self.W[-1] + self.b[-1])  # in (-1, 1)
        return self.joint_mid + self.joint_half_range * squashed, activations, squashed

    def predict(self, target_position, lengths):
        """Joint values for targets in the canonical limb frame. target_position: (..., 3), lengths: (..., n_params)."""
        return self._forward(self.features(target_position, lengths))[0]

    def loss_and_grads(self, target_position, lengths):
        """Mean squared foot-position error (in units of leg reach) and its gradient.

        The loss compares where the predicted angles put the foot with where it
        was asked to go. It never looks at the angles that generated the target.
        """
        batch = target_position.shape[0]
        joint_angles, activations, squashed = self._forward(self.features(target_position, lengths))
        error = (self.config.forward(joint_angles, lengths) - target_position) / self.config.reach  # foot error in units of reach
        loss = float(np.mean(np.sum(error * error, axis=1)))  # L = mean over batch of |error|^2

        # Back through forward kinematics: dL/dq = J^T dL/dfoot, where the foot position
        # enters the loss through error = (foot - target_position) / reach, so dL/dfoot = 2 error / (batch * reach).
        jac = self.config.jacobian(joint_angles, lengths)
        grad_q = np.einsum("nij,ni->nj", jac, error) * (2.0 / (batch * self.config.reach))
        # Through joint_angles = joint_mid + joint_half_range tanh(z): dq/dz = joint_half_range (1 - tanh^2).
        delta = grad_q * self.joint_half_range * (1.0 - squashed * squashed)

        # delta is dL/d(pre-activation) of the layer being processed. Weight gradient =
        # (layer input)^T delta; bias gradient = delta summed over the batch; and
        # propagating back through W then tanh multiplies by (1 - activation^2).
        grad_W, grad_b = [None] * len(self.W), [None] * len(self.b)
        for i in reversed(range(len(self.W))):
            grad_W[i] = activations[i].T @ delta
            grad_b[i] = delta.sum(axis=0)
            if i:
                delta = (delta @ self.W[i].T) * (1.0 - activations[i] ** 2)
        return loss, grad_W + grad_b

    def train_step(self, target_position, lengths, learning_rate: float) -> float:
        """One Adam update on a batch. Returns the loss before the update."""
        if self._opt is None:
            self._opt = Adam(self.params, learning_rate=learning_rate)
        loss, grads = self.loss_and_grads(target_position, lengths)
        self._opt.step(grads, learning_rate)
        return loss

    def state(self) -> dict:
        d = {f"W{i}": w for i, w in enumerate(self.W)}
        d.update({f"b{i}": b for i, b in enumerate(self.b)})
        d.update(self._config_state())
        return d


class Adam:
    """Adam optimiser (Kingma and Ba, 2015), updating the parameter arrays in place.

    For each parameter with gradient g, at step t:
        m = beta1 m + (1 - beta1) g              moving average of the gradient
        v = beta2 v + (1 - beta2) g^2            moving average of the squared gradient
        m_hat = m / (1 - beta1^t)                bias correction for the zero start
        v_hat = v / (1 - beta2^t)
        param -= learning_rate m_hat / (sqrt(v_hat) + eps)
    """

    def __init__(self, params, learning_rate=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params, self.learning_rate, self.beta1, self.beta2, self.eps = params, learning_rate, beta1, beta2, eps
        self.first_moment = [np.zeros_like(param) for param in params]  # m
        self.second_moment = [np.zeros_like(param) for param in params]  # v
        self.step_count = 0  # t

    def step(self, grads, learning_rate=None):
        learning_rate = self.learning_rate if learning_rate is None else learning_rate
        self.step_count += 1
        correction1 = 1 - self.beta1**self.step_count
        correction2 = 1 - self.beta2**self.step_count
        for param, grad, m, v in zip(self.params, grads, self.first_moment, self.second_moment):
            m *= self.beta1
            m += (1 - self.beta1) * grad
            v *= self.beta2
            v += (1 - self.beta2) * grad * grad
            param -= learning_rate * (m / correction1) / (np.sqrt(v / correction2) + self.eps)
