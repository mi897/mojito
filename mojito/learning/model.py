"""A small neural network that learns inverse kinematics for a family of legs.

Input : target tip position (3) + that limb's link lengths (n_params)
Output: joint values (dof), squashed so they always lie inside the joint limits

The sizes come from the limb description: a `spec.LimbSpec` built from a URDF,
or a `leg.LegConfig` for the original 3-joint leg. Saved weights carry the
LimbSpec they were trained for, and loading them against a URDF checks that
the two still describe the same limb.

This module is the NumPy backend: hand-written backpropagation, no dependency
beyond NumPy. mojito/learning/torch_model.py is the PyTorch backend. Both share
IKNetBase (mojito/learning/base.py) and the same weight files; mojito/learning/backend.py picks between them.
"""
from __future__ import annotations

import numpy as np

from ..geometry.limb_model import LimbModel
from ..geometry.spec import LimbSpec, SpecMismatch  # noqa: F401  (re-exported: imported from here by older code)
# IKNetBase and reconcile live in base.py; re-exported here so existing imports keep working.
from .base import IKNetBase, reconcile  # noqa: F401


class IKNet(IKNetBase):
    """NumPy implementation with hand-written backpropagation."""

    backend = "numpy"

    def __init__(self, config: LimbModel | None = None, hidden=(128, 128, 128), seed=0):
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
