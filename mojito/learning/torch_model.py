"""PyTorch backend for the leg IK network.

Same network, same loss, same weight files as the NumPy backend in model.py.
The differences are that gradients come from autograd rather than a
hand-derived Jacobian, and that it can run on a GPU.

Inputs and outputs of predict() are NumPy arrays, so everything built on top
(body.Robot, the gait code, the scripts) works unchanged with either backend.
For use inside a larger PyTorch model, call angles() and forward_kinematics()
directly: they take and return tensors and are differentiable.

Do not import this module directly unless you want PyTorch; use
mojito.make_model / mojito.load_model with backend="torch".
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from ..robot import leg
from .model import IKNet, IKNetBase


def forward_kinematics(joint_angles: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Differentiable foot position. Mirrors leg.forward (see its docstring for the equations). Shapes (..., 3)."""
    lengths = torch.broadcast_to(lengths, joint_angles.shape)
    abduction_offset, upper_length, lower_length = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    abduction, hip_pitch, knee = joint_angles[..., 0], joint_angles[..., 1], joint_angles[..., 2]
    sin_abduction, cos_abduction = torch.sin(abduction), torch.cos(abduction)
    x = -upper_length * torch.sin(hip_pitch) - lower_length * torch.sin(hip_pitch + knee)
    extension = upper_length * torch.cos(hip_pitch) + lower_length * torch.cos(hip_pitch + knee)
    y = abduction_offset * cos_abduction + extension * sin_abduction
    z = abduction_offset * sin_abduction - extension * cos_abduction
    return torch.stack([x, y, z], dim=-1)


class TorchIKNet(IKNetBase):
    """PyTorch implementation. Starts from the same weights as IKNet for a given seed."""

    backend = "torch"

    def __init__(self, config=None, hidden=(128, 128, 128), seed=0,
                 device="cpu", dtype=torch.float32):
        super().__init__(config, hidden)
        self.device = torch.device(device)
        self.dtype = dtype
        sizes = (self.in_dim,) + self.hidden + (self.out_dim,)
        layers = []
        for n_in, n_out in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(n_in, n_out), nn.Tanh()]  # the final Tanh squashes into the joint range
        self.net = nn.Sequential(*layers).to(device=self.device, dtype=self.dtype)
        self._nominal = self._tensor(self.config.nominal)
        self._joint_mid = self._tensor(self.joint_mid)
        self._joint_half_range = self._tensor(self.joint_half_range)
        self._opt = None
        reference = IKNet(self.config, self.hidden, seed=seed)  # identical initialisation
        self._set_weights(reference.W, reference.b)

    # ---- plumbing ----
    def _tensor(self, values) -> torch.Tensor:
        if isinstance(values, torch.Tensor):
            return values.to(device=self.device, dtype=self.dtype)
        return torch.tensor(np.asarray(values, dtype=np.float64), dtype=self.dtype, device=self.device)

    @property
    def _linears(self):
        return [module for module in self.net if isinstance(module, nn.Linear)]

    def _set_weights(self, W, b):
        with torch.no_grad():
            for layer, weight, bias in zip(self._linears, W, b):
                layer.weight.copy_(self._tensor(np.asarray(weight).T))  # torch stores (outputs, inputs)
                layer.bias.copy_(self._tensor(bias))
        self._opt = None

    def state(self) -> dict:
        state = {}
        for i, layer in enumerate(self._linears):
            state[f"W{i}"] = layer.weight.detach().cpu().numpy().T.astype(np.float64)
            state[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float64)
        state.update(self._config_state())
        return state

    # ---- tensors in, tensors out (differentiable) ----
    def forward_kinematics(self, joint_angles: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Differentiable tip position. URDF-derived limbs use the generic chain; the legacy leg its own formula."""
        if getattr(self.config, "backend_agnostic", False):
            return self.config.forward(joint_angles, lengths)
        return forward_kinematics(joint_angles, lengths)

    def angles(self, target_position: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Joint angles for canonical (left-leg) targets."""
        # Same dimensionless features as IKNet.features: position / reach and (length / nominal - 1) / tolerance.
        deviation = (lengths / self._nominal - 1.0) / self.config.length_tolerance
        features = torch.cat(
            [target_position / self.config.reach, torch.broadcast_to(deviation, target_position.shape[:-1] + deviation.shape[-1:])],
            dim=-1,
        )
        return self._joint_mid + self._joint_half_range * self.net(features)

    def loss(self, target_position: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mean squared foot-position error in units of leg reach. Same definition as IKNet."""
        error = (self.forward_kinematics(self.angles(target_position, lengths), lengths) - target_position) / self.config.reach
        return (error * error).sum(dim=-1).mean()

    # ---- NumPy in, NumPy out (the interface the rest of the project uses) ----
    def predict(self, target_position, lengths):
        with torch.no_grad():
            joint_angles = self.angles(self._tensor(target_position), self._tensor(lengths))
        return joint_angles.cpu().numpy().astype(np.float64)

    def loss_and_grads(self, target_position, lengths):
        """Loss and gradients laid out exactly like IKNet.loss_and_grads, for comparing backends."""
        self.net.zero_grad()
        loss = self.loss(self._tensor(target_position), self._tensor(lengths))
        loss.backward()
        grad_W = [layer.weight.grad.detach().cpu().numpy().T.astype(np.float64) for layer in self._linears]
        grad_b = [layer.bias.grad.detach().cpu().numpy().astype(np.float64) for layer in self._linears]
        return float(loss.item()), grad_W + grad_b

    def train_step(self, target_position, lengths, learning_rate: float) -> float:
        """One Adam update on a batch. Returns the loss before the update."""
        if self._opt is None:
            self._opt = torch.optim.Adam(self.net.parameters(), lr=learning_rate)
        for group in self._opt.param_groups:
            group["lr"] = learning_rate
        self._opt.zero_grad()
        loss = self.loss(self._tensor(target_position), self._tensor(lengths))
        loss.backward()
        self._opt.step()
        return float(loss.item())
