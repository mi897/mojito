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

from . import leg
from .model import IKNet, IKNetBase


def forward_kinematics(q: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    """Differentiable foot position. Mirrors leg.forward. Shapes (..., 3)."""
    lengths = torch.broadcast_to(lengths, q.shape)
    la, lu, ll = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    q1, q2, q3 = q[..., 0], q[..., 1], q[..., 2]
    s1, c1 = torch.sin(q1), torch.cos(q1)
    x = -lu * torch.sin(q2) - ll * torch.sin(q2 + q3)
    h = lu * torch.cos(q2) + ll * torch.cos(q2 + q3)
    return torch.stack([x, la * c1 + h * s1, la * s1 - h * c1], dim=-1)


class TorchIKNet(IKNetBase):
    """PyTorch implementation. Starts from the same weights as IKNet for a given seed."""

    backend = "torch"

    def __init__(self, cfg: leg.LegConfig | None = None, hidden=(128, 128, 128), seed=0,
                 device="cpu", dtype=torch.float32):
        super().__init__(cfg, hidden)
        self.device = torch.device(device)
        self.dtype = dtype
        sizes = (6,) + self.hidden + (3,)
        layers = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            layers += [nn.Linear(a, b), nn.Tanh()]  # the final Tanh squashes into the joint range
        self.net = nn.Sequential(*layers).to(device=self.device, dtype=self.dtype)
        self._nominal = self._tensor(self.cfg.nominal)
        self._q_mid = self._tensor(self.q_mid)
        self._q_half = self._tensor(self.q_half)
        self._opt = None
        reference = IKNet(self.cfg, self.hidden, seed=seed)  # identical initialisation
        self._set_weights(reference.W, reference.b)

    # ---- plumbing ----
    def _tensor(self, x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.tensor(np.asarray(x, dtype=np.float64), dtype=self.dtype, device=self.device)

    @property
    def _linears(self):
        return [m for m in self.net if isinstance(m, nn.Linear)]

    def _set_weights(self, W, b):
        with torch.no_grad():
            for lin, w, x in zip(self._linears, W, b):
                lin.weight.copy_(self._tensor(np.asarray(w).T))  # torch stores (outputs, inputs)
                lin.bias.copy_(self._tensor(x))
        self._opt = None

    def state(self) -> dict:
        d = {}
        for i, lin in enumerate(self._linears):
            d[f"W{i}"] = lin.weight.detach().cpu().numpy().T.astype(np.float64)
            d[f"b{i}"] = lin.bias.detach().cpu().numpy().astype(np.float64)
        d.update(self._config_state())
        return d

    # ---- tensors in, tensors out (differentiable) ----
    def angles(self, p: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Joint angles for canonical (left-leg) targets."""
        dev = (lengths / self._nominal - 1.0) / self.cfg.length_tolerance
        x = torch.cat([p / self.cfg.reach, torch.broadcast_to(dev, p.shape)], dim=-1)
        return self._q_mid + self._q_half * self.net(x)

    def loss(self, p: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Mean squared foot-position error in units of leg reach. Same definition as IKNet."""
        err = (forward_kinematics(self.angles(p, lengths), lengths) - p) / self.cfg.reach
        return (err * err).sum(dim=-1).mean()

    # ---- NumPy in, NumPy out (the interface the rest of the project uses) ----
    def predict(self, p, lengths):
        with torch.no_grad():
            q = self.angles(self._tensor(p), self._tensor(lengths))
        return q.cpu().numpy().astype(np.float64)

    def loss_and_grads(self, p, lengths):
        """Loss and gradients laid out exactly like IKNet.loss_and_grads, for comparing backends."""
        self.net.zero_grad()
        loss = self.loss(self._tensor(p), self._tensor(lengths))
        loss.backward()
        gW = [lin.weight.grad.detach().cpu().numpy().T.astype(np.float64) for lin in self._linears]
        gb = [lin.bias.grad.detach().cpu().numpy().astype(np.float64) for lin in self._linears]
        return float(loss.item()), gW + gb

    def train_step(self, p, lengths, lr: float) -> float:
        """One Adam update on a batch. Returns the loss before the update."""
        if self._opt is None:
            self._opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        for group in self._opt.param_groups:
            group["lr"] = lr
        self._opt.zero_grad()
        loss = self.loss(self._tensor(p), self._tensor(lengths))
        loss.backward()
        self._opt.step()
        return float(loss.item())
