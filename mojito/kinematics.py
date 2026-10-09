"""Forward kinematics, Jacobian and a numeric IK solver for any serial limb.

A limb is described by a `Chain`: an ordered list of elements, each a fixed
translation (a link, whose length is a parameter), a fixed rotation, and
optionally a joint (revolute or prismatic) about an axis. mojito/spec.py builds
chains from URDF files; nothing here knows about URDF or about learning.

`fk` and `jacobian` are written against whatever array library the inputs come
from. NumPy arrays give NumPy results; torch tensors give differentiable
torch results. Only `sin`, `cos`, `stack` and `@` are used.

Conventions (identical to mojito.leg, which tests use as an oracle):
    q       (..., dof)       joint values, in the chain's joint order
    params  (..., n_params)  link lengths; broadcast against q
    result  (..., 3)         tip position in the limb frame
The limb frame is the frame of the first joint (the "shoulder").
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Chain:
    """Constant description of a serial chain. All arrays are plain NumPy."""

    dirs: np.ndarray  # (E, 3) unit direction of each element's translation (zeros if none)
    param_index: tuple  # (E,) index into params of the length scaling this translation, or -1
    rot: np.ndarray  # (E, 3, 3) fixed rotation applied after the translation
    has_rot: tuple  # (E,) False where rot is the identity, to skip the multiply
    joint_index: tuple  # (E,) index into q if this element carries a joint, else -1
    axes: np.ndarray  # (E, 3) unit joint axis in the element frame (zeros if no joint)
    prismatic: tuple  # (E,) joint slides instead of rotating

    @property
    def dof(self) -> int:
        return sum(1 for j in self.joint_index if j >= 0)

    @property
    def n_params(self) -> int:
        return 1 + max(self.param_index, default=-1)


# ----------------------------------------------------------------- array library glue
def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _lib(x):
    if _is_torch(x):
        import torch

        return torch
    return np


def _const(xp, ref, a):
    """A constant array in the same library, dtype and device as `ref`."""
    if xp is np:
        return np.asarray(a, dtype=float)
    return xp.as_tensor(np.asarray(a, dtype=float), dtype=ref.dtype, device=ref.device)


def _cross(a, b, xp):
    return xp.stack(
        [a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
         a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
         a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]],
        -1,
    )


def _axis_rotation(xp, ref, axis, angle):
    """Rodrigues rotation about a constant unit axis. angle: (...,) -> (..., 3, 3)."""
    ax = np.asarray(axis, dtype=float)
    K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
    A = np.outer(ax, ax)
    I, K, A = (_const(xp, ref, m) for m in (np.eye(3), K, A))
    c, s = xp.cos(angle)[..., None, None], xp.sin(angle)[..., None, None]
    return c * I + s * K + (1.0 - c) * A


# ----------------------------------------------------------------- kinematics
def _walk(chain: Chain, q, params):
    """Run the chain once. Returns the tip and, per joint, its position and world axis."""
    xp = _lib(q)
    q = q if _is_torch(q) else np.asarray(q, dtype=float)
    params = params if _is_torch(params) else np.asarray(params, dtype=float)
    batch = q.shape[:-1]
    R = xp.broadcast_to(_const(xp, q, np.eye(3)), batch + (3, 3))
    t = xp.zeros(batch + (3,), dtype=q.dtype, **({"device": q.device} if xp is not np else {}))
    origins, world_axes = [None] * chain.dof, [None] * chain.dof
    for e in range(len(chain.param_index)):
        pi = chain.param_index[e]
        if pi >= 0:
            step = params[..., pi, None] * _const(xp, q, chain.dirs[e])
            t = t + (R @ step[..., None])[..., 0]
        if chain.has_rot[e]:
            R = R @ _const(xp, q, chain.rot[e])
        j = chain.joint_index[e]
        if j >= 0:
            axis = _const(xp, q, chain.axes[e])
            a_world = R @ axis
            origins[j], world_axes[j] = t, a_world
            if chain.prismatic[e]:
                t = t + a_world * q[..., j, None]
            else:
                R = R @ _axis_rotation(xp, q, chain.axes[e], q[..., j])
    return t, origins, world_axes


def fk(chain: Chain, q, params):
    """Tip position for joint values q. q: (..., dof), params: (..., n_params)."""
    return _walk(chain, q, params)[0]


def points(chain: Chain, q, params):
    """Limb origin, each joint's position and the tip, shape (..., dof + 2, 3). For drawing."""
    tip, origins, _ = _walk(chain, q, params)
    xp = _lib(q)
    return xp.stack([xp.zeros_like(tip)] + origins + [tip], -2)


def jacobian(chain: Chain, q, params):
    """d(tip position)/d(q), shape (..., 3, dof)."""
    xp = _lib(q)
    tip, origins, axes = _walk(chain, q, params)
    prismatic = [None] * chain.dof
    for e, j in enumerate(chain.joint_index):
        if j >= 0:
            prismatic[j] = chain.prismatic[e]
    cols = [axes[j] if prismatic[j] else _cross(axes[j], tip - origins[j], xp) for j in range(chain.dof)]
    return xp.stack(cols, -1)


# ----------------------------------------------------------------- numeric IK (NumPy)
def numeric_ik(chain: Chain, limits, p, params, q0=None, iters: int = 100, damping: float = 1e-2,
               tol: float = 1e-9):
    """Damped-least-squares Newton IK for a batch of targets (NumPy only).

    A local solver: it returns the solution nearest `q0` (default: the middle
    of the joint ranges), clipped to the limits. Use it as a reference for
    chains with no closed form, not as a fast or global solver.
    """
    p = np.asarray(p, dtype=float)
    limits = np.asarray(limits, dtype=float)
    lo, hi = limits[:, 0], limits[:, 1]
    q = np.broadcast_to(limits.mean(axis=1) if q0 is None else np.asarray(q0, dtype=float),
                        p.shape[:-1] + (chain.dof,)).copy()
    eye = np.eye(3)
    for _ in range(iters):
        err = p - fk(chain, q, params)
        if np.max(np.abs(err), initial=0.0) < tol:
            break
        J = jacobian(chain, q, params)
        JJt = J @ np.swapaxes(J, -1, -2) + (damping**2) * eye
        dq = np.swapaxes(J, -1, -2) @ np.linalg.solve(JJt, err[..., None])
        q = np.clip(q + dq[..., 0], lo, hi)
    return q
