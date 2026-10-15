"""Forward kinematics, Jacobian and a numeric IK solver for any serial limb.

A limb is described by a `Chain`: an ordered list of elements, each a fixed
translation (a link, whose length is a parameter), a fixed rotation, and
optionally a joint (revolute or prismatic) about an axis. mojito/geometry/spec.py builds
chains from URDF files; nothing here knows about URDF or about learning.

`fk` and `jacobian` are written against whatever array library the inputs come
from. NumPy arrays give NumPy results; torch tensors give differentiable
torch results. Only `sin`, `cos`, `stack` and `@` are used.

Conventions (identical to mojito.robot.leg, which tests use as an oracle):
    joint_values   (..., dof)       joint values (angles for revolute, distances for prismatic), in the chain's joint order
    link_lengths   (..., n_params)  link lengths; broadcast against joint_values
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
    param_index: tuple  # (E,) index into link_lengths of the length scaling this translation, or -1
    rot: np.ndarray  # (E, 3, 3) fixed rotation applied after the translation
    has_rot: tuple  # (E,) False where rot is the identity, to skip the multiply
    joint_index: tuple  # (E,) index into joint_values if this element carries a joint, else -1
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


def _const(xp, reference, values):
    """A constant array in the same library, dtype and device as `reference`."""
    if xp is np:
        return np.asarray(values, dtype=float)
    return xp.as_tensor(np.asarray(values, dtype=float), dtype=reference.dtype, device=reference.device)


def _cross(a, b, xp):
    """Cross product a x b along the last axis."""
    return xp.stack(
        [a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
         a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
         a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]],
        -1,
    )


def _axis_rotation(xp, reference, axis, angle):
    """Rodrigues rotation about a constant unit axis. angle: (...,) -> (..., 3, 3).

    R = cos(a) I + sin(a) [axis]x + (1 - cos(a)) axis axis^T,
    where [axis]x is the matrix that performs the cross product with `axis`.
    """
    axis = np.asarray(axis, dtype=float)
    cross_matrix = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    outer = np.outer(axis, axis)
    identity, cross_matrix, outer = (_const(xp, reference, m) for m in (np.eye(3), cross_matrix, outer))
    cos_angle, sin_angle = xp.cos(angle)[..., None, None], xp.sin(angle)[..., None, None]
    return cos_angle * identity + sin_angle * cross_matrix + (1.0 - cos_angle) * outer


# ----------------------------------------------------------------- kinematics
def _walk(chain: Chain, joint_values, link_lengths):
    """Run the chain once. Returns the tip and, per joint, its position and world axis.

    Keeps a running frame (position, rotation). Each element first translates
    along its direction (rotated into the world by the frame so far), then
    applies its fixed rotation, then its joint: a revolute joint rotates the
    frame about its axis, a prismatic one slides the position along it.
    """
    xp = _lib(joint_values)
    joint_values = joint_values if _is_torch(joint_values) else np.asarray(joint_values, dtype=float)
    link_lengths = link_lengths if _is_torch(link_lengths) else np.asarray(link_lengths, dtype=float)
    batch = joint_values.shape[:-1]
    rotation = xp.broadcast_to(_const(xp, joint_values, np.eye(3)), batch + (3, 3))  # frame orientation so far
    position = xp.zeros(batch + (3,), dtype=joint_values.dtype, **({"device": joint_values.device} if xp is not np else {}))
    joint_origins, joint_world_axes = [None] * chain.dof, [None] * chain.dof
    for element in range(len(chain.param_index)):
        length_index = chain.param_index[element]
        if length_index >= 0:
            offset = link_lengths[..., length_index, None] * _const(xp, joint_values, chain.dirs[element])
            position = position + (rotation @ offset[..., None])[..., 0]
        if chain.has_rot[element]:
            rotation = rotation @ _const(xp, joint_values, chain.rot[element])
        joint = chain.joint_index[element]
        if joint >= 0:
            axis_in_world = rotation @ _const(xp, joint_values, chain.axes[element])
            joint_origins[joint], joint_world_axes[joint] = position, axis_in_world
            if chain.prismatic[element]:
                position = position + axis_in_world * joint_values[..., joint, None]
            else:
                rotation = rotation @ _axis_rotation(xp, joint_values, chain.axes[element], joint_values[..., joint])
    return position, joint_origins, joint_world_axes


def fk(chain: Chain, joint_values, link_lengths):
    """Tip position for the given joint values. joint_values: (..., dof), link_lengths: (..., n_params)."""
    return _walk(chain, joint_values, link_lengths)[0]


def points(chain: Chain, joint_values, link_lengths):
    """Limb origin, each joint's position and the tip, shape (..., dof + 2, 3). For drawing."""
    tip, joint_origins, _ = _walk(chain, joint_values, link_lengths)
    xp = _lib(joint_values)
    return xp.stack([xp.zeros_like(tip)] + joint_origins + [tip], -2)


def jacobian(chain: Chain, joint_values, link_lengths):
    """d(tip position)/d(joint_values), shape (..., 3, dof).

    Column j is, for a joint with world axis a_j at origin o_j and tip position p:
        revolute:   a_j x (p - o_j)     (the tip moves at right angles to its lever arm)
        prismatic:  a_j                 (the tip slides along the axis)
    """
    xp = _lib(joint_values)
    tip, joint_origins, joint_world_axes = _walk(chain, joint_values, link_lengths)
    is_prismatic = [None] * chain.dof
    for element, joint in enumerate(chain.joint_index):
        if joint >= 0:
            is_prismatic[joint] = chain.prismatic[element]
    columns = [
        joint_world_axes[j] if is_prismatic[j] else _cross(joint_world_axes[j], tip - joint_origins[j], xp)
        for j in range(chain.dof)
    ]
    return xp.stack(columns, -1)


# ----------------------------------------------------------------- numeric IK (NumPy)
def numeric_ik(chain: Chain, limits, target_position, link_lengths, initial_joint_values=None,
               max_iterations: int = 100, damping: float = 1e-2, tolerance: float = 1e-9):
    """Damped-least-squares Newton IK for a batch of targets (NumPy only).

    A local solver: it returns the solution nearest `initial_joint_values` (default: the middle
    of the joint ranges), clipped to the limits. Use it as a reference for
    chains with no closed form, not as a fast or global solver.

    Each iteration solves for the joint step that moves the tip toward the target:
        step = J^T (J J^T + damping^2 I)^-1 error,    error = target - fk(joint_values)
    The damping term keeps the step finite near singular poses.
    """
    target_position = np.asarray(target_position, dtype=float)
    limits = np.asarray(limits, dtype=float)
    lower, upper = limits[:, 0], limits[:, 1]
    joint_values = np.broadcast_to(
        limits.mean(axis=1) if initial_joint_values is None else np.asarray(initial_joint_values, dtype=float),
        target_position.shape[:-1] + (chain.dof,),
    ).copy()
    identity = np.eye(3)
    for _ in range(max_iterations):
        error = target_position - fk(chain, joint_values, link_lengths)
        if np.max(np.abs(error), initial=0.0) < tolerance:
            break
        jac = jacobian(chain, joint_values, link_lengths)
        jac_t = np.swapaxes(jac, -1, -2)
        damped_gram = jac @ jac_t + (damping**2) * identity
        joint_step = jac_t @ np.linalg.solve(damped_gram, error[..., None])
        joint_values = np.clip(joint_values + joint_step[..., 0], lower, upper)
    return joint_values
