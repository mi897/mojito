"""Kinematics of one 3-DoF quadruped leg.

Frame (origin at the shoulder, fixed to the body):
    x forward, y lateral (away from the body for a left leg), z up.

Joints:
    q[0] abduction  - rotation about x
    q[1] hip pitch  - rotation about y
    q[2] knee       - rotation about y, relative to the upper leg

Link lengths are plain arrays passed in at call time, never module constants,
so every function works on a batch of *different* legs at once.

Everything is written for the left-hand ("canonical") leg. A right-hand leg is
its mirror image: flip the sign of y going in and of q[0] coming out. See
`mirror_target` and `mirror_angles`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Placeholder geometry in metres: (abduction offset, upper leg, lower leg).
NOMINAL_LENGTHS = np.array([0.04, 0.12, 0.12])

# Placeholder joint limits in radians, rows = joints, columns = (low, high).
# The knee only bends one way and stops short of straight, which keeps the
# inverse kinematics unique and away from the full-extension singularity.
JOINT_LIMITS = np.array(
    [
        [-0.6, 0.6],  # abduction
        [0.0, 1.5],  # hip pitch
        [-2.3, -0.3],  # knee
    ]
)


@dataclass(frozen=True)
class LegConfig:
    """Nominal leg plus how far individual legs may deviate from it."""

    nominal: np.ndarray = None  # (3,) abduction offset, upper, lower [m]
    limits: np.ndarray = None  # (3, 2) joint limits [rad]
    length_tolerance: float = 0.10  # each length varies by +/- this fraction

    def __post_init__(self):
        if self.nominal is None:
            object.__setattr__(self, "nominal", NOMINAL_LENGTHS.copy())
        if self.limits is None:
            object.__setattr__(self, "limits", JOINT_LIMITS.copy())

    @property
    def reach(self) -> float:
        """Length scale used to make positions dimensionless."""
        return float(self.nominal[1] + self.nominal[2])


def forward(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Foot position for joint angles q.  q: (..., 3), lengths: (..., 3)."""
    q = np.asarray(q, dtype=float)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), q.shape)
    la, lu, ll = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    s1, c1 = np.sin(q[..., 0]), np.cos(q[..., 0])
    s2, c2 = np.sin(q[..., 1]), np.cos(q[..., 1])
    s23, c23 = np.sin(q[..., 1] + q[..., 2]), np.cos(q[..., 1] + q[..., 2])

    x = -lu * s2 - ll * s23
    h = lu * c2 + ll * c23  # leg extension measured in the leg's own plane
    y = la * c1 + h * s1
    z = la * s1 - h * c1
    return np.stack([x, y, z], axis=-1)


def jacobian(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """d(foot position)/d(q), shape (..., 3, 3)."""
    q = np.asarray(q, dtype=float)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), q.shape)
    la, lu, ll = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    s1, c1 = np.sin(q[..., 0]), np.cos(q[..., 0])
    s2, c2 = np.sin(q[..., 1]), np.cos(q[..., 1])
    s23, c23 = np.sin(q[..., 1] + q[..., 2]), np.cos(q[..., 1] + q[..., 2])

    x = -lu * s2 - ll * s23
    h = lu * c2 + ll * c23
    y = la * c1 + h * s1
    z = la * s1 - h * c1

    J = np.zeros(q.shape[:-1] + (3, 3))
    J[..., 0, 1] = -h
    J[..., 0, 2] = -ll * c23
    J[..., 1, 0] = -z
    J[..., 1, 1] = x * s1
    J[..., 1, 2] = -ll * s23 * s1
    J[..., 2, 0] = y
    J[..., 2, 1] = -x * c1
    J[..., 2, 2] = ll * s23 * c1
    return J


def joint_positions(q: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Shoulder, hip, knee and foot positions, shape (..., 4, 3). For drawing."""
    q = np.asarray(q, dtype=float)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), q.shape)
    la, lu = lengths[..., 0], lengths[..., 1]
    s1, c1 = np.sin(q[..., 0]), np.cos(q[..., 0])
    s2, c2 = np.sin(q[..., 1]), np.cos(q[..., 1])
    shoulder = np.zeros(q.shape)
    hip = np.stack([np.zeros_like(la), la * c1, la * s1], axis=-1)
    hk = lu * c2
    knee = np.stack([-lu * s2, la * c1 + hk * s1, la * s1 - hk * c1], axis=-1)
    foot = forward(q, lengths)
    return np.stack([shoulder, hip, knee, foot], axis=-2)


def analytic_ik(p: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Closed-form inverse kinematics: the baseline the network is judged against.

    Returns the solution with the foot below the shoulder and the knee bending
    the way JOINT_LIMITS allows. Unreachable targets are clamped to the nearest
    reachable distance rather than returning NaN.
    """
    p = np.asarray(p, dtype=float)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), p.shape)
    la, lu, ll = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    x, y, z = p[..., 0], p[..., 1], p[..., 2]

    h = np.sqrt(np.maximum(y * y + z * z - la * la, 0.0))
    q1 = np.arctan2(z, y) + np.arctan2(h, la)
    cos_knee = (x * x + h * h - lu * lu - ll * ll) / (2.0 * lu * ll)
    q3 = -np.arccos(np.clip(cos_knee, -1.0, 1.0))
    q2 = np.arctan2(-x, h) - np.arctan2(ll * np.sin(q3), lu + ll * np.cos(q3))
    return np.stack([q1, q2, q3], axis=-1)


def mirror_target(p: np.ndarray) -> np.ndarray:
    """Map a right-leg target into the canonical left-leg frame (and back)."""
    return np.asarray(p, dtype=float) * np.array([1.0, -1.0, 1.0])


def mirror_angles(q: np.ndarray) -> np.ndarray:
    """Map canonical joint angles to the right-leg convention (and back)."""
    return np.asarray(q, dtype=float) * np.array([-1.0, 1.0, 1.0])


def sample_lengths(
    rng: np.random.Generator, n: int, cfg: LegConfig, tolerance: float | None = None
) -> np.ndarray:
    """n legs whose three lengths each deviate independently from nominal."""
    tol = cfg.length_tolerance if tolerance is None else tolerance
    return cfg.nominal * (1.0 + rng.uniform(-tol, tol, size=(n, 3)))


def sample_joint_angles(
    rng: np.random.Generator,
    lengths: np.ndarray,
    cfg: LegConfig,
    workspace_uniform: bool = True,
    oversample: int = 4,
) -> np.ndarray:
    """One reachable pose per leg in `lengths`.

    Angles drawn uniformly in joint space pile foot positions up wherever the
    leg is folded. With workspace_uniform=True we draw extra candidates and
    keep each with probability proportional to |det J|, the local volume the
    foot sweeps per unit of joint motion, which makes foot positions uniform
    over the reachable volume instead.
    """
    n = lengths.shape[0]
    lo, hi = cfg.limits[:, 0], cfg.limits[:, 1]
    if not workspace_uniform:
        return rng.uniform(lo, hi, size=(n, 3))

    q = rng.uniform(lo, hi, size=(n, oversample, 3))
    w = np.abs(np.linalg.det(jacobian(q, lengths[:, None, :])))
    w = w / w.sum(axis=1, keepdims=True)
    pick = (rng.random((n, 1)) > np.cumsum(w, axis=1)).sum(axis=1)
    pick = np.minimum(pick, oversample - 1)
    return q[np.arange(n), pick]


def sample_batch(rng: np.random.Generator, n: int, cfg: LegConfig, **kw):
    """(target positions, leg lengths, the joint angles that produced them)."""
    tolerance = kw.pop("tolerance", None)
    lengths = sample_lengths(rng, n, cfg, tolerance)
    q = sample_joint_angles(rng, lengths, cfg, **kw)
    return forward(q, lengths), lengths, q
