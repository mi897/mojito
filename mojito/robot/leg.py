"""Kinematics of one 3-DoF quadruped leg.

Frame (origin at the shoulder, fixed to the body):
    x forward, y lateral (away from the body for a left leg), z up.

Joints:
    joint_angles[0] abduction  - rotation about x
    joint_angles[1] hip pitch  - rotation about y
    joint_angles[2] knee       - rotation about y, relative to the upper leg
(joint_angles is the (..., 3) argument name used throughout this module.)

Link lengths are plain arrays passed in at call time, never module constants,
so every function works on a batch of *different* legs at once.

Everything is written for the left-hand ("canonical") leg. A right-hand leg is
its mirror image: flip the sign of y going in and of the abduction angle coming out. See
`mirror_target` and `mirror_angles`.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.kinematics import array_library, as_array
from ..geometry.limb_model import LimbModel, Mirror
# Placeholder geometry, defined once in mojito/geometry/placeholders.py (which also feeds the URDF generators):
# NOMINAL_LENGTHS = (abduction offset, upper leg, lower leg) in metres, JOINT_LIMITS = (3, 2) (low, high) in radians.
from ..geometry.placeholders import JOINT_LIMITS, NOMINAL_LENGTHS


@dataclass(frozen=True)
class LegConfig:
    """Nominal leg plus how far individual legs may deviate from it. Satisfies `limb_model.LimbModel`."""

    nominal: np.ndarray = field(default_factory=NOMINAL_LENGTHS.copy)  # (3,) abduction offset, upper, lower [m]
    limits: np.ndarray = field(default_factory=JOINT_LIMITS.copy)  # (3, 2) joint limits [rad]
    length_tolerance: float = 0.10  # each length varies by +/- this fraction

    def __post_init__(self):
        # default_factory covers omitted arguments; an explicit None (as scripts/learning/train.py passes for the
        # limits) also means "the placeholder", as it always has.
        if self.nominal is None:
            object.__setattr__(self, "nominal", NOMINAL_LENGTHS.copy())
        if self.limits is None:
            object.__setattr__(self, "limits", JOINT_LIMITS.copy())

    @property
    def reach(self) -> float:
        """Length scale used to make positions dimensionless."""
        return float(self.nominal[1] + self.nominal[2])

    # The same interface as spec.LimbSpec (limb_model.LimbModel), so networks and samplers take either.
    dof = 3
    n_params = 3
    backend_agnostic = True  # forward() and jacobian() accept NumPy arrays and torch tensors

    @property
    def mirror(self) -> Mirror:
        """Right legs are left legs with y and abduction flipped."""
        return Mirror()

    def forward(self, joint_angles, link_lengths):
        return forward(joint_angles, link_lengths)

    def jacobian(self, joint_angles, link_lengths):
        return jacobian(joint_angles, link_lengths)


def _foot_terms(joint_angles, lengths):
    """The intermediates `forward` and `jacobian` share, in whichever array library the inputs use.

    Returns (xp, terms) with terms a dict of sin_abduction, cos_abduction,
    sin_lower, cos_lower, lower_length, x, extension, y, z, each (...,).
    Only sin, cos and broadcast_to are called, so NumPy arrays and torch
    tensors (differentiably) both work.
    """
    xp = array_library(joint_angles)
    joint_angles = as_array(joint_angles)
    lengths = xp.broadcast_to(as_array(lengths), joint_angles.shape)
    abduction_offset, upper_length, lower_length = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    sin_abduction, cos_abduction = xp.sin(joint_angles[..., 0]), xp.cos(joint_angles[..., 0])
    sin_hip, cos_hip = xp.sin(joint_angles[..., 1]), xp.cos(joint_angles[..., 1])
    hip_plus_knee = joint_angles[..., 1] + joint_angles[..., 2]  # absolute angle of the lower leg
    sin_lower, cos_lower = xp.sin(hip_plus_knee), xp.cos(hip_plus_knee)

    x = -upper_length * sin_hip - lower_length * sin_lower
    extension = upper_length * cos_hip + lower_length * cos_lower
    y = abduction_offset * cos_abduction + extension * sin_abduction
    z = abduction_offset * sin_abduction - extension * cos_abduction
    return xp, dict(sin_abduction=sin_abduction, cos_abduction=cos_abduction, sin_lower=sin_lower,
                    cos_lower=cos_lower, lower_length=lower_length, x=x, extension=extension, y=y, z=z)


def forward(joint_angles, lengths):
    """Foot position for the given joint angles.  joint_angles: (..., 3), lengths: (..., 3).

    With abduction a, hip pitch b, knee k, abduction offset l_a, upper length
    l_u and lower length l_l, the foot lies at

        x = -l_u sin(b) - l_l sin(b + k)             (forward/back, in the leg plane)
        e =  l_u cos(b) + l_l cos(b + k)             (extension: distance down the leg
                                                      plane, measured from the hip)
        y =  l_a cos(a) + e sin(a)
        z =  l_a sin(a) - e cos(a)

    The hip pitch and knee swing the leg in a plane that contains the x axis;
    the abduction then rotates that whole plane about x. The knee angle is
    relative to the upper leg, so the lower leg points along b + k.

    NumPy arrays give a NumPy result; torch tensors give a differentiable torch
    result (the PyTorch backend trains through this function).
    """
    xp, terms = _foot_terms(joint_angles, lengths)
    return xp.stack([terms["x"], terms["y"], terms["z"]], -1)


def jacobian(joint_angles, lengths):
    """d(foot position)/d(joint_angles), shape (..., 3, 3). Row = x, y, z; column = joint.

    Differentiating `forward` (names as there, e = extension):

        dx/da = 0                    dx/db = -e              dx/dk = -l_l cos(b + k)
        dy/da = -z                   dy/db = x sin(a)        dy/dk = -l_l sin(b + k) sin(a)
        dz/da =  y                   dz/db = -x cos(a)       dz/dk =  l_l sin(b + k) cos(a)

    The abduction column is (0, -z, y): a rotation about the x axis moves the
    foot at right angles to its offset from that axis. Tests check this
    against finite differences. Accepts NumPy arrays or torch tensors, like `forward`.
    """
    xp, terms = _foot_terms(joint_angles, lengths)
    sin_abduction, cos_abduction, lower_length = terms["sin_abduction"], terms["cos_abduction"], terms["lower_length"]
    x, extension, y, z = terms["x"], terms["extension"], terms["y"], terms["z"]
    sin_lower, cos_lower = terms["sin_lower"], terms["cos_lower"]
    if xp is np:
        # NumPy fast path: fill one preallocated array, as before the torch path existed (stacking is slower).
        jac = np.zeros(x.shape + (3, 3))
        jac[..., 0, 1] = -extension
        jac[..., 0, 2] = -lower_length * cos_lower
        jac[..., 1, 0] = -z
        jac[..., 1, 1] = x * sin_abduction
        jac[..., 1, 2] = -lower_length * sin_lower * sin_abduction
        jac[..., 2, 0] = y
        jac[..., 2, 1] = -x * cos_abduction
        jac[..., 2, 2] = lower_length * sin_lower * cos_abduction
        return jac
    # Torch: in-place writes would break autograd, so the same entries are stacked.
    zero = xp.zeros_like(x)
    rows = [
        [zero, -extension, -lower_length * cos_lower],
        [-z, x * sin_abduction, -lower_length * sin_lower * sin_abduction],
        [y, -x * cos_abduction, lower_length * sin_lower * cos_abduction],
    ]
    return xp.stack([xp.stack(row, -1) for row in rows], -2)


def joint_positions(joint_angles: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Shoulder, hip, knee and foot positions, shape (..., 4, 3). For drawing.

    Same geometry as `forward`: the hip sits at distance l_a from the shoulder
    along the abduction-rotated y axis, and the knee is the upper leg's end,
    i.e. `forward` with the lower length set to zero.
    """
    joint_angles = np.asarray(joint_angles, dtype=float)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), joint_angles.shape)
    abduction_offset, upper_length = lengths[..., 0], lengths[..., 1]
    sin_abduction, cos_abduction = np.sin(joint_angles[..., 0]), np.cos(joint_angles[..., 0])
    sin_hip, cos_hip = np.sin(joint_angles[..., 1]), np.cos(joint_angles[..., 1])
    shoulder = np.zeros(joint_angles.shape)
    hip = np.stack([np.zeros_like(abduction_offset), abduction_offset * cos_abduction, abduction_offset * sin_abduction], axis=-1)
    upper_extension = upper_length * cos_hip  # extension of the knee alone
    knee = np.stack(
        [
            -upper_length * sin_hip,
            abduction_offset * cos_abduction + upper_extension * sin_abduction,
            abduction_offset * sin_abduction - upper_extension * cos_abduction,
        ],
        axis=-1,
    )
    foot = forward(joint_angles, lengths)
    return np.stack([shoulder, hip, knee, foot], axis=-2)


def analytic_ik(target_position: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Closed-form inverse kinematics: the baseline the network is judged against.

    Returns the solution with the foot below the shoulder and the knee bending
    the way JOINT_LIMITS allows. Unreachable targets are clamped to the nearest
    reachable distance rather than returning NaN.

    Derivation, inverting `forward` for a target position (x, y, z):

    1. y and z only see the offset l_a and the extension e, so
           y^2 + z^2 = l_a^2 + e^2          =>   e = sqrt(y^2 + z^2 - l_a^2)
       The abduction is the direction of (y, z) plus the angle the offset
       subtends: a = atan2(z, y) + atan2(e, l_a).
    2. In the leg plane the foot is at (x, e), a two-link arm. The law of
       cosines gives the knee:
           cos(k) = (x^2 + e^2 - l_u^2 - l_l^2) / (2 l_u l_l)
       with the negative root, since the knee bends one way only (k <= 0).
    3. The hip pitch is the direction to the foot, atan2(-x, e), minus the
       angle the bent lower leg adds: b = atan2(-x, e) - atan2(l_l sin(k), l_u + l_l cos(k)).
    """
    target_position = np.asarray(target_position, dtype=float)
    lengths = np.broadcast_to(np.asarray(lengths, dtype=float), target_position.shape)
    abduction_offset, upper_length, lower_length = lengths[..., 0], lengths[..., 1], lengths[..., 2]
    x, y, z = target_position[..., 0], target_position[..., 1], target_position[..., 2]

    extension = np.sqrt(np.maximum(y * y + z * z - abduction_offset**2, 0.0))
    abduction = np.arctan2(z, y) + np.arctan2(extension, abduction_offset)
    cos_knee = (x * x + extension * extension - upper_length**2 - lower_length**2) / (2.0 * upper_length * lower_length)
    knee = -np.arccos(np.clip(cos_knee, -1.0, 1.0))
    hip_pitch = np.arctan2(-x, extension) - np.arctan2(
        lower_length * np.sin(knee), upper_length + lower_length * np.cos(knee)
    )
    return np.stack([abduction, hip_pitch, knee], axis=-1)


def mirror_target(target_position: np.ndarray) -> np.ndarray:
    """Map a right-leg target into the canonical left-leg frame (and back)."""
    return np.asarray(target_position, dtype=float) * np.array([1.0, -1.0, 1.0])


def mirror_angles(joint_angles: np.ndarray) -> np.ndarray:
    """Map canonical joint angles to the right-leg convention (and back)."""
    return np.asarray(joint_angles, dtype=float) * np.array([-1.0, 1.0, 1.0])


def sample_lengths(
    rng: np.random.Generator, n_legs: int, config: LimbModel, tolerance: float | None = None
) -> np.ndarray:
    """n_legs legs whose three lengths each deviate independently from nominal."""
    fraction = config.length_tolerance if tolerance is None else tolerance
    return config.nominal * (1.0 + rng.uniform(-fraction, fraction, size=(n_legs, len(config.nominal))))


def sample_joint_angles(
    rng: np.random.Generator,
    lengths: np.ndarray,
    config: LimbModel,
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
    n_legs = lengths.shape[0]
    low, high = config.limits[:, 0], config.limits[:, 1]
    if not workspace_uniform:
        return rng.uniform(low, high, size=(n_legs, config.dof))

    candidates = rng.uniform(low, high, size=(n_legs, oversample, config.dof))
    jac = config.jacobian(candidates, lengths[:, None, :])
    if config.dof == 3:
        volume_weight = np.abs(np.linalg.det(jac))
    else:  # volume swept per unit of joint motion, for chains that are not square: sqrt(det(J^T J)) or sqrt(det(J J^T))
        jac_t = np.swapaxes(jac, -1, -2)
        gram = jac_t @ jac if config.dof < 3 else jac @ jac_t
        volume_weight = np.sqrt(np.maximum(np.linalg.det(gram), 0.0))
    probability = volume_weight / volume_weight.sum(axis=1, keepdims=True)
    # Inverse-CDF draw of one candidate per leg, with probability proportional to its weight.
    chosen = (rng.random((n_legs, 1)) > np.cumsum(probability, axis=1)).sum(axis=1)
    chosen = np.minimum(chosen, oversample - 1)
    return candidates[np.arange(n_legs), chosen]


def sample_batch(rng: np.random.Generator, n_samples: int, config: LimbModel, **sampler_options):
    """(target positions, leg lengths, the joint angles that produced them)."""
    tolerance = sampler_options.pop("tolerance", None)
    lengths = sample_lengths(rng, n_samples, config, tolerance)
    joint_angles = sample_joint_angles(rng, lengths, config, **sampler_options)
    return config.forward(joint_angles, lengths), lengths, joint_angles


def is_leg(config: LimbModel) -> bool:
    """True if `config` (a LegConfig or a LimbSpec) has exactly this module's 3-joint leg kinematics.

    Lets the closed-form IK and the leg-specific plots be used on URDF-derived limbs that turn out to be this leg.
    """
    if isinstance(config, LegConfig):
        return True
    if (config.dof, config.n_params) != (3, 3):
        return False
    rng = np.random.default_rng(0)
    joint_angles = rng.uniform(JOINT_LIMITS[:, 0], JOINT_LIMITS[:, 1], size=(16, 3))
    lengths = NOMINAL_LENGTHS * (1 + rng.uniform(-0.1, 0.1, size=(16, 3)))
    return bool(np.allclose(config.forward(joint_angles, lengths), forward(joint_angles, lengths), atol=1e-9))
