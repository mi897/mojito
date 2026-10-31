"""The interface every limb description offers, and how mirrored limbs relate.

Two classes describe a limb to the networks and samplers: `leg.LegConfig`
(the built-in 3-joint leg) and `spec.LimbSpec` (any limb read from a URDF).
`LimbModel` writes down what they have in common, so code that takes either
can say so in its signature. `Mirror` is the sign-flip relation between a
limb and the canonical limb of its group (a right leg and a left leg).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Mirror:
    """How a limb relates to the canonical limb of its group.

    FK_this(sign_q * joint_angles, link_lengths) == flip_p * FK_canonical(joint_angles, link_lengths)

    The defaults are the left/right leg mirror: flip y of the target, flip the abduction angle.
    """

    flip_p: tuple = (1.0, -1.0, 1.0)
    sign_q: tuple = (-1.0, 1.0, 1.0)

    def target(self, target_position):
        """Map a target for this limb into the canonical limb's frame (and back)."""
        return np.asarray(target_position, dtype=float) * np.asarray(self.flip_p)

    def angles(self, joint_angles):
        """Map canonical joint values to this limb's convention (and back)."""
        return np.asarray(joint_angles, dtype=float) * np.asarray(self.sign_q)


@runtime_checkable
class LimbModel(Protocol):
    """What a network or sampler needs to know about one limb (`leg.LegConfig` or `spec.LimbSpec`)."""

    @property
    def dof(self) -> int:
        """Number of joints: the network's output size."""

    @property
    def n_params(self) -> int:
        """Number of free link lengths: with the 3 target coordinates, the network's input size."""

    @property
    def nominal(self) -> np.ndarray:
        """(n_params,) nominal link lengths in metres; inputs are scaled by them."""

    @property
    def limits(self) -> np.ndarray:
        """(dof, 2) joint limits (low, high); network outputs are squashed into them."""

    @property
    def length_tolerance(self) -> float:
        """Fractional spread of each link length around nominal that the network is trained on."""

    @property
    def reach(self) -> float:
        """Length scale in metres that makes positions dimensionless."""

    @property
    def backend_agnostic(self) -> bool:
        """True if `forward` and `jacobian` accept torch tensors as well as NumPy arrays (and stay differentiable)."""

    @property
    def mirror(self) -> Mirror | None:
        """The default mirror for the other side of the robot, or None if the limb has none of its own."""

    def forward(self, joint_angles, link_lengths):
        """Tip position (..., 3) in the limb frame for joint values (..., dof) and link lengths (..., n_params)."""

    def jacobian(self, joint_angles, link_lengths):
        """d(tip position)/d(joint values), shape (..., 3, dof)."""
