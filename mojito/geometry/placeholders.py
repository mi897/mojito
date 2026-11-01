"""Placeholder dimensions of the example robots, in one place.

No robot exists yet: every number here is a stand-in until the owner supplies
real ones (`CLAUDE.md`, open question 1). The generators in `robots.py` write
them into the URDFs in `robots/`, which are then the source of truth for
geometry; `mojito.robot.leg` and `mojito.robot.body` re-export the same
values for the legacy 3-joint leg model and the quadruped body.

Two placeholder quadrupeds are written here: the Stage 0-1 one (`NOMINAL_LENGTHS`, `BODY_LENGTH`, `BODY_WIDTH`,
`STAND_HEIGHT`, also as `STAGE1_QUADRUPED`), and the larger one the Stage 2 simulator uses (`SIM_QUADRUPED`, with
masses). Both are `QuadrupedDesign`s; the simulation functions take `design=` and default to `SIM_QUADRUPED`.

Units: metres, radians and kilograms.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

# Leg link lengths: (abduction offset, upper leg, lower leg).
NOMINAL_LENGTHS = np.array([0.04, 0.12, 0.12])

# Leg joint limits, rows = joints (abduction, hip pitch, knee), columns = (low, high).
# The knee only bends one way and stops short of straight, which keeps the
# inverse kinematics unique and away from the full-extension singularity.
JOINT_LIMITS = np.array(
    [
        [-0.6, 0.6],  # abduction
        [0.0, 1.5],  # hip pitch
        [-2.3, -0.3],  # knee
    ]
)

# Length scale that makes leg positions dimensionless: upper + lower leg.
LEG_REACH = float(NOMINAL_LENGTHS[1] + NOMINAL_LENGTHS[2])

# Quadruped body: shoulder spacing.
BODY_LENGTH = 0.24  # front shoulders to rear shoulders
BODY_WIDTH = 0.10  # left shoulders to right shoulders

# Shoulder height above the ground when the Stage 0-1 quadruped stands (feet under the hips).
STAND_HEIGHT = 0.17


@dataclass(frozen=True)
class QuadrupedDesign:
    """One set of placeholder dimensions (and, for physics, masses) of the four-legged robot. Metres and kilograms.

    abduction_offset, upper_length, lower_length   the leg, the same for all four legs (`lengths` gives them as (3,))
    body_length, body_width                         shoulder spacing: front to rear, left to right
    stance_height                                   shoulder height above the ground when standing, feet under the hips
    masses                                          kg by role: "body", and per leg "hip", "thigh", "shank", "foot", the
                                                    roles `mojito.sim.mjcf` builds the model with; None for a design that is
                                                    geometry only. Do not change the dict in place: use `dataclasses.replace`.
    """
    abduction_offset: float
    upper_length: float
    lower_length: float
    body_length: float
    body_width: float
    stance_height: float
    masses: dict | None = field(default=None, hash=False)

    @property
    def lengths(self) -> np.ndarray:
        """(3,) abduction offset, upper, lower length."""
        return np.array([self.abduction_offset, self.upper_length, self.lower_length])

    @property
    def total_mass(self) -> float:
        """Body plus four legs, kg: masses["body"] + 4 (hip + thigh + shank + foot)."""
        return self.masses["body"] + 4 * sum(mass for role, mass in self.masses.items() if role != "body")


# The Stage 0-1 quadruped (geometry only: the leg network, the body and gait demos), from the numbers above.
STAGE1_QUADRUPED = QuadrupedDesign(
    abduction_offset=float(NOMINAL_LENGTHS[0]), upper_length=float(NOMINAL_LENGTHS[1]),
    lower_length=float(NOMINAL_LENGTHS[2]), body_length=BODY_LENGTH, body_width=BODY_WIDTH, stance_height=STAND_HEIGHT)

# The Stage 2 simulator's quadruped (docs/STAGE2.md, "Sizing"): legs and body length 1.25 times the Stage 0-1 ones, so the
# listed servos are not oversized against the legs, and a wider body. The stance height is the one the stand test and the
# crawl use; the tuned trot stands lower (`mojito.sim.gait_replay.TROT_STANCE_HEIGHT`). Servos are listed at 60 g.
SIM_QUADRUPED = QuadrupedDesign(
    abduction_offset=0.05, upper_length=0.15, lower_length=0.15, body_length=0.30, body_width=0.15, stance_height=0.2125,
    masses={"body": 0.8, "hip": 0.06, "thigh": 0.10, "shank": 0.06, "foot": 0.01})
