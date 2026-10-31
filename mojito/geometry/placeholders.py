"""Placeholder dimensions of the example robots, in one place.

No robot exists yet: every number here is a stand-in until the owner supplies
real ones (`CLAUDE.md`, open question 1). The generators in `robots.py` write
them into the URDFs in `robots/`, which are then the source of truth for
geometry; `mojito.robot.leg` and `mojito.robot.body` re-export the same
values for the legacy 3-joint leg model and the quadruped body.

Units: metres and radians.
"""
from __future__ import annotations

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
