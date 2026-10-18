"""Design sweep: stand and walk many sets of dimensions in physics (docs/STAGE2.md, work package 5).

A design is a set of leg lengths and a body size. For each design and total mass `evaluate_design` runs
`stand.run_stand` once and `gait_replay.run_gait` for the trot and the crawl at several commanded speeds,
and keeps the numbers that say whether it stood, whether it walked, and how close the servos got to stall.

Every run lasts `RUN_TIME` seconds of simulated time: the stand test holds for all of it; a walk stands for
1 s, blends for 0.5 s and walks for the rest, so the trot (1.7 s period) walks 5 cycles and the crawl (2.0 s) 4.25.

The gaits are not re-tuned for each design. Their timings (period, duty, swing height) and the servo
gains are the ones tuned on the 150 mm placeholder legs (`gait_replay.settings_for`); only the standing
height changes with the leg: each leg stands with the same knee bend as the placeholder leg does at the tuned
height, so a long or short leg is not forced to a different posture. A design that fails may therefore only need different timings.

Everything is a placeholder (dimensions, masses, servo figures, friction), see docs/STAGE2.md.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..robot import gaits, leg, statics
from . import actuators, gait_replay, mjcf, stand

REFERENCE_LENGTHS = np.array([0.05, 0.15, 0.15])  # the placeholder leg the standing heights were tuned on
RUN_TIME = 10.0              # simulated seconds per run
STAND_TIME, RAMP_TIME = 1.0, 0.5  # run_gait defaults: standing, then blending into the gait
ABDUCTION_OFFSET = 0.05      # m, the same for every design
SPEEDS = (0.05, 0.10, 0.15)  # commanded forward speeds, m/s
MASS_SCALES = (0.6, 1.0, 1.75)  # times DEFAULT_MASSES (1.72 kg), as in the trot_1p7_combined sweep
GAITS = {"trot": gaits.TROT, "crawl": gaits.CRAWL}
MASS_SPREAD = 0.05           # each role's mass times a uniform draw in 1 +- this, as in scripts/gait/sweeps/trot_1p7_combined.py
LATENCY = (0.018, 0.022)     # servo latency drawn uniformly from this range, s


@dataclass(frozen=True)
class Design:
    upper_length: float = 0.15
    lower_length: float = 0.15
    body_length: float = 0.30
    body_width: float = 0.15

    @property
    def lengths(self) -> np.ndarray:
        """(3,) abduction offset, upper, lower length in metres."""
        return np.array([ABDUCTION_OFFSET, self.upper_length, self.lower_length])

    @property
    def name(self) -> str:
        return (f"legs {1000 * self.upper_length:.0f}/{1000 * self.lower_length:.0f} mm, "
                f"body {1000 * self.body_length:.0f}x{1000 * self.body_width:.0f} mm")


def design_grid(leg_lengths=(0.10, 0.15, 0.20), body_sizes=((0.24, 0.12), (0.36, 0.18))) -> list:
    """Every pair of upper and lower length at the placeholder body, then the other body sizes at the placeholder legs."""
    designs = [Design(upper, lower) for upper in leg_lengths for lower in leg_lengths]
    designs += [Design(body_length=length, body_width=width) for length, width in body_sizes]
    return designs


def scaled_height(design: Design, base_height: float) -> float:
    """Height at which `design`'s legs stand with the knee bent as the placeholder leg's is at `base_height`.

    With the foot straight below the hip, the hip-to-foot distance in the leg plane is the height, and by the law of cosines
    height = sqrt(upper^2 + lower^2 + 2 upper lower cos(knee)), knee < 0 being the bend. Checked in
    tests/sim/test_design_sweep.py: the placeholder leg gives back `base_height`, and the knee from the IK is the same for every design.
    """
    knee = leg.analytic_ik(np.array([0.0, REFERENCE_LENGTHS[0], -base_height]), REFERENCE_LENGTHS)[2]
    return float(np.sqrt(design.upper_length**2 + design.lower_length**2
                         + 2 * design.upper_length * design.lower_length * np.cos(knee)))


def scaled_masses(scale: float) -> dict:
    """mjcf.DEFAULT_MASSES with every mass times `scale` (the body, servos and links all scale alike)."""
    return {role: mass * scale for role, mass in mjcf.DEFAULT_MASSES.items()}


def static_mass_bound(design: Design, height: float, stall_torque: float) -> float:
    """Largest total mass (kg) that four feet can hold statically at `stall_torque`: stall / (G * worst torque per unit weight).

    Same calculation as scripts/body/torque_envelope.py for the stand case (equal share per foot, no leg mass).
    Checked against that script's figure for the placeholder design in tests/sim/test_design_sweep.py.
    """
    lengths = design.lengths
    joint_angles = leg.analytic_ik(np.array([0.0, lengths[0], -height]), lengths)
    torque_per_weight = statics.joint_torques(joint_angles, lengths, 1 / 4)
    return float(stall_torque / (torque_per_weight.max() * statics.G))


def draw_masses_and_latency(mass_scale: float, draw: int) -> tuple:
    """Masses (every role times a uniform 0.95 to 1.05 and `mass_scale`) and latency for random draw number `draw`.

    The generator is seeded with `draw`, so draw k is the same in every design and cell and cells are compared on the same draws.
    """
    rng = np.random.default_rng(draw)
    masses = {role: mass * mass_scale * rng.uniform(1 - MASS_SPREAD, 1 + MASS_SPREAD)
              for role, mass in mjcf.DEFAULT_MASSES.items()}
    return masses, float(rng.uniform(*LATENCY))


def evaluate_design(design: Design, mass_scale: float = 1.0, speeds=SPEEDS, run_time: float = RUN_TIME, draw=None) -> dict:
    """Stand once and walk each gait at each speed; return a JSON-ready dict of the numbers kept per run.

    draw  None for the nominal masses and latency, or an integer: random draw number (see `draw_masses_and_latency`).
    """
    if draw is None:
        masses, latency = scaled_masses(mass_scale), None
    else:
        masses, latency = draw_masses_and_latency(mass_scale, draw)
    out = {"design": {"upper_length_m": design.upper_length, "lower_length_m": design.lower_length,
                      "abduction_offset_m": ABDUCTION_OFFSET, "body_length_m": design.body_length,
                      "body_width_m": design.body_width},
           "mass_scale": mass_scale, "draw": draw, "latency_s": latency, "walk": []}
    height = scaled_height(design, gait_replay.DEFAULT_STANCE_HEIGHT)
    servo = actuators.ServoParams() if latency is None else actuators.ServoParams(latency=latency)
    stood = stand.run_stand(design.lengths, height, design.body_length, design.body_width, masses=masses, servo=servo,
                            duration=run_time)
    out["stand"] = {"stance_height_m": height, "total_mass_kg": stood["total_mass_kg"], "stood": stood["stood"],
                    "tilt_deg": stood["tilt_deg"], "run_peak_fraction_of_limit": stood["run_peak_fraction_of_limit"],
                    "front_load_fraction": float(stood["foot_load_n"][:2].sum() / stood["foot_load_n"].sum()),
                    "static_bound_kg": static_mass_bound(design, height, servo.stall_torque)}
    for name, gait in GAITS.items():
        settings = gait_replay.settings_for(gait)
        if latency is not None:
            settings["servo"] = replace(settings["servo"], latency=latency)
        walk_height = scaled_height(design, settings["height"])
        for speed in speeds:
            r = gait_replay.run_gait(gait, (speed, 0.0), 0.0, cycles=(run_time - STAND_TIME - RAMP_TIME) / gait.period, lengths=design.lengths, height=walk_height,
                                     body_length=design.body_length, body_width=design.body_width, masses=masses,
                                     servo=settings["servo"])
            out["walk"].append({
                "gait": name, "speed_m_s": speed, "stance_height_m": walk_height, "walked": r["walked"],
                "within_joint_limits": r["targets_within_joint_limits"],
                "velocity_ratio": r["measured_velocity_m_s"][0] / speed,
                "max_tilt_deg": r["max_tilt_deg"], "peak_torque_fraction_of_limit": r["peak_torque_fraction_of_limit"],
                "fraction_of_walking_saturated": r["fraction_of_walking_saturated"],
                "feet_down_min": r["feet_down_min"], "slip_max_mm_per_step": r["slip_max_mm_per_step"]})
    return out
