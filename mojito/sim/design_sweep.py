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

import itertools
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


# Re-tuning the gait for a design (docs/STAGE2.md, work package 5, "Re-tuned per design").
# Grid searched per design, mass and gait: the knee bend the leg stands with (which sets the standing height), swing height,
# and servo gains. Period, duty, offsets and the stride (set by the commanded speed) stay as in gaits.TROT and gaits.CRAWL.
KNEE_GRID = (-1.30, -1.57, -1.80, -2.02, -2.25)  # rad; the placeholder limit is -2.3, the untuned crawl stands at -1.57 and the trot at -2.02
LIFT_GRID = (0.015, 0.022, 0.030)                # swing height, m
GAIN_GRID = ((10.0, 0.2), (30.0, 0.5), (50.0, 0.9))  # (kp, kd)
TUNE_SPEED = 0.10                                # commanded speed the search scores, m/s


@dataclass(frozen=True)
class GaitSettings:
    knee: float    # rad, knee angle with the foot straight below the hip (negative: bent)
    lift: float    # swing height, m
    kp: float      # N.m/rad
    kd: float      # N.m.s/rad


def settings_grid() -> list:
    return [GaitSettings(knee, lift, kp, kd) for knee, lift, (kp, kd) in itertools.product(KNEE_GRID, LIFT_GRID, GAIN_GRID)]


def default_settings(gait_name: str) -> GaitSettings:
    """The settings the untuned sweep used for `gait_name`, written as GaitSettings (for comparison)."""
    gait = GAITS[gait_name]
    base = gait_replay.settings_for(gait)
    knee = leg.analytic_ik(np.array([0.0, REFERENCE_LENGTHS[0], -base["height"]]), REFERENCE_LENGTHS)[2]
    return GaitSettings(float(knee), gait.lift, base["servo"].kp, base["servo"].kd)


def height_for_knee(design: Design, knee: float) -> float:
    """Hip-to-foot distance with the foot straight below the hip and the knee at `knee`: law of cosines, as in `scaled_height`."""
    return float(np.sqrt(design.upper_length**2 + design.lower_length**2
                         + 2 * design.upper_length * design.lower_length * np.cos(knee)))


def evaluate_walk(design: Design, gait_name: str, settings: GaitSettings, mass_scale: float, speed: float,
                  run_time: float = RUN_TIME, draw=None) -> dict:
    """One walk with `settings`; the same numbers per walk as `evaluate_design` keeps, plus the settings."""
    if draw is None:
        masses, latency = scaled_masses(mass_scale), None
    else:
        masses, latency = draw_masses_and_latency(mass_scale, draw)
    servo = actuators.ServoParams(kp=settings.kp, kd=settings.kd, **({} if latency is None else {"latency": latency}))
    gait = replace(GAITS[gait_name], lift=settings.lift)
    height = height_for_knee(design, settings.knee)
    r = gait_replay.run_gait(gait, (speed, 0.0), 0.0, cycles=(run_time - STAND_TIME - RAMP_TIME) / gait.period,
                             lengths=design.lengths, height=height, body_length=design.body_length,
                             body_width=design.body_width, masses=masses, servo=servo)
    return {"gait": gait_name, "speed_m_s": speed, "stance_height_m": height, "walked": r["walked"],
            "within_joint_limits": r["targets_within_joint_limits"],
            "velocity_ratio": r["measured_velocity_m_s"][0] / speed,
            "max_tilt_deg": r["max_tilt_deg"], "peak_torque_fraction_of_limit": r["peak_torque_fraction_of_limit"],
            "fraction_of_walking_saturated": r["fraction_of_walking_saturated"],
            "feet_down_min": r["feet_down_min"], "slip_max_mm_per_step": r["slip_max_mm_per_step"]}


def is_valid(walk: dict) -> bool:
    """Every commanded foot target inside the joint limits, body upright, positive measured speed."""
    return bool(walk["walked"] and walk["within_joint_limits"] and walk["velocity_ratio"] > 0.0)


def tuning_score(walk: dict) -> tuple:
    """Larger is better: invalid walks last, then closest to the commanded speed (ratio 1), then lowest peak torque."""
    if not is_valid(walk):
        return (0, -np.inf, 0.0)
    return (1, -abs(1.0 - walk["velocity_ratio"]), -walk["peak_torque_fraction_of_limit"])
