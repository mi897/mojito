"""How heavy can the robot be before the servos cannot hold it?

Static joint torques per newton of body weight for three stances, then the
largest total mass a given servo torque can hold. Placeholder dimensions, no
leg mass, no motion: a first bound for comparing designs, not a prediction.

    results/body/torque_envelope.json

    python scripts/body/torque_envelope.py [--height 0.2125] [--body-length 0.30] [--body-width 0.15]
"""
import argparse
import json

import numpy as np

from mojito.results import results_path, write_result
from mojito.robot import gaits, leg, statics
from mojito.sim import actuators  # NumPy only, no MuJoCo

# Stall torque from retailer listings for the DS3218 (unverified, see docs/STAGE2.md), as in the servo model
# (`actuators.ServoParams`): 19 kg*cm at 5 V and 21.5 kg*cm at 6.8 V, a straight line between them; the supply is
# kept within 5 to 6.0 V, so the model never needs a value outside the listed points.
SERVO_STALL_NM = {f"{v} V": actuators.ServoParams(voltage=v).stall_torque for v in (5.0, 6.0)}
DERATES = (1.0, 0.5)
# Stage 2 design placeholders (docs/STAGE2.md): 1.25x the Stage 0-1 placeholders so the
# servos (listed 40 x 20 x 40.5 mm) are not oversized against the legs. Still placeholders.
DESIGN = {"lengths": [0.05, 0.15, 0.15], "height": 0.2125, "body_length": 0.30, "body_width": 0.15}
NAMES = ("abduction", "hip_pitch", "knee")


def stance_q(lengths, height):
    """Joint angles of a left leg standing with the foot straight below the hip."""
    return leg.analytic_ik(np.array([0.0, lengths[0], -height]), lengths)


def cases(lengths, height, body_length, body_width):
    """Peak |joint torque| per joint over legs, per unit of body weight (metres), for each stance."""
    q = stance_q(lengths, height)
    hx, hy = body_length / 2, body_width / 2
    shoulders = np.array([[hx, hy], [hx, -hy], [-hx, hy], [-hx, -hy]])
    feet = shoulders + np.array([0.0, 1.0]) * np.array([lengths[0], -lengths[0], lengths[0], -lengths[0]])[:, None]
    out = {
        "stand_4_feet": statics.joint_torques(q, lengths, 1 / 4),
        "trot_2_feet": statics.joint_torques(q, lengths, 1 / 2),
    }
    # Crawl: three feet down, body leaning away from the lifted foot (gaits.CRAWL.sway).
    worst = np.zeros(3)
    min_load = np.inf
    for lifted in range(4):
        keep = [i for i in range(4) if i != lifted]
        home = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)[lifted]
        com = -np.sign(home) * gaits.CRAWL.sway  # body centre relative to the feet
        loads = statics.foot_loads(feet[keep], com)
        min_load = min(min_load, loads.min())
        for load in loads:
            worst = np.maximum(worst, statics.joint_torques(q, lengths, load))
    out["crawl_3_feet"] = worst
    return out, min_load


def max_mass(torque_per_weight, torque_nm):
    """Largest total mass (kg) whose static weight keeps every joint under torque_nm."""
    return torque_nm / (torque_per_weight.max() * statics.G)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=float, nargs=3, default=DESIGN["lengths"])
    ap.add_argument("--height", type=float, default=DESIGN["height"])
    ap.add_argument("--body-length", type=float, default=DESIGN["body_length"])
    ap.add_argument("--body-width", type=float, default=DESIGN["body_width"])
    args = ap.parse_args()
    lengths = np.array(args.lengths)

    result = {
        "what": "static joint torque per unit body weight, vertical foot loads only; no leg or servo mass, no motion, no friction limit",
        "lengths_m": lengths.tolist(), "stand_height_m": args.height,
        "body_length_m": args.body_length, "body_width_m": args.body_width,
        "servo_stall_nm_unverified": {k: round(v, 3) for k, v in SERVO_STALL_NM.items()},
        "derates": list(DERATES),
    }
    per, min_load = cases(lengths, args.height, args.body_length, args.body_width)
    result["crawl_min_foot_load_fraction"] = float(min_load)
    result["torque_per_weight_m"] = {c: dict(zip(NAMES, map(float, t))) for c, t in per.items()}
    result["max_total_mass_kg"] = {
        c: {f"{v} derate {d}": round(max_mass(t, stall * d), 2) for v, stall in SERVO_STALL_NM.items() for d in DERATES}
        for c, t in per.items()
    }

    # Scale all link lengths and the stand height together.
    scale = {}
    for s in (0.75, 1.0, 1.25, 1.5, 2.0):
        p, _ = cases(lengths * s, args.height * s, args.body_length, args.body_width)
        scale[str(s)] = {c: round(max_mass(t, SERVO_STALL_NM["5.0 V"] * 0.5), 2) for c, t in p.items()}
    result["max_mass_kg_by_leg_scale_5V_derate_0.5"] = scale

    # Body length to width ratio at fixed width, 5 V, derate 0.5.
    ratio = {}
    for r in (1.0, 1.5, 2.0, 2.4):
        p, ml = cases(lengths, args.height, args.body_width * r, args.body_width)
        ratio[str(r)] = {"crawl_max_mass_kg": round(max_mass(p["crawl_3_feet"], SERVO_STALL_NM["5.0 V"] * 0.5), 2),
                         "crawl_min_foot_load_fraction": round(float(ml), 3)}
    result["body_length_over_width_effect"] = ratio

    write_result(results_path("body", "torque_envelope.json"), result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
