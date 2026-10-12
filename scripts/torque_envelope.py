"""How heavy can the robot be before the servos cannot hold it?

Static joint torques per newton of body weight for three stances, then the
largest total mass a given servo torque can hold. Placeholder dimensions, no
leg mass, no motion: a first bound for comparing designs, not a prediction.

    results/torque_envelope.json

    python scripts/torque_envelope.py [--height 0.17] [--body-length 0.24] [--body-width 0.10]
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import gaits, leg, statics  # noqa: E402

KGCM = 0.0980665  # N*m per kg*cm
# Stall torque from retailer listings for the DS3218 (unverified, see docs/STAGE2.md):
# 19 kg*cm at 5 V and 21.5 kg*cm at 6.8 V. Between them a straight line; the supply is
# kept within 5 to 6.5 V, so the model never needs a value outside the listed points.
LISTED_V, LISTED_KGCM = (5.0, 6.8), (19.0, 21.5)
SERVO_KGCM = {f"{v} V": float(np.interp(v, LISTED_V, LISTED_KGCM)) for v in (5.0, 6.5)}
DERATES = (1.0, 0.5)
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
    ap.add_argument("--lengths", type=float, nargs=3, default=leg.NOMINAL_LENGTHS.tolist())
    ap.add_argument("--height", type=float, default=0.17)
    ap.add_argument("--body-length", type=float, default=0.24)
    ap.add_argument("--body-width", type=float, default=0.10)
    args = ap.parse_args()
    lengths = np.array(args.lengths)

    result = {
        "what": "static joint torque per unit body weight, vertical foot loads only; no leg or servo mass, no motion, no friction limit",
        "lengths_m": lengths.tolist(), "stand_height_m": args.height,
        "body_length_m": args.body_length, "body_width_m": args.body_width,
        "servo_stall_nm_unverified": {k: round(v * KGCM, 3) for k, v in SERVO_KGCM.items()},
        "derates": list(DERATES),
    }
    per, min_load = cases(lengths, args.height, args.body_length, args.body_width)
    result["crawl_min_foot_load_fraction"] = float(min_load)
    result["torque_per_weight_m"] = {c: dict(zip(NAMES, map(float, t))) for c, t in per.items()}
    result["max_total_mass_kg"] = {
        c: {f"{v} derate {d}": round(max_mass(t, kg * KGCM * d), 2) for v, kg in SERVO_KGCM.items() for d in DERATES}
        for c, t in per.items()
    }

    # Scale all link lengths and the stand height together.
    scale = {}
    for s in (0.75, 1.0, 1.25, 1.5, 2.0):
        p, _ = cases(lengths * s, args.height * s, args.body_length, args.body_width)
        scale[str(s)] = {c: round(max_mass(t, SERVO_KGCM["5.0 V"] * KGCM * 0.5), 2) for c, t in p.items()}
    result["max_mass_kg_by_leg_scale_5V_derate_0.5"] = scale

    # Body length to width ratio at fixed width, 5 V, derate 0.5.
    ratio = {}
    for r in (1.0, 1.5, 2.0, 2.4):
        p, ml = cases(lengths, args.height, args.body_width * r, args.body_width)
        ratio[str(r)] = {"crawl_max_mass_kg": round(max_mass(p["crawl_3_feet"], SERVO_KGCM["5.0 V"] * KGCM * 0.5), 2),
                         "crawl_min_foot_load_fraction": round(float(ml), 3)}
    result["body_length_over_width_effect"] = ratio

    os.makedirs("results", exist_ok=True)
    with open("results/torque_envelope.json", "w") as f:
        json.dump(result, f, indent=1)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
