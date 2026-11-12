"""Mechanical fit of the first robot, part 2: servo-box clearance in MuJoCo, and whether the electronics fit in the body.

Servo boxes (listed 40 x 20 x 40.5 mm, shaft offset assumed) in the layout of `mojito/robot/fit.py` are swept over the joint
range, one leg at a time, and over the commanded trot and crawl; the smallest gap between boxes on different bodies is
reported (negative is overlap). The battery and electronics of `mojito/robot/budget.py` are shelf-packed into the space between
the plates and the front and rear abduction servos. All placeholder; needs mujoco.

    results/body/fit_check.json

    uv run --extra sim python scripts/body/fit_check.py [--points 9]
"""
import argparse
import json

from mojito.geometry.placeholders import SIM_QUADRUPED
from mojito.results import results_path, write_result
from mojito.robot import budget, fit, gaits
from mojito.sim import clearance

GAIT_CASES = (  # (gait, velocity m/s, yaw rate rad/s): the commands the physics replays use
    (gaits.TROT, (0.15, 0.0), 0.0), (gaits.TROT, (0.0, 0.10), 0.0), (gaits.TROT, (0.0, 0.0), 0.6),
    (gaits.CRAWL, (0.10, 0.0), 0.0), (gaits.CRAWL, (0.0, 0.0), 0.3),
)


def rows(pairs, key):
    """The pairs closer than the distance cap, nearest first, in millimetres."""
    near = sorted(((k, v) for k, v in pairs.items() if v["min_gap_m"] < clearance.DISTANCE_CAP), key=lambda kv: kv[1]["min_gap_m"])
    return [{"boxes": list(k), "min_gap_mm": round(v["min_gap_m"] * 1e3, 1), key: v[key]} for k, v in near]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, default=9, help="grid points per joint for the joint-range sweep")
    args = ap.parse_args()
    design = SIM_QUADRUPED
    result = {
        "what": "servo-box clearance in MuJoCo and electronics packing. Placeholder layout and sizes: box size is a retailer listing "
                "(unverified), shaft position, which side the servo bodies sit on and every electronics size are my assumptions",
        "servo_box_m": {"length": fit.ServoSpec().length, "width": fit.ServoSpec().width, "height": fit.ServoSpec().height,
                        "shaft_offset": fit.ServoSpec().shaft_offset},
        "cap_mm": clearance.DISTANCE_CAP * 1e3, "layouts": {},
    }
    for label, layout in (("hip_servo_body_outboard (default)", fit.ServoLayout()),
                          ("hip_servo_body_inboard", fit.ServoLayout(hip_body_inward=True))):
        sweep = clearance.sweep(design, layout, points=args.points)
        contact = clearance.first_contact(design, layout)
        gait_rows = {}
        for gait, velocity, yaw in GAIT_CASES:
            swept = clearance.gait_sweep(gait, velocity, yaw, design, layout)
            gait_rows[f"{gait.name} v={list(velocity)} yaw={yaw}"] = rows(swept["pairs"], "phase")[:3]
        result["layouts"][label] = {
            "layout": layout.__dict__, "joint_range_poses_per_leg": args.points ** 3,
            "joint_range_pairs_near": rows(sweep["pairs"], "pose"),
            "front_left_abduction_vs_hip_servo_first_contact_rad": {k: v for k, v in contact.items()},
            "gait_nearest_pairs": gait_rows,
        }
    parts = [p for p in budget.default_parts(design) if p.where == "body" and p.size and p.group != "servo" and not p.name.startswith("Hall")]  # servos and Hall boards sit at the joints
    items = [(p.name, p.size) for p in parts for _ in range(p.count)]
    free = fit.free_body_space(design)
    packed = fit.pack_in_box(items, free)
    result["electronics_in_body"] = {
        "free_space_mm": [round(x * 1e3, 1) for x in free], "parts": [name for name, _ in items],
        "fits": packed["fits"], "unplaced": packed["unplaced"], "footprint_fraction": round(packed["footprint_fraction"], 3),
        "volume_fraction": round(sum(a * b * c for _, (a, b, c) in items) / (free[0] * free[1] * free[2]), 3),
        "not_counted": "wiring, connectors, standoffs, cable room, airflow, a second layer",
    }
    write_result(results_path("body", "fit_check.json"), result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
