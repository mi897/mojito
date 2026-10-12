"""Replay the Stage 1 trot and crawl in MuJoCo through the servo model.

Placeholder dimensions, masses (1.72 kg), servo figures (5 V, unverified), friction. Writes
results/gait_replay.json. Needs mujoco.

    python scripts/gait_replay.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import gait_replay, gaits  # noqa: E402

# (gait, forward velocity m/s, yaw rate rad/s, cycles)
CASES = [
    (gaits.TROT, 0.0, 0.0, 4), (gaits.TROT, 0.05, 0.0, 4), (gaits.TROT, 0.15, 0.0, 4), (gaits.TROT, 0.30, 0.0, 4),
    (gaits.TROT, 0.0, 1.2, 4),
    (gaits.CRAWL, 0.0, 0.0, 2), (gaits.CRAWL, 0.02, 0.0, 2), (gaits.CRAWL, 0.04, 0.0, 2), (gaits.CRAWL, 0.08, 0.0, 2),
]


def summary(r):
    return {
        "gait": r["gait"], "commanded_velocity_m_s": r["commanded_velocity_m_s"],
        "commanded_yaw_rate_rad_s": r["commanded_yaw_rate_rad_s"],
        "walked": r["walked"], "max_tilt_deg": round(r["max_tilt_deg"], 1),
        "min_body_height_mm": round(1000 * r["min_body_height_m"], 1),
        "measured_forward_lateral_velocity_m_s": [round(x, 3) for x in r["measured_velocity_m_s"]],
        "measured_yaw_rate_rad_s": round(r["measured_yaw_rate_rad_s"], 3),
        "peak_torque_fraction_of_limit": round(r["peak_torque_fraction_of_limit"], 3),
        "fraction_of_walking_saturated": round(r["fraction_of_walking_saturated"], 3),
        "max_joint_gap_deg": round(r["max_joint_gap_deg"], 1),
        "feet_down_min": r["feet_down_min"], "feet_down_mean": round(r["feet_down_mean"], 2),
        "slip_max_mm_per_step": round(r["slip_max_mm_per_step"], 2),
        "slip_mean_mm_per_step": round(r["slip_mean_mm_per_step"], 3),
    }


def main():
    out = {"what": "trot and crawl foot trajectories (gaits.py) through closed-form IK and the servo model into the "
                   "actuated MuJoCo model; the body is free. 1 s stand, 0.5 s blend, then whole gait cycles; "
                   "velocity and yaw rate are measured over the walking phase. Placeholder dimensions, masses, "
                   "servo figures and friction.",
           "lengths_m": [0.05, 0.15, 0.15], "stance_height_m": 0.2125, "timestep_s": 0.002, "runs": []}
    for gait, vx, yaw_rate, cycles in CASES:
        r = gait_replay.run_gait(gait, (vx, 0.0), yaw_rate, cycles=cycles)
        s = summary(r)
        out["runs"].append(s)
        print(f"{s['gait']:5s} v={vx:.2f} w={yaw_rate:.1f}  walked={s['walked']}  measured v={s['measured_forward_lateral_velocity_m_s']}"
              f"  yaw={s['measured_yaw_rate_rad_s']}  torque={s['peak_torque_fraction_of_limit']}  slip={s['slip_max_mm_per_step']} mm/step")
    os.makedirs("results", exist_ok=True)
    with open("results/gait_replay.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
