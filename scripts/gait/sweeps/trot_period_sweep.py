"""Trot period sweep in physics: 0.5 s (the Stage 1 trot) against 1.0 to 2.0 s. Writes results/gait/sweeps/period/trot_period_sweep.json.

Same model as scripts/gait/gait_replay.py (placeholder dimensions, masses 1.72 kg, 5 V, kp 10, kd 0.2, unverified servo
figures). Duty 0.5 and the diagonal offsets stay as in `gaits.TROT`; only `period` changes. One run per cell, no
randomisation. The baseline runs 4 cycles, the longer periods 3 (so each walks 6 s or more).

    python scripts/gait/sweeps/trot_period_sweep.py        # needs mujoco; a few minutes
"""
import dataclasses
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from mojito.robot import gaits
from mojito.sim import gait_replay  # noqa: E402

PERIODS = (0.5, 1.0, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0)
# (forward velocity m/s, yaw rate rad/s)
COMMANDS = ((0.0, 0.0), (0.05, 0.0), (0.15, 0.0), (0.30, 0.0), (0.0, 0.6), (0.0, 1.2))


def main():
    out = {"what": "gaits.TROT with only the period changed (duty 0.5, diagonal offsets), through closed-form IK and the "
                   "servo model into the actuated MuJoCo model, body free. 1 s stand, 0.5 s blend, then whole cycles "
                   "(4 at 0.5 s, 3 otherwise). Placeholder dimensions, masses, servo figures and friction. One run per cell.",
           "periods_s": list(PERIODS), "runs": []}
    for period in PERIODS:
        gait = dataclasses.replace(gaits.TROT, period=period, lift=0.03)  # 30 mm, as when the sweep was made
        for vx, yaw_rate in COMMANDS:
            r = gait_replay.run_gait(gait, (vx, 0.0), yaw_rate, cycles=4 if period < 1.0 else 3)
            s = {"period_s": period, "commanded_velocity_m_s": vx, "commanded_yaw_rate_rad_s": yaw_rate,
                 "walked": r["walked"], "measured_velocity_m_s": round(r["measured_velocity_m_s"][0], 3),
                 "measured_yaw_rate_rad_s": round(r["measured_yaw_rate_rad_s"], 3),
                 "max_tilt_deg": round(r["max_tilt_deg"], 1), "min_body_height_mm": round(1000 * r["min_body_height_m"], 1),
                 "peak_torque_fraction_of_limit": round(r["peak_torque_fraction_of_limit"], 3),
                 "fraction_of_walking_saturated": round(r["fraction_of_walking_saturated"], 3),
                 "feet_down_min": r["feet_down_min"], "feet_down_mean": round(r["feet_down_mean"], 2),
                 "slip_max_mm_per_step": round(r["slip_max_mm_per_step"], 2)}
            out["runs"].append(s)
            print(f"T={period} v={vx} w={yaw_rate} walked={s['walked']} v={s['measured_velocity_m_s']:+.3f} yaw={s['measured_yaw_rate_rad_s']:+.2f} "
                  f"tilt={s['max_tilt_deg']} torque={s['peak_torque_fraction_of_limit']} sat={s['fraction_of_walking_saturated']}", flush=True)
    with open("results/gait/sweeps/period/trot_period_sweep.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
