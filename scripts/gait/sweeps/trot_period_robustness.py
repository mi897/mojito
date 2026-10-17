"""Does the 1.7 s trot hold up away from the default servo, mass and swing height? Writes results/gait/sweeps/period/trot_period_robustness.json.

`trot_period_sweep.py` chose 1.7 s at one setting (kp 10, 1.72 kg, 30 mm swing). Here the same period sweep is repeated
over the physics lab's grid: 3 servo gains x 3 masses x 2 swing heights, trot at 0.15 m/s, duty 0.5, diagonal offsets.
Only `period` changes. One run per cell, no randomisation. The 0.5 s baseline runs 4 cycles, the others 3.

    python scripts/gait/sweeps/trot_period_robustness.py        # needs mujoco; 144 runs
"""
import dataclasses
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from mojito.robot import gaits
from mojito.sim import actuators, gait_replay, mjcf  # noqa: E402

PERIODS = (0.5, 1.0, 1.5, 1.7, 2.0, 2.5, 3.0, 4.0)
KP = {10: 0.2, 30: 0.5, 100: 1.0}  # kp -> kd, as in scripts/website/site_sweep.py
MASS_SCALE = (0.6, 1.0, 1.75)  # times the placeholder masses (1.72 kg)
LIFT = (0.03, 0.06)  # m
SPEED = 0.15  # m/s


def main():
    runs = []
    for period, kp, scale, lift in itertools.product(PERIODS, KP, MASS_SCALE, LIFT):
        gait = dataclasses.replace(gaits.TROT, period=period, lift=lift)
        masses = {k: v * scale for k, v in mjcf.DEFAULT_MASSES.items()}
        servo = actuators.ServoParams(kp=float(kp), kd=KP[kp])
        r = gait_replay.run_gait(gait, (SPEED, 0.0), 0.0, cycles=4 if period < 1.0 else 3, masses=masses, servo=servo)
        s = {"period_s": period, "kp": kp, "mass_scale": scale, "lift_m": lift, "total_mass_kg": round(r["total_mass_kg"], 2),
             "walked": r["walked"], "measured_velocity_m_s": round(r["measured_velocity_m_s"][0], 3),
             "max_tilt_deg": round(r["max_tilt_deg"], 1),
             "peak_torque_fraction_of_limit": round(r["peak_torque_fraction_of_limit"], 3),
             "fraction_of_walking_saturated": round(r["fraction_of_walking_saturated"], 3),
             "feet_down_min": r["feet_down_min"]}
        runs.append(s)
        print(f"T={period} kp={kp} m={s['total_mass_kg']} lift={lift} walked={s['walked']} v={s['measured_velocity_m_s']:+.3f} "
              f"tilt={s['max_tilt_deg']}", flush=True)
    out = {"what": "gaits.TROT with only the period changed (and swing height, servo gain and mass as listed), commanded 0.15 m/s, "
                   "through closed-form IK and the servo model into the actuated MuJoCo model, body free. Same set-up as "
                   "trot_period_sweep.json. Placeholder dimensions, masses, servo figures and friction. One run per cell.",
           "commanded_velocity_m_s": SPEED, "periods_s": list(PERIODS), "kp": list(KP), "kd": KP,
           "mass_scale": list(MASS_SCALE), "lift_m": list(LIFT), "runs": runs}
    with open("results/gait/sweeps/period/trot_period_robustness.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
