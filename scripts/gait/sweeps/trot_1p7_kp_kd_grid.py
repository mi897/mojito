"""kd sweeps at 14 values of kp for the 1.7 s trot, 10 runs per cell. Writes results/gait/sweeps/servo_gains/trot_1p7_kp_kd_grid.json.

The earlier kp sweep tied kd to kp (kd = 0.5 * sqrt(kp / 30)). Here kp and kd are independent: kp 20 to 150 in steps of 10,
and for each kp a kd grid wide enough to show where the trot gets worse on both sides (0 to 3 N·m·s/rad, denser from 0.5 to 2 where
the speed falls away). Everything else is the baseline of trot_1p7_one_at_a_time.py: period 1.7 s, duty 0.5, diagonal offsets, commanded
0.15 m/s, stance height 0.19 m, swing height 30 mm, 1.72 kg. Each cell is run 10 times with the same random mass (+-5%) and latency
(18 to 22 ms) draws as that script (draw i uses seed i, the same draws in every cell); the spread is from those draws only.

    python scripts/gait/sweeps/trot_1p7_kp_kd_grid.py        # needs mujoco; 2380 runs, parallel over the CPU cores, about two hours
"""
import dataclasses
import json
import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from mojito.robot import gaits  # noqa: E402
from mojito.sim import actuators, gait_replay, mjcf  # noqa: E402

PERIOD, DUTY, SPEED, LIFT, HEIGHT = 1.7, 0.5, 0.15, 0.03, 0.19
ITERATIONS = 10
MASS_SPREAD = 0.05
LATENCY = (0.018, 0.022)
KP = [float(k) for k in range(20, 151, 10)]
KD = [0.0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 3.0]


def one(cell):
    kp, kd, iteration = cell
    rng = np.random.default_rng(iteration)
    masses = {k: v * rng.uniform(1 - MASS_SPREAD, 1 + MASS_SPREAD) for k, v in mjcf.DEFAULT_MASSES.items()}
    latency = rng.uniform(*LATENCY)
    gait = dataclasses.replace(gaits.TROT, period=PERIOD, duty=DUTY, lift=LIFT)
    r = gait_replay.run_gait(gait, (SPEED, 0.0), 0.0, cycles=3, height=HEIGHT, masses=masses,
                             servo=actuators.ServoParams(kp=kp, kd=kd, latency=latency))
    v = r["measured_velocity_m_s"][0]
    return {"kp": kp, "kd": kd, "iteration": iteration, "walked": r["walked"],
            "measured_velocity_m_s": round(v, 4), "velocity_ratio": round(v / SPEED, 3),
            "max_tilt_deg": round(r["max_tilt_deg"], 2),
            "peak_torque_fraction_of_limit": round(r["peak_torque_fraction_of_limit"], 3),
            "fraction_of_walking_saturated": round(r["fraction_of_walking_saturated"], 3),
            "feet_down_min": r["feet_down_min"], "slip_max_mm_per_step": round(r["slip_max_mm_per_step"], 2)}


def main():
    cells = [(kp, kd, i) for kp in KP for kd in KD for i in range(ITERATIONS)]
    runs = []
    with multiprocessing.Pool() as pool:
        for n, r in enumerate(pool.imap(one, cells, chunksize=1), 1):
            runs.append(r)
            if n % 100 == 0:
                print(f"{n}/{len(cells)}", flush=True)
    out = {"what": "gaits.TROT at 1.7 s, duty 0.5, commanded 0.15 m/s, 0.19 m stance, 30 mm swing; kp and kd varied independently, 10 runs "
                   "per cell, each a random draw of body masses (+-5%) and servo latency (18 to 22 ms), the same 10 draws in every cell. "
                   "Through closed-form IK and the servo model into the actuated MuJoCo model, body free. Placeholder dimensions, "
                   "masses (1.72 kg nominal), servo figures and friction.",
           "iterations": ITERATIONS, "mass_spread": MASS_SPREAD, "latency_s_range": list(LATENCY), "kp": KP, "kd": KD, "runs": runs}
    with open("results/gait/sweeps/servo_gains/trot_1p7_kp_kd_grid.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{sum(r['walked'] for r in runs)} of {len(runs)} walked")


if __name__ == "__main__":
    main()
