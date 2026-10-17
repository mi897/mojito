"""Push the best 1.7 s trot settings further, one parameter at a time, with repeats. Writes
results/gait/sweeps/parameters/trot_1p7_one_at_a_time.json.

Baseline (everything not being swept): period 1.7 s, duty 0.5, diagonal offsets, commanded 0.15 m/s, stance height 0.19 m,
swing height 30 mm, kp 30 with kd 0.5, 1.72 kg. Four sweeps, each changing only the named parameter:

  kp      15 to 150 in steps of 15 (10 values), kd tied to kp as kd = 0.5 * sqrt(kp / 30) (the baseline's 0.5 at kp 30)
  kd      0.1 to 2.0 (10 values) at kp 30
  lift    swing height 10 to 30 mm in steps of 2 mm (11 values)
  height  stance height 0.09 to 0.19 m in steps of 0.01 (11 values)

The simulation is deterministic, so repeating an identical run gives identical numbers. Each value is therefore run
10 times with a different random draw of two small perturbations (draw i uses seed i, and the same 10 draws are used at
every value, so values are compared on the same perturbations): every body mass scaled by a uniform factor in 0.95 to 1.05,
and the servo latency drawn uniformly from 18 to 22 ms (the baseline is 20 ms). The spread in the results is the spread from these
perturbations only; it is not a measurement of anything on a real robot.

    python scripts/gait/sweeps/trot_1p7_one_at_a_time.py        # needs mujoco; 420 runs, parallel over the CPU cores
"""
import dataclasses
import json
import math
import multiprocessing
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))
from mojito.robot import gaits  # noqa: E402
from mojito.sim import actuators, gait_replay, mjcf  # noqa: E402

PERIOD, DUTY, SPEED = 1.7, 0.5, 0.15
BASE = {"kp": 30.0, "kd": 0.5, "lift": 0.03, "height": 0.19}
ITERATIONS = 10
MASS_SPREAD = 0.05  # each mass times a uniform factor in 1 +- this
LATENCY = (0.018, 0.022)  # s, uniform
SWEEPS = {
    "kp": [float(k) for k in range(15, 151, 15)],
    "kd": [0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 1.0, 1.25, 1.5, 2.0],
    "lift": [round(0.010 + 0.002 * i, 3) for i in range(11)],
    "height": [round(0.09 + 0.01 * i, 2) for i in range(11)],
}


def kd_for(kp):
    return 0.5 * math.sqrt(kp / 30.0)


def one(cell):
    sweep, value, iteration = cell
    p = dict(BASE)
    p[sweep] = value
    if sweep == "kp":
        p["kd"] = kd_for(value)
    rng = np.random.default_rng(iteration)
    masses = {k: v * rng.uniform(1 - MASS_SPREAD, 1 + MASS_SPREAD) for k, v in mjcf.DEFAULT_MASSES.items()}
    latency = rng.uniform(*LATENCY)
    gait = dataclasses.replace(gaits.TROT, period=PERIOD, duty=DUTY, lift=p["lift"])
    r = gait_replay.run_gait(gait, (SPEED, 0.0), 0.0, cycles=3, height=p["height"], masses=masses,
                             servo=actuators.ServoParams(kp=p["kp"], kd=p["kd"], latency=latency))
    v = r["measured_velocity_m_s"][0]
    return {"sweep": sweep, "value": value, "iteration": iteration, "kp": p["kp"], "kd": round(p["kd"], 4),
            "lift_m": p["lift"], "height_m": p["height"], "total_mass_kg": round(r["total_mass_kg"], 3),
            "latency_s": round(latency, 4), "walked": r["walked"],
            "within_joint_limits": r["targets_within_joint_limits"],
            "measured_velocity_m_s": round(v, 4), "velocity_ratio": round(v / SPEED, 3),
            "max_tilt_deg": round(r["max_tilt_deg"], 2),
            "min_body_height_m": round(r["min_body_height_m"], 4),
            "peak_torque_fraction_of_limit": round(r["peak_torque_fraction_of_limit"], 3),
            "fraction_of_walking_saturated": round(r["fraction_of_walking_saturated"], 3),
            "feet_down_min": r["feet_down_min"], "slip_max_mm_per_step": round(r["slip_max_mm_per_step"], 2)}


def main():
    cells = [(s, v, i) for s, vals in SWEEPS.items() for v in vals for i in range(ITERATIONS)]
    with multiprocessing.Pool() as pool:
        runs = []
        for n, r in enumerate(pool.imap(one, cells, chunksize=1), 1):
            runs.append(r)
            if n % 20 == 0:
                print(f"{n}/{len(cells)}", flush=True)
    out = {"what": "gaits.TROT at 1.7 s, duty 0.5, commanded 0.15 m/s; one parameter swept at a time from the baseline, 10 runs per "
                   "value, each run a random draw of body masses (+-5%) and servo latency (18 to 22 ms); the same 10 draws at every "
                   "value. Through closed-form IK and the servo model into the actuated MuJoCo model, body free. Placeholder "
                   "dimensions, masses (1.72 kg nominal), servo figures and friction.",
           "baseline": {"period_s": PERIOD, "duty": DUTY, "speed_m_s": SPEED, "kp": BASE["kp"], "kd": BASE["kd"],
                        "lift_m": BASE["lift"], "height_m": BASE["height"]},
           "kd_rule": "kd = 0.5 * sqrt(kp / 30) in the kp sweep", "iterations": ITERATIONS,
           "mass_spread": MASS_SPREAD, "latency_s_range": list(LATENCY), "sweeps": SWEEPS, "runs": runs}
    with open("results/gait/sweeps/parameters/trot_1p7_one_at_a_time.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{sum(r['walked'] for r in runs)} of {len(runs)} walked")


if __name__ == "__main__":
    main()
