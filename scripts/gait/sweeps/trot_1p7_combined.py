"""Do the best values from the one-at-a-time sweeps add up when used together? Writes results/gait/sweeps/combined/trot_1p7_combined.json.

Nine settings of (kp, kd, swing height, stance height), each run at four commanded speeds and three masses, 10 runs per cell:

  baseline          kp 30, kd 0.5, swing 30 mm, stance 0.19 m
  best              kp 50, kd 0.9, swing 22 mm, stance 0.16 m   (the best value of each from trot_1p7_one_at_a_time and trot_1p7_kp_kd_grid)
  best_stance_0.17  best, with the stance 0.17 m (more margin from the joint limits)
  best_baseline_gains / best_baseline_swing / best_baseline_stance   best with one group put back to the baseline
  gains_only / swing_only / stance_only                              baseline with one group changed to the best

Period 1.7 s, duty 0.5, diagonal offsets. Speeds 0.05, 0.10, 0.15, 0.20 m/s. Masses are the placeholder masses (1.72 kg) times 0.6, 1.0, 1.75.
Each run is a random draw (seed = iteration) of every body mass (times a uniform 0.95 to 1.05) and servo latency (uniform 18 to 22 ms),
the same 10 draws in every cell, so cells are compared on the same draws. The spread is from those draws only.

    python scripts/gait/sweeps/trot_1p7_combined.py        # needs mujoco; 1080 runs, parallel over the CPU cores, about an hour
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

PERIOD, DUTY = 1.7, 0.5
ITERATIONS = 10
MASS_SPREAD = 0.05
LATENCY = (0.018, 0.022)
SPEEDS = (0.05, 0.10, 0.15, 0.20)
MASS_SCALE = (0.6, 1.0, 1.75)
BASELINE = {"kp": 30.0, "kd": 0.5, "lift": 0.03, "height": 0.19}
BEST = {"kp": 50.0, "kd": 0.9, "lift": 0.022, "height": 0.16}
GAINS, SWING, STANCE = ("kp", "kd"), ("lift",), ("height",)


def mix(base, other, keys):
    out = dict(base)
    out.update({k: other[k] for k in keys})
    return out


CONFIGS = {
    "baseline": BASELINE,
    "best": BEST,
    "best_stance_0.17": {**BEST, "height": 0.17},
    "best_baseline_gains": mix(BEST, BASELINE, GAINS),
    "best_baseline_swing": mix(BEST, BASELINE, SWING),
    "best_baseline_stance": mix(BEST, BASELINE, STANCE),
    "gains_only": mix(BASELINE, BEST, GAINS),
    "swing_only": mix(BASELINE, BEST, SWING),
    "stance_only": mix(BASELINE, BEST, STANCE),
}


def one(cell):
    name, speed, scale, iteration = cell
    p = CONFIGS[name]
    rng = np.random.default_rng(iteration)
    masses = {k: v * scale * rng.uniform(1 - MASS_SPREAD, 1 + MASS_SPREAD) for k, v in mjcf.DEFAULT_MASSES.items()}
    latency = rng.uniform(*LATENCY)
    gait = dataclasses.replace(gaits.TROT, period=PERIOD, duty=DUTY, lift=p["lift"])
    r = gait_replay.run_gait(gait, (speed, 0.0), 0.0, cycles=3, height=p["height"], masses=masses,
                             servo=actuators.ServoParams(kp=p["kp"], kd=p["kd"], latency=latency))
    v = r["measured_velocity_m_s"][0]
    return {"config": name, "speed_m_s": speed, "mass_scale": scale, "iteration": iteration, "total_mass_kg": round(r["total_mass_kg"], 3),
            "walked": r["walked"], "within_joint_limits": r["targets_within_joint_limits"],
            "measured_velocity_m_s": round(v, 4), "velocity_ratio": round(v / speed, 3),
            "max_tilt_deg": round(r["max_tilt_deg"], 2),
            "peak_torque_fraction_of_limit": round(r["peak_torque_fraction_of_limit"], 3),
            "fraction_of_walking_saturated": round(r["fraction_of_walking_saturated"], 3),
            "feet_down_min": r["feet_down_min"], "slip_max_mm_per_step": round(r["slip_max_mm_per_step"], 2)}


def main():
    cells = [(n, s, m, i) for n in CONFIGS for s in SPEEDS for m in MASS_SCALE for i in range(ITERATIONS)]
    runs = []
    with multiprocessing.Pool() as pool:
        for n, r in enumerate(pool.imap(one, cells, chunksize=1), 1):
            runs.append(r)
            if n % 100 == 0:
                print(f"{n}/{len(cells)}", flush=True)
    out = {"what": "gaits.TROT at 1.7 s, duty 0.5; nine (kp, kd, swing, stance) settings at four commanded speeds and three masses, 10 runs "
                   "per cell, each a random draw of body masses (+-5%) and servo latency (18 to 22 ms), the same 10 draws in every cell. "
                   "Through closed-form IK and the servo model into the actuated MuJoCo model, body free. Placeholder dimensions, "
                   "masses (1.72 kg nominal times the mass scale), servo figures and friction.",
           "iterations": ITERATIONS, "mass_spread": MASS_SPREAD, "latency_s_range": list(LATENCY), "speeds_m_s": list(SPEEDS),
           "mass_scale": list(MASS_SCALE), "configs": CONFIGS, "runs": runs}
    with open("results/gait/sweeps/combined/trot_1p7_combined.json", "w") as f:
        json.dump(out, f, indent=1)
    print(f"{sum(r['walked'] for r in runs)} of {len(runs)} walked")


if __name__ == "__main__":
    main()
