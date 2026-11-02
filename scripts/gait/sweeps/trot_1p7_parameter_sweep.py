"""Which other parameters matter for the 1.7 s trot? Writes results/gait/sweeps/parameters/trot_1p7_parameter_sweep.json.

Period stays 1.7 s (offsets diagonal). Varied together: duty, commanded speed, stance height, servo gain and swing
height. Mass stays at the placeholder 1.72 kg (`trot_period_robustness.py` covers mass). One run per cell, 3 cycles,
no randomisation.

    uv run --extra sim python scripts/gait/sweeps/trot_1p7_parameter_sweep.py        # needs mujoco; 192 runs, parallel over the CPU cores
"""
import dataclasses
import itertools
import multiprocessing

from mojito.results import results_path, write_result
from mojito.robot import gaits
from mojito.sim import actuators, gait_replay

PERIOD = 1.7
DUTY = (0.5, 0.55, 0.6, 0.7)  # 0.5 leaves no time with four feet down; more does
SPEED = (0.05, 0.10, 0.15, 0.20)  # m/s
HEIGHT = (0.19, 0.2125, 0.23)  # m, stance height of the body
KP = {10: 0.2, 30: 0.5}  # kp -> kd, as in trot_period_robustness.py
LIFT = (0.03, 0.06)  # m


def one(cell):
    duty, speed, height, kp, lift = cell
    gait = dataclasses.replace(gaits.TROT, period=PERIOD, duty=duty, lift=lift)
    r = gait_replay.run_gait(gait, (speed, 0.0), 0.0, cycles=3, height=height,
                             servo=actuators.ServoParams(kp=float(kp), kd=KP[kp]))
    v = r["measured_velocity_m_s"][0]
    return {"duty": duty, "speed_m_s": speed, "height_m": height, "kp": kp, "lift_m": lift,
            "walked": r["walked"], "measured_velocity_m_s": round(v, 3), "velocity_ratio": round(v / speed, 2),
            "max_tilt_deg": round(r["max_tilt_deg"], 1),
            "peak_torque_fraction_of_limit": round(r["peak_torque_fraction_of_limit"], 3),
            "fraction_of_walking_saturated": round(r["fraction_of_walking_saturated"], 3),
            "feet_down_min": r["feet_down_min"], "slip_max_mm_per_step": round(r["slip_max_mm_per_step"], 2)}


def main():
    cells = list(itertools.product(DUTY, SPEED, HEIGHT, KP, LIFT))
    with multiprocessing.Pool() as pool:
        runs = pool.map(one, cells, chunksize=1)
    out = {"what": "gaits.TROT at a 1.7 s period with duty, commanded speed, stance height, servo gain and swing height "
                   "varied as listed, through closed-form IK and the servo model into the actuated MuJoCo model, body free. "
                   "Placeholder dimensions, masses (1.72 kg), servo figures and friction. One run per cell, 3 cycles.",
           "period_s": PERIOD, "duty": list(DUTY), "speed_m_s": list(SPEED), "height_m": list(HEIGHT),
           "kp": list(KP), "kd": KP, "lift_m": list(LIFT), "runs": runs}
    write_result(results_path("gait", "sweeps", "parameters", "trot_1p7_parameter_sweep.json"), out)
    print(f"{sum(r['walked'] for r in runs)} of {len(runs)} walked")


if __name__ == "__main__":
    main()
