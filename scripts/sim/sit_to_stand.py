"""Sit-to-stand test: rise from a deep crouch to the standing height under gravity, through the servo model.

The feet stay under the hips; the height follows a minimum-jerk profile (see `stand.run_sit_to_stand`).
Placeholder dimensions, masses and servo figures. "Sitting" is a crouch at 140 mm: the placeholder knee
limit stops the 150 mm legs lower than about 130 mm.

    results/stand/sit_to_stand.json
    results/stand/sit_to_stand.png

    python scripts/sim/sit_to_stand.py
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.robot import statics
from mojito.sim import actuators, stand  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1fa6a0"
INK = "#333333"
LENGTHS = np.array([0.05, 0.15, 0.15])
SIT, STAND = 0.14, 0.2125
RISE_TIMES = (0.5, 1.0, 2.0, 4.0)
MASSES = (1.5, 2.0, 3.0, 4.0, 5.0)
# (kp, kd, command resolution in rad or None for the PCA9685 default 0.66 deg). Gains are placeholders, so several are
# shown; the third separates the effect of command quantisation, which a stiff gain turns into torque steps.
GAINS = ((10.0, 0.2, None), (100.0, 1.0, None), (100.0, 1.0, 1e-4))
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "stand")


def light(total):
    """All of `total` in the body, negligible legs, so the static calculation applies."""
    return {"body": total - 4 * 4e-4, "hip": 1e-4, "thigh": 1e-4, "shank": 1e-4, "foot": 1e-4}


def _name(kp, kd, res):
    return f"kp {kp:g} N.m/rad, kd {kd:g}, " + ("command resolution 0.66 deg" if res is None else f"command resolution {np.degrees(res):.3f} deg")


def row(r, mass, rise):
    return {"requested_mass_kg": mass, "rise_time_s": rise,
            "sit_hold_peak_fraction_of_limit": round(r["sit_hold_peak_fraction"], 3),
            "rise_peak_fraction_of_limit": round(r["rise_peak_fraction"], 3),
            "rise_peak_time_after_start_s": round(r["rise_peak_time_s"], 2),
            "fraction_of_rise_with_a_servo_saturated": round(r["fraction_of_rise_saturated"], 2),
            "max_tilt_deg": round(r["max_tilt_deg"], 1),
            "final_height_error_mm": round(1000 * r["final_height_error_m"], 1),
            "stood": r["stood"]}


def main():
    out = {"what": "sit-to-stand in MuJoCo through the servo model, 5 V, listed stall torque (unverified); legs of "
                   "negligible mass; minimum-jerk height profile; 1 s hold crouched, rise, 1 s hold standing. "
                   "Fractions are of the 1.86 N.m stall torque; 'stood' means within 50 mm of the target height and "
                   "never tilted past 20 degrees. Not shown: anything about a real robot (masses, gains and "
                   "servo figures are placeholders), or about a lying-down start.",
           "lengths_m": LENGTHS.tolist(), "sit_height_m": SIT, "stand_height_m": STAND}

    # Static torque at the crouch against the measured hold, light robot 1.5 kg, for reference.
    r = stand.run_sit_to_stand(SIT, STAND, 2.0, masses=light(1.5))
    from mojito.robot import body
    robot = body.Robot(body.AnalyticSolver(), lengths=LENGTHS, body_length=0.30, body_width=0.15)
    q_sit, q_stand = robot.solve(robot.stance(SIT))[0], robot.solve(robot.stance(STAND))[0]
    out["static_knee_torque_per_kg_nm"] = {
        "crouched": round(float(statics.joint_torques(q_sit, LENGTHS, 9.81 / 4)[2]), 4),
        "standing": round(float(statics.joint_torques(q_stand, LENGTHS, 9.81 / 4)[2]), 4),
        "note": "equal quarter of the weight per foot, per kg of robot"}
    out["sweeps"] = {}
    series = {}
    for kp, kd, res in GAINS:
        rows = []
        for mass in MASSES:
            for rise in RISE_TIMES:
                r = stand.run_sit_to_stand(SIT, STAND, rise, masses=light(mass),
                                           servo=actuators.ServoParams(kp=kp, kd=kd, **({} if res is None else {"resolution": res})))
                rows.append(row(r, mass, rise))
                series[(kp, res, mass, rise)] = r
        out["sweeps"][_name(kp, kd, res)] = rows
    with open(os.path.join(OUT, "sit_to_stand.json"), "w") as f:
        json.dump(out, f, indent=1)
    for name, rows in out["sweeps"].items():
        print(name)
        for s in rows:
            print(f'  {s["requested_mass_kg"]:3.1f} kg rise {s["rise_time_s"]:3.1f} s  sit {s["sit_hold_peak_fraction_of_limit"]:.2f}  '
                  f'rise peak {s["rise_peak_fraction_of_limit"]:.2f}  saturated {s["fraction_of_rise_with_a_servo_saturated"]:.2f}  '
                  f'stood={s["stood"]}')

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
    for ax, (kp, kd, res) in zip(axes, GAINS):
        for mass, colour in ((1.5, AQUA), (3.0, BLUE), (4.0, ORANGE)):
            r = series[(kp, res, mass, 2.0)]
            ax.plot(r["time_s"], r["peak_torque_nm"] / r["servo_limit_nm"], color=colour, lw=1.6,
                    label=f'{mass:g} kg{"" if r["stood"] else " (did not stand)"}')
        ax.axhline(1.0, color=INK, ls="--", lw=0.8)
        ax.set_title(_name(kp, kd, res).replace(".m/rad", "·m/rad") + "\n2 s rise from t = 1 s", fontsize=9, color=INK)
        ax.set_xlabel("time [s]", color=INK)
        ax.legend(frameon=False, fontsize=8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].set_ylabel("largest joint torque / stall torque", color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "sit_to_stand.png"), dpi=130)


if __name__ == "__main__":
    main()
