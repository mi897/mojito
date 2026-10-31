"""Stand test: hold the IK standing posture under gravity with the servo model.

Placeholder dimensions (Stage 2 design set), placeholder masses, unverified servo
figures. Writes results/stand/stand_test.json.

    uv run --extra sim python scripts/stand/stand_test.py
"""
import numpy as np

from mojito.results import defaults_of, results_path, write_result
from mojito.robot import statics
from mojito.sim import actuators, stand

STAND_DEFAULTS = defaults_of(stand.run_stand)  # the runs below change masses, servo and (section 4) height only
LENGTHS = np.array(STAND_DEFAULTS["lengths"])
CROUCH_HEIGHT, CROUCH_DURATION = 0.14, 2.0  # m, s: the crouch used by the sit-to-stand test, held for a shorter run
LEG_KG = 4e-4  # per leg in the "light legs" runs: essentially zero, so the static calculation applies


def light(total):
    """Masses with all of `total` in the body and negligible legs."""
    return {"body": total - 4 * LEG_KG, "hip": 1e-4, "thigh": 1e-4, "shank": 1e-4, "foot": 1e-4}


def summary(r):
    return {
        "total_mass_kg": round(r["total_mass_kg"], 4),
        "servo_limit_nm": round(r["servo_limit_nm"], 4),
        "peak_torque_nm_by_joint_max_over_legs": {n: round(float(r["torque_peak_nm"][:, j].max()), 4)
                                                   for j, n in enumerate(("abduction", "hip_pitch", "knee"))},
        "peak_torque_fraction_of_limit_last_window": round(r["torque_fraction_of_limit"], 4),
        "peak_torque_fraction_of_limit_whole_run": round(r["run_peak_fraction_of_limit"], 4),
        "stood": r["stood"], "max_tilt_deg_last_window": round(r["tilt_deg"], 1),
        "saturated": r["saturated"],
        "foot_load_n_FL_FR_RL_RR": [round(float(x), 3) for x in r["foot_load_n"]],
        "foot_load_share_percent_FL_FR_RL_RR": [round(100 * float(x) / r["weight_n"], 1) for x in r["foot_load_n"]],
        "max_knee_gap_deg": round(float(np.degrees(np.abs(r["gap_rad"][:, 2]).max())), 3),
        "body_height_drop_mm": round(1000 * r["body_height_drop_m"], 2),
    }


def main():
    out = {"what": "four-foot stand, IK posture commanded through the servo model in MuJoCo; "
                   f"mean over the last {STAND_DEFAULTS['window']:g} s of a {STAND_DEFAULTS['duration']:g} s run "
                   f"({CROUCH_DURATION:g} s for the crouch_{CROUCH_HEIGHT * 1000:.0f}mm runs). "
                   "Placeholder dimensions, masses and servo figures.",
           "lengths_m": LENGTHS.tolist(), "stance_height_m": STAND_DEFAULTS["height"]}

    # 1. Check against the static calculation: negligible leg mass, 1.5 kg, 5 V.
    r = stand.run_stand(masses=light(1.5))
    check = []
    for i in range(4):
        right = i % 2 == 1
        q = r["q_actual"][i] * np.array([-1.0 if right else 1.0, 1.0, 1.0])
        pred = statics.joint_torques(q, LENGTHS, r["foot_load_n"][i])
        check.append({"leg": ("FL", "FR", "RL", "RR")[i], "measured_nm": np.round(r["torque_nm"][i], 4).tolist(),
                      "static_from_measured_foot_load_nm": np.round(pred, 4).tolist()})
    even = statics.joint_torques(r["q_command"][0], LENGTHS, 1.5 * statics.G / 4)
    out["static_check_1p5kg_light_legs"] = {
        "run": summary(r), "per_leg": check,
        "even_quarter_load_prediction_nm": np.round(even, 4).tolist(),
        "note": "the feet do not share the weight equally (front/rear differ because the knees bend one way), "
                "so the equal-quarter prediction is not exact; the prediction from the measured foot loads is."}

    # 2. Default placeholder masses (servos, links), both ends of the supply range.
    out["default_masses"] = {}
    for v in (5.0, 6.0):
        r = stand.run_stand(servo=actuators.ServoParams(voltage=v))
        out["default_masses"][f"{v} V"] = summary(r)

    # 3. Mass sweep, light legs, 5 V, for the default position-loop gain and a stiffer one.
    #    kp and kd are placeholders, so the second run shows how much the answer depends on them.
    out["mass_sweep_5v_light_legs"] = {}
    for kp, kd in ((10.0, 0.2), (100.0, 1.0)):
        rows = []
        for mass in (1.0, 1.5, 2.0, 3.0, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0):
            r = stand.run_stand(masses=light(mass), servo=actuators.ServoParams(kp=kp, kd=kd))
            s = summary(r)
            s["requested_mass_kg"] = mass
            rows.append(s)
        out["mass_sweep_5v_light_legs"][f"kp {kp} N.m/rad, kd {kd}"] = rows

    # 4. Weight distribution at the crouch used by the sit-to-stand test (140 mm), light legs and default masses.
    crouch = dict(height=CROUCH_HEIGHT, duration=CROUCH_DURATION)
    out[f"crouch_{CROUCH_HEIGHT * 1000:.0f}mm"] = {"light_legs_1p5kg": summary(stand.run_stand(masses=light(1.5), **crouch)),
                                                  "default_masses": summary(stand.run_stand(**crouch))}

    write_result(results_path("stand", "stand_test.json"), out)
    for name, rows in out["mass_sweep_5v_light_legs"].items():
        print(name)
        for s in rows:
            print(f'  {s["requested_mass_kg"]:4.1f} kg  stood={s["stood"]!s:5}  whole-run peak '
                  f'{s["peak_torque_fraction_of_limit_whole_run"]:.2f} of limit  knee gap {s["max_knee_gap_deg"]} deg')


if __name__ == "__main__":
    main()
