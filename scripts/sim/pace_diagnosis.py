"""Why does the pace (left legs together, then right) go backward? (docs/STAGE2.md, work package 4)

Hypothesis: with only two feet on one side down, the support is a line along that side, away from the centre of
mass. The body is not held up laterally, so it rolls onto the lifted feet until they touch the floor, they carry
the load although commanded to swing, and a loaded swing foot moves forward in the body frame, pushing the body back.
Checks, each from a MuJoCo run (placeholder dimensions, masses, servo figures, friction; one run per entry):
  1. load share on feet commanded to swing, and the 'ask' of scripts/sim/gait_speed_diagnosis.py, pace against trot;
  2. roll range of the body against the roll that lifts the swing foot back to the floor, 2 atan(lift / track);
  3. whether load sits on the swing side (mean left-load share while the left legs are commanded to stand);
  4. variants: period, duty, swing height;
  5. a lateral body shift toward the stance side (the crawl's lean idea), which should unload the swing feet.

    python scripts/sim/pace_diagnosis.py        # writes results/gait/pace_diagnosis.json
"""
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.robot import gaits
from mojito.sim import actuators, gait_replay  # noqa: E402

TIMESTEP = 0.002
PACE = dataclasses.replace(gaits.TROT, name="pace", offsets=(0.0, 0.5, 0.0, 0.5))  # FL RL together, FR RR together
Servo = actuators.ServoParams
IDEAL = Servo(kp=100.0, kd=1.0, latency=0.0, resolution=1e-4, torque_scale=10.0)


def measure(gait, velocity=0.1, cycles=3, **kw):
    r = gait_replay.run_gait(gait, (velocity, 0.0), cycles=cycles, **kw)
    t = r["time_s"]
    walking = t >= r["walk_start_s"]
    loads = r["foot_load_n"][1:] * walking[1:, None]
    command_vx = np.diff(r["feet_body_command_m"][:, :, 0], axis=0) / TIMESTEP
    commanded = r["commanded_contact"]
    rotation = r["rotation"]
    roll = np.degrees(np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2]))
    left = r["foot_load_n"][:, [0, 2]].sum(1)
    right = r["foot_load_n"][:, [1, 3]].sum(1)
    left_share = left / np.maximum(left + right, 1e-9)
    left_stance = commanded[:, 0] & walking
    right_stance = ~commanded[:, 0] & walking
    return {
        "measured_m_s": round(r["measured_velocity_m_s"][0], 3),
        "load_weighted_ask_m_s": round(float(-(loads * command_vx).sum() / loads.sum()), 3),
        "load_share_on_feet_commanded_to_swing": round(float((loads * ~commanded[1:]).sum() / loads.sum()), 3),
        "roll_range_deg": round(float(np.ptp(roll[walking])), 1),
        "left_load_share_while_left_commanded_down": round(float(left_share[left_stance].mean()), 2),
        "left_load_share_while_right_commanded_down": round(float(left_share[right_stance].mean()), 2),
        "feet_commanded_down_mean": round(float(commanded[walking].sum(1).mean()), 2),
        "max_tilt_deg": round(r["max_tilt_deg"], 1), "walked": r["walked"],
        "_track_m": float(np.mean(np.abs(r["feet_body_actual_m"][walking][:, :, 1]))) * 2,
    }


def main():
    out = {"what": __doc__.strip().split("\n")[0], "velocity_commanded_m_s": 0.1}
    out["trot_vs_pace"] = {f"{g.name} {n}": measure(g, **kw)
                           for g in (gaits.TROT, PACE)
                           for n, kw in (("kp 10", {}), ("kp 100", dict(servo=Servo(kp=100.0, kd=1.0))), ("ideal servo", dict(servo=IDEAL)))}
    rows = {}
    for name, gait in (("pace period 0.5 (base)", PACE), ("period 1.0", dataclasses.replace(PACE, period=1.0)),
                       ("period 2.0", dataclasses.replace(PACE, period=2.0)), ("duty 0.6", dataclasses.replace(PACE, duty=0.6)),
                       ("duty 0.7", dataclasses.replace(PACE, duty=0.7)), ("swing height 60 mm", dataclasses.replace(PACE, lift=0.06))):
        rows[name] = measure(gait)
        rows[name]["roll_range_if_swing_foot_just_reaches_floor_deg"] = round(
            float(2 * np.degrees(np.arctan(gait.lift / rows[name]["_track_m"]))), 1)
    out["pace_variants"] = rows
    original = gaits.body_sway
    out["pace_with_lateral_shift_toward_stance_side"] = {}
    try:
        for shift in (0.03, 0.06):
            def lean(gait, times, home, shift=shift):
                times = np.atleast_1d(np.asarray(times, float))
                result = np.zeros((len(times), 3))
                phase = (times / gait.period) % 1.0
                result[:, 1] = -shift * np.tanh(3 * np.cos(2 * np.pi * (phase - 0.25)))  # sign found by trying both
                return result
            gaits.body_sway = lean
            out["pace_with_lateral_shift_toward_stance_side"][f"{shift * 1000:.0f} mm"] = measure(PACE)
    finally:
        gaits.body_sway = original
    for group in out.values():
        if isinstance(group, dict):
            for row in group.values():
                if isinstance(row, dict):
                    row.pop("_track_m", None)
    for group in ("trot_vs_pace", "pace_variants", "pace_with_lateral_shift_toward_stance_side"):
        for name, row in out[group].items():
            print(f"{name:26s} measured {row['measured_m_s']:+.3f} ask {row['load_weighted_ask_m_s']:+.3f} "
                  f"swing-load {row['load_share_on_feet_commanded_to_swing']:.2f} roll {row['roll_range_deg']:5.1f}  "
                  f"left share (left down/right down) {row['left_load_share_while_left_commanded_down']}/{row['left_load_share_while_right_commanded_down']}"
                  + (f"  roll if foot just reaches floor {row['roll_range_if_swing_foot_just_reaches_floor_deg']}" if "roll_range_if_swing_foot_just_reaches_floor_deg" in row else ""))
    with open(os.path.join(os.path.dirname(__file__), "..", "..", "results", "gait", "pace_diagnosis.json"), "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
