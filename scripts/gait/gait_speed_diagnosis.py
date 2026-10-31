"""Why does the replayed gait move slower than commanded? (docs/STAGE2.md, work package 4)

Idea: a planted foot is fixed in the world, so a planted foot whose commanded body-frame velocity is `v_foot`
asks the body to move at `-v_foot`. Averaging that ask over the loaded feet, weighted by their normal force,
gives a speed the commanded motion can deliver:

    ask = - sum_over_time_and_feet(load * commanded_foot_vx) / sum_over_time_and_feet(load)

`ask` falls below the commanded speed when feet that are commanded to swing still carry load (they ask for the
body to move backward). Whatever falls short of `ask` is the servo not following the command.
Runs variants of the servo and the gait, writes results/gait/gait_speed_diagnosis.json. Needs mujoco.

    python scripts/gait/gait_speed_diagnosis.py
"""
import dataclasses

import numpy as np

from mojito.results import defaults_of, results_path, write_result
from mojito.robot import body, gaits
from mojito.sim import actuators, gait_replay

RUN_DEFAULTS = defaults_of(gait_replay.run_gait)  # every run here keeps run_gait's timestep, stand and blend times and height
TIMESTEP = RUN_DEFAULTS["timestep"]
STAND, RAMP = RUN_DEFAULTS["stand_time"], RUN_DEFAULTS["ramp_time"]
# The diagnosis was made with the original 30 mm swing, 0.2125 m stance and kp 10 servo; `gaits.TROT` is now 22 mm (and tuned in gait_replay).
TROT_30MM = dataclasses.replace(gaits.TROT, lift=0.03)
Servo = actuators.ServoParams
IDEAL = Servo(kp=100.0, kd=1.0, latency=0.0, resolution=1e-4, torque_scale=10.0)  # not a real servo: a tracking upper bound

VARIANTS = {
    "baseline (kp 10, kd 0.2, latency 20 ms, PCA9685 steps)": dict(),
    "latency 0": dict(servo=Servo(latency=0.0)),
    "fine command resolution": dict(servo=Servo(resolution=1e-4)),
    "kp 100, kd 1": dict(servo=Servo(kp=100.0, kd=1.0)),
    "ideal servo (kp 100, kd 1, latency 0, fine, 10x stall torque)": dict(servo=IDEAL),
    "swing height 60 mm": dict(lift=0.06),
    "ideal servo, swing height 60 mm": dict(servo=IDEAL, lift=0.06),
}


def analyse(gait, velocity, **kw):
    lift = kw.pop("lift", None)
    if lift is not None:
        gait = dataclasses.replace(gait, lift=lift)
    r = gait_replay.run_gait(gait, (velocity, 0.0), cycles=3 if gait.name == "trot" else 2, **kw)
    t = r["time_s"]
    walking = t >= STAND + RAMP
    home = body.Robot(body.AnalyticSolver()).stance(RUN_DEFAULTS["height"])
    _, commanded_contact, _ = gaits.foot_targets(gait, np.maximum(t - STAND, 0.0), home, (velocity, 0.0), 0.0)
    loads = r["foot_load_n"][1:] * walking[1:, None]
    command_vx = np.diff(r["feet_body_command_m"][:, :, 0], axis=0) / TIMESTEP  # commanded body-frame foot velocity
    ask = float(-(loads * command_vx).sum() / loads.sum())
    swing_share = float((loads * ~commanded_contact[1:]).sum() / loads.sum())
    return {
        "commanded_m_s": velocity, "measured_m_s": round(r["measured_velocity_m_s"][0], 3),
        "load_weighted_ask_m_s": round(ask, 3),
        "load_share_on_feet_commanded_to_swing": round(swing_share, 3),
        "max_tilt_deg": round(r["max_tilt_deg"], 1), "walked": r["walked"],
    }


def main():
    out = {"what": "Why the replayed gait is slower than commanded. 'ask' is the body speed the loaded feet's "
                   "commanded motion would give if they stayed planted and rigid (see this script's docstring). "
                   "Placeholder dimensions, masses (1.72 kg), servo figures and friction; one run per entry.",
           "trot": {}, "crawl": {}}
    for name, kw in VARIANTS.items():
        out["trot"][name] = [analyse(TROT_30MM, v, **dict(kw)) for v in (0.0, 0.15)]
    for name in ("baseline (kp 10, kd 0.2, latency 20 ms, PCA9685 steps)", "kp 100, kd 1"):
        out["crawl"][name] = [analyse(gaits.CRAWL, v, **dict(VARIANTS[name])) for v in (0.0, 0.04)]
    for gait in ("trot", "crawl"):
        for name, rows in out[gait].items():
            print(f"{gait:5s} {name[:48]:48s}", " | ".join(
                f"cmd {x['commanded_m_s']:.2f}: measured {x['measured_m_s']:+.3f} ask {x['load_weighted_ask_m_s']:+.3f} "
                f"swing-load {x['load_share_on_feet_commanded_to_swing']:.2f}" for x in rows))
    write_result(results_path("gait", "gait_speed_diagnosis.json"), out, indent=2)


if __name__ == "__main__":
    main()
