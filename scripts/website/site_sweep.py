"""Grid of MuJoCo gait replays for the website's physics playground (needs mujoco).

The website cannot run MuJoCo, so every combination a visitor can pick is run here, once, and the
measured result and a 25 Hz recording of the motion are stored. The page replays the nearest stored run.
Nothing is interpolated between runs, and each cell is one run with no randomisation.

Grid: gait and forward speed, servo gain (kd = kp/10 except kp 10 where kd = 0.2, as in docs/STAGE2.md),
total mass (the placeholder masses scaled together), swing height. Everything else is as in
`scripts/sim/gait_replay.py`: placeholder 50/150/150 mm legs, 300 x 150 mm body, 5 V, unverified servo figures.

    python scripts/website/site_sweep.py            # writes results/gait/physics_sweep.json (about 2 minutes)

Frame layout (25 Hz, from 0.5 s into the run): integers, to keep the file small
    [ox mm, oy mm, oz mm, roll 0.01 deg, pitch 0.01 deg, yaw 0.01 deg,
     12 numbers: body-frame foot position of FL FR RL RR in mm (x, y, z each),
     4 numbers: pad load of FL FR RL RR in 0.1 N,
     commanded stance as a 4-bit number (bit k set = leg k commanded down; FL is bit 0)]
"""
import argparse
import dataclasses
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.robot import gaits
from mojito.sim import actuators, gait_replay, mjcf  # noqa: E402

PACE = gaits.Gait("pace", period=0.5, duty=0.5, offsets=(0.0, 0.5, 0.0, 0.5))
GAITS = {  # name -> (gait, forward speeds in m/s, cycles)
    "trot": (gaits.TROT, (0.0, 0.15, 0.30), 4),
    "crawl": (gaits.CRAWL, (0.0, 0.04, 0.08), 2),
    "pace": (PACE, (0.0, 0.10), 4),
}
KP = {10: 0.2, 30: 0.5, 100: 1.0}  # kp -> kd
MASS_SCALE = (0.6, 1.0, 1.75)  # times the placeholder masses (1.72 kg)
LIFT = (0.03, 0.06)  # m
FRAME_STEP = 20  # physics steps per recorded frame: 20 x 2 ms = FRAME_DT (25 Hz)
FRAME_DT = 0.04
RECORD_FROM = 0.5  # s


def euler(rotation):
    """Roll, pitch, yaw (rad) of a body-to-world rotation R = Rz(yaw) Ry(pitch) Rx(roll)."""
    roll = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = -np.arcsin(np.clip(rotation[..., 2, 0], -1.0, 1.0))
    yaw = np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])
    return roll, pitch, yaw


def one_run(name, speed, kp, scale, lift):
    gait, _, cycles = GAITS[name]
    gait = dataclasses.replace(gait, lift=lift)
    masses = {k: v * scale for k, v in mjcf.DEFAULT_MASSES.items()}
    servo = actuators.ServoParams(kp=float(kp), kd=KP[kp])
    r = gait_replay.run_gait(gait, (speed, 0.0), 0.0, cycles=cycles, masses=masses, servo=servo)
    t, dt = r["time_s"], r["timestep_s"]
    pick = np.arange(int(round(RECORD_FROM / dt)), len(t), FRAME_STEP)
    roll, pitch, yaw = euler(r["rotation"][pick])
    contact_bits = (r["commanded_contact"][pick] * (1 << np.arange(4))).sum(1)
    frames = np.concatenate([
        1000 * r["origin_m"][pick], np.degrees(np.stack([roll, pitch, yaw], 1)) * 100,
        1000 * r["feet_body_actual_m"][pick].reshape(len(pick), 12), 10 * r["foot_load_n"][pick],
        contact_bits[:, None]], axis=1)
    return {
        "gait": name, "speed_m_s": speed, "kp": kp, "kd": KP[kp], "mass_scale": scale, "lift_m": lift,
        "total_mass_kg": round(r["total_mass_kg"], 2), "walked": r["walked"],
        "measured_speed_m_s": round(r["measured_velocity_m_s"][0], 3),
        "measured_lateral_m_s": round(r["measured_velocity_m_s"][1], 3),
        "max_tilt_deg": round(r["max_tilt_deg"], 1), "min_body_height_mm": round(1000 * r["min_body_height_m"], 1),
        "peak_torque_fraction": round(r["peak_torque_fraction_of_limit"], 3),
        "fraction_saturated": round(r["fraction_of_walking_saturated"], 3),
        "max_joint_gap_deg": round(r["max_joint_gap_deg"], 1),
        "feet_down_min": r["feet_down_min"], "feet_down_mean": round(r["feet_down_mean"], 2),
        "frames": np.rint(frames).astype(int).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default="results/gait/physics_sweep.json")
    args = parser.parse_args()
    runs = []
    for name, (_, speeds, _) in GAITS.items():
        for speed, kp, scale, lift in itertools.product(speeds, KP, MASS_SCALE, LIFT):
            run = one_run(name, speed, kp, scale, lift)
            runs.append(run)
            print(f"{name:5s} v={speed:.2f} kp={kp:3d} m={run['total_mass_kg']:.2f} lift={lift:.2f}  "
                  f"walked={run['walked']}  measured={run['measured_speed_m_s']:+.3f}  tilt={run['max_tilt_deg']}  "
                  f"torque={run['peak_torque_fraction']}", flush=True)
    out = {
        "what": "Grid of MuJoCo gait replays (mojito.sim.gait_replay.run_gait), one run per cell, no randomisation, no "
                "interpolation between cells. Placeholder 50/150/150 mm legs, 300 x 150 mm body, 5 V, unverified servo "
                "figures, placeholder friction. Mass is the placeholder masses scaled together (1.72 kg at scale 1).",
        "frame_dt_s": FRAME_DT, "record_from_s": RECORD_FROM,
        "frame_columns": "ox oy oz [mm], roll pitch yaw [0.01 deg], 12 foot positions in the body frame [mm] "
                         "(FL FR RL RR, x y z), 4 pad loads [0.1 N], commanded-stance bits (FL = bit 0)",
        "axes": {"gait": {k: list(v[1]) for k, v in GAITS.items()}, "kp": list(KP), "kd": KP,
                 "mass_scale": list(MASS_SCALE), "lift_m": list(LIFT)},
        "lengths_m": [0.05, 0.15, 0.15], "body_m": [0.30, 0.15], "stance_height_m": 0.2125,
        "runs": runs,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, separators=(",", ":"))
    print(f"{len(runs)} runs, {os.path.getsize(args.out) / 1e6:.2f} MB -> {args.out}")


if __name__ == "__main__":
    main()
