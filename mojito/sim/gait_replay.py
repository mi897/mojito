"""Replay a scripted gait in physics (docs/STAGE2.md, work package 4).

The Stage 1 foot trajectories (`gaits.foot_targets`, plus the crawl lean from
`gaits.body_sway`) are turned into joint targets by the closed-form IK, passed
through `actuators.Servo`, and applied to the actuated MuJoCo model on the floor.
Nothing holds the body up or moves it along: it has to push itself with its feet.

Each run stands for `stand_time` s, blends from the standing feet to the gait
over `ramp_time` s, then walks for `cycles` gait periods.

Everything is a placeholder (dimensions, masses, servo figures, friction), see docs/STAGE2.md.
"""
from __future__ import annotations

import numpy as np

from ..robot import body, gaits
from . import actuators, mjcf, stand


# Settings for the trot that did best in physics (docs/STAGE2.md, "Do the best values add up?"): 0.16 m stance, kp 50, kd 0.9.
# The swing height (22 mm) is in `gaits.TROT`. Other gaits keep the original stance height and servo defaults.
TROT_STANCE_HEIGHT = 0.16
TROT_SERVO = dict(kp=50.0, kd=0.9)
DEFAULT_STANCE_HEIGHT = 0.2125


def settings_for(gait) -> dict:
    """Keyword arguments (`height`, `servo`) for `run_gait`: the tuned ones for the trot, the original ones for any other gait.

    Keyed on the name, so variants made with `dataclasses.replace(gaits.TROT, ...)` that keep the name "trot" (period and
    swing-height sweeps) get the tuned settings; `gaits.TROT_STAGE1` is named "trot_stage1" and gets the original ones.
    """
    if gait.name == "trot":
        return {"height": TROT_STANCE_HEIGHT, "servo": actuators.ServoParams(**TROT_SERVO)}
    return {"height": DEFAULT_STANCE_HEIGHT, "servo": actuators.ServoParams()}


def _smoothstep(x):
    return stand.smoothstep(x)


def _foot_contact_and_positions(mj_model, data, pad, mujoco):
    """Per leg: normal force on the pad (N, 0 if not touching) and the foot tip in the world."""
    loads = np.zeros(4)
    wrench = np.zeros(6)
    for c in range(data.ncon):
        con = data.contact[c]
        for geom in (con.geom1, con.geom2):
            if geom in pad:
                mujoco.mj_contactForce(mj_model, data, c, wrench)
                loads[pad[geom]] += wrench[0]
    feet = np.array([data.site(f"{p}_foot").xpos for p in body.LEG_NAMES])
    return loads, feet


def run_gait(gait, velocity=(0.0, 0.0), yaw_rate=0.0, cycles=4, stand_time=1.0, ramp_time=0.5,
             lengths=(0.05, 0.15, 0.15), height=0.2125, body_length=0.30, body_width=0.15, masses=None,
             servo=None, timestep=0.002, contact_force=0.5, record_every=0) -> dict:
    """Walk `gait` at body-frame `velocity` (m/s) and `yaw_rate` (rad/s) and report what the physics did.

    A foot is "planted" at a step when it carries more than `contact_force` newtons in this step and the
    last one. Slip is the horizontal distance a planted foot moved in a step.
    The commanded foot in the body frame is (gaits.foot_targets, shifted by gaits.body_sway); the
    command blends from the standing feet over `ramp_time`.
    "walked" means: body upright within 30 degrees throughout, never lower than half the stance height,
    and the walking phase finished.
    """
    import mujoco

    lengths = np.asarray(lengths, float)
    robot, m, d, qadr, dadr, aid, pad, start_z = stand._open(
        lengths, height, body_length, body_width, masses, timestep)
    sv = actuators.Servo(servo or actuators.ServoParams())
    limit = sv.params.stall_torque
    home = robot.stance(height)
    walk_time = cycles * gait.period
    steps = int(round((stand_time + ramp_time + walk_time) / timestep))
    stand_steps = int(round(stand_time / timestep))
    step_ok = True

    log = {k: [] for k in ("t", "origin", "yaw", "tilt", "tau", "loads", "feet", "sat", "gap", "feet_body", "command", "rotation")}
    frames = []
    commanded_contact = []
    for k in range(steps):
        t = k * timestep
        t_gait = max(0.0, t - stand_time)
        blend = float(_smoothstep((t - stand_time) / ramp_time)) if ramp_time > 0 else float(t >= stand_time)
        feet, contact, _ = gaits.foot_targets(gait, t_gait, home, velocity, yaw_rate)
        commanded_contact.append(contact[0])
        if blend == 0.0:
            targets = home
        else:
            sway = gaits.body_sway(gait, t_gait, home)
            walking = body.apply_posture(feet, offset=sway)[0]
            targets = home + blend * (walking - home)
        if not robot.reachable(targets).all():
            step_ok = False
        joint_targets = robot.solve(targets).ravel()
        tau = sv.step(joint_targets, d.qpos[qadr], d.qvel[dadr], timestep)
        d.ctrl[aid] = tau
        mujoco.mj_step(m, d)
        origin = d.site("body_origin")
        loads, feet_world = _foot_contact_and_positions(m, d, pad, mujoco)
        rotation = origin.xmat.reshape(3, 3)
        log["t"].append(t)
        log["origin"].append(origin.xpos.copy())
        log["yaw"].append(np.arctan2(rotation[1, 0], rotation[0, 0]))
        log["tilt"].append(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        log["tau"].append(np.abs(tau))
        log["loads"].append(loads)
        log["feet"].append(feet_world)
        log["sat"].append(sv.last_saturated.any())
        log["feet_body"].append((feet_world - origin.xpos) @ rotation)  # R^T (foot - origin), body frame
        log["command"].append(targets.copy())
        log["rotation"].append(rotation.copy())
        log["gap"].append(np.abs(d.qpos[qadr] - joint_targets))
        if record_every and k % record_every == 0:
            frame = stand._snapshot(d)
            frame["loads"] = loads.copy()  # normal force on each pad, FL FR RL RR
            frame["legs3d"] = np.array([[d.body(f"{p}_hip").xpos, d.body(f"{p}_shank").xpos, d.site(f"{p}_foot").xpos]
                                        for p in body.LEG_NAMES])  # (4, 3, 3): shoulder, knee, foot of each leg
            frames.append(frame)

    t = np.array(log["t"])
    origin = np.array(log["origin"])
    yaw = np.unwrap(np.array(log["yaw"]))
    tilt = np.array(log["tilt"])
    tau = np.array(log["tau"])
    loads = np.array(log["loads"])
    feet = np.array(log["feet"])
    walking = t >= stand_time + ramp_time
    planted = (loads[1:] > contact_force) & (loads[:-1] > contact_force)
    slip = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)  # (T-1, 4) metres per step
    slip_walking = slip[planted & walking[1:, None]]
    first, last = np.flatnonzero(walking)[[0, -1]]
    elapsed = t[last] - t[first]
    displacement = origin[last, :2] - origin[first, :2]
    rotation_first = np.array([[np.cos(yaw[first]), -np.sin(yaw[first])], [np.sin(yaw[first]), np.cos(yaw[first])]])
    forward_lateral = rotation_first.T @ displacement  # displacement in the body frame at the start of walking
    grounded_count = (loads[walking] > contact_force).sum(1)
    stood_height = origin[stand_steps - 1, 2]
    upright = bool(tilt.max() < 30.0 and origin[:, 2].min() > 0.5 * (height + mjcf.FOOT_RADIUS))
    return {
        "gait": gait.name, "commanded_velocity_m_s": tuple(velocity), "commanded_yaw_rate_rad_s": yaw_rate,
        "total_mass_kg": float(m.body_subtreemass[1]), "servo_limit_nm": limit,
        "walked": bool(upright and step_ok),
        "targets_within_joint_limits": bool(step_ok),
        "max_tilt_deg": float(tilt.max()),
        "min_body_height_m": float(origin[:, 2].min()),
        "stood_body_height_m": float(stood_height),
        "measured_velocity_m_s": (forward_lateral / elapsed).tolist(),
        "measured_yaw_rate_rad_s": float((yaw[last] - yaw[first]) / elapsed),
        "peak_torque_fraction_of_limit": float(tau[walking].max() / limit),
        "peak_torque_fraction_whole_run": float(tau.max() / limit),
        "fraction_of_walking_saturated": float(np.mean(np.array(log["sat"])[walking])),
        "max_joint_gap_deg": float(np.degrees(np.array(log["gap"])[walking].max())),
        "feet_down_min": int(grounded_count.min()),
        "feet_down_mean": float(grounded_count.mean()),
        "slip_max_mm_per_step": float(1000 * slip_walking.max()) if slip_walking.size else 0.0,
        "slip_mean_mm_per_step": float(1000 * slip_walking.mean()) if slip_walking.size else 0.0,
        "slip_planted_steps": int(slip_walking.size),
        "timestep_s": timestep,
        "walk_start_s": stand_time + ramp_time,
        "commanded_contact": np.array(commanded_contact),
        "feet_body_actual_m": np.array(log["feet_body"]), "feet_body_command_m": np.array(log["command"]),
        "rotation": np.array(log["rotation"]),
        "time_s": t, "origin_m": origin, "yaw_rad": yaw, "tilt_deg": tilt, "torque_nm": tau, "foot_load_n": loads,
        "frames": frames,
    }
