"""Stand test: hold the IK standing posture under gravity with the servo model (docs/STAGE2.md, work package 3).

Builds the actuated MuJoCo model for a set of dimensions, commands the standing
joint angles from the closed-form IK through `actuators.Servo`, runs, and
reports per-joint torque, command-versus-actual gap, foot loads and body height.

Everything is a placeholder (dimensions, masses, servo figures), see docs/STAGE2.md.
"""
from __future__ import annotations

import numpy as np

from . import actuators, body, mjcf, robots


def per_leg_names():
    return [f"{p}_{j}" for p in body.LEG_NAMES for j in ("abduction", "hip_pitch", "knee")]


def run_stand(lengths=(0.05, 0.15, 0.15), height=0.2125, body_length=0.30, body_width=0.15, masses=None,
              servo=None, duration=3.0, window=0.5, timestep=0.002) -> dict:
    """Stand on four feet and report the last `window` seconds.

    lengths  (3,) abduction offset, upper, lower; all four legs the same.
    masses   overrides for mjcf.DEFAULT_MASSES.
    servo    an actuators.ServoParams (default: 5 V, listed stall torque).
    "stood" means the body stayed within 20 degrees of upright and within 50 mm of the commanded height.
    Returns arrays in leg order FL, FR, RL, RR with joints (abduction, hip pitch, knee).
    """
    import mujoco

    lengths = np.asarray(lengths, float)
    robot = body.Robot(body.AnalyticSolver(), lengths=lengths, body_length=body_length, body_width=body_width)
    q_cmd = robot.solve(robot.stance(height))  # (4, 3)
    model = robots.quadruped_urdf(body_length, body_width, lengths=lengths)
    start_z = height + mjcf.FOOT_RADIUS  # the pad centre is the foot tip; its lowest point is one radius lower
    m = mujoco.MjModel.from_xml_string(
        mjcf.mjcf_from_urdf(model, masses=masses, height=start_z, timestep=timestep, actuated=True))
    d = mujoco.MjData(m)
    names = per_leg_names()
    mjcf.set_pose(m, d, names, q_cmd.ravel(), base_pos=(0.0, 0.0, start_z))
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
    pad = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{p}_foot_pad"): i for i, p in enumerate(body.LEG_NAMES)}

    sv = actuators.Servo(servo or actuators.ServoParams())
    sv.reset()
    target = q_cmd.ravel()
    steps, win = int(round(duration / timestep)), int(round(window / timestep))
    tau_log, q_log, load_log, z_log, sat_log, tilt_log = [], [], [], [], [], []
    f6 = np.zeros(6)
    run_peak = 0.0
    for k in range(steps):
        tau = sv.step(target, d.qpos[qadr], d.qvel[dadr], timestep)
        d.ctrl[aid] = tau
        mujoco.mj_step(m, d)
        run_peak = max(run_peak, float(np.abs(tau).max()))
        if k >= steps - win:
            tau_log.append(tau)
            q_log.append(d.qpos[qadr].copy())
            z_log.append(d.site("body_origin").xpos[2])
            tilt_log.append(np.degrees(np.arccos(np.clip(d.site("body_origin").xmat[8], -1.0, 1.0))))
            sat_log.append(sv.last_saturated.copy())
            loads = np.zeros(4)
            for c in range(d.ncon):
                con = d.contact[c]
                for g in (con.geom1, con.geom2):
                    if g in pad:
                        mujoco.mj_contactForce(m, d, c, f6)
                        loads[pad[g]] += f6[0]
            load_log.append(loads)
    tau = np.array(tau_log).reshape(-1, 4, 3)
    q = np.array(q_log).reshape(-1, 4, 3)
    limit = sv.p.stall_torque
    mass = float(m.body_subtreemass[1])
    return {
        "total_mass_kg": mass,
        "weight_n": mass * 9.81,
        "servo_limit_nm": limit,
        "torque_nm": np.abs(tau).mean(0),         # mean |torque| per joint over the window
        "torque_peak_nm": np.abs(tau).max(0),
        "torque_fraction_of_limit": float(np.abs(tau).max() / limit),   # last window only
        "run_peak_fraction_of_limit": run_peak / limit,                   # whole run, includes the settling transient
        "saturated": bool(np.array(sat_log).any()),
        "q_command": q_cmd,
        "q_actual": q.mean(0),
        "gap_rad": (q.mean(0) - q_cmd),           # actual minus commanded
        "foot_load_n": np.array(load_log).mean(0),
        "body_height_m": float(np.mean(z_log)),
        "body_height_drop_m": float(start_z - np.mean(z_log)),  # below the height the pads touch the floor at
        "stance_height_m": height,
        "tilt_deg": float(np.max(tilt_log)),     # angle between the body's up axis and vertical
        "stood": bool(np.max(tilt_log) < 20.0 and start_z - np.mean(z_log) < 0.05),
    }
