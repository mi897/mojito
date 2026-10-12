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


def _snapshot(d) -> dict:
    """Side-view drawing data: shoulder, knee and foot of the left legs, and the body pose."""
    return {"legs": {p: np.array([d.body(f"{p}_hip").xpos, d.body(f"{p}_shank").xpos, d.site(f"{p}_foot").xpos])
                     for p in ("FL", "RL")},
            "origin": d.site("body_origin").xpos.copy(), "R": d.site("body_origin").xmat.reshape(3, 3).copy()}


def _open(lengths, height, body_length, body_width, masses, timestep):
    """Actuated MuJoCo model standing at `height` (feet under the hips), plus index lookups."""
    import mujoco

    robot = body.Robot(body.AnalyticSolver(), lengths=lengths, body_length=body_length, body_width=body_width)
    q0 = robot.solve(robot.stance(height))
    model = robots.quadruped_urdf(body_length, body_width, lengths=lengths)
    start_z = height + mjcf.FOOT_RADIUS  # the pad centre is the foot tip; its lowest point is one radius lower
    m = mujoco.MjModel.from_xml_string(
        mjcf.mjcf_from_urdf(model, masses=masses, height=start_z, timestep=timestep, actuated=True))
    d = mujoco.MjData(m)
    names = per_leg_names()
    mjcf.set_pose(m, d, names, q0.ravel(), base_pos=(0.0, 0.0, start_z))
    jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    qadr, dadr = m.jnt_qposadr[jid], m.jnt_dofadr[jid]
    aid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in names]
    pad = {mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"{p}_foot_pad"): i for i, p in enumerate(body.LEG_NAMES)}
    return robot, m, d, qadr, dadr, aid, pad, start_z


def run_stand(lengths=(0.05, 0.15, 0.15), height=0.2125, body_length=0.30, body_width=0.15, masses=None,
              servo=None, duration=3.0, window=0.5, timestep=0.002, record_every=0) -> dict:
    """Stand on four feet and report the last `window` seconds.

    lengths  (3,) abduction offset, upper, lower; all four legs the same.
    masses   overrides for mjcf.DEFAULT_MASSES.
    servo    an actuators.ServoParams (default: 5 V, listed stall torque).
    "stood" means the body stayed within 20 degrees of upright and within 50 mm of the commanded height.
    Returns arrays in leg order FL, FR, RL, RR with joints (abduction, hip pitch, knee).
    """
    import mujoco

    lengths = np.asarray(lengths, float)
    robot, m, d, qadr, dadr, aid, pad, start_z = _open(lengths, height, body_length, body_width, masses, timestep)
    q_cmd = robot.solve(robot.stance(height))  # (4, 3)
    sv = actuators.Servo(servo or actuators.ServoParams())
    sv.reset()
    target = q_cmd.ravel()
    steps, win = int(round(duration / timestep)), int(round(window / timestep))
    tau_log, q_log, load_log, z_log, sat_log, tilt_log = [], [], [], [], [], []
    f6 = np.zeros(6)
    run_peak = 0.0
    frames = []
    trace = []
    for k in range(steps):
        tau = sv.step(target, d.qpos[qadr], d.qvel[dadr], timestep)
        d.ctrl[aid] = tau
        mujoco.mj_step(m, d)
        run_peak = max(run_peak, float(np.abs(tau).max()))
        trace.append(float(np.abs(tau).max()) / sv.p.stall_torque)
        if record_every and k % record_every == 0:
            frames.append(_snapshot(d))
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
        "frames": frames,
        "torque_trace": np.array(trace)[::record_every] if record_every else np.array(trace),  # largest |torque| / limit per step
    }


def smoothstep(x):
    """0 to 1 with zero slope at both ends (minimum-jerk profile)."""
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


def run_sit_to_stand(sit_height=0.14, stand_height=0.2125, rise_time=2.0, settle=1.0, hold=1.0,
                     lengths=(0.05, 0.15, 0.15), body_length=0.30, body_width=0.15, masses=None,
                     servo=None, timestep=0.002, record_every=0) -> dict:
    """Start crouched at `sit_height`, hold `settle` s, rise to `stand_height` over `rise_time` s, hold `hold` s.

    The feet stay under the hips; the commanded height follows a minimum-jerk profile and the joint
    targets come from the closed-form IK at each step. "Sitting" here is a deep crouch: the placeholder
    knee limit (-2.3 rad) stops the 150 mm legs from going lower than about 0.13 m.
    Succeeds ("stood") if the body ends within 50 mm of the target height (the servo position loop sags a few
    millimetres to centimetres under load, see run_stand) and never tilted past 20 degrees.
    """
    import mujoco

    lengths = np.asarray(lengths, float)
    robot, m, d, qadr, dadr, aid, pad, start_z = _open(lengths, sit_height, body_length, body_width, masses, timestep)
    if not robot.reachable(robot.stance(sit_height)).all():
        raise ValueError(f"sit height {sit_height} m is outside the joint limits")
    sv = actuators.Servo(servo or actuators.ServoParams())
    sv.reset()
    limit = sv.p.stall_torque
    t_end = settle + rise_time + hold
    steps = int(round(t_end / timestep))
    cache = {}
    t_log, tau_log, z_log, tilt_log, sat_log, h_log = [], [], [], [], [], []
    frames = []
    for k in range(steps):
        t = k * timestep
        h = sit_height + (stand_height - sit_height) * smoothstep((t - settle) / rise_time)
        key = round(h, 6)
        if key not in cache:
            cache[key] = robot.solve(robot.stance(h)).ravel()
        tau = sv.step(cache[key], d.qpos[qadr], d.qvel[dadr], timestep)
        d.ctrl[aid] = tau
        mujoco.mj_step(m, d)
        if record_every and k % record_every == 0:
            frames.append(_snapshot(d))
        t_log.append(t)
        tau_log.append(np.abs(tau).max())
        h_log.append(h)
        z_log.append(d.site("body_origin").xpos[2])
        tilt_log.append(np.degrees(np.arccos(np.clip(d.site("body_origin").xmat[8], -1.0, 1.0))))
        sat_log.append(sv.last_saturated.any())
    t_log, tau_log, z_log = np.array(t_log), np.array(tau_log), np.array(z_log)
    rising = (t_log >= settle) & (t_log <= settle + rise_time)
    final_err = float(stand_height + mjcf.FOOT_RADIUS - np.mean(z_log[-int(0.25 / timestep):]))
    return {
        "total_mass_kg": float(m.body_subtreemass[1]),
        "servo_limit_nm": limit,
        "sit_hold_peak_fraction": float(tau_log[(t_log < settle) & (t_log > 0.3 * settle)].max() / limit),
        "rise_peak_fraction": float(tau_log[rising].max() / limit),
        "rise_peak_time_s": float(t_log[rising][np.argmax(tau_log[rising])] - settle),
        "fraction_of_rise_saturated": float(np.mean(np.array(sat_log)[rising])),
        "max_tilt_deg": float(np.max(tilt_log)),
        "final_height_error_m": final_err,
        "stood": bool(final_err < 0.05 and np.max(tilt_log) < 20.0),
        "time_s": t_log, "body_z_m": z_log, "commanded_height_m": np.array(h_log), "peak_torque_nm": tau_log,
        "frames": frames,
    }
