"""Replay a scripted gait in physics (docs/STAGE2.md, work package 4).

The Stage 1 foot trajectories (`gaits.foot_targets`, plus the crawl lean from
`gaits.body_sway`) are turned into joint targets by the closed-form IK, passed
through `actuators.Servo`, and applied to the actuated MuJoCo model on the floor.
Nothing holds the body up or moves it along: it has to push itself with its feet.

Each run stands for `stand_time` s, blends from the standing feet to the gait
over `ramp_time` s, then walks for `cycles` gait periods. The command is open loop,
so the foot and joint targets of every step are computed before the physics runs.

Everything is a placeholder (dimensions, masses, servo figures, friction), see docs/STAGE2.md.
"""
from __future__ import annotations

import numpy as np

from ..geometry.placeholders import SIM_QUADRUPED
from ..robot import body, gaits
from . import actuators, mjcf, stand
from .robot import SimRobot


# Settings for the trot that did best in physics (docs/STAGE2.md, "Do the best values add up?"): 0.16 m stance, kp 50, kd 0.9.
# The swing height (22 mm) is in `gaits.TROT`. Other gaits keep the placeholder's stance height and the servo defaults.
TROT_STANCE_HEIGHT = 0.16
TROT_SERVO = dict(kp=50.0, kd=0.9)
DEFAULT_STANCE_HEIGHT = SIM_QUADRUPED.stance_height  # 0.2125 m, every gait but the tuned trot


def settings_for(gait) -> dict:
    """Keyword arguments (`height`, `servo`) for `run_gait`: the tuned ones for the trot, the original ones for any other gait.

    Keyed on the name, so variants made with `dataclasses.replace(gaits.TROT, ...)` that keep the name "trot" (period and
    swing-height sweeps) get the tuned settings; `gaits.TROT_STAGE1` is named "trot_stage1" and gets the original ones.
    """
    if gait.name == "trot":
        return {"height": TROT_STANCE_HEIGHT, "servo": actuators.ServoParams(**TROT_SERVO)}
    return {"height": DEFAULT_STANCE_HEIGHT, "servo": actuators.ServoParams()}


def commanded_feet(gait, home, times, stand_time, ramp_time, velocity=(0.0, 0.0), yaw_rate=0.0):
    """Body-frame foot targets (T, 4, 3) and commanded stance (T, 4) of a replay at `times` (T,) s, in one batched call.

    Before `stand_time` the feet stay at `home` (4, 3); from then the gait runs at gait time t - stand_time, and the
    command blends from home to it over `ramp_time`:
        targets = home + blend (walking - home),  walking = gaits.foot_targets shifted by -gaits.body_sway,
        blend = smoothstep((t - stand_time) / ramp_time)
    (a step with blend 0 is exactly `home`). Element by element the same arithmetic as one call per step.
    """
    times = np.asarray(times, float)
    gait_times = np.maximum(times - stand_time, 0.0)
    # The blend per step through the scalar smoothstep, as the per-step loop computed it before.
    blend = np.array([float(stand.smoothstep((time - stand_time) / ramp_time)) if ramp_time > 0 else float(time >= stand_time)
                      for time in times.tolist()])
    feet, contact, _ = gaits.foot_targets(gait, gait_times, home, velocity, yaw_rate)
    sway = gaits.body_sway(gait, gait_times, home)  # (T, 3)
    walking = body.apply_posture(feet, offset=sway)
    targets = home + blend[:, None, None] * (walking - home)
    targets[blend == 0.0] = home
    return targets, contact


def run_gait(gait, velocity=(0.0, 0.0), yaw_rate=0.0, cycles=4, stand_time=1.0, ramp_time=0.5, design=SIM_QUADRUPED,
             height=None, masses=None, servo=None, timestep=0.002, contact_force=0.5, record_every=0) -> dict:
    """Walk `gait` at body-frame `velocity` (m/s) and `yaw_rate` (rad/s) and report what the physics did.

    design  a placeholders.QuadrupedDesign; height (standing height, m) defaults to design.stance_height and masses
            (kg by role) to design.masses.
    A foot is "planted" at a step when it carries more than `contact_force` newtons in this step and the
    last one. Slip is the horizontal distance a planted foot moved in a step.
    The commanded foot in the body frame is (gaits.foot_targets, shifted by gaits.body_sway); the
    command blends from the standing feet over `ramp_time` (`commanded_feet`).
    "walked" means: body upright within 30 degrees throughout, never lower than half the stance height,
    and the walking phase finished.
    """
    sim_robot = SimRobot(design, height, masses, timestep)
    height = sim_robot.stance_height
    kinematics = sim_robot.kinematics
    servo_model = actuators.Servo(servo or actuators.ServoParams())
    limit = servo_model.params.stall_torque
    home = kinematics.stance(height)
    walk_time = cycles * gait.period
    n_steps = int(round((stand_time + ramp_time + walk_time) / timestep))
    stand_steps = int(round(stand_time / timestep))

    # The command does not depend on what the physics does: every step's targets at once.
    times = np.arange(n_steps) * timestep
    targets, commanded_contact = commanded_feet(gait, home, times, stand_time, ramp_time, velocity, yaw_rate)
    targets_within_limits = bool(kinematics.reachable(targets).all())
    joint_targets = kinematics.solve(targets).reshape(n_steps, 12)

    log = {key: [] for key in ("origin", "rotation", "yaw", "tilt", "torque", "loads", "feet", "feet_body", "saturated", "gap")}
    frames = []
    for step_index in range(n_steps):
        torques = servo_model.step(joint_targets[step_index], sim_robot.joint_angles(), sim_robot.joint_velocities(), timestep)
        sim_robot.step(torques)
        loads = sim_robot.foot_loads()
        origin, rotation, feet_world = sim_robot.origin(), sim_robot.body_rotation(), sim_robot.feet_world()
        log["origin"].append(origin)
        log["rotation"].append(rotation)
        log["yaw"].append(sim_robot.yaw())
        log["tilt"].append(sim_robot.tilt_deg())
        log["torque"].append(np.abs(torques))
        log["loads"].append(loads)
        log["feet"].append(feet_world)
        log["feet_body"].append((feet_world - origin) @ rotation)  # R^T (foot - origin), body frame
        log["saturated"].append(servo_model.last_saturated.any())
        log["gap"].append(np.abs(sim_robot.joint_angles() - joint_targets[step_index]))
        if record_every and step_index % record_every == 0:
            frame = sim_robot.snapshot()
            frame["loads"] = loads.copy()  # normal force on each pad, FL FR RL RR
            frame["legs3d"] = sim_robot.legs_world()  # (4, 3, 3): shoulder, knee, foot of each leg
            frames.append(frame)

    origin = np.array(log["origin"])
    rotation = np.array(log["rotation"])
    yaw = np.unwrap(np.array(log["yaw"]))
    tilt = np.array(log["tilt"])
    torque = np.array(log["torque"])
    loads = np.array(log["loads"])
    feet = np.array(log["feet"])
    walking = times >= stand_time + ramp_time
    planted = (loads[1:] > contact_force) & (loads[:-1] > contact_force)
    slip = np.linalg.norm(np.diff(feet[..., :2], axis=0), axis=-1)  # (T-1, 4) metres per step
    slip_walking = slip[planted & walking[1:, None]]
    first, last = np.flatnonzero(walking)[[0, -1]]
    elapsed = times[last] - times[first]
    displacement = origin[last, :2] - origin[first, :2]
    rotation_first = np.array([[np.cos(yaw[first]), -np.sin(yaw[first])], [np.sin(yaw[first]), np.cos(yaw[first])]])
    forward_lateral = rotation_first.T @ displacement  # displacement in the body frame at the start of walking
    grounded_count = (loads[walking] > contact_force).sum(1)
    stood_height = origin[stand_steps - 1, 2]
    upright = bool(tilt.max() < 30.0 and origin[:, 2].min() > 0.5 * (height + mjcf.FOOT_RADIUS))
    return {
        "gait": gait.name, "commanded_velocity_m_s": tuple(velocity), "commanded_yaw_rate_rad_s": yaw_rate,
        "total_mass_kg": sim_robot.total_mass, "servo_limit_nm": limit,
        "walked": bool(upright and targets_within_limits),
        "targets_within_joint_limits": targets_within_limits,
        "max_tilt_deg": float(tilt.max()),
        "min_body_height_m": float(origin[:, 2].min()),
        "stood_body_height_m": float(stood_height),
        "measured_velocity_m_s": (forward_lateral / elapsed).tolist(),
        "measured_yaw_rate_rad_s": float((yaw[last] - yaw[first]) / elapsed),
        "peak_torque_fraction_of_limit": float(torque[walking].max() / limit),
        "peak_torque_fraction_whole_run": float(torque.max() / limit),
        "fraction_of_walking_saturated": float(np.mean(np.array(log["saturated"])[walking])),
        "max_joint_gap_deg": float(np.degrees(np.array(log["gap"])[walking].max())),
        "feet_down_min": int(grounded_count.min()),
        "feet_down_mean": float(grounded_count.mean()),
        "slip_max_mm_per_step": float(1000 * slip_walking.max()) if slip_walking.size else 0.0,
        "slip_mean_mm_per_step": float(1000 * slip_walking.mean()) if slip_walking.size else 0.0,
        "slip_planted_steps": int(slip_walking.size),
        "timestep_s": timestep,
        "walk_start_s": stand_time + ramp_time,
        "commanded_contact": commanded_contact,
        "feet_body_actual_m": np.array(log["feet_body"]), "feet_body_command_m": targets,
        "rotation": rotation,
        "time_s": times, "origin_m": origin, "yaw_rad": yaw, "tilt_deg": tilt, "torque_nm": torque, "foot_load_n": loads,
        "frames": frames,
    }
