"""Stand test: hold the IK standing posture under gravity with the servo model (docs/STAGE2.md, work package 3).

Builds the actuated MuJoCo model for a design (`robot.SimRobot`), commands the standing joint angles from the
closed-form IK through `actuators.Servo`, runs, and reports per-joint torque, command-versus-actual gap, foot loads and
body height.

Everything is a placeholder (dimensions, masses, servo figures), see docs/STAGE2.md.
"""
from __future__ import annotations

import numpy as np

from ..geometry.placeholders import SIM_QUADRUPED
from ..robot import statics
from . import actuators, mjcf
from .robot import SimRobot


def run_stand(design=SIM_QUADRUPED, height=None, masses=None, servo=None, duration=3.0, window=0.5, timestep=0.002,
              record_every=0) -> dict:
    """Stand on four feet and report the last `window` seconds.

    design   a placeholders.QuadrupedDesign: leg lengths (all four legs the same), body size, masses.
    height   standing height in m (default: design.stance_height).
    masses   kg by role, overriding the design's (mjcf.DEFAULT_MASSES for the placeholder).
    servo    an actuators.ServoParams (default: 5 V, listed stall torque).
    "stood" means the body stayed within 20 degrees of upright and within 50 mm of the commanded height.
    Returns arrays in leg order FL, FR, RL, RR with joints (abduction, hip pitch, knee).
    """
    sim_robot = SimRobot(design, height, masses, timestep)
    height = sim_robot.stance_height
    commanded_joint_angles = sim_robot.kinematics.solve(sim_robot.kinematics.stance(height))  # (4, 3)
    servo_model = actuators.Servo(servo or actuators.ServoParams())
    joint_targets = commanded_joint_angles.ravel()
    n_steps, n_window = int(round(duration / timestep)), int(round(window / timestep))
    torque_log, angle_log, load_log, height_log, saturated_log, tilt_log = [], [], [], [], [], []
    run_peak = 0.0
    frames = []
    trace = []
    for step_index in range(n_steps):
        torques = servo_model.step(joint_targets, sim_robot.joint_angles(), sim_robot.joint_velocities(), timestep)
        sim_robot.step(torques)
        run_peak = max(run_peak, float(np.abs(torques).max()))
        trace.append(float(np.abs(torques).max()) / servo_model.params.stall_torque)
        if record_every and step_index % record_every == 0:
            frames.append(sim_robot.snapshot())
        if step_index >= n_steps - n_window:
            torque_log.append(torques)
            angle_log.append(sim_robot.joint_angles())
            height_log.append(sim_robot.data.site("body_origin").xpos[2])
            tilt_log.append(sim_robot.tilt_deg())
            saturated_log.append(servo_model.last_saturated.copy())
            load_log.append(sim_robot.foot_loads())
    torques = np.array(torque_log).reshape(-1, 4, 3)
    joint_angles = np.array(angle_log).reshape(-1, 4, 3)
    limit = servo_model.params.stall_torque
    mass = sim_robot.total_mass
    start_height = sim_robot.start_height
    return {
        "total_mass_kg": mass,
        "weight_n": mass * statics.G,
        "servo_limit_nm": limit,
        "torque_nm": np.abs(torques).mean(0),         # mean |torque| per joint over the window
        "torque_peak_nm": np.abs(torques).max(0),
        "torque_fraction_of_limit": float(np.abs(torques).max() / limit),   # last window only
        "run_peak_fraction_of_limit": run_peak / limit,                   # whole run, includes the settling transient
        "saturated": bool(np.array(saturated_log).any()),
        "q_command": commanded_joint_angles,
        "q_actual": joint_angles.mean(0),
        "gap_rad": (joint_angles.mean(0) - commanded_joint_angles),           # actual minus commanded
        "foot_load_n": np.array(load_log).mean(0),
        "body_height_m": float(np.mean(height_log)),
        "body_height_drop_m": float(start_height - np.mean(height_log)),  # below the height the pads touch the floor at
        "stance_height_m": height,
        "tilt_deg": float(np.max(tilt_log)),     # angle between the body's up axis and vertical
        "stood": bool(np.max(tilt_log) < 20.0 and start_height - np.mean(height_log) < 0.05),
        "frames": frames,
        "torque_trace": np.array(trace)[::record_every] if record_every else np.array(trace),  # largest |torque| / limit per step
    }


def smoothstep(x):
    """0 to 1 with zero slope at both ends (minimum-jerk profile)."""
    x = np.clip(x, 0.0, 1.0)
    return x**3 * (10 - 15 * x + 6 * x**2)


def run_sit_to_stand(sit_height=0.14, stand_height=None, rise_time=2.0, settle=1.0, hold=1.0, design=SIM_QUADRUPED,
                     masses=None, servo=None, timestep=0.002, record_every=0) -> dict:
    """Start crouched at `sit_height`, hold `settle` s, rise to `stand_height` over `rise_time` s, hold `hold` s.

    design        a placeholders.QuadrupedDesign; stand_height defaults to design.stance_height.
    The feet stay under the hips; the commanded height follows a minimum-jerk profile and the joint
    targets come from the closed-form IK at each step. "Sitting" here is a deep crouch: the placeholder
    knee limit (-2.3 rad) stops the 150 mm legs from going lower than about 0.13 m.
    Succeeds ("stood") if the body ends within 50 mm of the target height (the servo position loop sags a few
    millimetres to centimetres under load, see run_stand) and never tilted past 20 degrees.
    """
    stand_height = design.stance_height if stand_height is None else stand_height
    sim_robot = SimRobot(design, sit_height, masses, timestep)
    kinematics = sim_robot.kinematics
    if not kinematics.reachable(kinematics.stance(sit_height)).all():
        raise ValueError(f"sit height {sit_height} m is outside the joint limits")
    servo_model = actuators.Servo(servo or actuators.ServoParams())
    limit = servo_model.params.stall_torque
    end_time = settle + rise_time + hold
    n_steps = int(round(end_time / timestep))
    joint_targets_at = {}  # commanded height (rounded to 1 um) -> joint targets, so a held height is solved once
    time_log, torque_log, height_log, tilt_log, saturated_log, commanded_log = [], [], [], [], [], []
    frames = []
    for step_index in range(n_steps):
        time = step_index * timestep
        commanded_height = sit_height + (stand_height - sit_height) * smoothstep((time - settle) / rise_time)
        key = round(commanded_height, 6)
        if key not in joint_targets_at:
            joint_targets_at[key] = kinematics.solve(kinematics.stance(commanded_height)).ravel()
        torques = servo_model.step(joint_targets_at[key], sim_robot.joint_angles(), sim_robot.joint_velocities(), timestep)
        sim_robot.step(torques)
        if record_every and step_index % record_every == 0:
            frames.append(sim_robot.snapshot())
        time_log.append(time)
        torque_log.append(np.abs(torques).max())
        commanded_log.append(commanded_height)
        height_log.append(sim_robot.data.site("body_origin").xpos[2])
        tilt_log.append(sim_robot.tilt_deg())
        saturated_log.append(servo_model.last_saturated.any())
    time_log, torque_log, height_log = np.array(time_log), np.array(torque_log), np.array(height_log)
    rising = (time_log >= settle) & (time_log <= settle + rise_time)
    final_error = float(stand_height + mjcf.FOOT_RADIUS - np.mean(height_log[-int(0.25 / timestep):]))
    return {
        "total_mass_kg": sim_robot.total_mass,
        "servo_limit_nm": limit,
        "sit_hold_peak_fraction": float(torque_log[(time_log < settle) & (time_log > 0.3 * settle)].max() / limit),
        "rise_peak_fraction": float(torque_log[rising].max() / limit),
        "rise_peak_time_s": float(time_log[rising][np.argmax(torque_log[rising])] - settle),
        "fraction_of_rise_saturated": float(np.mean(np.array(saturated_log)[rising])),
        "max_tilt_deg": float(np.max(tilt_log)),
        "final_height_error_m": final_error,
        "stood": bool(final_error < 0.05 and np.max(tilt_log) < 20.0),
        "time_s": time_log, "body_z_m": height_log, "commanded_height_m": np.array(commanded_log), "peak_torque_nm": torque_log,
        "frames": frames,
    }
