"""Actuator models for the simulator (docs/STAGE2.md, work package 2). Sensors are in `sensors.py`.

Pure NumPy, no MuJoCo: each model takes joint state and returns torques or
readings, so `step` can be called from any simulation loop.

Every figure in `ServoParams` is an unverified retailer number for the
"DSServo DS3218 20KG" (docs/STAGE2.md, "Servo model"). Treat them as defaults to
be replaced by a datasheet or a measurement, never as the robot's.

Interface (what a later DC-motor model must also provide):
    reset(n_joints)                                         forget history; n_joints joints from then on
    step(target_angles, joint_angles, joint_velocities, dt)  commanded joint targets and true state -> joint torques
    reports                       names of the signals the actuator itself feeds back
A hobby servo reports nothing, so `reports` is empty; a motor with current
sensing would list ("position", "velocity", "current").
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass

import numpy as np

KGCM = 0.0980665  # N·m per kg·cm
V_MIN, V_MAX = 5.0, 6.0  # supply range the owner keeps to; outside it the model refuses to run


def _lerp(x, x0, y0, x1, y1):
    """Straight line through (x0, y0) and (x1, y1), evaluated at x: y0 + (y1 - y0) (x - x0) / (x1 - x0)."""
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)


@dataclass
class ServoParams:
    """Listed figures and model choices. `voltage` must lie in 5–6 V."""
    voltage: float = 5.0
    # torque and speed are given at two listed points (5 V and 6.8 V); between them a straight line (an assumption)
    stall_5v: float = 19.0 * KGCM     # N·m
    stall_68v: float = 21.5 * KGCM    # N·m
    noload_5v: float = np.radians(60.0) / 0.16    # rad/s, from 0.16 s/60°
    noload_68v: float = np.radians(60.0) / 0.14   # rad/s, from 0.14 s/60°
    torque_scale: float = 1.0         # e.g. 0.5 for a "continuous" torque assumption
    kp: float = 10.0                  # N·m/rad: position loop gain (gap at stall = stall / kp)
    kd: float = 0.2                   # N·m·s/rad
    latency: float = 0.02             # s between command and servo seeing it (assumed)
    resolution: float = np.radians(270.0) / 410.0  # rad: PCA9685 counts over the 270° range
    centre: float = 0.0               # command that maps to count zero, so quantisation is on a grid

    def __post_init__(self):
        if not V_MIN <= self.voltage <= V_MAX:
            raise ValueError(f"supply voltage {self.voltage} V is outside the modelled {V_MIN}-{V_MAX} V range")

    @property
    def stall_torque(self) -> float:
        return self.torque_scale * _lerp(self.voltage, 5.0, self.stall_5v, 6.8, self.stall_68v)

    @property
    def noload_speed(self) -> float:
        return _lerp(self.voltage, 5.0, self.noload_5v, 6.8, self.noload_68v)


class Servo:
    """Position servo: quantised, delayed target -> PD torque, limited by a torque-speed line.

    The available torque falls linearly from the stall value at rest to zero at the
    no-load speed (standard brushed-motor behaviour), and is further limited in
    the direction of motion only: braking is not reduced.
    """
    reports: tuple = ()

    def __init__(self, params: ServoParams | None = None, n_joints: int = 12):
        self.params = params or ServoParams()
        self.reset(n_joints)

    def reset(self, n_joints: int = 12) -> None:
        """Forget queued commands and the held target; `last_saturated` becomes `n_joints` False values.

        `n_joints` only sizes `last_saturated` until the first `step`, which takes any number of joints and sets
        `last_saturated` to the size of its targets.
        """
        self._queue: deque = deque()  # (step index at which the command arrives, quantised target)
        self._step_index = 0
        self._held = None
        self.last_saturated = np.zeros(n_joints, bool)

    def quantise(self, target_angles):
        """Snap `target_angles` to the nearest multiple of the resolution, counted from `centre`:
        centre + round((target_angles - centre) / resolution) * resolution."""
        step = self.params.resolution
        return self.params.centre + np.round((np.asarray(target_angles, float) - self.params.centre) / step) * step

    def step(self, target_angles, joint_angles, joint_velocities, dt: float) -> np.ndarray:
        """Joint torques for one time step of `dt` s. `dt` must stay the same between `reset` calls: the latency is
        counted in steps of the current `dt`, so a command already queued keeps the step count it was given."""
        params = self.params
        # A command sent at step k is seen at the first step j whose time is at least its send time plus the latency:
        # j dt >= k dt + latency, so j = k + ceil(latency / dt). Counting whole steps keeps that exact over long runs,
        # where an accumulated float time would drift. The 1e-9 lets a latency that is a multiple of dt up to rounding
        # (0.02 s at 0.002 s) count as that multiple (10 steps), not one more. tests/sim/test_actuators.py checks it.
        n_delay_steps = math.ceil(params.latency / dt - 1e-9)
        self._queue.append((self._step_index + n_delay_steps, self.quantise(target_angles)))
        seen = None
        while self._queue and self._queue[0][0] <= self._step_index:
            seen = self._queue.popleft()[1]
        if seen is not None:
            self._held = seen
        elif self._held is None:  # nothing has arrived yet: hold where the joint is
            self._held = self.quantise(joint_angles)
        self._step_index += 1
        joint_angles, joint_velocities = np.asarray(joint_angles, float), np.asarray(joint_velocities, float)
        # PD law on the delayed, quantised target:  torque = kp (target - angle) - kd velocity
        torque = params.kp * (self._held - joint_angles) - params.kd * joint_velocities
        speed_fraction = np.abs(joint_velocities) / params.noload_speed  # 0 at rest, 1 at the no-load speed
        stall_torque = params.stall_torque
        # Torque-speed line: available torque = stall_torque (1 - |joint velocity| / noload_speed), floored at zero.
        # It limits torque pushing along the motion; opposing torque (braking) is not reduced.
        along_motion = np.sign(torque) == np.sign(joint_velocities)
        limit = np.where(along_motion, stall_torque * np.clip(1.0 - speed_fraction, 0.0, 1.0), stall_torque)
        self.last_saturated = np.abs(torque) >= limit
        return np.clip(torque, -limit, limit)
