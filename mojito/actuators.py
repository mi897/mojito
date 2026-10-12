"""Actuator models for the simulator (docs/STAGE2.md, work package 2). Sensors are in `sensors.py`.

Pure NumPy, no MuJoCo: each model takes joint state and returns torques or
readings, so `step` can be called from any simulation loop.

Every figure in `ServoParams` is an unverified retailer number for the
"DSServo DS3218 20KG" (docs/STAGE2.md, "Servo model"). Treat them as defaults to
be replaced by a datasheet or a measurement, never as the robot's.

Interface (what a later DC-motor model must also provide):
    reset(n)                      forget history for n joints
    step(target, q, qd, dt)       commanded joint targets and true state -> joint torques
    reports                       names of the signals the actuator itself feeds back
A hobby servo reports nothing, so `reports` is empty; a motor with current
sensing would list ("position", "velocity", "current").
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

KGCM = 0.0980665  # N·m per kg·cm
V_MIN, V_MAX = 5.0, 6.0  # supply range the owner keeps to; outside it the model refuses to run


def _lerp(v, v0, y0, v1, y1):
    return y0 + (y1 - y0) * (v - v0) / (v1 - v0)


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

    def __init__(self, params: ServoParams | None = None):
        self.p = params or ServoParams()
        self._queue: deque = deque()
        self._t = 0.0
        self._held = None

    def reset(self, n: int = 12) -> None:
        self._queue.clear()
        self._t = 0.0
        self._held = None

    def quantise(self, target):
        r = self.p.resolution
        return self.p.centre + np.round((np.asarray(target, float) - self.p.centre) / r) * r

    def step(self, target, q, qd, dt: float) -> np.ndarray:
        p = self.p
        self._queue.append((self._t + p.latency, self.quantise(target)))
        seen = None
        while self._queue and self._queue[0][0] <= self._t + 1e-12:
            seen = self._queue.popleft()[1]
        if seen is not None:
            self._held = seen
        elif self._held is None:  # nothing has arrived yet: hold where the joint is
            self._held = self.quantise(q)
        self._t += dt
        q, qd = np.asarray(q, float), np.asarray(qd, float)
        tau = p.kp * (self._held - q) - p.kd * qd
        speed = np.abs(qd) / p.noload_speed
        tmax = p.stall_torque
        # torque pushing along the motion is cut by the torque-speed line; opposing torque (braking) is not
        along = np.sign(tau) == np.sign(qd)
        limit = np.where(along, tmax * np.clip(1.0 - speed, 0.0, 1.0), tmax)
        self.last_saturated = np.abs(tau) >= limit
        return np.clip(tau, -limit, limit)
