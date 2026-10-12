"""Joint-angle sensor and IMU models for the simulator (docs/STAGE2.md, work package 2).

Pure NumPy, no MuJoCo. These produce what the policy may observe: measured, never
true, values. All parameters are placeholders until the hardware is chosen (the
Hall-effect chip and the IMU model are tabled).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np


@dataclass
class JointSensorParams:
    """Hall-effect joint angle sensor (type tabled; all parameters open)."""
    resolution: float = 2 * np.pi / 4096  # 12-bit
    noise: float = np.radians(0.1)        # rad, standard deviation per reading
    offset: float = 0.0                   # rad, added to every joint (or a per-joint array)
    nonlinearity: float = 0.0             # rad, amplitude of a once-per-turn sinusoidal error
    delay: float = 0.002                  # s (bus read time)


class JointSensor:
    """Measured angle = quantise(joint_angles + offset + a·sin(joint_angles) + noise), seen `delay` seconds late."""
    def __init__(self, params: JointSensorParams | None = None, seed: int = 0):
        self.params = params or JointSensorParams()
        self.rng = np.random.default_rng(seed)
        self._queue: deque = deque()
        self._time = 0.0
        self._last = None

    def reset(self) -> None:
        self._queue.clear()
        self._time = 0.0
        self._last = None

    def read(self, joint_angles, dt: float) -> np.ndarray:
        params = self.params
        joint_angles = np.asarray(joint_angles, float)
        # reading = joint_angles + offset + a sin(joint_angles) + noise, then rounded to the nearest multiple of the resolution
        raw = joint_angles + params.offset + params.nonlinearity * np.sin(joint_angles) + self.rng.normal(0.0, params.noise, joint_angles.shape)
        raw = np.round(raw / params.resolution) * params.resolution
        self._queue.append((self._time + params.delay, raw))
        while self._queue and self._queue[0][0] <= self._time + 1e-12:
            self._last = self._queue.popleft()[1]
        self._time += dt
        if self._last is None:  # nothing has arrived yet: report the first reading rather than nothing
            self._last = raw
        return self._last


@dataclass
class ImuParams:
    gyro_noise: float = np.radians(0.1)     # rad/s per reading
    gyro_bias_walk: float = np.radians(0.01)  # rad/s per sqrt(s): bias random walk
    accel_noise: float = 0.05               # m/s²


class Imu:
    """Body angular rate (with a drifting bias) and specific force, both in the body frame."""
    def __init__(self, params: ImuParams | None = None, seed: int = 0):
        self.params = params or ImuParams()
        self.rng = np.random.default_rng(seed)
        self.bias = np.zeros(3)

    def read(self, gyro, accel, dt: float):
        params = self.params
        # Random walk: after dt seconds the bias moves by a Gaussian step of standard
        # deviation gyro_bias_walk * sqrt(dt) (variance grows linearly with time).
        self.bias = self.bias + self.rng.normal(0.0, params.gyro_bias_walk * np.sqrt(dt), 3)
        gyro_reading = np.asarray(gyro, float) + self.bias + self.rng.normal(0.0, params.gyro_noise, 3)
        accel_reading = np.asarray(accel, float) + self.rng.normal(0.0, params.accel_noise, 3)
        return gyro_reading, accel_reading
