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
    """Measured angle = quantise(q + offset + a·sin(q) + noise), seen `delay` seconds late."""
    def __init__(self, params: JointSensorParams | None = None, seed: int = 0):
        self.p = params or JointSensorParams()
        self.rng = np.random.default_rng(seed)
        self._queue: deque = deque()
        self._t = 0.0
        self._last = None

    def reset(self) -> None:
        self._queue.clear()
        self._t = 0.0
        self._last = None

    def read(self, q, dt: float) -> np.ndarray:
        p = self.p
        q = np.asarray(q, float)
        raw = q + p.offset + p.nonlinearity * np.sin(q) + self.rng.normal(0.0, p.noise, q.shape)
        raw = np.round(raw / p.resolution) * p.resolution
        self._queue.append((self._t + p.delay, raw))
        while self._queue and self._queue[0][0] <= self._t + 1e-12:
            self._last = self._queue.popleft()[1]
        self._t += dt
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
        self.p = params or ImuParams()
        self.rng = np.random.default_rng(seed)
        self.bias = np.zeros(3)

    def read(self, gyro, accel, dt: float):
        p = self.p
        self.bias = self.bias + self.rng.normal(0.0, p.gyro_bias_walk * np.sqrt(dt), 3)
        g = np.asarray(gyro, float) + self.bias + self.rng.normal(0.0, p.gyro_noise, 3)
        a = np.asarray(accel, float) + self.rng.normal(0.0, p.accel_noise, 3)
        return g, a
