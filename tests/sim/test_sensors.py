"""Run with:  python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.sim import sensors as S  # noqa: E402

from .clock_helpers import fast_forward  # noqa: E402


class TestSensors(unittest.TestCase):
    def test_sensor_statistics(self):
        p = S.JointSensorParams(noise=0.01, resolution=1e-6, offset=0.02, delay=0.0)
        sens = S.JointSensor(p, seed=1)
        r = np.array([sens.read(np.full(1000, 0.3), 0.001) for _ in range(5)]).ravel()
        self.assertAlmostEqual(r.mean(), 0.32, delta=0.002)
        self.assertAlmostEqual(r.std(), 0.01, delta=0.001)

    def test_sensor_quantises_and_delays(self):
        sens = S.JointSensor(S.JointSensorParams(noise=0.0, resolution=0.1, delay=0.003), seed=0)
        out = [sens.read(np.array([0.04 * k]), 0.001)[0] for k in range(6)]
        np.testing.assert_allclose(out[:3], out[0])  # nothing newer has arrived yet
        self.assertAlmostEqual(out[5], 0.1, places=9)  # reading from 0.003 s ago was 0.08 -> 0.1

    def seen_read_steps(self, delay, dt, n_steps, start=0):
        """Per read, the index (from `start`) of the read whose angle the sensor reports. Read k measures (k + 1) * 1e-6 rad
        with no noise; before anything has arrived the sensor reports the first reading (index 0)."""
        sens = S.JointSensor(S.JointSensorParams(noise=0.0, resolution=1e-9, delay=delay))
        if start:
            fast_forward(sens, start, dt)
        out = np.array([sens.read(np.array([(k + 1) * 1e-6]), dt)[0] for k in range(n_steps)])
        return np.round(out / 1e-6).astype(int) - 1

    def test_reading_arrives_at_the_first_step_at_or_after_read_time_plus_delay(self):
        # Reference in exact rational arithmetic: read j reports the latest k with j dt >= k dt + delay.
        from fractions import Fraction
        import math
        for delay, dt in (("0.002", "0.002"), ("0.003", "0.001"), ("0.0025", "0.002"), ("0", "0.001"), ("0.02", "0.005")):
            n_delay_steps = math.ceil(Fraction(delay) / Fraction(dt))
            seen = self.seen_read_steps(float(delay), float(dt), 500)
            expected = np.array([max(j - n_delay_steps, 0) for j in range(500)])
            np.testing.assert_array_equal(seen, expected, err_msg=f"delay {delay} s, dt {dt} s")

    def test_delay_is_still_whole_steps_after_8192_s(self):
        # as for the servo: an accumulated float time compared within 1e-12 s gave 5 steps instead of 4 from step 1,638,399
        seen = self.seen_read_steps(0.02, 0.005, 60, start=1_638_380)
        np.testing.assert_array_equal(seen[4:], np.arange(60 - 4))

    def test_gyro_bias_drifts_like_a_random_walk(self):
        end = []
        for seed in range(200):
            imu = S.Imu(S.ImuParams(gyro_noise=0.0, gyro_bias_walk=0.01, accel_noise=0.0), seed=seed)
            for _ in range(1000):
                g, _ = imu.read(np.zeros(3), np.zeros(3), 0.001)
            end.append(g)
        # after 1 s the spread is bias_walk * sqrt(1 s)
        self.assertAlmostEqual(np.std(end), 0.01, delta=0.002)


if __name__ == "__main__":
    unittest.main()
