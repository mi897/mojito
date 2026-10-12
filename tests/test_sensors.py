"""Run with:  python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import sensors as S  # noqa: E402


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
