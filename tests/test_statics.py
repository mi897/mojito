"""Run with:  python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import leg, statics  # noqa: E402


class TestStatics(unittest.TestCase):
    def test_torques_match_moment_arms(self):
        # Foot under the hip with zero abduction: each torque is load times a horizontal
        # (or lateral) lever arm read off the joint positions, independent of the Jacobian.
        L = leg.NOMINAL_LENGTHS
        q = leg.analytic_ik(np.array([0.05, L[0], -0.17]), L)
        q[0] = 0.0
        q = leg.analytic_ik(leg.forward(q, L), L)  # a reachable pose with abduction near zero
        shoulder, hip, knee, foot = leg.joint_positions(q, L)
        F = 3.0
        tau = statics.joint_torques(q, L, F)
        np.testing.assert_allclose(tau[1], F * abs(foot[0] - hip[0]), atol=1e-9)
        np.testing.assert_allclose(tau[2], F * abs(foot[0] - knee[0]), atol=1e-9)
        # abduction: lever arm is the foot's distance from the abduction axis (x), in y-z
        np.testing.assert_allclose(tau[0], F * abs(foot[1]), atol=1e-9)

    def test_torque_is_linear_in_load(self):
        L = leg.NOMINAL_LENGTHS
        q = np.array([0.1, 0.6, -1.2])
        np.testing.assert_allclose(statics.joint_torques(q, L, 4.0), 4 * statics.joint_torques(q, L, 1.0), atol=1e-12)

    def test_three_foot_loads_balance_weight_and_moments(self):
        feet = np.array([[0.12, 0.09], [0.12, -0.09], [-0.12, 0.09]])
        com = np.array([-0.01, -0.02])
        f = statics.foot_loads(feet, com, weight=10.0)
        self.assertAlmostEqual(f.sum(), 10.0)
        np.testing.assert_allclose(f @ feet, 10.0 * com, atol=1e-12)

    def test_centroid_gives_equal_loads_and_outside_com_goes_negative(self):
        feet = np.array([[0.12, 0.09], [0.12, -0.09], [-0.12, 0.09]])
        np.testing.assert_allclose(statics.foot_loads(feet, feet.mean(axis=0)), 1 / 3, atol=1e-12)
        self.assertLess(statics.foot_loads(feet, np.array([-0.2, -0.2])).min(), 0)


if __name__ == "__main__":
    unittest.main()
