"""Run with:  python -m unittest discover tests   (skipped if MuJoCo is not installed)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import actuators, body, stand, statics  # noqa: E402

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mujoco = None

LENGTHS = np.array([0.05, 0.15, 0.15])


def light(total):
    return {"body": total - 4 * 4e-4, "hip": 1e-4, "thigh": 1e-4, "shank": 1e-4, "foot": 1e-4}


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestStand(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = stand.run_stand(masses=light(1.5), duration=2.0)

    def test_foot_loads_add_up_to_the_weight(self):
        self.assertAlmostEqual(self.r["foot_load_n"].sum(), self.r["weight_n"], delta=0.01 * self.r["weight_n"])

    def test_servo_torque_matches_static_torque_from_the_measured_foot_loads(self):
        r = self.r
        for i in range(4):
            sign = np.array([-1.0 if i % 2 else 1.0, 1.0, 1.0])  # right legs are mirrored
            pred = statics.joint_torques(r["q_actual"][i] * sign, LENGTHS, r["foot_load_n"][i])
            np.testing.assert_allclose(r["torque_nm"][i], pred, atol=0.01)
            np.testing.assert_allclose(r["torque_nm"][i, [0, 2]], pred[[0, 2]], rtol=0.03)

    def test_equal_quarter_prediction_is_close_but_not_exact(self):
        # the static calculation assumes each foot carries a quarter; the knees bend one way so front and rear differ
        even = statics.joint_torques(self.r["q_command"][0], LENGTHS, self.r["weight_n"] / 4)
        knee = self.r["torque_nm"][:, 2]
        np.testing.assert_allclose(knee, even[2], rtol=0.12)
        self.assertGreater(knee.max() / knee.min(), 1.02)

    def test_stands_upright_and_commanded_gap_is_small_for_a_light_robot(self):
        self.assertTrue(self.r["stood"])
        self.assertLess(np.abs(self.r["gap_rad"]).max(), np.radians(5))
        self.assertLess(self.r["torque_fraction_of_limit"], 0.5)

    def test_heavier_robot_sags_more_and_too_heavy_does_not_stand(self):
        light_gap = np.abs(self.r["gap_rad"][:, 2]).max()
        heavy = stand.run_stand(masses=light(3.0), duration=2.0)
        self.assertGreater(np.abs(heavy["gap_rad"][:, 2]).max(), light_gap)
        too_heavy = stand.run_stand(masses=light(9.0), duration=2.0)  # past the 7.2 kg static bound at 5 V
        self.assertFalse(too_heavy["stood"])
        self.assertTrue(too_heavy["saturated"] or too_heavy["run_peak_fraction_of_limit"] > 0.99)

    def test_higher_voltage_raises_the_limit(self):
        r6 = stand.run_stand(masses=light(1.5), servo=actuators.ServoParams(voltage=6.0), duration=1.0)
        self.assertGreater(r6["servo_limit_nm"], self.r["servo_limit_nm"])


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestSitToStand(unittest.TestCase):
    fine = actuators.ServoParams(resolution=1e-4)  # command steps small enough not to cause torque steps

    def test_crouch_hold_torque_matches_static_torque_from_measured_loads(self):
        r = stand.run_stand(masses=light(1.5), height=0.14, duration=2.0)
        for i in range(4):
            sign = np.array([-1.0 if i % 2 else 1.0, 1.0, 1.0])
            pred = statics.joint_torques(r["q_actual"][i] * sign, LENGTHS, r["foot_load_n"][i])
            np.testing.assert_allclose(r["torque_nm"][i, [0, 2]], pred[[0, 2]], rtol=0.03)
        self.assertAlmostEqual(r["foot_load_n"].sum(), r["weight_n"], delta=0.01 * r["weight_n"])

    def test_crouch_needs_more_knee_torque_than_standing(self):
        robot = body.Robot(body.AnalyticSolver(), lengths=LENGTHS, body_length=0.3, body_width=0.15)
        knee = [statics.joint_torques(robot.solve(robot.stance(h))[0], LENGTHS, 1.0)[2] for h in (0.14, 0.2125)]
        self.assertGreater(knee[0], 1.2 * knee[1])  # per newton of foot load: deeper crouch, longer lever arm (about 1.25x)

    def test_slow_rise_peaks_at_the_crouch_and_fast_rise_peaks_higher(self):
        slow = stand.run_sit_to_stand(masses=light(3.0), rise_time=4.0, servo=self.fine)
        fast = stand.run_sit_to_stand(masses=light(3.0), rise_time=0.5, servo=self.fine)
        self.assertTrue(slow["stood"] and fast["stood"])
        self.assertAlmostEqual(slow["rise_peak_fraction"], slow["sit_hold_peak_fraction"], delta=0.05)
        self.assertGreater(fast["rise_peak_fraction"], slow["rise_peak_fraction"] + 0.05)

    def test_too_heavy_cannot_rise(self):
        r = stand.run_sit_to_stand(masses=light(6.0), rise_time=2.0, servo=self.fine)
        self.assertFalse(r["stood"])
        self.assertGreater(r["fraction_of_rise_saturated"], 0.5)

    def test_sit_height_outside_joint_limits_is_refused(self):
        with self.assertRaises(ValueError):
            stand.run_sit_to_stand(sit_height=0.08)


if __name__ == "__main__":
    unittest.main()
