"""Run with:  python -m unittest discover tests   (skipped if MuJoCo is not installed)"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import gait_replay, gaits  # noqa: E402

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mujoco = None


def light(total):
    return {"body": total - 4 * 4e-4, "hip": 1e-4, "thigh": 1e-4, "shank": 1e-4, "foot": 1e-4}


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestGaitReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trot = gait_replay.run_gait(gaits.TROT, (0.15, 0.0), cycles=2)
        cls.crawl = gait_replay.run_gait(gaits.CRAWL, (0.04, 0.0), cycles=1)

    def test_default_robot_stays_up_in_both_gaits(self):
        for r in (self.trot, self.crawl):
            self.assertTrue(r["walked"])
            self.assertLess(r["max_tilt_deg"], 15.0)

    def test_planted_feet_do_not_move_in_the_world(self):
        # conservation property from CLAUDE.md: a foot carrying load stays put (here to a few mm per 2 ms step at most;
        # a swing foot moves tens of mm per second, a slipping foot far more)
        for r in (self.trot, self.crawl):
            self.assertGreater(r["slip_planted_steps"], 100)
            self.assertLess(r["slip_max_mm_per_step"], 3.0)

    def test_commanded_forward_speed_moves_the_body_forward_more_than_standing_in_place(self):
        still = gait_replay.run_gait(gaits.TROT, (0.0, 0.0), cycles=2)
        self.assertGreater(self.trot["measured_velocity_m_s"][0], still["measured_velocity_m_s"][0] + 0.03)

    def test_turning_command_turns_the_body_the_right_way(self):
        r = gait_replay.run_gait(gaits.TROT, (0.0, 0.0), yaw_rate=1.2, cycles=2)
        self.assertGreater(r["measured_yaw_rate_rad_s"], 0.3)

    def test_torque_is_reported_against_the_servo_limit(self):
        for r in (self.trot, self.crawl):
            self.assertGreater(r["peak_torque_fraction_of_limit"], 0.1)
            self.assertLessEqual(r["peak_torque_fraction_whole_run"], 1.0 + 1e-6)

    def test_a_much_heavier_robot_saturates_the_servos(self):
        r = gait_replay.run_gait(gaits.TROT, (0.15, 0.0), cycles=2, masses=light(8.0))
        self.assertGreater(r["fraction_of_walking_saturated"], 0.2)
        self.assertFalse(r["walked"])


if __name__ == "__main__":
    unittest.main()
