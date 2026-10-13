"""Run with:  python -m unittest discover tests   (skipped if MuJoCo is not installed)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import actuators, gait_replay, gaits  # noqa: E402

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

    def test_stiffer_servo_closes_the_crawl_speed_gap(self):
        # docs/STAGE2.md, "Why the body falls short": in the crawl the commanded motion asks for the commanded speed,
        # so the shortfall at kp 10 is the servo not following; kp 100 follows it
        stiff = actuators.ServoParams(kp=100.0, kd=1.0)
        r = gait_replay.run_gait(gaits.CRAWL, (0.04, 0.0), cycles=1, servo=stiff)
        self.assertAlmostEqual(r["measured_velocity_m_s"][0], 0.04, delta=0.01)
        self.assertGreater(r["measured_velocity_m_s"][0], self.crawl["measured_velocity_m_s"][0] + 0.005)

    def test_commanded_contact_matches_the_gait_timing(self):
        r = self.trot
        walking = r["time_s"] >= r["walk_start_s"]
        np.testing.assert_allclose(r["commanded_contact"][walking].mean(0), gaits.TROT.duty, atol=0.02)
        # diagonal pairs (FL, RR) and (FR, RL) are in step
        np.testing.assert_array_equal(r["commanded_contact"][:, 0], r["commanded_contact"][:, 3])
        np.testing.assert_array_equal(r["commanded_contact"][:, 1], r["commanded_contact"][:, 2])

    def test_pace_rolls_onto_its_swing_feet_and_the_roll_matches_the_lift_over_the_track(self):
        # docs/STAGE2.md, "Why the pace goes backward": two feet on one side leave the body unsupported laterally, so it rolls
        # until the lifted feet reach the floor: roll range about 2 atan(lift / track), here lift 0.03 m, track about 0.25 m
        import dataclasses

        pace = dataclasses.replace(gaits.TROT, name="pace", offsets=(0.0, 0.5, 0.0, 0.5))
        r = gait_replay.run_gait(pace, (0.1, 0.0), cycles=2)
        walking = r["time_s"] >= r["walk_start_s"]
        rotation = r["rotation"]
        roll = np.degrees(np.arctan2(rotation[:, 2, 1], rotation[:, 2, 2]))[walking]
        expected = 2 * np.degrees(np.arctan(pace.lift / 0.25))
        self.assertAlmostEqual(np.ptp(roll), expected, delta=0.2 * expected)
        loads = r["foot_load_n"][walking]
        swing = ~r["commanded_contact"][walking]
        self.assertGreater((loads * swing).sum() / loads.sum(), 0.4)  # trot: about 0.2
        self.assertLess(r["measured_velocity_m_s"][0], 0.0)  # commanded +0.1 m/s

    def test_a_much_heavier_robot_saturates_the_servos(self):
        r = gait_replay.run_gait(gaits.TROT, (0.15, 0.0), cycles=2, masses=light(8.0))
        self.assertGreater(r["fraction_of_walking_saturated"], 0.2)
        self.assertFalse(r["walked"])


if __name__ == "__main__":
    unittest.main()
