"""Run with:  python -m unittest discover tests   (simulation tests skipped if MuJoCo is not installed)"""
import json
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.robot import leg  # noqa: E402
from mojito.sim import actuators, design_sweep, gait_replay  # noqa: E402

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mujoco = None


class TestDesignGrid(unittest.TestCase):
    def test_grid_is_nine_leg_pairs_then_two_bodies(self):
        grid = design_sweep.design_grid()
        self.assertEqual(len(grid), 11)
        self.assertEqual({(d.upper_length, d.lower_length) for d in grid[:9]},
                         {(u, l) for u in (0.10, 0.15, 0.20) for l in (0.10, 0.15, 0.20)})
        self.assertTrue(all(d.body_length == 0.30 for d in grid[:9]))
        self.assertTrue(all((d.upper_length, d.lower_length) == (0.15, 0.15) for d in grid[9:]))

    def test_masses_scale_every_role(self):
        scaled = design_sweep.scaled_masses(2.0)
        for role, mass in design_sweep.mjcf.DEFAULT_MASSES.items():
            self.assertAlmostEqual(scaled[role], 2 * mass)


class TestDraws(unittest.TestCase):
    def test_a_draw_is_repeatable_and_inside_the_stated_ranges(self):
        masses, latency = design_sweep.draw_masses_and_latency(1.75, 3)
        self.assertEqual((masses, latency), design_sweep.draw_masses_and_latency(1.75, 3))
        self.assertNotEqual(masses, design_sweep.draw_masses_and_latency(1.75, 4)[0])
        self.assertGreaterEqual(latency, design_sweep.LATENCY[0])
        self.assertLessEqual(latency, design_sweep.LATENCY[1])
        for role, mass in design_sweep.mjcf.DEFAULT_MASSES.items():
            ratio = masses[role] / (1.75 * mass)
            self.assertTrue(1 - design_sweep.MASS_SPREAD <= ratio <= 1 + design_sweep.MASS_SPREAD)


class TestScaledHeight(unittest.TestCase):
    def test_placeholder_leg_gives_back_the_tuned_height(self):
        for base in (gait_replay.DEFAULT_STANCE_HEIGHT, gait_replay.TROT_STANCE_HEIGHT):
            self.assertAlmostEqual(design_sweep.scaled_height(design_sweep.Design(), base), base, places=12)

    def test_every_design_stands_with_the_placeholder_knee_bend_found_by_the_ik(self):
        """Independent check of the law-of-cosines height: closed-form IK at that height returns the same knee angle."""
        base = gait_replay.DEFAULT_STANCE_HEIGHT
        reference_knee = leg.analytic_ik(np.array([0.0, 0.05, -base]), design_sweep.REFERENCE_LENGTHS)[2]
        for design in design_sweep.design_grid():
            height = design_sweep.scaled_height(design, base)
            joint_angles = leg.analytic_ik(np.array([0.0, design_sweep.ABDUCTION_OFFSET, -height]), design.lengths)
            self.assertAlmostEqual(joint_angles[2], reference_knee, places=9)


class TestTuningHelpers(unittest.TestCase):
    def test_default_settings_reproduce_the_untuned_heights(self):
        for name, height in (("trot", gait_replay.TROT_STANCE_HEIGHT), ("crawl", gait_replay.DEFAULT_STANCE_HEIGHT)):
            settings = design_sweep.default_settings(name)
            self.assertAlmostEqual(design_sweep.height_for_knee(design_sweep.Design(), settings.knee), height, places=12)

    def test_knee_from_ik_at_the_height_for_a_knee_returns_that_knee(self):
        """Independent check of the law of cosines: closed-form IK at the height gives back the knee angle."""
        for design in design_sweep.design_grid():
            for knee in design_sweep.KNEE_GRID:
                height = design_sweep.height_for_knee(design, knee)
                joint_angles = leg.analytic_ik(np.array([0.0, design_sweep.ABDUCTION_OFFSET, -height]), design.lengths)
                self.assertAlmostEqual(joint_angles[2], knee, places=9)

    def test_grid_size_and_score_order(self):
        self.assertEqual(len(design_sweep.settings_grid()), 45)
        good = {"walked": True, "within_joint_limits": True, "velocity_ratio": 0.95, "peak_torque_fraction_of_limit": 0.5}
        slow = dict(good, velocity_ratio=0.6)
        fast_invalid = dict(good, within_joint_limits=False, velocity_ratio=1.0)
        self.assertGreater(design_sweep.tuning_score(good), design_sweep.tuning_score(slow))
        self.assertGreater(design_sweep.tuning_score(slow), design_sweep.tuning_score(fast_invalid))
        self.assertGreater(design_sweep.tuning_score(good), design_sweep.tuning_score(dict(good, peak_torque_fraction_of_limit=0.9)))


class TestStaticBound(unittest.TestCase):
    def test_placeholder_design_matches_torque_envelope(self):
        """scripts/body/torque_envelope.py gives the stand-case bound at the 5 V stall torque for the same design."""
        path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "body", "torque_envelope.json")
        with open(path) as f:
            envelope = json.load(f)
        servo = actuators.ServoParams()
        bound = design_sweep.static_mass_bound(design_sweep.Design(), 0.2125, servo.stall_torque)
        expected = envelope["max_total_mass_kg"]["stand_4_feet"]["5.0 V derate 1.0"]
        self.assertAlmostEqual(bound, expected, delta=0.01)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestEvaluateDesign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.result = design_sweep.evaluate_design(design_sweep.Design(), 1.0, speeds=(0.10,), run_time=5.0)

    def test_placeholder_design_stands_at_the_stored_stand_test_figures(self):
        path = os.path.join(os.path.dirname(__file__), "..", "..", "results", "stand", "stand_test.json")
        with open(path) as f:
            stored = json.load(f)
        self.assertAlmostEqual(self.result["stand"]["total_mass_kg"], 1.72, places=2)
        self.assertTrue(self.result["stand"]["stood"])
        self.assertEqual(stored["lengths_m"], design_sweep.Design().lengths.tolist())

    def test_one_walk_per_gait_and_speed(self):
        self.assertEqual([(w["gait"], w["speed_m_s"]) for w in self.result["walk"]], [("trot", 0.10), ("crawl", 0.10)])

    def test_evaluate_walk_with_default_settings_matches_evaluate_design(self):
        walk = design_sweep.evaluate_walk(design_sweep.Design(), "trot", design_sweep.default_settings("trot"), 1.0, 0.10, run_time=5.0)
        self.assertAlmostEqual(walk["velocity_ratio"], self.result["walk"][0]["velocity_ratio"], places=6)

    def test_trot_at_placeholder_dimensions_walks(self):
        trot = self.result["walk"][0]
        self.assertTrue(trot["walked"])
        self.assertGreater(trot["velocity_ratio"], 0.5)


if __name__ == "__main__":
    unittest.main()
