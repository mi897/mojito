"""Run with:  python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import IKNet, body, gaits, leg  # noqa: E402

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "weights", "ik_leg.npz")


class TestBody(unittest.TestCase):
    def setUp(self):
        self.exact = body.Robot(body.AnalyticSolver())
        self.home = self.exact.stance()

    def test_rotation_is_orthonormal_and_ordered_zyx(self):
        R = body.rotation(0.3, -0.2, 0.7)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(body.rotation(yaw=np.pi / 2) @ [1, 0, 0], [0, 1, 0], atol=1e-12)
        np.testing.assert_allclose(body.rotation(pitch=np.pi / 2) @ [1, 0, 0], [0, 0, -1], atol=1e-12)

    def test_stance_is_symmetric_and_solved_exactly(self):
        np.testing.assert_allclose(self.home[0] * [1, -1, 1], self.home[1])
        q = self.exact.solve(self.home)
        np.testing.assert_allclose(self.exact.feet(q), self.home, atol=1e-12)
        np.testing.assert_allclose(q[0] * [-1, 1, 1], q[1], atol=1e-12)  # mirrored abduction

    def test_posture_keeps_feet_fixed_in_the_world(self):
        pose = dict(roll=0.2, pitch=-0.15, yaw=0.3)
        offset = (0.01, -0.02, 0.02)
        targets = body.apply_posture(self.home, offset, **pose)
        self.assertTrue(self.exact.reachable(targets).all())
        feet = self.exact.feet(self.exact.solve(targets))
        np.testing.assert_allclose(body.to_world(feet, offset, **pose), self.home, atol=1e-10)

    def test_per_leg_lengths_can_differ(self):
        lengths = leg.NOMINAL_LENGTHS * np.array([[1, 1, 1], [1, 1.08, 0.95], [1.05, 0.93, 1], [0.95, 1, 1.07]])
        robot = body.Robot(body.AnalyticSolver(), lengths)
        targets = body.apply_posture(self.home, roll=0.1, yaw=0.2)
        np.testing.assert_allclose(robot.feet(robot.solve(targets)), targets, atol=1e-10)
        robot.lengths[0, 2] *= 0.9  # change a link on the fly: same angles now miss
        self.assertGreater(robot.tracking_error(targets, self.exact.solve(targets))[0], 1e-3)

    def test_out_of_reach_target_is_flagged(self):
        far = self.home + [0, 0, -0.2]
        self.assertFalse(self.exact.reachable(far).any())

    @unittest.skipUnless(os.path.exists(WEIGHTS), "trained weights not present")
    def test_network_tracks_postures_within_half_a_millimetre(self):
        robot = body.Robot(IKNet.load(WEIGHTS))
        rng = np.random.default_rng(0)
        rpy = rng.uniform(-1, 1, size=(500, 3)) * np.radians([15, 12, 18])
        targets = body.apply_posture(np.broadcast_to(self.home, (500, 4, 3)), (0, 0, 0), *rpy.T)
        self.assertTrue(robot.reachable(targets).all())
        self.assertLess(robot.tracking_error(targets).max(), 0.5e-3)


class TestGaits(unittest.TestCase):
    def setUp(self):
        self.robot = body.Robot(body.AnalyticSolver())
        self.home = self.robot.stance()

    def run_gait(self, gait, v, w, n=600):
        t = np.linspace(0, 2 * gait.period, n, endpoint=False)
        feet, contact, _ = gaits.foot_targets(gait, t, self.home, v, w)
        pos, yaw = gaits.body_path(t, v, w)
        return t, feet, contact, pos, yaw

    def test_grounded_feet_do_not_slide(self):
        for gait in (gaits.TROT, gaits.CRAWL):
            for v, w in [((0.1, 0.0), 0.0), ((0.0, 0.0), 0.8), ((0.06, 0.03), 0.5)]:
                t, feet, contact, pos, yaw = self.run_gait(gait, v, w)
                world = body.to_world(feet, pos, yaw=yaw)
                planted = contact[1:] & contact[:-1]
                self.assertLess(np.linalg.norm(np.diff(world, axis=0), axis=-1)[planted].max(), 1e-9)
                np.testing.assert_allclose(world[..., 2][contact], -body.STAND_HEIGHT, atol=1e-12)

    def test_contact_patterns(self):
        _, _, contact, _, _ = self.run_gait(gaits.TROT, (0.1, 0), 0)
        self.assertTrue(np.all(contact.sum(1) == 2))
        self.assertTrue(np.all(contact[:, 0] == contact[:, 3]) and np.all(contact[:, 1] == contact[:, 2]))
        _, _, contact, _, _ = self.run_gait(gaits.CRAWL, (0.04, 0), 0)
        self.assertTrue(np.all(contact.sum(1) >= 3))
        np.testing.assert_allclose(contact.mean(0), gaits.CRAWL.duty, atol=0.01)

    def test_swing_lifts_and_is_continuous(self):
        t, feet, contact, _, _ = self.run_gait(gaits.TROT, (0.15, 0), 0, n=2000)
        self.assertAlmostEqual(feet[..., 2].max() + body.STAND_HEIGHT, gaits.TROT.lift, places=4)
        self.assertLess(np.linalg.norm(np.diff(feet, axis=0), axis=-1).max(), 2e-3)

    def test_gait_targets_are_reachable(self):
        for gait, v, w in [(gaits.TROT, (0.15, 0), 0), (gaits.TROT, (0, 0), 1.2), (gaits.CRAWL, (0.04, 0), 0)]:
            t, feet, _, _, _ = self.run_gait(gait, v, w)
            targets = body.apply_posture(feet, offset=gaits.body_sway(gait, t, self.home))
            self.assertTrue(self.robot.reachable(targets).all())

    def test_crawl_keeps_body_centre_over_its_feet(self):
        g = gaits.CRAWL
        t, feet, contact, pos, yaw = self.run_gait(g, (0.04, 0), 0)
        sway = gaits.body_sway(g, t, self.home)
        margin = gaits.stability_margin(body.to_world(feet, pos, yaw=yaw)[..., :2], contact, (pos + sway)[:, :2])
        self.assertGreater(margin.min(), 0.02)
        self.assertEqual(np.abs(gaits.body_sway(gaits.TROT, t, self.home)).max(), 0.0)

    def test_stability_margin_geometry(self):
        sq = np.array([[[1, 1], [1, -1], [-1, 1], [-1, -1]]], float)
        all_down = np.ones((1, 4), bool)
        self.assertAlmostEqual(gaits.stability_margin(sq, all_down, np.array([[0.0, 0.0]]))[0], 1.0)
        self.assertAlmostEqual(gaits.stability_margin(sq, all_down, np.array([[1.5, 0.0]]))[0], -0.5)


if __name__ == "__main__":
    unittest.main()
