"""Run with:  python -m unittest discover tests"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito import IKNet  # noqa: E402
from mojito.robot import body, gaits, leg

WEIGHTS = os.path.join(os.path.dirname(__file__), "..", "..", "weights", "ik_leg.npz")


def _reference_stability_margin(feet_xy, contact, point_xy):
    """The per-time-step loop implementation that gaits.stability_margin replaced, kept as the independent reference."""
    walk_order = [0, 1, 3, 2]  # FL, FR, RR, RL: around the body, clockwise seen from above
    margin = np.empty(feet_xy.shape[0])
    for i in range(feet_xy.shape[0]):
        grounded = np.array([feet_xy[i, k] for k in walk_order if contact[i, k]])
        point = point_xy[i]
        if len(grounded) == 0:
            margin[i] = -np.inf
            continue
        distinct = [foot for j, foot in enumerate(grounded) if np.linalg.norm(foot - grounded[j - 1]) > 1e-12]
        grounded = np.array(distinct) if distinct else grounded[:1]
        if len(grounded) < 3:
            start, end = grounded[0], grounded[-1]
            edge = end - start
            fraction = np.clip(np.dot(point - start, edge) / max(np.dot(edge, edge), 1e-12), 0, 1)
            margin[i] = -np.linalg.norm(point - (start + fraction * edge))
            continue
        signed_distances = []
        for j in range(len(grounded)):
            start, end = grounded[j], grounded[(j + 1) % len(grounded)]
            edge = end - start
            cross = edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0])
            signed_distances.append(cross / np.linalg.norm(edge))
        margin[i] = (-np.array(signed_distances)).min()
    return margin


class TestBody(unittest.TestCase):
    def setUp(self):
        self.exact = body.Robot(body.AnalyticSolver())
        self.home = self.exact.stance()

    def test_rotation_is_the_product_of_elementary_rotations_and_matches_urdf_rpy(self):
        from mojito.geometry import spec

        roll, pitch, yaw = 0.3, -0.2, 0.7
        c, s_ = np.cos, np.sin
        rx = np.array([[1, 0, 0], [0, c(roll), -s_(roll)], [0, s_(roll), c(roll)]])
        ry = np.array([[c(pitch), 0, s_(pitch)], [0, 1, 0], [-s_(pitch), 0, c(pitch)]])
        rz = np.array([[c(yaw), -s_(yaw), 0], [s_(yaw), c(yaw), 0], [0, 0, 1]])
        np.testing.assert_allclose(body.rotation(roll, pitch, yaw), rz @ ry @ rx, atol=1e-15)
        urdf_matrix = spec.rpy_matrix((roll, pitch, yaw))
        self.assertIsInstance(urdf_matrix, np.ndarray)
        self.assertEqual(urdf_matrix.shape, (3, 3))
        np.testing.assert_array_equal(urdf_matrix, body.rotation(roll, pitch, yaw))
        batch = body.rotation(np.array([roll, 0.0]), pitch, yaw)
        np.testing.assert_array_equal(batch[0], body.rotation(roll, pitch, yaw))

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
    def test_trot_is_1p7_s_22_mm_and_the_original_stage1_trot_differs_in_period_and_swing_height(self):
        self.assertEqual(gaits.TROT.period, 1.7)
        self.assertEqual(gaits.TROT.lift, 0.022)
        self.assertEqual(gaits.TROT_STAGE1.period, 0.5)
        self.assertEqual(gaits.TROT_STAGE1.lift, 0.03)
        for field in ("duty", "offsets", "sway"):
            self.assertEqual(getattr(gaits.TROT_STAGE1, field), getattr(gaits.TROT, field))
        # same foot path shape: a stance still covers velocity * stance_time, which grows with the period
        self.assertAlmostEqual(gaits.TROT.stance_time, 0.85)

    def test_sequence_offsets_reproduce_the_crawl_and_validate(self):
        g = gaits.CRAWL
        # lift order FR, RL, FL, RR with the first lift at phase 0.1 is the built-in crawl
        np.testing.assert_allclose(gaits.sequence_offsets((1, 2, 0, 3), g.duty), g.offsets, atol=1e-12)
        np.testing.assert_allclose(g.swing_starts(), [0.6, 0.1, 0.35, 0.85], atol=1e-12)
        self.assertAlmostEqual(g.stance_time + g.swing_time, g.period)
        with self.assertRaises(ValueError):
            gaits.Gait("bad", period=1.0, duty=1.0, offsets=(0, 0, 0, 0))
        with self.assertRaises(ValueError):
            gaits.Gait("bad", period=1.0, duty=0.5, offsets=(0, 0, 0))

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

    def test_stability_margin_with_no_foot_down_is_minus_infinity(self):
        sq = np.array([[[1, 1], [1, -1], [-1, 1], [-1, -1]]], float)
        self.assertEqual(gaits.stability_margin(sq, np.zeros((1, 4), bool), np.array([[0.0, 0.0]]))[0], -np.inf)

    def test_stability_margin_counts_coincident_feet_once(self):
        point = np.array([[0.5, 0.2]])
        # FL and FR on the same spot (1, 0), RL at (-1, 0): the support is the segment from (-1, 0) to (1, 0), 0.2 below the point
        feet = np.array([[[1, 0], [1, 0], [-1, 0], [5, 5]]], float)
        three = np.array([[True, True, True, False]])
        self.assertAlmostEqual(gaits.stability_margin(feet, three, point)[0], -0.2)
        # two feet on one spot: minus the distance to that spot
        two = np.array([[True, True, False, False]])
        self.assertAlmostEqual(gaits.stability_margin(feet, two, point)[0], -np.hypot(0.5, 0.2))
        # four feet, RR on RL: the triangle FL (1, 1), FR (1, -1), RL (-1, -1); the point (0.8, -0.5) is nearest the edge x = 1 (0.2;
        # 0.5 from y = -1, 1.3 / sqrt 2 from y = x)
        feet = np.array([[[1, 1], [1, -1], [-1, -1], [-1, -1]]], float)
        margin = gaits.stability_margin(feet, np.ones((1, 4), bool), np.array([[0.8, -0.5]]))[0]
        self.assertAlmostEqual(margin, 0.2)

    def assert_matches_reference(self, feet, contact, point):
        # Same operations except np.linalg.norm vs sqrt of a sum of squares, so equal to rounding: tolerance 1e-12.
        new = gaits.stability_margin(feet, contact, point)
        old = _reference_stability_margin(feet, contact, point)
        self.assertEqual(new.shape, (feet.shape[0],))
        np.testing.assert_array_equal(np.isinf(new), np.isinf(old))
        np.testing.assert_array_equal(np.isnan(new), np.isnan(old))
        finite = np.isfinite(old)
        np.testing.assert_array_equal(new[~finite & ~np.isnan(old)], old[~finite & ~np.isnan(old)])
        np.testing.assert_allclose(new[finite], old[finite], rtol=0, atol=1e-12)

    def test_stability_margin_matches_loop_reference_on_every_contact_pattern(self):
        rng = np.random.default_rng(0)
        patterns = np.array([[(code >> k) & 1 for k in range(4)] for code in range(16)], bool)
        for _ in range(50):
            feet = rng.uniform(-1, 1, (16, 4, 2))
            self.assert_matches_reference(feet, patterns, rng.uniform(-1.5, 1.5, (16, 2)))

    def test_stability_margin_matches_loop_reference_on_random_batches(self):
        rng = np.random.default_rng(1)
        n = 5000
        feet = rng.uniform(-1, 1, (n, 4, 2))
        contact = rng.random((n, 4)) < 0.6
        self.assert_matches_reference(feet, contact, rng.uniform(-1.5, 1.5, (n, 2)))
        # a rectangle-like stance with small jitter, as a walking robot has, mostly inside
        base = np.array([[0.2, 0.1], [0.2, -0.1], [-0.2, 0.1], [-0.2, -0.1]])
        feet = base + rng.normal(0, 0.02, (n, 4, 2))
        self.assert_matches_reference(feet, rng.random((n, 4)) < 0.8, rng.normal(0, 0.05, (n, 2)))

    def test_stability_margin_matches_loop_reference_on_coincident_feet(self):
        rng = np.random.default_rng(2)
        patterns = np.array([[(code >> k) & 1 for k in range(4)] for code in range(16)], bool)
        # which feet share a position: neighbours, wrap-around (RL on FL), diagonals, all four, and a few more
        groups = [(0, 1), (1, 3), (3, 2), (2, 0), (0, 3), (1, 2), (0, 1, 3), (0, 1, 3, 2), (0, 2, 3)]
        for group in groups:
            feet = rng.uniform(-1, 1, (16, 4, 2))
            for k in group[1:]:
                feet[:, k] = feet[:, group[0]]
            self.assert_matches_reference(feet, patterns, rng.uniform(-1.5, 1.5, (16, 2)))
        # two coincident pairs at once
        feet = rng.uniform(-1, 1, (16, 4, 2))
        feet[:, 1], feet[:, 3] = feet[:, 0], feet[:, 2]
        self.assert_matches_reference(feet, patterns, rng.uniform(-1.5, 1.5, (16, 2)))
        # all grounded feet equal, other feet elsewhere
        feet = np.zeros((16, 4, 2)) + np.array([0.3, -0.2])
        self.assert_matches_reference(feet, patterns, rng.uniform(-1.5, 1.5, (16, 2)))
        # grounded feet within the 1e-12 merge threshold
        feet = np.zeros((16, 4, 2))
        feet[:, 1] = 5e-13
        feet[:, 2] = (1.0, 0.0)
        self.assert_matches_reference(feet, patterns, rng.uniform(-1.5, 1.5, (16, 2)))

    def test_stability_margin_diagonal_coincident_feet_keep_non_convex_polygon(self):
        # FL and RR on the same spot, FR and RL elsewhere: not merged (only neighbours are), so the four-edge
        # polygon FL, FR, RR, RL is non-convex; the answer is the minimum over its four edges, as in the reference.
        feet = np.array([[[1, 1], [1, -1], [-1, 1], [1, 1]]], float)
        point = np.array([[0.5, 0.2]])
        self.assert_matches_reference(feet, np.ones((1, 4), bool), point)

    def test_stability_margin_no_foot_down_is_minus_infinity_in_a_batch(self):
        rng = np.random.default_rng(3)
        margin = gaits.stability_margin(rng.uniform(-1, 1, (7, 4, 2)), np.zeros((7, 4), bool), rng.uniform(-1, 1, (7, 2)))
        self.assertTrue(np.all(margin == -np.inf))

    def test_stability_margin_is_zero_on_an_edge(self):
        rng = np.random.default_rng(4)
        sq = np.array([[1, 1], [1, -1], [-1, 1], [-1, -1]], float)
        feet = np.tile(sq, (200, 1, 1)) * rng.uniform(0.5, 2, (200, 1, 1))
        # a point a fraction along the edge FL -> FR, and along FR -> RR
        fraction = rng.random((200, 1))
        on_front = feet[:, 0] + fraction * (feet[:, 1] - feet[:, 0])
        on_right = feet[:, 1] + fraction * (feet[:, 3] - feet[:, 1])
        for point in (on_front, on_right):
            margin = gaits.stability_margin(feet, np.ones((200, 4), bool), point)
            np.testing.assert_allclose(margin, 0.0, rtol=0, atol=1e-12)
        # also on the segment between two grounded feet
        margin = gaits.stability_margin(feet, np.tile([True, True, False, False], (200, 1)), on_front)
        np.testing.assert_allclose(margin, 0.0, rtol=0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
