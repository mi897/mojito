"""Run with:  python -m unittest discover tests"""
import os
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import IKNet, leg  # noqa: E402


class TestLeg(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.cfg = leg.LegConfig()

    def test_rest_pose(self):
        la, lu, ll = self.cfg.nominal
        np.testing.assert_allclose(leg.forward(np.zeros(3), self.cfg.nominal), [0, la, -(lu + ll)], atol=1e-12)

    def test_jacobian_matches_finite_differences(self):
        p, lengths, q = leg.sample_batch(self.rng, 200, self.cfg)
        J = leg.jacobian(q, lengths)
        eps = 1e-6
        for j in range(3):
            d = np.zeros(3)
            d[j] = eps
            num = (leg.forward(q + d, lengths) - leg.forward(q - d, lengths)) / (2 * eps)
            np.testing.assert_allclose(J[:, :, j], num, atol=1e-8)

    def test_analytic_ik_recovers_angles(self):
        for uniform in (True, False):
            p, lengths, q = leg.sample_batch(self.rng, 20000, self.cfg, workspace_uniform=uniform)
            np.testing.assert_allclose(leg.analytic_ik(p, lengths), q, atol=1e-7)

    def test_samples_respect_limits_with_varied_lengths(self):
        p, lengths, q = leg.sample_batch(self.rng, 20000, self.cfg)
        self.assertTrue(np.all(q >= self.cfg.limits[:, 0]) and np.all(q <= self.cfg.limits[:, 1]))
        dev = np.abs(lengths / self.cfg.nominal - 1)
        self.assertLessEqual(dev.max(), self.cfg.length_tolerance + 1e-12)
        self.assertGreater(dev.std(), 0.01)

    def test_workspace_resampling_favours_open_poses(self):
        lengths = leg.sample_lengths(self.rng, 50000, self.cfg)
        flat = leg.sample_joint_angles(self.rng, lengths, self.cfg, workspace_uniform=False)
        even = leg.sample_joint_angles(self.rng, lengths, self.cfg, workspace_uniform=True)
        det = lambda q: np.abs(np.linalg.det(leg.jacobian(q, lengths))).mean()
        self.assertGreater(det(even), 1.04 * det(flat))

    def test_joint_positions_end_at_foot(self):
        p, lengths, q = leg.sample_batch(self.rng, 100, self.cfg)
        pts = leg.joint_positions(q, lengths)
        np.testing.assert_allclose(pts[:, 3], p, atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(pts[:, 1] - pts[:, 0], axis=1), lengths[:, 0], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(pts[:, 2] - pts[:, 1], axis=1), lengths[:, 1], atol=1e-12)
        np.testing.assert_allclose(np.linalg.norm(pts[:, 3] - pts[:, 2], axis=1), lengths[:, 2], atol=1e-12)

    def test_right_leg_is_mirror_of_left(self):
        p, lengths, q = leg.sample_batch(self.rng, 100, self.cfg)
        # A right leg has its abduction offset on the other side.
        right = leg.forward(leg.mirror_angles(q), lengths * np.array([-1, 1, 1]))
        np.testing.assert_allclose(right, leg.mirror_target(p), atol=1e-12)


class TestModel(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)
        self.net = IKNet(hidden=(16, 16), seed=3)

    def test_backprop_matches_finite_differences(self):
        p, lengths, _ = leg.sample_batch(self.rng, 32, self.net.cfg)
        _, grads = self.net.loss_and_grads(p, lengths)
        eps = 1e-6
        for param, grad in zip(self.net.params, grads):
            for _ in range(5):
                idx = tuple(self.rng.integers(0, s) for s in param.shape)
                old = param[idx]
                param[idx] = old + eps
                up, _ = self.net.loss_and_grads(p, lengths)
                param[idx] = old - eps
                down, _ = self.net.loss_and_grads(p, lengths)
                param[idx] = old
                self.assertAlmostEqual((up - down) / (2 * eps), grad[idx], delta=1e-7)

    def test_outputs_always_within_joint_limits(self):
        wild = self.rng.normal(0, 5, size=(1000, 3))
        q = self.net.predict(wild, self.net.cfg.nominal)
        lim = self.net.cfg.limits
        self.assertTrue(np.all(q >= lim[:, 0]) and np.all(q <= lim[:, 1]))

    def test_save_load_and_json_roundtrip(self):
        p, lengths, _ = leg.sample_batch(self.rng, 16, self.net.cfg)
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "net.npz")
            self.net.save(path)
            again = IKNet.load(path)
        np.testing.assert_allclose(again.predict(p, lengths), self.net.predict(p, lengths))
        self.assertIn('"limits"', self.net.to_json())

    def test_right_leg_prediction_mirrors_left(self):
        p, lengths, _ = leg.sample_batch(self.rng, 16, self.net.cfg)
        left = self.net.predict_leg(p, lengths)
        right = self.net.predict_leg(leg.mirror_target(p), lengths, right=True)
        np.testing.assert_allclose(right, leg.mirror_angles(left))


if __name__ == "__main__":
    unittest.main()
