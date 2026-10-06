"""Backend switch, and NumPy/PyTorch agreement.

The PyTorch tests run only where PyTorch is installed:
    python -m unittest discover tests
"""
import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
import mojito  # noqa: E402
from mojito import IKNet, backend, body, leg  # noqa: E402

HAS_TORCH = importlib.util.find_spec("torch") is not None
WEIGHTS = os.path.join(ROOT, "weights", "ik_leg.npz")


class TestSwitch(unittest.TestCase):
    def tearDown(self):
        backend._current = None

    def test_default_is_numpy(self):
        env = {k: v for k, v in os.environ.items() if k != "MOJITO_BACKEND"}
        out = subprocess.run([sys.executable, "-c", "import mojito; print(mojito.get_backend())"],
                             cwd=ROOT, env=env, capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "numpy")

    def test_environment_variable_selects_backend(self):
        out = subprocess.run([sys.executable, "-c", "import mojito; print(mojito.get_backend())"],
                             cwd=ROOT, env=dict(os.environ, MOJITO_BACKEND="torch"), capture_output=True, text=True, check=True)
        self.assertEqual(out.stdout.strip(), "torch")

    def test_set_backend_and_per_call_override(self):
        mojito.set_backend("numpy")
        self.assertEqual(mojito.get_backend(), "numpy")
        mojito.set_backend("pytorch")  # accepted spelling
        self.assertEqual(mojito.get_backend(), "torch")
        self.assertIsInstance(mojito.make_model(hidden=(8,), backend="numpy"), IKNet)
        with self.assertRaises(ValueError):
            mojito.set_backend("jax")

    def test_numpy_model_round_trip_and_training_step(self):
        net = mojito.make_model(hidden=(16, 16), seed=1, backend="numpy")
        self.assertEqual(net.backend, "numpy")
        rng = np.random.default_rng(0)
        p, lengths, _ = leg.sample_batch(rng, 256, net.cfg)
        first = net.train_step(p, lengths, 1e-3)
        for _ in range(50):
            last = net.train_step(p, lengths, 1e-3)
        self.assertLess(last, first)
        with tempfile.TemporaryDirectory() as d:
            net.save(os.path.join(d, "n.npz"))
            again = mojito.load_model(os.path.join(d, "n.npz"), backend="numpy")
        np.testing.assert_allclose(again.predict(p, lengths), net.predict(p, lengths))

    @unittest.skipIf(HAS_TORCH, "PyTorch is installed")
    def test_missing_torch_gives_a_clear_message(self):
        with self.assertRaisesRegex(ImportError, "needs PyTorch"):
            mojito.make_model(backend="torch")


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TestTorchMatchesNumpy(unittest.TestCase):
    def setUp(self):
        import torch

        self.torch = torch
        self.rng = np.random.default_rng(5)
        self.np_net = mojito.make_model(hidden=(16, 16), seed=4, backend="numpy")
        self.t_net = mojito.make_model(hidden=(16, 16), seed=4, backend="torch", dtype=torch.float64)
        self.p, self.lengths, self.q = leg.sample_batch(self.rng, 64, self.np_net.cfg)

    def test_forward_kinematics_agree(self):
        from mojito.torch_model import forward_kinematics

        got = forward_kinematics(self.torch.tensor(self.q), self.torch.tensor(self.lengths)).numpy()
        np.testing.assert_allclose(got, leg.forward(self.q, self.lengths), atol=1e-12)

    def test_same_seed_gives_same_predictions(self):
        np.testing.assert_allclose(self.t_net.predict(self.p, self.lengths), self.np_net.predict(self.p, self.lengths), atol=1e-10)
        one = self.t_net.predict(self.p[0], self.np_net.cfg.nominal)  # unbatched, shared lengths
        np.testing.assert_allclose(one, self.np_net.predict(self.p[0], self.np_net.cfg.nominal), atol=1e-10)

    def test_autograd_matches_hand_derived_gradients(self):
        loss_n, grads_n = self.np_net.loss_and_grads(self.p, self.lengths)
        loss_t, grads_t = self.t_net.loss_and_grads(self.p, self.lengths)
        self.assertAlmostEqual(loss_n, loss_t, places=12)
        for a, b in zip(grads_n, grads_t):
            np.testing.assert_allclose(b, a, atol=1e-10)

    def test_training_follows_the_same_path(self):
        for _ in range(20):
            p, lengths, _ = leg.sample_batch(self.rng, 128, self.np_net.cfg)
            ln = self.np_net.train_step(p, lengths, 1e-3)
            lt = self.t_net.train_step(p, lengths, 1e-3)
            self.assertAlmostEqual(ln, lt, places=8)
        np.testing.assert_allclose(self.t_net.predict(self.p, self.lengths), self.np_net.predict(self.p, self.lengths), atol=1e-6)

    def test_weight_files_are_interchangeable(self):
        with tempfile.TemporaryDirectory() as d:
            self.np_net.save(os.path.join(d, "n.npz"))
            self.t_net.save(os.path.join(d, "t.npz"))
            t_from_n = mojito.load_model(os.path.join(d, "n.npz"), backend="torch", dtype=self.torch.float64)
            n_from_t = mojito.load_model(os.path.join(d, "t.npz"), backend="numpy")
        ref = self.np_net.predict(self.p, self.lengths)
        np.testing.assert_allclose(t_from_n.predict(self.p, self.lengths), ref, atol=1e-10)
        np.testing.assert_allclose(n_from_t.predict(self.p, self.lengths), ref, atol=1e-10)
        self.assertEqual(t_from_n.to_json(), self.np_net.to_json())

    @unittest.skipUnless(os.path.exists(WEIGHTS), "trained weights not present")
    def test_trained_weights_run_under_torch_in_float32(self):
        n = mojito.load_model(WEIGHTS, backend="numpy")
        t = mojito.load_model(WEIGHTS, backend="torch")  # default float32
        p, lengths, _ = leg.sample_batch(self.rng, 2000, n.cfg)
        np.testing.assert_allclose(t.predict(p, lengths), n.predict(p, lengths), atol=1e-4)
        err = np.linalg.norm(leg.forward(t.predict(p, lengths), lengths) - p, axis=1)
        self.assertLess(np.median(err), 0.5e-3)
        robot = body.Robot(t)
        self.assertLess(robot.tracking_error(robot.stance()).max(), 0.5e-3)
        self.assertEqual(t.predict_leg(p[:5], lengths[:5], right=True).shape, (5, 3))


if __name__ == "__main__":
    unittest.main()
