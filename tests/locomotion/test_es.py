import unittest

import numpy as np

from mojito.locomotion import es
from mojito.locomotion.policy import MLPPolicy


class TestES(unittest.TestCase):
    def test_gradient_points_uphill_on_a_quadratic(self):
        """For F(x) = -|x - target|^2 the true gradient is 2 (target - x); the ES estimate must align with it."""
        rng = np.random.default_rng(0)
        target, x, sigma = np.array([1.0, -2.0, 0.5, 3.0]), np.zeros(4), 0.1
        noise = rng.normal(size=(400, 4))
        value = lambda y: -np.sum((y - target) ** 2, axis=-1)  # noqa: E731
        gradient = es.es_gradient(value(x + sigma * noise), value(x - sigma * noise), noise, sigma)
        true_gradient = 2 * (target - x)
        cosine = gradient @ true_gradient / np.linalg.norm(gradient) / np.linalg.norm(true_gradient)
        self.assertGreater(cosine, 0.95)

    def test_centred_ranks(self):
        np.testing.assert_allclose(es.centred_ranks([5.0, -1.0, 2.0]), [0.5, -0.5, 0.0])

    def test_adam_converges_on_quadratic(self):
        optimiser, x = es.Adam(2, learning_rate=0.1), np.array([3.0, -4.0])
        for _ in range(400):
            x = optimiser.step(x, -2 * x)  # ascent on -|x|^2
        self.assertLess(np.linalg.norm(x), 0.1)


class TestPolicy(unittest.TestCase):
    def test_starts_at_zero_action(self):
        self.assertTrue(np.all(MLPPolicy(47, 12).act(np.ones(47)) == 0.0))

    def test_parameter_round_trip_and_save(self):
        policy = MLPPolicy(5, 3, hidden=(4,), seed=1)
        policy.set_parameters(np.random.default_rng(2).normal(size=policy.n_parameters))
        observation = np.arange(5.0)
        restored = MLPPolicy(5, 3, hidden=(4,))
        restored.set_parameters(policy.parameters())
        np.testing.assert_allclose(restored.act(observation), policy.act(observation))
        self.assertLessEqual(np.abs(policy.act(observation)).max(), 1.0)
        with self.assertRaises(ValueError):
            policy.set_parameters(np.zeros(3))


if __name__ == "__main__":
    unittest.main()
