import unittest

import numpy as np

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mujoco = None

if mujoco is not None:
    from mojito.locomotion import env as loco
    from mojito.locomotion.rollout import rollout


@unittest.skipIf(mujoco is None, "needs mujoco")
class TestEnv(unittest.TestCase):
    def setUp(self):
        self.env = loco.LocomotionEnv(loco.EnvConfig(episode_time=1.0, randomisation=loco.Randomisation.none()))

    def test_observation_size(self):
        self.assertEqual(self.env.reset(0).shape, (loco.N_OBSERVATIONS,))

    def test_zero_action_is_the_scripted_foot_target(self):
        """With action 0 the commanded feet are the scripted trot's: the joint targets follow `scripted_feet` through the IK."""
        self.env.reset(0, command=[0.1, 0.0, 0.0])
        feet = self.env.scripted_feet(0.5)
        direct = loco.commanded_feet(self.env.config.gait, self.env.home, [0.5], 0.0, loco.RAMP_TIME, (0.1, 0.0), 0.0)[0][0]
        np.testing.assert_allclose(feet, direct)
        np.testing.assert_allclose(self.env.scripted_feet(0.0), self.env.home)  # blend is zero at policy time 0

    def test_deterministic_given_seed(self):
        first = rollout(self.env, None, 3)
        second = rollout(self.env, None, 3)
        self.assertEqual(first["return"], second["return"])

    def test_scripted_trot_walks_forward_without_falling(self):
        result = rollout(self.env, None, 0, command=[0.15, 0.0, 0.0])
        self.assertFalse(result["fell"])
        self.assertGreater(result["mean_velocity_m_s"][0], 0.0)

    def test_observation_has_no_ground_truth_velocity(self):
        """Changing the true body velocity must not change the observation (only sensors do)."""
        self.env.reset(0)
        before = self.env.observation().copy()
        self.env._velocity_world = np.array([5.0, 5.0, 5.0])
        np.testing.assert_array_equal(before, self.env.observation())

    def test_randomisation_changes_the_robot(self):
        env = loco.LocomotionEnv(loco.EnvConfig(episode_time=0.2))
        env.reset(1)
        first = dict(env.domain)
        env.reset(2)
        self.assertNotEqual(first, env.domain)
        self.assertAlmostEqual(env.physics.total_mass, loco.SIM_QUADRUPED.total_mass * env.domain["mass_scale"], places=6)


if __name__ == "__main__":
    unittest.main()
