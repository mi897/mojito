"""Run with:  python -m unittest discover tests   (skipped if MuJoCo is not installed)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import gait_replay, gaits, live_sim  # noqa: E402

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mujoco = None


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestLiveSim(unittest.TestCase):
    def test_constant_parameters_reproduce_the_batch_replay(self):
        sim = live_sim.LiveSim(live_sim.PRESETS["trot"])
        sim.step(500)  # stand for 1 s, as run_gait does
        sim.set_params(walk=True)
        sim.step(1000)
        r = gait_replay.run_gait(gaits.TROT, (0.15, 0.0), cycles=4, stand_time=1.0, ramp_time=0.5)
        origin = sim.data.site("body_origin").xpos
        # the target-speed limit and the accumulated cycle position make this close, not bit-identical
        np.testing.assert_allclose(origin, r["origin_m"][1499], atol=2e-3)

    def test_changing_the_period_does_not_jump_the_phase(self):
        sim = live_sim.LiveSim(live_sim.PRESETS["trot"])
        sim.set_params(walk=True)
        sim.step(900)
        before = sim.cycles
        sim.set_params(period=1.3)
        sim.step(1)
        self.assertAlmostEqual(sim.cycles - before, sim.timestep / 1.3, places=9)

    def test_parameters_are_clamped_and_mass_change_asks_for_a_rebuild(self):
        sim = live_sim.LiveSim()
        self.assertFalse(sim.set_params(duty=5.0, kp=-3.0))
        self.assertEqual(sim.params["duty"], live_sim.LIMITS["duty"][1])
        self.assertEqual(sim.params["kp"], live_sim.LIMITS["kp"][0])
        self.assertTrue(sim.set_params(mass=3.0))
        sim.reset()
        self.assertAlmostEqual(sim.snapshot()["total_mass_kg"], 3.0, places=2)

    def test_servo_gains_change_live(self):
        sim = live_sim.LiveSim()
        sim.set_params(kp=55.0, kd=0.7)
        self.assertEqual((sim.servo.params.kp, sim.servo.params.kd), (55.0, 0.7))

    def test_random_changes_while_walking_stay_finite(self):
        rng = np.random.default_rng(0)
        sim = live_sim.LiveSim(live_sim.PRESETS["trot"])
        sim.set_params(walk=True)
        for _ in range(40):
            sim.set_params(period=rng.uniform(0.3, 2.0), duty=rng.uniform(0.3, 0.9), lift=rng.uniform(0, 0.06),
                           vx=rng.uniform(-0.2, 0.3), yaw_rate=rng.uniform(-1, 1), offsets=list(rng.uniform(0, 1, 4)),
                           kp=rng.uniform(5, 100))
            sim.step(100)
            snap = sim.snapshot()
            self.assertTrue(np.isfinite(np.array(snap["legs"])).all())
            self.assertTrue(np.isfinite(snap["origin"]).all())

    def test_snapshot_is_json_ready(self):
        import json

        sim = live_sim.LiveSim()
        sim.step(50)
        json.dumps(sim.snapshot())


if __name__ == "__main__":
    unittest.main()
