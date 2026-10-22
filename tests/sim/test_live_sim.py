"""Run with:  python -m unittest discover tests   (skipped if MuJoCo is not installed)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.robot import gaits
from mojito.sim import gait_replay, live_sim  # noqa: E402

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
        r = gait_replay.run_gait(gaits.TROT, (0.15, 0.0), cycles=4, stand_time=1.0, ramp_time=0.5,
                                 **gait_replay.settings_for(gaits.TROT))
        origin = sim.data.site("body_origin").xpos
        # the target-speed limit and the accumulated cycle position make this close, not bit-identical
        np.testing.assert_allclose(origin, r["origin_m"][1499], atol=2e-3)

    def test_presets_match_the_batch_replay_settings(self):
        # the live explorer's presets must start from the same settings the replays use, trot tuned, crawl and pace original
        for name, gait in (("trot", gaits.TROT), ("crawl", gaits.CRAWL)):
            settings = gait_replay.settings_for(gait)
            preset = live_sim.PRESETS[name]
            self.assertEqual(preset["height"], settings["height"])
            self.assertEqual(preset["kp"], settings["servo"].kp)
            self.assertEqual(preset["kd"], settings["servo"].kd)
            self.assertEqual(preset["lift"], gait.lift)
        self.assertEqual(live_sim.PRESETS["pace"]["kp"], gait_replay.settings_for(gaits.CRAWL)["servo"].kp)

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

    def test_drive_limits_are_inside_the_slider_limits(self):
        for name, full in live_sim.DRIVE.items():
            self.assertIn(name, live_sim.PRESETS)
            for key, value in full.items():
                low, high = live_sim.LIMITS[key]
                self.assertTrue(low <= -value and value <= high, (name, key))

    def test_full_drive_command_does_not_fall(self):
        # the page sends exactly these values at full stick (docs/STAGE2.md): all three at once, the worst case
        for name, full in live_sim.DRIVE.items():
            sim = live_sim.LiveSim(dict(live_sim.PRESETS[name], walk=True, **full))
            sim.step(int(4.0 / sim.timestep))
            self.assertFalse(sim.snapshot()["fell"], name)

    def test_pose_tilts_the_body_on_planted_feet(self):
        # conservation: the feet stay where they are in the world while the body tilts over them
        sim = live_sim.LiveSim(dict(live_sim.PRESETS["trot"], walk=False, body_roll=0.15, body_pitch=-0.15, body_yaw=0.2))
        sim.step(int(1.5 / sim.timestep))
        before = np.array(sim.snapshot()["legs"])[:, 2]
        sim.step(int(1.5 / sim.timestep))
        snap = sim.snapshot()
        self.assertLess(np.abs(np.array(snap["legs"])[:, 2] - before).max(), 0.005)
        self.assertEqual(snap["feet_down"], 4)
        self.assertAlmostEqual(snap["roll_deg"], np.degrees(0.15), delta=2.0)
        self.assertAlmostEqual(snap["pitch_deg"], np.degrees(-0.15), delta=2.0)

    def test_full_pose_stick_corners_hold(self):
        # the page sends these values at full stick; every corner of the stick box: no saturation, all four feet down
        import itertools

        for signs in itertools.product((-1, 1), repeat=3):
            sim = live_sim.LiveSim(dict(live_sim.PRESETS["trot"], walk=False, **{k: s * v for (k, v), s in zip(live_sim.POSE.items(), signs)}))
            sim.step(int(3.0 / sim.timestep))
            snap = sim.snapshot()
            self.assertEqual(snap["feet_down"], 4, signs)
            self.assertFalse(snap["saturated"] or snap["fell"], signs)

    def test_pose_limits_are_inside_the_slider_limits(self):
        for key, value in live_sim.POSE.items():
            low, high = live_sim.LIMITS[key]
            self.assertTrue(low <= -value and value <= high, key)


if __name__ == "__main__":
    unittest.main()
