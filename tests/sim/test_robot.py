"""SimRobot, the placeholder designs and the batched replay command.  Run with:  python -m unittest discover tests"""
import dataclasses
import unittest

import numpy as np

from mojito.geometry import placeholders
from mojito.robot import body, gaits, leg, statics
from mojito.sim import actuators, gait_replay, mjcf, stand
from mojito.sim import robot as sim_robot_module

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


class TestDesigns(unittest.TestCase):
    def test_stage1_design_is_the_kinematic_placeholder(self):
        design = placeholders.STAGE1_QUADRUPED
        np.testing.assert_array_equal(design.lengths, leg.NOMINAL_LENGTHS)
        self.assertEqual((design.body_length, design.body_width), (body.BODY_LENGTH, body.BODY_WIDTH))
        self.assertEqual(design.stance_height, body.STAND_HEIGHT)
        self.assertIsNone(design.masses)

    def test_sim_design_feeds_the_model_defaults(self):
        design = placeholders.SIM_QUADRUPED
        self.assertIs(mjcf.DEFAULT_MASSES, design.masses)
        self.assertEqual(gait_replay.DEFAULT_STANCE_HEIGHT, design.stance_height)
        # body plus four legs of hip, thigh, shank and foot, summed by hand
        self.assertAlmostEqual(design.total_mass, 0.8 + 4 * (0.06 + 0.10 + 0.06 + 0.01), places=12)
        self.assertEqual(mjcf.GRAVITY, -statics.G)

    def test_a_design_is_hashable_and_comparable(self):
        changed = dataclasses.replace(placeholders.SIM_QUADRUPED, upper_length=0.2)
        self.assertNotEqual(changed, placeholders.SIM_QUADRUPED)
        self.assertEqual(len({changed, placeholders.SIM_QUADRUPED, dataclasses.replace(changed)}), 2)


class TestAngles(unittest.TestCase):
    def test_roll_pitch_yaw_inverts_the_body_rotation(self):
        rng = np.random.default_rng(5)
        for _ in range(20):
            roll, pitch, yaw = rng.uniform(-1.2, 1.2, 3)
            rotation = body.rotation(roll, pitch, yaw)
            np.testing.assert_allclose(sim_robot_module.roll_pitch_yaw(rotation), (roll, pitch, yaw), atol=1e-12)
            # the up axis R e_z makes the angle arccos(cos(roll) cos(pitch)) with vertical
            self.assertAlmostEqual(float(sim_robot_module.tilt_deg(rotation)), np.degrees(np.arccos(np.cos(roll) * np.cos(pitch))),
                                   places=9)


class TestCommandedFeet(unittest.TestCase):
    def test_batched_command_equals_one_call_per_step(self):
        # the per-step computation run_gait used before it was batched, written out again
        kinematics = body.Robot(body.AnalyticSolver(), lengths=placeholders.SIM_QUADRUPED.lengths,
                                body_length=placeholders.SIM_QUADRUPED.body_length, body_width=placeholders.SIM_QUADRUPED.body_width)
        home = kinematics.stance(0.2125)
        stand_time, ramp_time, timestep = 0.1, 0.05, 0.002
        times = np.arange(400) * timestep
        for gait, velocity, yaw_rate in ((gaits.CRAWL, (0.04, 0.01), 0.0), (gaits.TROT, (0.15, 0.0), 0.6)):
            targets, contact = gait_replay.commanded_feet(gait, home, times, stand_time, ramp_time, velocity, yaw_rate)
            for step_index in range(len(times)):
                time = step_index * timestep
                gait_time = max(0.0, time - stand_time)
                blend = float(stand.smoothstep((time - stand_time) / ramp_time))
                feet, step_contact, _ = gaits.foot_targets(gait, gait_time, home, velocity, yaw_rate)
                if blend == 0.0:
                    expected = home
                else:
                    walking = body.apply_posture(feet, offset=gaits.body_sway(gait, gait_time, home))[0]
                    expected = home + blend * (walking - home)
                np.testing.assert_array_equal(targets[step_index], expected)
                np.testing.assert_array_equal(contact[step_index], step_contact[0])
                np.testing.assert_array_equal(kinematics.solve(targets[step_index]), kinematics.solve(expected))


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestSimRobot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sim_robot = sim_robot_module.SimRobot()
        servo = actuators.Servo(actuators.ServoParams())
        joint_targets = cls.sim_robot.kinematics.solve(cls.sim_robot.kinematics.stance(cls.sim_robot.stance_height)).ravel()
        for _ in range(500):  # hold the standing posture for 1 s
            cls.sim_robot.step(servo.step(joint_targets, cls.sim_robot.joint_angles(), cls.sim_robot.joint_velocities(), 0.002))

    def test_starts_at_the_design_stance(self):
        fresh = sim_robot_module.SimRobot()
        self.assertEqual(fresh.stance_height, placeholders.SIM_QUADRUPED.stance_height)
        self.assertAlmostEqual(fresh.origin()[2], placeholders.SIM_QUADRUPED.stance_height + mjcf.FOOT_RADIUS, places=12)
        np.testing.assert_allclose(fresh.feet_world()[:, 2], mjcf.FOOT_RADIUS, atol=1e-9)  # pads just touching the floor
        self.assertAlmostEqual(fresh.total_mass, placeholders.SIM_QUADRUPED.total_mass, places=9)
        self.assertEqual(len(fresh.pad_legs), 4)

    def test_foot_loads_carry_the_weight(self):
        loads = self.sim_robot.foot_loads()
        self.assertTrue((loads > 0).all())
        self.assertAlmostEqual(loads.sum() / (self.sim_robot.total_mass * statics.G), 1.0, delta=0.02)

    def test_readouts_agree_with_each_other(self):
        sim_robot = self.sim_robot
        rotation, origin = sim_robot.body_rotation(), sim_robot.origin()
        self.assertLess(sim_robot.tilt_deg(), 2.0)
        self.assertAlmostEqual(sim_robot.yaw(), float(sim_robot_module.yaw(rotation)))
        # feet_world in the body frame is mjcf.foot_positions
        feet_body = (sim_robot.feet_world() - origin) @ rotation
        np.testing.assert_allclose(feet_body, mjcf.foot_positions(sim_robot.model, sim_robot.data, [f"{n}_foot" for n in body.LEG_NAMES]))
        np.testing.assert_array_equal(sim_robot.legs_world()[:, 2], sim_robot.feet_world())
        snapshot = sim_robot.snapshot()
        np.testing.assert_array_equal(snapshot["legs"]["RL"], sim_robot.legs_world()[2])
        np.testing.assert_array_equal(snapshot["R"], rotation)

    def test_a_turned_base_reads_back_its_yaw_and_tilt(self):
        fresh = sim_robot_module.SimRobot()
        roll, yaw = 0.1, 0.3
        mjcf.set_pose(fresh.model, fresh.data, sim_robot_module.joint_names(), fresh.joint_angles(),
                      base_pos=(0.0, 0.0, 0.5), base_rotation=body.rotation(roll, 0.0, yaw))
        self.assertAlmostEqual(fresh.yaw(), yaw, places=9)
        self.assertAlmostEqual(fresh.tilt_deg(), np.degrees(roll), places=7)


if __name__ == "__main__":
    unittest.main()
