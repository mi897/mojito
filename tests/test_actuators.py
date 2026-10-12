"""Run with:  python -m unittest discover tests   (MuJoCo tests skipped if it is not installed)"""
import os
import sys
import unittest

import numpy as np
from scipy.optimize import brentq

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import actuators as A, body, mjcf, robots  # noqa: E402

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

G = 9.81


class TestServoMaths(unittest.TestCase):
    def test_voltage_range_enforced(self):
        for v in (4.9, 6.1, 6.8):
            with self.assertRaises(ValueError):
                A.ServoParams(voltage=v)

    def test_torque_and_speed_interpolation(self):
        p5, p6 = A.ServoParams(voltage=5.0), A.ServoParams(voltage=6.0)
        self.assertAlmostEqual(p5.stall_torque, 19 * A.KGCM)
        # straight line between 5 V (19) and 6.8 V (21.5): 20.39 kg.cm at 6 V
        self.assertAlmostEqual(p6.stall_torque / A.KGCM, 19 + 2.5 / 1.8, places=9)
        self.assertGreater(p6.noload_speed, p5.noload_speed)
        self.assertAlmostEqual(p5.noload_speed, np.radians(60) / 0.16)

    def test_quantisation_is_on_a_grid_within_half_a_step(self):
        s = A.Servo(A.ServoParams(resolution=0.01))
        x = np.random.default_rng(0).uniform(-1, 1, 100)
        err = s.quantise(x) - x
        self.assertLessEqual(np.abs(err).max(), 0.005 + 1e-12)
        np.testing.assert_allclose(s.quantise(x) / 0.01, np.round(s.quantise(x) / 0.01), atol=1e-9)

    def test_latency_delays_the_command(self):
        s = A.Servo(A.ServoParams(latency=0.02, kp=1.0, kd=0.0, resolution=1e-9))
        s.reset()
        out = [s.step([1.0], [0.0], [0.0], 0.005)[0] for _ in range(8)]
        # first call has nothing to hold except the current position (0): zero torque, until the command arrives
        self.assertEqual(out[0], 0.0)
        self.assertAlmostEqual(out[-1], 1.0, places=6)
        first = next(i for i, x in enumerate(out) if x > 0)
        self.assertEqual(first, 4)  # 0.02 s / 0.005 s

    def test_torque_falls_to_zero_at_noload_speed(self):
        p = A.ServoParams(kp=1000.0, kd=0.0, latency=0.0)
        s = A.Servo(p)
        s.reset()
        tau = s.step([1.0], [0.0], [p.noload_speed], 0.001)[0]
        self.assertAlmostEqual(tau, 0.0)
        s.reset()
        self.assertAlmostEqual(s.step([1.0], [0.0], [0.0], 0.001)[0], p.stall_torque)

    def test_sensor_statistics(self):
        p = A.JointSensorParams(noise=0.01, resolution=1e-6, offset=0.02, delay=0.0)
        sens = A.JointSensor(p, seed=1)
        r = np.array([sens.read(np.full(1000, 0.3), 0.001) for _ in range(5)]).ravel()
        self.assertAlmostEqual(r.mean(), 0.32, delta=0.002)
        self.assertAlmostEqual(r.std(), 0.01, delta=0.001)

    def test_sensor_quantises_and_delays(self):
        sens = A.JointSensor(A.JointSensorParams(noise=0.0, resolution=0.1, delay=0.003), seed=0)
        out = [sens.read(np.array([0.04 * k]), 0.001)[0] for k in range(6)]
        np.testing.assert_allclose(out[:3], out[0])  # nothing newer has arrived yet
        self.assertAlmostEqual(out[5], 0.1, places=9)  # reading from 0.003 s ago was 0.08 -> 0.1

    def test_gyro_bias_drifts_like_a_random_walk(self):
        imu = A.Imu(A.ImuParams(gyro_noise=0.0, gyro_bias_walk=0.01, accel_noise=0.0), seed=3)
        end = []
        for seed in range(200):
            imu = A.Imu(A.ImuParams(gyro_noise=0.0, gyro_bias_walk=0.01, accel_noise=0.0), seed=seed)
            for _ in range(1000):
                g, _ = imu.read(np.zeros(3), np.zeros(3), 0.001)
            end.append(g)
        # after 1 s the spread is bias_walk * sqrt(1 s)
        self.assertAlmostEqual(np.std(end), 0.01, delta=0.002)


ARM_XML = """
<mujoco><compiler angle="radian"/><option timestep="0.001" gravity="0 0 -9.81"/>
<worldbody><body name="arm"><joint name="j" type="hinge" axis="0 1 0" range="-1.5 1.5" armature="0.005" damping="0.01"/>
<geom type="capsule" fromto="0 0 0 {L} 0 0" size="0.005" mass="0.0001" contype="0" conaffinity="0"/>
<geom type="sphere" pos="{L} 0 0" size="0.01" mass="{m}" contype="0" conaffinity="0"/></body></worldbody>
<actuator><motor name="j" joint="j" gear="1"/></actuator></mujoco>"""


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestServoHoldingMass(unittest.TestCase):
    """A servo holding a mass on a horizontal arm. The arm swings about +y, so a downward load is a positive rotation."""

    def run_hold(self, m, L, steps=4000, **kw):
        mod = mujoco.MjModel.from_xml_string(ARM_XML.format(m=m, L=L))
        d = mujoco.MjData(mod)
        servo = A.Servo(A.ServoParams(latency=0.0, resolution=1e-9, **kw))
        servo.reset()
        for _ in range(steps):
            d.ctrl[0] = servo.step([0.0], d.qpos[:1], d.qvel[:1], mod.opt.timestep)[0]
            mujoco.mj_step(mod, d)
        return d.qpos[0], d.ctrl[0], servo.p

    def test_settles_at_closed_form_deflection_below_the_limit(self):
        m, L = 0.5, 0.12  # 0.59 N·m of load against 1.86 N·m
        q, tau, p = self.run_hold(m, L)
        # gravity torque about +y is +m g L cos(q); the servo gives kp (0 - q); they balance
        expected = brentq(lambda x: p.kp * (0.0 - x) + m * G * L * np.cos(x), -1.4, 1.4)
        self.assertAlmostEqual(q, expected, delta=2e-3)
        self.assertAlmostEqual(abs(tau), m * G * L * np.cos(q), delta=5e-3)
        self.assertLess(abs(tau), p.stall_torque)

    def test_saturates_and_sags_to_where_the_lever_arm_matches_the_limit(self):
        m, L = 2.0, 0.12  # 2.35 N·m at the horizontal against 1.86 N·m available
        q, tau, p = self.run_hold(m, L)
        self.assertAlmostEqual(abs(tau), p.stall_torque, delta=1e-3)  # a little under: the arm is still moving slightly
        # the arm sags until the load torque m g L cos(q) falls to the stall torque
        self.assertAlmostEqual(q, np.arccos(p.stall_torque / (m * G * L)), delta=5e-3)
        # and with the kp error that large the command is far from the position reached
        self.assertGreater(abs(q), p.stall_torque / p.kp)

    def test_gap_between_command_and_actual_at_the_limit(self):
        # just under the limit the gap is about stall/kp: the quantity the plan wants to see
        L = 0.12
        m = 0.95 * 1.8632 / (G * L)
        q, tau, p = self.run_hold(m, L, steps=6000)
        self.assertLess(abs(q), 1.5)
        self.assertGreater(abs(q), 0.5 * p.stall_torque / p.kp)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestActuatedRobot(unittest.TestCase):
    def test_actuated_model_has_motors_and_imu(self):
        model = robots.quadruped_urdf(0.30, 0.15)
        m = mujoco.MjModel.from_xml_string(mjcf.mjcf_from_urdf(model, height=0.3, actuated=True))
        self.assertEqual(m.nu, 12)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        self.assertEqual(d.sensor("accel").data.shape, (3,))
        self.assertEqual(d.sensor("gyro").data.shape, (3,))


if __name__ == "__main__":
    unittest.main()
