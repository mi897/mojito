"""Run with:  python -m unittest discover tests   (MuJoCo tests skipped if it is not installed)"""
import math
import os
import sys
import unittest

import numpy as np

try:
    from scipy.optimize import brentq
except ImportError:  # pragma: no cover
    brentq = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.geometry import robots
from mojito.robot import body
from mojito.sim import actuators as A, mjcf  # noqa: E402

from .clock_helpers import fast_forward  # noqa: E402

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

    def seen_send_steps(self, latency, dt, n_steps):
        """Per step, the index of the step whose command the servo holds (-1 before any has arrived).
        Step k commands the angle (k + 1) * 1e-6 rad; with kp 1, no damping and the joint at 0 the torque is that angle."""
        s = A.Servo(A.ServoParams(latency=latency, kp=1.0, kd=0.0, resolution=1e-9), n_joints=1)
        out = np.array([s.step([(k + 1) * 1e-6], [0.0], [0.0], dt)[0] for k in range(n_steps)])
        return np.round(out / 1e-6).astype(int) - 1

    def test_command_is_seen_at_the_first_step_at_or_after_send_time_plus_latency(self):
        # Reference in exact rational arithmetic: step j sees the latest k with j dt >= k dt + latency.
        from fractions import Fraction
        for latency, dt in (("0.02", "0.002"), ("0.0195", "0.002"), ("0.0181", "0.002"), ("0.022", "0.002"), ("0", "0.002"),
                            ("0.02", "0.005")):
            delay = math.ceil(Fraction(latency) / Fraction(dt))
            n_steps = 2000
            seen = self.seen_send_steps(float(latency), float(dt), n_steps)
            expected = np.array([j - delay if j >= delay else -1 for j in range(n_steps)])
            np.testing.assert_array_equal(seen, expected, err_msg=f"latency {latency} s, dt {dt} s")

    def test_latency_is_still_whole_steps_after_8192_s(self):
        # An accumulated float time drifts by more than 1e-12 s here: comparing it gave 5 steps instead of 4 from step 1,638,399.
        s = A.Servo(A.ServoParams(latency=0.02, kp=1.0, kd=0.0, resolution=1e-9), n_joints=1)
        start, dt = 1_638_380, 0.005
        fast_forward(s, start, dt)
        out = np.array([s.step([(k + 1) * 1e-6], [0.0], [0.0], dt)[0] for k in range(60)])
        seen = np.round(out / 1e-6).astype(int) - 1  # index (from `start`) of the command the servo holds
        np.testing.assert_array_equal(seen[4:], np.arange(60 - 4))

    def test_latency_stays_a_whole_number_of_steps_over_a_long_run(self):
        # 200 s at 0.002 s: the delay is still exactly 0.02 / 0.002 = 10 steps at the end
        seen = self.seen_send_steps(0.02, 0.002, 100_000)
        np.testing.assert_array_equal(seen[-1000:], np.arange(100_000 - 1000, 100_000) - 10)

    def test_last_saturated_exists_before_the_first_step_and_reset_sizes_it(self):
        s = A.Servo()
        np.testing.assert_array_equal(s.last_saturated, np.zeros(12, bool))
        s.reset(3)
        np.testing.assert_array_equal(s.last_saturated, np.zeros(3, bool))
        self.assertFalse(A.Servo(n_joints=1).last_saturated.any())

    def test_torque_falls_to_zero_at_noload_speed(self):
        p = A.ServoParams(kp=1000.0, kd=0.0, latency=0.0)
        s = A.Servo(p)
        s.reset()
        tau = s.step([1.0], [0.0], [p.noload_speed], 0.001)[0]
        self.assertAlmostEqual(tau, 0.0)
        s.reset()
        self.assertAlmostEqual(s.step([1.0], [0.0], [0.0], 0.001)[0], p.stall_torque)


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
        return d.qpos[0], d.ctrl[0], servo.params

    @unittest.skipIf(brentq is None, "scipy not installed")
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
