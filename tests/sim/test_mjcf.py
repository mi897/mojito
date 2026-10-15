"""Run with:  python -m unittest discover tests   (skipped if MuJoCo is not installed)"""
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.geometry import robots
from mojito.robot import body, leg
from mojito.sim import mjcf  # noqa: E402

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None

LEGS = body.LEG_NAMES


def per_leg_urdf(lengths, body_length, body_width):
    """Quadruped URDF whose four legs have different lengths (4, 3)."""
    model = robots.quadruped_urdf(body_length, body_width)
    for i, p in enumerate(LEGS):
        la, lu, ll = lengths[i]
        side = -1.0 if p.endswith("R") else 1.0
        model.joints[f"{p}_hip_pitch"].xyz = (0.0, side * la, 0.0)
        model.joints[f"{p}_knee"].xyz = (0.0, 0.0, -lu)
        model.joints[f"{p}_foot"].xyz = (0.0, 0.0, -ll)
    return model


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestMjcf(unittest.TestCase):
    def build(self, lengths, bl=0.30, bw=0.15):
        model = per_leg_urdf(lengths, bl, bw)
        m = mujoco.MjModel.from_xml_string(mjcf.mjcf_from_urdf(model, height=0.3))
        return model, m, mujoco.MjData(m)

    def test_feet_match_kinematics_for_random_lengths_angles_and_base_poses(self):
        rng = np.random.default_rng(0)
        for _ in range(25):
            lengths = np.array([0.05, 0.15, 0.15]) * rng.uniform(0.7, 1.3, (4, 3))
            bl, bw = rng.uniform(0.2, 0.4), rng.uniform(0.1, 0.2)
            model, m, d = self.build(lengths, bl, bw)
            robot = body.Robot(body.AnalyticSolver(), lengths=lengths, body_length=bl, body_width=bw)
            lo, hi = robot.limits[:, 0], robot.limits[:, 1]
            q = np.empty((4, 3))
            for i in range(4):
                q[i] = rng.uniform(lo, hi)
                if body.IS_RIGHT[i]:
                    q[i, 0] = rng.uniform(-hi[0], -lo[0])  # right-leg abduction limits are flipped
            R = body.rotation(*rng.uniform(-0.5, 0.5, 3))
            pos = rng.uniform(-1, 1, 3)
            mjcf.set_pose(m, d, [f"{p}_{j}" for p in LEGS for j in ("abduction", "hip_pitch", "knee")],
                          q.ravel(), pos, R)
            feet = mjcf.foot_positions(m, d, [f"{p}_foot" for p in LEGS])
            np.testing.assert_allclose(feet, robot.feet(q), atol=1e-9)
            # the same feet seen from the world: base position plus rotated body-frame position
            world = np.array([d.site(f"{p}_foot").xpos for p in LEGS])
            np.testing.assert_allclose(world, pos + robot.feet(q) @ R.T, atol=1e-9)

    def test_joint_limits_come_from_the_urdf(self):
        model, m, _ = self.build(np.tile([0.05, 0.15, 0.15], (4, 1)))
        for name in mjcf.joint_order(model):
            lo, hi = model.joints[name].limits
            np.testing.assert_allclose(m.jnt_range[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)], (lo, hi))

    def test_link_length_change_moves_the_foot(self):
        a = self.build(np.tile([0.05, 0.15, 0.15], (4, 1)))
        b = self.build(np.tile([0.05, 0.15, 0.20], (4, 1)))
        names = [f"{p}_{j}" for p in LEGS for j in ("abduction", "hip_pitch", "knee")]
        zs = []
        for _, m, d in (a, b):
            mjcf.set_pose(m, d, names, np.tile([0.0, 0.5, -1.0], 4))
            zs.append(mjcf.foot_positions(m, d, ["FL_foot"])[0, 2])
        self.assertLess(zs[1], zs[0] - 0.04)

    def test_standing_pose_rests_on_the_floor_under_gravity(self):
        # A rigid-ish check of contact and mass: dropped with the stance pose, the base settles above
        # the floor and the four pads carry the weight (sum of vertical contact force = weight).
        L = np.tile([0.05, 0.15, 0.15], (4, 1))
        model, m, d = self.build(L)
        robot = body.Robot(body.AnalyticSolver(), lengths=L, body_length=0.30, body_width=0.15)
        q = robot.solve(robot.stance(0.2125))
        names = [f"{p}_{j}" for p in LEGS for j in ("abduction", "hip_pitch", "knee")]
        mjcf.set_pose(m, d, names, q.ravel(), (0, 0, 0.2125 + mjcf.FOOT_RADIUS))
        d.qvel[:] = 0
        # hold the joints with stiff PD so the test is about contact and mass, not the servo model
        qadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        dadr = [m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in names]
        target = q.ravel()
        for _ in range(1500):
            d.qfrc_applied[dadr] = 30.0 * (target - d.qpos[qadr]) - 0.15 * d.qvel[dadr]
            mujoco.mj_step(m, d)
        total = 0.0
        f6 = np.zeros(6)
        for i in range(d.ncon):
            mujoco.mj_contactForce(m, d, i, f6)
            total += f6[0]  # normal component in the contact frame
        weight = m.body_subtreemass[1] * 9.81
        self.assertAlmostEqual(total / weight, 1.0, delta=0.03)
        self.assertEqual(d.ncon, 4)
        self.assertGreater(d.qpos[2], 0.15)


if __name__ == "__main__":
    unittest.main()
