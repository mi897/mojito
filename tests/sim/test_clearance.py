"""Run with:  python -m unittest discover tests   (skipped if MuJoCo is not installed)"""
import unittest

import numpy as np

from mojito.geometry import robots
from mojito.geometry.placeholders import SIM_QUADRUPED
from mojito.robot import fit, gaits
from mojito.sim import mjcf

try:
    import mujoco

    from mojito.sim import clearance
except ImportError:  # pragma: no cover
    mujoco = None


def point_to_box(points, center, rotation, half):
    """Distance from points (n, 3) to a solid box, in closed form: clamp in the box frame."""
    local = (points - center) @ rotation
    outside = np.maximum(np.abs(local) - half, 0.0)
    return np.linalg.norm(outside, axis=1)


def surface_samples(center, rotation, half, step=0.002):
    """Points on the surface of a box, `step` apart."""
    points = []
    for axis in range(3):
        others = [i for i in range(3) if i != axis]
        grids = [np.linspace(-half[i], half[i], max(int(2 * half[i] / step) + 1, 2)) for i in others]
        gu, gv = np.meshgrid(*grids)
        for sign in (-1.0, 1.0):
            face = np.zeros((gu.size, 3))
            face[:, axis] = sign * half[axis]
            face[:, others[0]], face[:, others[1]] = gu.ravel(), gv.ravel()
            points.append(face)
    return center + np.vstack(points) @ rotation.T


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestClearance(unittest.TestCase):
    def test_servo_boxes_add_no_mass(self):
        urdf_model = robots.quadruped_urdf(SIM_QUADRUPED.body_length, SIM_QUADRUPED.body_width, lengths=SIM_QUADRUPED.lengths)
        plain = mujoco.MjModel.from_xml_string(mjcf.mjcf_from_urdf(urdf_model))
        with_boxes = clearance.build()[0]
        self.assertAlmostEqual(float(with_boxes.body_mass.sum()), float(plain.body_mass.sum()), places=9)
        self.assertEqual(with_boxes.ngeom, plain.ngeom + 12)

    def test_shaft_is_on_the_joint_axis_in_the_world(self):
        model, data, boxes, urdf_model = clearance.build()
        q = np.array([0.3, 0.7, -1.1] * 4)
        mjcf.set_pose(model, data, mjcf.joint_order(urdf_model), q, base_pos=(0.1, -0.2, 0.4))
        for box in boxes:
            geom = data.geom(box.name)
            rotation = geom.xmat.reshape(3, 3)
            shaft = geom.xpos + rotation @ (np.array([fit.ServoSpec().shaft_offset, 0.0, fit.ServoSpec().height / 2]))
            joint = data.joint(f"{box.leg}_{box.joint}")
            np.testing.assert_allclose(shaft, joint.xanchor, atol=1e-9)
            np.testing.assert_allclose(np.abs(rotation[:, 2] @ joint.xaxis), 1.0, atol=1e-9)

    def test_distance_matches_sampled_surface_distance(self):
        model, data, boxes, urdf_model = clearance.build()
        names = mjcf.joint_order(urdf_model)
        rng = np.random.default_rng(3)
        checked = 0
        for _ in range(12):
            q = np.concatenate([clearance.stance_angles(SIM_QUADRUPED, i) for i in range(4)])
            q[:3] = rng.uniform([-0.6, 0.0, -2.3], [0.6, 1.5, -0.3])
            mjcf.set_pose(model, data, names, q)
            for a, b in (("FL_servo_abduction", "FL_servo_hip_pitch"), ("FL_servo_knee", "body_box")):
                geom_a, geom_b = data.geom(a), data.geom(b)
                gap = clearance.box_distance(model, data, a, b, cap=1.0)
                if gap <= 0.0:
                    continue
                rot_a, rot_b = geom_a.xmat.reshape(3, 3), geom_b.xmat.reshape(3, 3)
                half_a, half_b = model.geom_size[geom_a.id], model.geom_size[geom_b.id]
                sampled = min(point_to_box(surface_samples(geom_a.xpos, rot_a, half_a), geom_b.xpos, rot_b, half_b).min(),
                              point_to_box(surface_samples(geom_b.xpos, rot_b, half_b, 0.005), geom_a.xpos, rot_a, half_a).min())
                self.assertAlmostEqual(gap, sampled, delta=0.003)  # surface samples 2 mm apart, so within a few mm
                checked += 1
        self.assertGreater(checked, 10)

    def test_known_overlap_is_negative(self):
        model, data, boxes, urdf_model = clearance.build(layout=fit.ServoLayout(hip_body_inward=True))
        mjcf.set_pose(model, data, mjcf.joint_order(urdf_model), np.concatenate([clearance.stance_angles(SIM_QUADRUPED, i) for i in range(4)]))
        self.assertLess(clearance.box_distance(model, data, "FL_servo_abduction", "FL_servo_hip_pitch"), 0.0)

    def test_inboard_hip_servo_touches_the_abduction_servo_and_outboard_never_does(self):
        inboard = clearance.first_contact(layout=fit.ServoLayout(hip_body_inward=True))
        self.assertEqual(inboard["positive"], 0.0)
        outboard = clearance.first_contact()
        self.assertIsNone(outboard["positive"])
        self.assertIsNone(outboard["negative"])

    def test_sweep_pair_set_and_cap(self):
        result = clearance.sweep(points=3)
        self.assertEqual(result["poses"], 4 * 27)
        # 12 servo boxes and the body box make 78 pairs; the 4 abduction servos and the body box share the root body: 10 pairs not compared
        self.assertEqual(len(result["pairs"]), 78 - 10)
        for value in result["pairs"].values():
            self.assertLessEqual(value["min_gap_m"], clearance.DISTANCE_CAP)
        # nothing in the same rigid body is compared: the four abduction servos are all in the root body
        self.assertNotIn(("FL_servo_abduction", "FR_servo_abduction"), result["pairs"])

    def test_trot_in_place_stays_clear(self):
        result = clearance.gait_sweep(gaits.TROT, samples=40)
        self.assertGreater(min(v["min_gap_m"] for v in result["pairs"].values()), 0.0)


if __name__ == "__main__":
    unittest.main()
