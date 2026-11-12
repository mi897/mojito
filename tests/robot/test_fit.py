import unittest

import numpy as np

from mojito.geometry.placeholders import SIM_QUADRUPED
from mojito.robot import fit


def body_frame_center(box):
    """Box centre in the body frame at zero joint angles: link frames are translations (hip at the shoulder, thigh offset sideways)."""
    side = 1.0 if box.leg[1] == "L" else -1.0
    front = 1.0 if box.leg[0] == "F" else -1.0
    shoulder = np.array([front * SIM_QUADRUPED.body_length / 2, side * SIM_QUADRUPED.body_width / 2, 0.0])
    if box.joint == "abduction":
        return box.center
    thigh_origin = np.array([0.0, side * SIM_QUADRUPED.lengths[0], 0.0])
    return shoulder + box.center + (thigh_origin if box.joint == "knee" else 0.0)


def overlap_gap(a, b):
    """Largest separation along the body axes between two axis-aligned boxes in the body frame (negative: they overlap)."""
    return (np.abs(body_frame_center(a) - body_frame_center(b)) - (np.abs(a.rotation) @ a.half_size + np.abs(b.rotation) @ b.half_size)).max()


class TestServoBoxes(unittest.TestCase):
    def setUp(self):
        self.boxes = fit.servo_boxes(SIM_QUADRUPED)

    def test_twelve_boxes_in_leg_order(self):
        self.assertEqual([b.name for b in self.boxes],
                         [f"{leg}_servo_{joint}" for leg in fit.LEG_NAMES for joint in fit.JOINTS])

    def test_rotations_are_proper(self):
        for box in self.boxes:
            np.testing.assert_allclose(box.rotation.T @ box.rotation, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(box.rotation), 1.0)

    def test_left_and_right_legs_mirror(self):
        mirror = np.diag([1.0, -1.0, 1.0])
        for index in (0, 1, 2, 6, 7, 8):
            left, right = self.boxes[index], self.boxes[index + 3]
            np.testing.assert_allclose(right.center, mirror @ left.center, atol=1e-12)
            # a mirror turns a right-handed frame left-handed; the box itself is symmetric, so compare the box's axis lines
            np.testing.assert_allclose(np.abs(right.rotation), np.abs(mirror @ left.rotation), atol=1e-12)

    def test_shaft_axis_is_the_joint_axis(self):
        # the shaft (box w axis) is along x for the abduction joint and along y for hip pitch and knee (robots._leg_joints)
        axis = {"abduction": [1, 0, 0], "hip_pitch": [0, 1, 0], "knee": [0, 1, 0]}
        for box in self.boxes:
            np.testing.assert_allclose(np.abs(box.rotation[:, 2]), axis[box.joint], atol=1e-12)

    def test_output_face_sits_in_the_joint_plane(self):
        spec = fit.ServoSpec()
        offset, upper, _ = SIM_QUADRUPED.lengths
        for box in self.boxes:
            face_centre = box.center + box.rotation[:, 2] * spec.height / 2
            shaft = face_centre + box.rotation[:, 0] * spec.shaft_offset
            side = 1.0 if box.leg[1] == "L" else -1.0
            front = 1.0 if box.leg[0] == "F" else -1.0
            expected = {"abduction": [front * SIM_QUADRUPED.body_length / 2, side * SIM_QUADRUPED.body_width / 2, 0.0],
                        "hip_pitch": [0.0, side * offset, 0.0], "knee": [0.0, 0.0, -upper]}[box.joint]
            np.testing.assert_allclose(shaft, expected, atol=1e-12)
            np.testing.assert_allclose(box.center + box.rotation[:, 2] * spec.height / 2 - face_centre, 0.0, atol=1e-12)

    def test_default_layout_has_no_static_overlap_of_hip_and_abduction_servo(self):
        # in the standing pose both boxes are axis-aligned here, so a separating-axis check on the three axes is exact
        for leg_index in range(4):
            a, b = self.boxes[3 * leg_index], self.boxes[3 * leg_index + 1]
            self.assertGreater(overlap_gap(a, b), 0.0)

    def test_inboard_hip_servo_overlaps_the_abduction_servo(self):
        boxes = fit.servo_boxes(SIM_QUADRUPED, fit.ServoLayout(hip_body_inward=True))
        self.assertLess(overlap_gap(boxes[0], boxes[1]), 0.0)  # standing pose only; the MuJoCo test covers the sweep


class TestPacking(unittest.TestCase):
    def test_exact_tiling_fits(self):
        parts = [(f"p{i}", (0.05, 0.02, 0.01)) for i in range(8)]
        result = fit.pack_in_box(parts, (0.10, 0.08, 0.02))  # 2 x 4 tiles of 0.05 x 0.02
        self.assertTrue(result["fits"])
        self.assertAlmostEqual(result["footprint_fraction"], 1.0)

    def test_one_part_too_many_does_not_fit(self):
        parts = [(f"p{i}", (0.05, 0.02, 0.01)) for i in range(9)]
        result = fit.pack_in_box(parts, (0.10, 0.08, 0.02))
        self.assertFalse(result["fits"])
        self.assertEqual(len(result["unplaced"]), 1)

    def test_too_tall_part_is_unplaced(self):
        self.assertEqual(fit.pack_in_box([("tall", (0.05, 0.05, 0.05))], (0.2, 0.2, 0.04))["unplaced"], ["tall"])

    def test_footprint_fraction_is_area_over_free_area(self):
        parts = [("a", (0.04, 0.03, 0.01)), ("b", (0.02, 0.02, 0.02))]
        result = fit.pack_in_box(parts, (0.2, 0.1, 0.05))
        self.assertAlmostEqual(result["footprint_fraction"], (0.03 * 0.04 + 0.02 * 0.02) / (0.2 * 0.1))

    def test_free_space_of_the_default_body(self):
        length, width, height = fit.free_body_space(SIM_QUADRUPED)
        self.assertAlmostEqual(length, 0.30 - 2 * 0.0405)
        self.assertAlmostEqual(width, 0.15)
        self.assertAlmostEqual(height, 0.06 - 0.006)


if __name__ == "__main__":
    unittest.main()
