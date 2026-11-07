"""Where the servos sit, and whether the electronics fit in the body: the geometry of the mechanical fit check.

Pure NumPy, no MuJoCo (`mojito.sim.clearance` measures distances between the boxes in MuJoCo). Everything here is a
placeholder layout (docs/STAGE2.md, "Servo layout and fit"), not a drawing of the robot.

A servo is a box: `length` x `width` x `height` (listed 40 x 20 x 40.5 mm). Its output shaft runs along the height,
through the middle of the width, and sits `shaft_offset` from the box centre along the length (listed servos have
the shaft near one end). The output face is the end of the height where the horn is. Box local axes: u along the length
(toward the shaft), w along the shaft (toward the horn), v = w x u. The output face sits in the joint plane.

Layout (owner's decision A: knee servo at the knee): the abduction servos sit in the body with the shaft along x at the
shoulder, the hip-pitch servo on the hip link, the knee servo on the thigh. Link frames at zero joint angle are plain
translations of the body frame (`mojito.geometry.robots`), so a box is placed with a centre and a rotation in its link's frame.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

LEG_NAMES = ("FL", "FR", "RL", "RR")  # leg order of the whole project
JOINTS = ("abduction", "hip_pitch", "knee")


@dataclass(frozen=True)
class ServoSpec:
    """Box size and shaft position of a servo. Listed DS3218 figures (unverified), metres."""
    length: float = 0.040
    width: float = 0.020
    height: float = 0.0405
    shaft_offset: float = 0.010  # shaft to box centre along the length: an assumption, the listings do not give it

    @property
    def half_size(self) -> np.ndarray:
        return np.array([self.length, self.width, self.height]) / 2


@dataclass(frozen=True)
class ServoLayout:
    """The choices of the default layout; flip one to try the alternative.

    hip_body_inward     hip-pitch servo body on the body side of the thigh plane (horn pointing out). False is the default:
                        inward overlaps the abduction servo even at zero abduction (results/body/fit_check.json)
    hip_long_part_out   its long part (30 mm of the 40 mm) extends away from the body centre along x
    knee_body_inward    knee servo body on the body side of the thigh plane
    knee_long_part_up   its long part extends up the thigh (the shaft is near the lower end)
    """
    hip_body_inward: bool = False
    hip_long_part_out: bool = True
    knee_body_inward: bool = True
    knee_long_part_up: bool = True


@dataclass(frozen=True)
class ServoBox:
    leg: str
    joint: str
    link: str            # the body it is attached to: the root link, or "<leg>_hip" or "<leg>_thigh"
    center: np.ndarray   # in that link's frame, at zero joint angles
    rotation: np.ndarray  # (3, 3), box local axes (u, v, w) as columns, in the link's frame
    half_size: np.ndarray

    @property
    def name(self) -> str:
        return f"{self.leg}_servo_{self.joint}"


def _box(leg, joint, link, joint_point, horn_direction, length_direction, spec: ServoSpec) -> ServoBox:
    """A servo whose output face is at joint_point with the horn pointing along horn_direction."""
    w = np.asarray(horn_direction, float)
    u = np.asarray(length_direction, float)
    assert abs(w @ u) < 1e-12, "the length direction must be perpendicular to the shaft"
    v = np.cross(w, u)
    center = np.asarray(joint_point, float) - w * spec.height / 2 - u * spec.shaft_offset
    return ServoBox(leg, joint, link, center, np.column_stack([u, v, w]), spec.half_size)


def servo_boxes(design, layout: ServoLayout | None = None, spec: ServoSpec | None = None, root: str = "body") -> list:
    """The 12 servo boxes of `design` (a `placeholders.QuadrupedDesign`), leg order FL, FR, RL, RR, joint order as `JOINTS`."""
    layout, spec = layout or ServoLayout(), spec or ServoSpec()
    offset, upper, _ = (float(x) for x in design.lengths)
    boxes = []
    for leg, (front, left) in zip(LEG_NAMES, ((1, 1), (1, -1), (-1, 1), (-1, -1))):
        front, side = float(front), float(left)  # +1 front or left, -1 rear or right
        x, y = front * design.body_length / 2, side * design.body_width / 2
        x_axis, y_axis, z_axis = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
        # abduction: shaft along x, body toward the body centre, shaft near the outer (lateral) end of the length
        boxes.append(_box(leg, "abduction", root, (x, y, 0.0), front * x_axis, side * y_axis, spec))
        # hip pitch: shaft along y, on the hip link, whose frame is at the shoulder
        horn = side * y_axis if layout.hip_body_inward else -side * y_axis
        along = -front * x_axis if layout.hip_long_part_out else front * x_axis
        boxes.append(_box(leg, "hip_pitch", f"{leg}_hip", (0.0, side * offset, 0.0), horn, along, spec))
        # knee: shaft along y, on the thigh, whose frame is at the hip-pitch joint
        horn = side * y_axis if layout.knee_body_inward else -side * y_axis
        along = -z_axis if layout.knee_long_part_up else z_axis
        boxes.append(_box(leg, "knee", f"{leg}_thigh", (0.0, 0.0, -upper), horn, along, spec))
    return boxes


def free_body_space(design, spec: ServoSpec | None = None, body_height: float = 0.06, plate_thickness: float = 0.003) -> tuple:
    """(length, width, height) of the space between the plates and between the front and rear abduction servos.

    The abduction servos reach `spec.height` inward along x from each shoulder; the width is the shoulder spacing.
    Walls, standoffs and cable room are not subtracted. `body_height` is the placeholder body box of `mojito.sim.mjcf`."""
    spec = spec or ServoSpec()
    return (design.body_length - 2 * spec.height, design.body_width, body_height - 2 * plate_thickness)


def pack_in_box(parts, free_size) -> dict:
    """Shelf-pack boxes lying flat into a free box, as a quick yes/no on whether the parts fit.

    parts      list of (name, (a, b, c)): box sizes in metres; each is laid with its smallest side vertical and may turn
               by 90 degrees about the vertical.
    free_size  (length, width, height) of the free space.
    Rows of parts run along the length; a part goes in the first row with room. A shelf packing is not the best packing,
    so "does not fit" means this packing failed, not that none exists."""
    length, width, height = free_size
    sized = []
    for name, size in parts:
        flat = sorted(size)  # smallest first: height, then the two footprint sides
        sized.append((name, flat[2], flat[1], flat[0]))
    sized.sort(key=lambda s: (-s[1], -s[2]))
    rows, placements, unplaced = [], [], []  # a row: [y0, row_width, x_used]
    for name, long_side, short_side, tall in sized:
        if tall > height or long_side > length:
            unplaced.append(name)
            continue
        for row in rows:
            if row[2] + long_side <= length and short_side <= row[1]:
                placements.append((name, row[2], row[0], long_side, short_side))
                row[2] += long_side
                break
        else:
            y0 = sum(r[1] for r in rows)
            if y0 + short_side > width:
                unplaced.append(name)
                continue
            rows.append([y0, short_side, long_side])
            placements.append((name, 0.0, y0, long_side, short_side))
    used = sum(p[3] * p[4] for p in placements)
    return {"fits": not unplaced, "unplaced": unplaced, "placements": placements,
            "footprint_used_m2": used, "free_footprint_m2": length * width, "footprint_fraction": used / (length * width)}
