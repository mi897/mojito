"""Do the servo boxes collide as the legs move? Distances between boxes in MuJoCo over the joint range.

The servo boxes come from `mojito.robot.fit` and are added to the generated model (`mjcf.mjcf_from_urdf(servo_boxes=...)`).
Distances are `mujoco.mj_geomDistance`: positive is the gap between two boxes, negative is penetration depth. Needs mujoco.

One leg at a time sweeps a grid over its own joint limits while the other three stand; every pair of boxes on different
rigid bodies is measured at every pose. The sweep covers the full joint range, which is more than the gaits use, so a
collision here is a collision somewhere in the range, not necessarily in a walk.
"""
from __future__ import annotations

import itertools

import mujoco
import numpy as np

from ..geometry import robots
from ..geometry.placeholders import SIM_QUADRUPED
from ..robot import fit, leg
from . import mjcf

BODY_BOX = "body_box"  # the placeholder body envelope of mjcf (the abduction servos sit in it by design: same body, not measured)
DISTANCE_CAP = 0.05  # m: gaps larger than this are reported as the cap


def build(design=SIM_QUADRUPED, layout=None, spec=None):
    """(MjModel, MjData, boxes, urdf_model) for `design` with servo boxes added."""
    boxes = fit.servo_boxes(design, layout, spec)
    urdf_model = robots.quadruped_urdf(design.body_length, design.body_width, lengths=design.lengths)
    model = mujoco.MjModel.from_xml_string(mjcf.mjcf_from_urdf(urdf_model, servo_boxes=boxes))
    return model, mujoco.MjData(model), boxes, urdf_model


def box_distance(model, data, name_a: str, name_b: str, cap: float = DISTANCE_CAP) -> float:
    """Signed distance between two named geoms at the pose in `data` (metres, capped at `cap`)."""
    geom_a = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name_a)
    geom_b = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name_b)
    fromto = np.zeros(6)
    return float(min(mujoco.mj_geomDistance(model, data, geom_a, geom_b, cap, fromto), cap))


def stance_angles(design, leg_index: int) -> np.ndarray:
    """Joint angles of leg `leg_index` standing with the foot under the hip (right legs mirrored)."""
    q = leg.analytic_ik(np.array([0.0, design.lengths[0], -design.stance_height]), np.asarray(design.lengths, float))
    return leg.mirror_angles(q) if leg_index % 2 == 1 else q


def sweep(design=SIM_QUADRUPED, layout=None, spec=None, points: int = 7, cap: float = DISTANCE_CAP) -> dict:
    """Smallest distance for every pair of boxes on different bodies, over a joint-range grid, one leg at a time.

    Returns {"pairs": {(box_a, box_b): {"min_gap_m", "pose"}}, "poses": number of poses, "boxes": the boxes}.
    `pose` is (leg, joint angles) at the smallest gap."""
    model, data, boxes, urdf_model = build(design, layout, spec)
    joint_names = mjcf.joint_order(urdf_model)
    root = urdf_model.root()
    owner = {box.name: box.link for box in boxes} | {BODY_BOX: root}
    pairs = [(a, b) for a, b in itertools.combinations(owner, 2) if owner[a] != owner[b]]
    result = {pair: {"min_gap_m": cap, "pose": None} for pair in pairs}
    poses = 0
    for leg_index, leg_name in enumerate(fit.LEG_NAMES):
        limits = np.array([[urdf_model.joints[f"{leg_name}_{j}"].limits[0], urdf_model.joints[f"{leg_name}_{j}"].limits[1]]
                           for j in fit.JOINTS])
        grids = [np.linspace(lo, hi, points) for lo, hi in limits]
        for angles in itertools.product(*grids):
            q = np.concatenate([stance_angles(design, i) for i in range(4)])
            q[3 * leg_index:3 * leg_index + 3] = angles
            mjcf.set_pose(model, data, joint_names, q)
            poses += 1
            for a, b in pairs:
                gap = box_distance(model, data, a, b, cap)
                if gap < result[(a, b)]["min_gap_m"]:
                    result[(a, b)] = {"min_gap_m": gap, "pose": (leg_name, [float(x) for x in angles])}
    return {"pairs": result, "poses": poses, "boxes": boxes}


def first_contact(design=SIM_QUADRUPED, layout=None, spec=None, leg_index: int = 0, joint_index: int = 0,
                  pair: tuple | None = None, step: float = 0.005) -> dict:
    """Smallest |angle| (rad) at which `pair` of boxes touches as one joint of one leg moves either way from standing; None if never.

    Defaults to the abduction joint of the front left leg against its hip-pitch servo. Other joints stay in the stance.
    Returns {"positive": angle or None, "negative": angle or None, "limits": (low, high)}."""
    model, data, boxes, urdf_model = build(design, layout, spec)
    names = mjcf.joint_order(urdf_model)
    leg_name = fit.LEG_NAMES[leg_index]
    a, b = pair or (f"{leg_name}_servo_abduction", f"{leg_name}_servo_hip_pitch")
    q = np.concatenate([stance_angles(design, i) for i in range(4)])
    low, high = urdf_model.joints[f"{leg_name}_{fit.JOINTS[joint_index]}"].limits
    out = {"limits": (float(low), float(high))}
    for key, sign, limit in (("positive", 1.0, high), ("negative", -1.0, -low)):
        out[key] = None
        for angle in np.arange(0.0, max(limit, 0.0) + step, step):
            pose = q.copy()
            pose[3 * leg_index + joint_index] = q[3 * leg_index + joint_index] + sign * angle
            mjcf.set_pose(model, data, names, pose)
            if box_distance(model, data, a, b) <= 0.0:
                out[key] = float(angle)
                break
    return out


def gait_sweep(gait, velocity=(0.0, 0.0), yaw_rate: float = 0.0, design=SIM_QUADRUPED, layout=None, spec=None,
               samples: int = 200, cap: float = DISTANCE_CAP) -> dict:
    """The same pair distances over one cycle of `gait`, at the commanded joint angles (the closed-form IK of the commanded feet).

    The angles are what the controller asks for, not what the servos reach under load; the stance height is the one the
    physics replay uses for this gait (`gait_replay.settings_for`). Returns {"pairs": {(a, b): {"min_gap_m", "phase"}}, "poses"}."""
    from ..robot import body
    from . import gait_replay

    model, data, boxes, urdf_model = build(design, layout, spec)
    names = mjcf.joint_order(urdf_model)
    kinematics = body.Robot(body.AnalyticSolver(), lengths=design.lengths, body_length=design.body_length, body_width=design.body_width)
    home = kinematics.stance(gait_replay.settings_for(gait)["height"])
    times = np.linspace(0.0, gait.period, samples, endpoint=False)
    targets, _ = gait_replay.commanded_feet(gait, home, times, 0.0, 0.0, velocity, yaw_rate)
    angles = kinematics.solve(targets).reshape(samples, 12)
    owner = {box.name: box.link for box in boxes} | {BODY_BOX: urdf_model.root()}
    pairs = [(a, b) for a, b in itertools.combinations(owner, 2) if owner[a] != owner[b]]
    result = {pair: {"min_gap_m": cap, "phase": None} for pair in pairs}
    for index in range(samples):
        mjcf.set_pose(model, data, names, angles[index])
        for a, b in pairs:
            gap = box_distance(model, data, a, b, cap)
            if gap < result[(a, b)]["min_gap_m"]:
                result[(a, b)] = {"min_gap_m": gap, "phase": float(index / samples)}
    return {"pairs": result, "poses": samples}
