"""MuJoCo model (MJCF) generated from a URDF model.

The URDF stays the source of truth for geometry: bodies, joint axes, offsets and
limits are read from a `urdf.URDFModel`, so changing a link length in the
generator changes the simulator too. What a URDF as used here does not carry
(masses, collision shapes, friction) comes from the arguments below.

Every mass and size here is a placeholder, like the dimensions (docs/STAGE2.md).

Sites written into the model:
    body_origin   the origin of the root link: the centre of the four shoulders
    <leg>_foot    the tip of each leg (a link called "<leg>_foot" in the URDF)

Foot position in the body frame is R_baseᵀ (foot_world - body_origin_world);
see `foot_positions`. Ground truth for rewards and evaluation, never policy input.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import numpy as np

from ..geometry.placeholders import SIM_QUADRUPED
from ..geometry.urdf import URDFError, URDFModel
from ..robot.statics import G

# Placeholder masses in kg, by role: a plain-dict copy of those of the simulator's placeholder quadruped (geometry/placeholders.py).
DEFAULT_MASSES = dict(SIM_QUADRUPED.masses)
FOOT_RADIUS = 0.012
BODY_THICKNESS = 0.06  # height of the box drawn for the body, m
LINK_RADIUS = 0.01
FLOOR_FRICTION = 1.0
FOOT_FRICTION = 1.0
GRAVITY = -G           # m/s^2, along z (statics.G is the magnitude)
JOINT_DAMPING = 0.05   # N·m·s/rad
JOINT_ARMATURE = 0.001  # kg·m^2
FLOOR_HALF_SIZE = 5.0  # m, the floor plane drawn out to +-this in x and y
FRICTION_TORSION, FRICTION_ROLLING = 0.005, 0.0001


def _numbers(values) -> str:
    """Space-separated numbers, as MJCF attributes take them."""
    return " ".join(f"{float(value):.9g}" for value in values)


def _role(link: str, root: str) -> str:
    if link == root:
        return "body"
    for role in ("hip", "thigh", "shank", "foot"):
        if link.endswith("_" + role):
            return role
    return "shank"


def mjcf_from_urdf(model: URDFModel, body_size=None, masses=None,
                   height: float = 0.0, timestep: float = 0.002, actuated: bool = False, servo_boxes=()) -> str:
    """MJCF text for a floating-base robot described by `model`.

    body_size  (length, width, height) of the box drawn for the root link; default the simulator's placeholder body
               (placeholders.SIM_QUADRUPED) and BODY_THICKNESS.
    masses     overrides for DEFAULT_MASSES.
    height     starting height of the root link above the floor [m].
    servo_boxes  `fit.ServoBox`es drawn as massless, non-colliding boxes on their links (for clearance checks, `sim.clearance`);
               the link masses are unchanged.
    actuated   add a torque motor per movable joint (ctrl = N·m, driven by `actuators.Servo`) and a
               gyro and accelerometer at body_origin (sensors "gyro", "accel").
    """
    if body_size is None:
        body_size = (SIM_QUADRUPED.body_length, SIM_QUADRUPED.body_width, BODY_THICKNESS)
    mass_of = {**DEFAULT_MASSES, **(masses or {})}  # kg by role
    root = model.root()
    mujoco_element = ET.Element("mujoco", model=model.name)
    ET.SubElement(mujoco_element, "compiler", angle="radian", autolimits="true")
    ET.SubElement(mujoco_element, "option", timestep=str(timestep), gravity=f"0 0 {GRAVITY}")
    default = ET.SubElement(mujoco_element, "default")
    ET.SubElement(default, "joint", damping=str(JOINT_DAMPING), armature=str(JOINT_ARMATURE))
    # Only feet and the floor collide for now; the rest of the geometry is for looks and inertia.
    ET.SubElement(default, "geom", contype="0", conaffinity="0")
    worldbody = ET.SubElement(mujoco_element, "worldbody")
    ET.SubElement(worldbody, "geom", name="floor", type="plane", size=_numbers((FLOOR_HALF_SIZE, FLOOR_HALF_SIZE, 0.1)), contype="1", conaffinity="1",
                  friction=f"{FLOOR_FRICTION} {FRICTION_TORSION} {FRICTION_ROLLING}")

    def build(parent_element: ET.Element, link: str, joint) -> None:
        """Add `link` below `parent_element`, hung from `joint` (None for the root)."""
        child_joints = model.children(link)
        if joint is None:
            body_element = ET.SubElement(parent_element, "body", name=link, pos=_numbers((0.0, 0.0, height)))
            ET.SubElement(body_element, "freejoint", name="base")
            ET.SubElement(body_element, "site", name="body_origin", size="0.004", rgba="1 0 0 1")
            ET.SubElement(body_element, "geom", name="body_box", type="box", size=_numbers(np.array(body_size) / 2), mass=str(mass_of["body"]),
                          rgba="0.5 0.5 0.55 1")
        else:
            body_element = ET.SubElement(parent_element, "body", name=link, pos=_numbers(joint.xyz))
            if joint.movable:
                lower, upper = joint.limits
                ET.SubElement(body_element, "joint", name=joint.name, type="slide" if joint.type == "prismatic" else "hinge",
                              axis=_numbers(joint.axis), range=_numbers((lower, upper)))
            role = _role(link, root)
            if role == "foot":
                ET.SubElement(body_element, "site", name=link, size="0.004", rgba="0 0.6 1 1")
                ET.SubElement(body_element, "geom", name=f"{link}_pad", type="sphere", size=str(FOOT_RADIUS),
                              mass=str(mass_of["foot"]), contype="1", conaffinity="1",
                              friction=f"{FOOT_FRICTION} {FRICTION_TORSION} {FRICTION_ROLLING}", rgba="0.1 0.1 0.1 1")
            else:
                # a capsule from this link's origin to the origin of its first child, or a ball if it has none
                ends = [np.asarray(child_joint.xyz, float) for child_joint in child_joints]
                end = ends[0] if ends else np.zeros(3)
                if np.linalg.norm(end) > 1e-9:
                    ET.SubElement(body_element, "geom", type="capsule", fromto=_numbers(np.concatenate([np.zeros(3), end])),
                                  size=str(LINK_RADIUS), mass=str(mass_of[role]), rgba="0.8 0.8 0.85 1")
                else:
                    ET.SubElement(body_element, "geom", type="sphere", size=str(LINK_RADIUS * 1.5), mass=str(mass_of[role]),
                                  rgba="0.8 0.8 0.85 1")
        for box in servo_boxes:
            if box.link == link:
                x_axis, y_axis = box.rotation[:, 0], box.rotation[:, 1]
                ET.SubElement(body_element, "geom", name=box.name, type="box", size=_numbers(box.half_size), pos=_numbers(box.center),
                              xyaxes=_numbers(np.concatenate([x_axis, y_axis])), density="0", rgba="0.9 0.5 0.1 0.6")
        for child_joint in child_joints:
            build(body_element, child_joint.child, child_joint)

    build(worldbody, root, None)
    if actuated:
        actuator_element = ET.SubElement(mujoco_element, "actuator")
        for name in joint_order(model):
            ET.SubElement(actuator_element, "motor", name=name, joint=name, gear="1")
        sensor_element = ET.SubElement(mujoco_element, "sensor")
        ET.SubElement(sensor_element, "gyro", name="gyro", site="body_origin")
        ET.SubElement(sensor_element, "accelerometer", name="accel", site="body_origin")
    return ET.tostring(mujoco_element, encoding="unicode")


def joint_order(model: URDFModel) -> list:
    """Names of the movable joints, in URDF file order (the order of qpos after the 7 base values)."""
    return [j.name for j in model.joints.values() if j.movable]


def set_pose(mj_model, data, joint_names, joint_angles, base_pos=(0.0, 0.0, 0.0), base_rotation=None) -> None:
    """Put the base at base_pos with a body-to-world rotation matrix and set joint angles, then run kinematics."""
    import mujoco

    data.qpos[:3] = base_pos
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.eye(3).ravel() if base_rotation is None else np.asarray(base_rotation, float).ravel())
    data.qpos[3:7] = quat
    for name, value in zip(joint_names, joint_angles):
        joint_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise URDFError(f"joint {name!r} is not in the MuJoCo model")
        data.qpos[mj_model.jnt_qposadr[joint_id]] = value
    mujoco.mj_forward(mj_model, data)


def foot_positions(mj_model, data, feet) -> np.ndarray:
    """Foot positions in the body frame, R_baseᵀ (foot_world - body_origin_world). Shape (len(feet), 3)."""
    origin = data.site("body_origin").xpos
    body_to_world = data.site("body_origin").xmat.reshape(3, 3)
    return np.array([body_to_world.T @ (data.site(foot).xpos - origin) for foot in feet])
