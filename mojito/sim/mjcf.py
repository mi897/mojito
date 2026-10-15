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

from ..geometry.urdf import URDFModel, URDFError

# Placeholder masses in kg, by role. Servos are listed at 60 g (docs/STAGE2.md).
DEFAULT_MASSES = {"body": 0.8, "hip": 0.06, "thigh": 0.10, "shank": 0.06, "foot": 0.01}
FOOT_RADIUS = 0.012
LINK_RADIUS = 0.01
FLOOR_FRICTION = 1.0
FOOT_FRICTION = 1.0


def _v(a) -> str:
    return " ".join(f"{float(x):.9g}" for x in a)


def _role(link: str, root: str) -> str:
    if link == root:
        return "body"
    for role in ("hip", "thigh", "shank", "foot"):
        if link.endswith("_" + role):
            return role
    return "shank"


def mjcf_from_urdf(model: URDFModel, body_size=(0.30, 0.15, 0.06), masses=None,
                   height: float = 0.0, timestep: float = 0.002, actuated: bool = False) -> str:
    """MJCF text for a floating-base robot described by `model`.

    body_size  (length, width, height) of the box drawn for the root link.
    masses     overrides for DEFAULT_MASSES.
    height     starting height of the root link above the floor [m].
    actuated   add a torque motor per movable joint (ctrl = N·m, driven by `actuators.Servo`) and a
               gyro and accelerometer at body_origin (sensors "gyro", "accel").
    """
    m = {**DEFAULT_MASSES, **(masses or {})}
    root = model.root()
    mj = ET.Element("mujoco", model=model.name)
    ET.SubElement(mj, "compiler", angle="radian", autolimits="true")
    ET.SubElement(mj, "option", timestep=str(timestep), gravity="0 0 -9.81")
    default = ET.SubElement(mj, "default")
    ET.SubElement(default, "joint", damping="0.05", armature="0.001")
    # Only feet and the floor collide for now; the rest of the geometry is for looks and inertia.
    ET.SubElement(default, "geom", contype="0", conaffinity="0")
    wb = ET.SubElement(mj, "worldbody")
    ET.SubElement(wb, "geom", name="floor", type="plane", size="5 5 0.1", contype="1", conaffinity="1",
                  friction=f"{FLOOR_FRICTION} 0.005 0.0001")

    def build(parent_el: ET.Element, link: str, joint) -> None:
        """Add `link` below `parent_el`, hung from `joint` (None for the root)."""
        kids = model.children(link)
        if joint is None:
            el = ET.SubElement(parent_el, "body", name=link, pos=_v((0.0, 0.0, height)))
            ET.SubElement(el, "freejoint", name="base")
            ET.SubElement(el, "site", name="body_origin", size="0.004", rgba="1 0 0 1")
            ET.SubElement(el, "geom", type="box", size=_v(np.array(body_size) / 2), mass=str(m["body"]),
                          rgba="0.5 0.5 0.55 1")
        else:
            el = ET.SubElement(parent_el, "body", name=link, pos=_v(joint.xyz))
            if joint.movable:
                lo, hi = joint.limits
                ET.SubElement(el, "joint", name=joint.name, type="slide" if joint.type == "prismatic" else "hinge",
                              axis=_v(joint.axis), range=_v((lo, hi)))
            role = _role(link, root)
            if role == "foot":
                ET.SubElement(el, "site", name=link, size="0.004", rgba="0 0.6 1 1")
                ET.SubElement(el, "geom", name=f"{link}_pad", type="sphere", size=str(FOOT_RADIUS),
                              mass=str(m["foot"]), contype="1", conaffinity="1",
                              friction=f"{FOOT_FRICTION} 0.005 0.0001", rgba="0.1 0.1 0.1 1")
            else:
                # a capsule from this link's origin to the origin of its first child, or a ball if it has none
                ends = [np.asarray(k.xyz, float) for k in kids]
                end = ends[0] if ends else np.zeros(3)
                if np.linalg.norm(end) > 1e-9:
                    ET.SubElement(el, "geom", type="capsule", fromto=_v(np.concatenate([np.zeros(3), end])),
                                  size=str(LINK_RADIUS), mass=str(m[role]), rgba="0.8 0.8 0.85 1")
                else:
                    ET.SubElement(el, "geom", type="sphere", size=str(LINK_RADIUS * 1.5), mass=str(m[role]),
                                  rgba="0.8 0.8 0.85 1")
        for k in kids:
            build(el, k.child, k)

    build(wb, root, None)
    if actuated:
        act = ET.SubElement(mj, "actuator")
        for name in joint_order(model):
            ET.SubElement(act, "motor", name=name, joint=name, gear="1")
        sens = ET.SubElement(mj, "sensor")
        ET.SubElement(sens, "gyro", name="gyro", site="body_origin")
        ET.SubElement(sens, "accelerometer", name="accel", site="body_origin")
    return ET.tostring(mj, encoding="unicode")


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
        jid = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise URDFError(f"joint {name!r} is not in the MuJoCo model")
        data.qpos[mj_model.jnt_qposadr[jid]] = value
    mujoco.mj_forward(mj_model, data)


def foot_positions(mj_model, data, feet) -> np.ndarray:
    """Foot positions in the body frame, R_baseᵀ (foot_world - body_origin_world). Shape (len(feet), 3)."""
    import mujoco

    origin = data.site("body_origin").xpos
    R = data.site("body_origin").xmat.reshape(3, 3)
    return np.array([R.T @ (data.site(f).xpos - origin) for f in feet])
