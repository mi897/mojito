"""The actuated MuJoCo quadruped on the floor, with the lookups and readouts the stand test, the gait replay and the live
simulation share (docs/STAGE2.md, work packages 3 and 4).

`SimRobot` builds the model for a `placeholders.QuadrupedDesign` (the URDF from `robots.quadruped_urdf`, the MJCF from
`mjcf.mjcf_from_urdf`), puts it standing at a height with the feet under the hips, and then offers:

    joint_angles(), joint_velocities()   (12,) in leg order FL, FR, RL, RR, joints (abduction, hip pitch, knee)
    step(torques)                        apply the 12 motor torques (N·m) and advance one physics step
    foot_loads()                         normal force on each foot pad (N), summed over its contacts
    origin(), body_rotation()            body_origin site in the world: position and body-to-world rotation R
    feet_world(), legs_world()           foot tips (4, 3); shoulder, knee and foot of each leg (4, 3, 3)
    tilt_deg(), yaw()                    angle between the body's up axis and vertical; heading
    snapshot()                           side-view drawing data of the left legs

These are simulation ground truth, for rewards, evaluation and drawing, never policy input (CLAUDE.md).
Everything is a placeholder (dimensions, masses, servo figures, friction), see docs/STAGE2.md.
"""
from __future__ import annotations

import numpy as np

from ..geometry import robots
from ..geometry.placeholders import SIM_QUADRUPED
from ..robot import body
from . import mjcf

JOINT_KINDS = ("abduction", "hip_pitch", "knee")


def joint_names() -> list:
    """The 12 actuated joints, "<leg>_<joint>", in leg order FL, FR, RL, RR and joint order abduction, hip pitch, knee."""
    return [f"{leg_name}_{joint}" for leg_name in body.LEG_NAMES for joint in JOINT_KINDS]


class SimRobot:
    """The actuated quadruped of `design`, standing with its feet under the hips at shoulder height `height`.

    design    a placeholders.QuadrupedDesign (default: the simulator's placeholder)
    height    standing height in m (default: design.stance_height); the pads start just touching the floor
    masses    kg by role, overriding design.masses role by role; a role neither gives gets mjcf.DEFAULT_MASSES
    timestep  physics step, s

    Attributes: `kinematics` (a body.Robot with the closed-form IK and the design's dimensions), `model`, `data`,
    `qpos_addresses`, `dof_addresses`, `actuator_ids` (each (12,), in `joint_names()` order), `pad_legs` (foot-pad geom
    id -> leg index), `start_height` (body height at the start, m).
    """

    def __init__(self, design=SIM_QUADRUPED, height=None, masses=None, timestep=0.002):
        import mujoco

        self._mujoco = mujoco
        self.design = design
        self.stance_height = design.stance_height if height is None else height
        self.timestep = timestep
        self.kinematics = body.Robot(body.AnalyticSolver(), lengths=design.lengths, body_length=design.body_length,
                                     body_width=design.body_width)
        initial_joint_angles = self.kinematics.solve(self.kinematics.stance(self.stance_height))
        urdf_model = robots.quadruped_urdf(design.body_length, design.body_width, lengths=design.lengths)
        self.start_height = self.stance_height + mjcf.FOOT_RADIUS  # the pad centre is the foot tip; its lowest point is one radius lower
        masses = {**(design.masses or {}), **(masses or {})}  # mjcf_from_urdf puts DEFAULT_MASSES under these
        self.model = mujoco.MjModel.from_xml_string(
            mjcf.mjcf_from_urdf(urdf_model, body_size=(design.body_length, design.body_width, mjcf.BODY_THICKNESS),
                                masses=masses, height=self.start_height, timestep=timestep, actuated=True))
        self.data = mujoco.MjData(self.model)
        names = joint_names()
        mjcf.set_pose(self.model, self.data, names, initial_joint_angles.ravel(), base_pos=(0.0, 0.0, self.start_height))
        joint_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in names]
        self.qpos_addresses = self.model.jnt_qposadr[joint_ids]
        self.dof_addresses = self.model.jnt_dofadr[joint_ids]
        self.actuator_ids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name) for name in names]
        self.pad_legs = {mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"{leg_name}_foot_pad"): index
                         for index, leg_name in enumerate(body.LEG_NAMES)}
        self._wrench = np.zeros(6)  # contact force and torque in the contact frame, normal force first

    # ----- driving -----
    def joint_angles(self) -> np.ndarray:
        """(12,) measured joint angles, rad (a copy)."""
        return self.data.qpos[self.qpos_addresses]

    def joint_velocities(self) -> np.ndarray:
        """(12,) joint velocities, rad/s (a copy)."""
        return self.data.qvel[self.dof_addresses]

    def apply_torques(self, torques) -> None:
        """Set the 12 motor commands (N·m), in `joint_names()` order."""
        self.data.ctrl[self.actuator_ids] = torques

    def step(self, torques) -> None:
        """Apply `torques` (N·m) and advance the physics by one timestep."""
        self.apply_torques(torques)
        self._mujoco.mj_step(self.model, self.data)

    # ----- readouts (simulation ground truth) -----
    @property
    def total_mass(self) -> float:
        """Mass of the whole robot as built, kg."""
        return float(self.model.body_subtreemass[1])

    def foot_loads(self) -> np.ndarray:
        """(4,) normal force on each foot pad (N, 0 when it does not touch), summed over its contacts."""
        loads = np.zeros(4)
        for contact_index in range(self.data.ncon):
            contact = self.data.contact[contact_index]
            for geom in (contact.geom1, contact.geom2):
                if geom in self.pad_legs:
                    self._mujoco.mj_contactForce(self.model, self.data, contact_index, self._wrench)
                    loads[self.pad_legs[geom]] += self._wrench[0]
        return loads

    def origin(self) -> np.ndarray:
        """(3,) the body_origin site (centre of the four shoulders) in the world, m (a copy)."""
        return self.data.site("body_origin").xpos.copy()

    def body_rotation(self) -> np.ndarray:
        """(3, 3) body-to-world rotation R = Rz(yaw) Ry(pitch) Rx(roll) (a copy)."""
        return self.data.site("body_origin").xmat.reshape(3, 3).copy()

    def feet_world(self) -> np.ndarray:
        """(4, 3) foot tips in the world, FL FR RL RR, m."""
        return np.array([self.data.site(f"{leg_name}_foot").xpos for leg_name in body.LEG_NAMES])

    def legs_world(self) -> np.ndarray:
        """(4, 3, 3) shoulder, knee and foot of each leg in the world, FL FR RL RR, m."""
        return np.array([[self.data.body(f"{leg_name}_hip").xpos, self.data.body(f"{leg_name}_shank").xpos,
                          self.data.site(f"{leg_name}_foot").xpos] for leg_name in body.LEG_NAMES])

    def tilt_deg(self) -> float:
        """Angle between the body's up axis and vertical, degrees: arccos(R[2, 2])."""
        return float(tilt_deg(self.data.site("body_origin").xmat.reshape(3, 3)))

    def yaw(self) -> float:
        """Heading of the body's x axis in the world, rad: atan2(R[1, 0], R[0, 0])."""
        return float(yaw(self.data.site("body_origin").xmat.reshape(3, 3)))

    def snapshot(self) -> dict:
        """Side-view drawing data: shoulder, knee and foot of the left legs, and the body pose."""
        return {"legs": {leg_name: np.array([self.data.body(f"{leg_name}_hip").xpos, self.data.body(f"{leg_name}_shank").xpos,
                                             self.data.site(f"{leg_name}_foot").xpos])
                         for leg_name in ("FL", "RL")},
                "origin": self.origin(), "R": self.body_rotation()}


def tilt_deg(rotation):
    """Angle between the body's up axis and vertical, degrees, of a body-to-world rotation (..., 3, 3): arccos(R[2, 2])."""
    return np.degrees(np.arccos(np.clip(rotation[..., 2, 2], -1.0, 1.0)))


def yaw(rotation):
    """Heading (rad) of a body-to-world rotation (..., 3, 3), R = Rz(yaw) Ry(pitch) Rx(roll): atan2(R[1, 0], R[0, 0])."""
    return np.arctan2(rotation[..., 1, 0], rotation[..., 0, 0])


def roll_pitch_yaw(rotation):
    """Roll, pitch, yaw (rad) of a body-to-world rotation (..., 3, 3), R = Rz(yaw) Ry(pitch) Rx(roll).

    roll = atan2(R[2, 1], R[2, 2]), pitch = asin(-R[2, 0]), yaw = atan2(R[1, 0], R[0, 0]).
    """
    roll = np.arctan2(rotation[..., 2, 1], rotation[..., 2, 2])
    pitch = np.arcsin(np.clip(-rotation[..., 2, 0], -1.0, 1.0))
    return roll, pitch, yaw(rotation)
