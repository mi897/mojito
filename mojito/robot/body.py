"""Whole-body kinematics: four legs on a rigid body.

Body frame: origin at the centre of the four shoulders, x forward, y left, z up.
Legs are always ordered FL, FR, RL, RR (front-left, front-right, rear-left,
rear-right).

Each leg solves its inverse kinematics in its own shoulder frame with the
single shared leg model. Right legs are mirror images of left legs, so their
targets are flipped in y on the way in and their abduction angle on the way out.
"""
from __future__ import annotations

import numpy as np

from ..geometry.kinematics import rpy_rotation
# Placeholder body, shoulder spacing in metres (defined in mojito/geometry/placeholders.py, which the URDF generator also reads):
# BODY_LENGTH front shoulders to rear shoulders, BODY_WIDTH left shoulders to right shoulders.
from ..geometry.placeholders import BODY_LENGTH, BODY_WIDTH
from . import leg

LEG_NAMES = ("FL", "FR", "RL", "RR")
IS_RIGHT = np.array([False, True, False, True])
_MIRROR = np.where(IS_RIGHT[:, None], [1.0, -1.0, 1.0], 1.0)  # (4, 3) target flip: y of right legs
_MIRROR_ANGLES = np.where(IS_RIGHT[:, None], [-1.0, 1.0, 1.0], 1.0)  # (4, 3) angle flip: abduction of right legs

STAND_HEIGHT = 0.17  # shoulder height above the ground when standing


def rotation(roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """Body-to-world rotation matrix, R = Rz(yaw) Ry(pitch) Rx(roll). Shape (..., 3, 3).

    The same convention as URDF rpy; one implementation, `geometry.kinematics.rpy_rotation`.
    """
    return rpy_rotation(roll, pitch, yaw)


class AnalyticSolver:
    """Closed-form leg IK with the same call signature as the network."""

    def predict(self, target_position, lengths):
        return leg.analytic_ik(target_position, lengths)


class Robot:
    """Geometry of the quadruped plus whichever leg solver drives it.

    solver   anything with predict(target_position, lengths) -> joint_angles for a canonical (left) leg:
             a trained IKNet, or AnalyticSolver() for the exact reference.
    lengths  (3,) for four identical legs or (4, 3) for one row per leg.
             Plain attribute: change it at any time.
    """

    def __init__(self, solver, lengths=None, body_length=BODY_LENGTH, body_width=BODY_WIDTH,
                 limits=None):
        self.solver = solver
        config = getattr(solver, "config", None)
        nominal = config.nominal if config is not None else leg.NOMINAL_LENGTHS
        self.lengths = np.broadcast_to(np.asarray(nominal if lengths is None else lengths, float), (4, 3)).copy()
        self.limits = np.asarray(limits if limits is not None else (config.limits if config is not None else leg.JOINT_LIMITS))
        hx, hy = body_length / 2, body_width / 2
        self.shoulders = np.array([[hx, hy, 0.0], [hx, -hy, 0.0], [-hx, hy, 0.0], [-hx, -hy, 0.0]])

    # ---- frames ----
    def _to_canonical(self, feet_body):
        return (np.asarray(feet_body, float) - self.shoulders) * _MIRROR

    def stance(self, height=STAND_HEIGHT, spread=0.0) -> np.ndarray:
        """Default foot positions in the body frame: under each hip. Shape (4, 3)."""
        side = np.where(IS_RIGHT, -1.0, 1.0)
        feet = self.shoulders.copy()
        feet[:, 1] += side * (self.lengths[:, 0] + spread)
        feet[:, 2] = -height
        return feet

    # ---- inverse and forward kinematics for all four legs ----
    def solve(self, feet_body) -> np.ndarray:
        """Joint angles (..., 4, 3) that the solver gives for body-frame foot targets (..., 4, 3)."""
        joint_angles = self.solver.predict(self._to_canonical(feet_body), self.lengths)
        return joint_angles * _MIRROR_ANGLES

    def joint_points(self, joint_angles) -> np.ndarray:
        """Shoulder, hip, knee, foot of every leg in the body frame. Shape (..., 4, 4, 3)."""
        joint_angles = np.asarray(joint_angles, float) * _MIRROR_ANGLES
        pts = leg.joint_positions(joint_angles, self.lengths)  # canonical shoulder frame
        return pts * _MIRROR[:, None, :] + self.shoulders[:, None, :]

    def feet(self, joint_angles) -> np.ndarray:
        """Where the given joint angles actually put the feet, body frame. Shape (..., 4, 3)."""
        return self.joint_points(joint_angles)[..., 3, :]

    def tracking_error(self, feet_body, joint_angles=None) -> np.ndarray:
        """Distance in metres between requested and achieved foot positions. Shape (..., 4)."""
        joint_angles = self.solve(feet_body) if joint_angles is None else joint_angles
        return np.linalg.norm(self.feet(joint_angles) - np.asarray(feet_body, float), axis=-1)

    def reachable(self, feet_body) -> np.ndarray:
        """True where a target lies inside the joint limits, judged by the exact solution."""
        target_position = self._to_canonical(feet_body)
        joint_angles = leg.analytic_ik(target_position, self.lengths)
        hit = np.linalg.norm(leg.forward(joint_angles, self.lengths) - target_position, axis=-1) < 1e-9
        return hit & np.all((joint_angles >= self.limits[:, 0]) & (joint_angles <= self.limits[:, 1]), axis=-1)

    def limit_margin(self, joint_angles) -> np.ndarray:
        """Smallest distance in radians from any joint to its limit. Shape (..., 4)."""
        joint_angles = np.asarray(joint_angles, float) * _MIRROR_ANGLES
        return np.minimum(joint_angles - self.limits[:, 0], self.limits[:, 1] - joint_angles).min(axis=-1)


def apply_posture(feet_body, offset=(0.0, 0.0, 0.0), roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """Move the body while the feet stay where they are on the ground.

    feet_body are foot positions (..., 4, 3) in the frame of the *unposed*
    body. The body is then shifted by `offset` and rotated by roll/pitch/yaw
    about its own centre. Returns the same feet in the posed body's frame,
    which is what the legs have to reach.
    """
    feet = np.asarray(feet_body, float)
    R = rotation(roll, pitch, yaw)
    rel = feet - np.asarray(offset, float)[..., None, :]
    return np.einsum("...ji,...kj->...ki", R, rel)  # R^T applied to each foot


def to_world(points_body, position=(0.0, 0.0, 0.0), roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """Body-frame points (..., N, 3) expressed in the world for a body at the given pose."""
    R = rotation(roll, pitch, yaw)
    return np.einsum("...ij,...kj->...ki", R, np.asarray(points_body, float)) + np.asarray(position, float)[..., None, :]


class URDFRobot:
    """Whole-body kinematics for any robot described by a URDF (any number of limbs, any DoF).

    Built from a `spec.RobotSpec`. `solvers` maps each model group to anything with
    predict(target_position, params) -> joint_angles for that group's canonical limb (a trained IKNet, or
    `limb.numeric_ik`-based solver); a single solver is accepted when there is one group.
    The body frame is the URDF base link's frame. Limbs keep the order of the manifest.

    `params` maps limb name -> link lengths and starts from the URDF values.
    It is a plain dict: update it at any time, for example with calibrated lengths.
    """

    def __init__(self, robot, solvers):
        self.spec = robot
        if not isinstance(solvers, dict):
            if len(robot.groups) != 1:
                raise ValueError(f"robot has {len(robot.groups)} model groups; pass solvers as {{group: model}}")
            solvers = {next(iter(robot.groups)): solvers}
        missing = set(robot.groups) - set(solvers)
        if missing:
            raise ValueError(f"no solver for group(s) {sorted(missing)}")
        self.solvers = solvers
        self.names = list(robot.limbs)
        self.params = robot.params()
        self._group = {m.name: g for g in robot.groups.values() for m in g.members}
        self._member = {m.name: m for g in robot.groups.values() for m in g.members}

    def limb(self, name):
        return self.spec.limbs[name]

    def _to_limb_frame(self, name, p_body):
        l = self.limb(name)
        return (np.asarray(p_body, float) - l.mount_t) @ l.mount_R  # R^T applied row-wise

    def _from_limb_frame(self, name, p_limb):
        l = self.limb(name)
        return np.asarray(p_limb, float) @ l.mount_R.T + l.mount_t

    def stance(self, height=STAND_HEIGHT) -> dict:
        """Feet under each limb's zero pose, at `height` below the base. name -> (3,)."""
        out = {}
        for n in self.names:
            l = self.limb(n)
            foot = self._from_limb_frame(n, l.forward(np.zeros(l.dof), self.params[n]))
            foot[2] = -height
            out[n] = foot
        return out

    def solve(self, feet_body: dict) -> dict:
        """Joint values per limb for body-frame tip targets (name -> (..., 3))."""
        out = {}
        for n in self.names:
            m, g = self._member[n], self._group[n]
            target_position = self._to_limb_frame(n, feet_body[n])
            if m.mirror is not None:
                target_position = m.mirror.target(target_position)
            joint_angles = self.solvers[g.name].predict(target_position, self.params[n])
            out[n] = m.mirror.angles(joint_angles) if m.mirror is not None else joint_angles
        return out

    def joint_points(self, joint_angles: dict) -> dict:
        """Origin, joints and tip of every limb in the body frame. name -> (..., dof + 2, 3)."""
        return {n: self._from_limb_frame(n, self.limb(n).points(np.asarray(joint_angles[n], float), self.params[n]))
                for n in self.names}

    def feet(self, joint_angles: dict) -> dict:
        return {n: points[..., -1, :] for n, points in self.joint_points(joint_angles).items()}

    def tracking_error(self, feet_body: dict, joint_angles: dict | None = None) -> dict:
        """Distance in metres between requested and achieved tips. name -> (...,)."""
        joint_angles = self.solve(feet_body) if joint_angles is None else joint_angles
        f = self.feet(joint_angles)
        return {n: np.linalg.norm(f[n] - np.asarray(feet_body[n], float), axis=-1) for n in self.names}

    def limit_margin(self, joint_angles: dict) -> dict:
        """Smallest distance from any joint to its limit, per limb."""
        return {n: np.minimum(np.asarray(joint_angles[n], float) - self.limb(n).limits[:, 0],
                              self.limb(n).limits[:, 1] - np.asarray(joint_angles[n], float)).min(axis=-1)
                for n in self.names}
