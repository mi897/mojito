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

from . import leg

LEG_NAMES = ("FL", "FR", "RL", "RR")
IS_RIGHT = np.array([False, True, False, True])
_MIRROR = np.where(IS_RIGHT[:, None], [1.0, -1.0, 1.0], 1.0)  # (4, 3)

# Placeholder body: shoulder spacing in metres.
BODY_LENGTH = 0.24  # front shoulders to rear shoulders
BODY_WIDTH = 0.10  # left shoulders to right shoulders
STAND_HEIGHT = 0.17  # shoulder height above the ground when standing


def rotation(roll=0.0, pitch=0.0, yaw=0.0) -> np.ndarray:
    """Body-to-world rotation matrix, R = Rz(yaw) Ry(pitch) Rx(roll). Shape (..., 3, 3)."""
    roll, pitch, yaw = np.broadcast_arrays(*(np.asarray(a, dtype=float) for a in (roll, pitch, yaw)))
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    R = np.empty(roll.shape + (3, 3))
    R[..., 0, 0], R[..., 0, 1], R[..., 0, 2] = cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr
    R[..., 1, 0], R[..., 1, 1], R[..., 1, 2] = sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr
    R[..., 2, 0], R[..., 2, 1], R[..., 2, 2] = -sp, cp * sr, cp * cr
    return R


class AnalyticSolver:
    """Closed-form leg IK with the same call signature as the network."""

    def predict(self, p, lengths):
        return leg.analytic_ik(p, lengths)


class Robot:
    """Geometry of the quadruped plus whichever leg solver drives it.

    solver   anything with predict(p, lengths) -> q for a canonical (left) leg:
             a trained IKNet, or AnalyticSolver() for the exact reference.
    lengths  (3,) for four identical legs or (4, 3) for one row per leg.
             Plain attribute: change it at any time.
    """

    def __init__(self, solver, lengths=None, body_length=BODY_LENGTH, body_width=BODY_WIDTH,
                 limits=None):
        self.solver = solver
        cfg = getattr(solver, "cfg", None)
        nominal = cfg.nominal if cfg is not None else leg.NOMINAL_LENGTHS
        self.lengths = np.broadcast_to(np.asarray(nominal if lengths is None else lengths, float), (4, 3)).copy()
        self.limits = np.asarray(limits if limits is not None else (cfg.limits if cfg is not None else leg.JOINT_LIMITS))
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
        q = self.solver.predict(self._to_canonical(feet_body), self.lengths)
        return q * np.where(IS_RIGHT[:, None], [-1.0, 1.0, 1.0], 1.0)

    def joint_points(self, q) -> np.ndarray:
        """Shoulder, hip, knee, foot of every leg in the body frame. Shape (..., 4, 4, 3)."""
        q = np.asarray(q, float) * np.where(IS_RIGHT[:, None], [-1.0, 1.0, 1.0], 1.0)
        pts = leg.joint_positions(q, self.lengths)  # canonical shoulder frame
        return pts * _MIRROR[:, None, :] + self.shoulders[:, None, :]

    def feet(self, q) -> np.ndarray:
        """Where joint angles q actually put the feet, body frame. Shape (..., 4, 3)."""
        return self.joint_points(q)[..., 3, :]

    def tracking_error(self, feet_body, q=None) -> np.ndarray:
        """Distance in metres between requested and achieved foot positions. Shape (..., 4)."""
        q = self.solve(feet_body) if q is None else q
        return np.linalg.norm(self.feet(q) - np.asarray(feet_body, float), axis=-1)

    def reachable(self, feet_body) -> np.ndarray:
        """True where a target lies inside the joint limits, judged by the exact solution."""
        p = self._to_canonical(feet_body)
        q = leg.analytic_ik(p, self.lengths)
        hit = np.linalg.norm(leg.forward(q, self.lengths) - p, axis=-1) < 1e-9
        return hit & np.all((q >= self.limits[:, 0]) & (q <= self.limits[:, 1]), axis=-1)

    def limit_margin(self, q) -> np.ndarray:
        """Smallest distance in radians from any joint to its limit. Shape (..., 4)."""
        q = np.asarray(q, float) * np.where(IS_RIGHT[:, None], [-1.0, 1.0, 1.0], 1.0)
        return np.minimum(q - self.limits[:, 0], self.limits[:, 1] - q).min(axis=-1)


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
