"""Static joint torques for a robot standing still, from foot loads.

Geometry only plus gravity: no leg mass, no motion, no friction limit. It says
how much torque the joints must hold, not whether the robot can walk.
"""
import numpy as np

from . import leg

G = 9.81  # m/s^2


def foot_loads(feet_xy, com_xy, weight=1.0) -> np.ndarray:
    """Vertical load on each of three feet that balances weight at com_xy.

    feet_xy: (3, 2) foot positions in the horizontal plane; com_xy: (2,).
    Solves sum(f) = W, sum(f x) = W cx, sum(f y) = W cy. A negative entry means
    the centre of mass is outside the support triangle (the robot would tip).
    """
    feet_xy = np.asarray(feet_xy, float)
    A = np.vstack([np.ones(3), feet_xy[:, 0], feet_xy[:, 1]])
    b = weight * np.array([1.0, com_xy[0], com_xy[1]])
    return np.linalg.solve(A, b)


def joint_torques(q, lengths, load) -> np.ndarray:
    """|joint torque| (abduction, hip pitch, knee) holding a vertical load on the foot.

    q: (3,) joint angles of a canonical (left) leg; load: upward force on the foot.
    Torque = J^T F for a force F = (0, 0, load) in the leg frame. Right legs are
    mirrors, and a mirror leaves the vertical component alone, so they match.
    """
    J = leg.jacobian(np.asarray(q, float), np.asarray(lengths, float))
    return np.abs(J.T @ np.array([0.0, 0.0, load]))
