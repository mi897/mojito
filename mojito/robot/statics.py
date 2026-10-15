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
    Solves sum(f) = W, sum(f x) = W cx, sum(f y) = W cy (W = weight, cx, cy =
    com_xy). A negative entry means
    the centre of mass is outside the support triangle (the robot would tip).
    """
    feet_xy = np.asarray(feet_xy, float)
    # Three equations in the three loads f_i, one row each for force balance
    # and the two moment balances about the origin:
    #   sum f_i = W,   sum f_i x_i = W cx,   sum f_i y_i = W cy
    equilibrium_matrix = np.vstack([np.ones(3), feet_xy[:, 0], feet_xy[:, 1]])
    required = weight * np.array([1.0, com_xy[0], com_xy[1]])
    return np.linalg.solve(equilibrium_matrix, required)


def joint_torques(joint_angles, lengths, load) -> np.ndarray:
    """|joint torque| (abduction, hip pitch, knee) holding a vertical load on the foot.

    joint_angles: (3,) joint angles of a canonical (left) leg; load: upward force on the foot.
    Torque = J^T F for a force F = (0, 0, load) in the leg frame. Right legs are
    mirrors, and a mirror leaves the vertical component alone, so they match.
    """
    jac = leg.jacobian(np.asarray(joint_angles, float), np.asarray(lengths, float))
    foot_force = np.array([0.0, 0.0, load])  # F in the leg frame, z up
    return np.abs(jac.T @ foot_force)  # tau = J^T F, by virtual work: tau . dq = F . dp
