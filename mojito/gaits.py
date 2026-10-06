"""Scripted gaits: where each foot should be, in the body frame, at time t.

These are kinematic reference motions. They say nothing about forces or
balance; they exist to exercise the leg solver over realistic foot paths and
to give later stages something to imitate.

The command is a body-frame velocity (vx forward, vy left, in m/s) and a yaw
rate (rad/s). A foot in stance is held fixed on the ground, so in the body
frame it moves exactly opposite to the body. A foot in swing lifts and travels
back to where the next stance begins.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Gait:
    name: str
    period: float  # seconds per full cycle
    duty: float  # fraction of the cycle each foot spends on the ground
    offsets: tuple  # phase at which each leg (FL, FR, RL, RR) starts its stance
    lift: float = 0.03  # swing height in metres
    sway: float = 0.0  # body shift away from the swinging leg, metres (crawl)


# Diagonal pairs move together; two feet down at all times.
TROT = Gait("trot", period=0.5, duty=0.5, offsets=(0.0, 0.5, 0.5, 0.0))
# One foot in the air at a time, in the order RL, FL, RR, FR.
# The body leans away from that foot while all four are down.
CRAWL = Gait("crawl", period=2.0, duty=0.85, offsets=(0.75, 0.25, 0.5, 0.0), lift=0.03, sway=0.025)


def _planar_flow(tau, v, omega):
    """Rotation angle and translation of a ground point seen from the body, tau seconds before it reaches its mid-stance position."""
    ang = omega * tau
    small = np.abs(omega) < 1e-9
    w = np.where(small, 1.0, omega)
    a = np.where(small, tau, np.sin(ang) / w)  # integral of cos
    b = np.where(small, 0.0, (1.0 - np.cos(ang)) / w)  # integral of sin
    shift = np.stack([a * v[0] - b * v[1], b * v[0] + a * v[1]], axis=-1)
    return ang, shift


def _stance_point(home, tau, v, omega):
    """Body-frame position of a grounded foot that sits at `home` when tau = 0."""
    ang, shift = _planar_flow(tau, v, omega)
    c, s = np.cos(ang), np.sin(ang)
    x = c * home[..., 0] - s * home[..., 1] + shift[..., 0]
    y = s * home[..., 0] + c * home[..., 1] + shift[..., 1]
    return np.stack([x, y, np.broadcast_to(home[..., 2], x.shape)], axis=-1)


def foot_targets(gait: Gait, t, home, velocity=(0.0, 0.0), yaw_rate=0.0):
    """Foot targets in the body frame.

    t         scalar or (T,) times in seconds
    home      (4, 3) neutral foot positions (see Robot.stance)
    returns   feet (T, 4, 3), contact (T, 4) booleans, phase-in-swing (T, 4)
    """
    t = np.atleast_1d(np.asarray(t, float))[:, None]  # (T, 1)
    v = np.asarray(velocity, float)
    home = np.asarray(home, float)[None]  # (1, 4, 3)
    phase = (t / gait.period - np.asarray(gait.offsets)[None]) % 1.0  # (T, 4)
    contact = phase < gait.duty
    t_stance = gait.duty * gait.period

    # Stance: s runs 0 -> 1, the foot passes `home` at s = 0.5.
    s = np.where(contact, phase / gait.duty, 0.0)
    stance = _stance_point(home, (0.5 - s) * t_stance, v, yaw_rate)

    # Swing: from where stance ended to where the next one starts.
    u = np.where(contact, 0.0, (phase - gait.duty) / (1.0 - gait.duty))
    start = _stance_point(home, np.full_like(u, -0.5 * t_stance), v, yaw_rate)
    end = _stance_point(home, np.full_like(u, 0.5 * t_stance), v, yaw_rate)
    blend = (u - np.sin(2 * np.pi * u) / (2 * np.pi))[..., None]  # starts and ends at rest
    swing = start + (end - start) * blend
    swing[..., 2] += gait.lift * np.sin(np.pi * u) ** 2

    feet = np.where(contact[..., None], stance, swing)
    return feet, contact, u


def body_sway(gait: Gait, t, home):
    """Horizontal body shift (T, 3) that leans away from the leg in the air.

    Only for gaits that lift one foot at a time (gait.sway > 0 and no
    overlapping swings); zero otherwise. The lean is held for the whole of a
    leg's swing and blends smoothly to the next leg's lean while all four feet
    are down, so the body is already across before a foot leaves the ground.
    """
    t = np.atleast_1d(np.asarray(t, float))
    out = np.zeros((t.shape[0], 3))
    swing_len = 1.0 - gait.duty
    starts = (np.asarray(gait.offsets) + gait.duty) % 1.0  # phase at which each leg lifts
    order = np.argsort(starts)
    gaps = np.diff(np.append(starts[order], starts[order][0] + 1.0)) - swing_len
    if gait.sway == 0.0 or np.any(gaps < -1e-9):
        return out
    lean = -np.sign(np.asarray(home, float)[:, :2]) * gait.sway  # (4, 2) target per swinging leg
    phase = (t / gait.period) % 1.0
    for k, i in enumerate(order):
        nxt = order[(k + 1) % 4]
        since = (phase - starts[i]) % 1.0  # phase since leg i lifted
        hold = since < swing_len
        move = (~hold) & (since < swing_len + gaps[k])
        out[hold, :2] = lean[i]
        if gaps[k] > 1e-9:
            x = (since[move] - swing_len) / gaps[k]
            w = (x * x * (3 - 2 * x))[:, None]
            out[move, :2] = lean[i] * (1 - w) + lean[nxt] * w
    return out


def body_path(t, velocity=(0.0, 0.0), yaw_rate=0.0):
    """World position (T, 3) and yaw (T,) of a body that starts at the origin."""
    t = np.atleast_1d(np.asarray(t, float))
    ang, shift = _planar_flow(t, np.asarray(velocity, float), yaw_rate)
    return np.concatenate([shift, np.zeros((t.shape[0], 1))], axis=1), ang


def stability_margin(feet_xy, contact, point_xy):
    """Distance from a point to the edge of the support polygon, per time step.

    Positive when the point is inside the polygon of grounded feet. With fewer
    than three feet down the support is a line, and the result is minus the
    distance to it. Feet must be given in FL, FR, RL, RR order.
    """
    order = [0, 1, 3, 2]  # walk around the body
    out = np.empty(feet_xy.shape[0])
    for i in range(feet_xy.shape[0]):
        pts = np.array([feet_xy[i, k] for k in order if contact[i, k]])
        c = point_xy[i]
        if len(pts) < 3:
            a, b = pts[0], pts[-1]
            ab = b - a
            u = np.clip(np.dot(c - a, ab) / max(np.dot(ab, ab), 1e-12), 0, 1)
            out[i] = -np.linalg.norm(c - (a + u * ab))
            continue
        d = []
        for j in range(len(pts)):
            a, b = pts[j], pts[(j + 1) % len(pts)]
            e = b - a
            d.append((e[0] * (c[1] - a[1]) - e[1] * (c[0] - a[0])) / np.linalg.norm(e))
        # Feet are visited clockwise seen from above, so inside means every cross product is negative.
        out[i] = (-np.array(d)).min()
    return out
