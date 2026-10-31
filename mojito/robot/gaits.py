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

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Gait:
    name: str
    period: float  # seconds per full cycle
    duty: float  # fraction of the cycle each foot spends on the ground
    offsets: tuple  # phase at which each leg (FL, FR, RL, RR) starts its stance
    lift: float = 0.03  # swing height in metres
    sway: float = 0.0  # body shift away from the swinging leg, metres (crawl)

    def __post_init__(self):
        if self.period <= 0 or not 0.0 < self.duty < 1.0 or len(self.offsets) != 4:
            raise ValueError("a gait needs period > 0, 0 < duty < 1 and four offsets (FL, FR, RL, RR)")

    @property
    def stance_time(self) -> float:
        """Seconds each foot is on the ground per cycle: duty * period."""
        return self.duty * self.period

    @property
    def swing_time(self) -> float:
        """Seconds each foot is in the air per cycle: (1 - duty) * period."""
        return (1.0 - self.duty) * self.period

    def swing_starts(self) -> np.ndarray:
        """Cycle phase at which each leg (FL, FR, RL, RR) lifts: its stance start plus the duty, mod 1."""
        return (np.asarray(self.offsets, float) + self.duty) % 1.0


# Diagonal pairs move together; two feet down at all times.
# The period is 1.7 s. Stage 1 first used 0.5 s, set before any physics; a period sweep in MuJoCo showed that was not
# optimal and 1.7 s was chosen from it (docs/STAGE2.md, "Longer trot periods"). The swing height is 22 mm, from the one-at-a-time
# and combined sweeps ("Pushing the best settings further", "Do the best values add up?"); it was 30 mm before.
# TROT_STAGE1 keeps the original 0.5 s and 30 mm for comparison. It has its own name so that code keyed on the name "trot"
# (gait_replay.settings_for: 0.16 m stance, kp 50, kd 0.9, all tuned for the 1.7 s trot) does not treat it as the tuned trot.
TROT = Gait("trot", period=1.7, duty=0.5, offsets=(0.0, 0.5, 0.5, 0.0), lift=0.022)
TROT_STAGE1 = replace(TROT, name="trot_stage1", period=0.5, lift=0.03)
# One foot in the air at a time, in the order RL, FL, RR, FR.
# The body leans away from that foot while all four are down.
CRAWL = Gait("crawl", period=2.0, duty=0.85, offsets=(0.75, 0.25, 0.5, 0.0), lift=0.03, sway=0.025)


def sequence_offsets(order, duty, first_lift=0.1):
    """Offsets for a gait in which the legs lift one after another at equal spacing.

    order       the four leg indices (0 FL, 1 FR, 2 RL, 3 RR) in the order they lift
    first_lift  cycle phase at which the first leg lifts (a common shift only changes where the cycle starts)

    Leg order[k] lifts at phase  first_lift + k/4  and starts its stance a swing later, so its offset
    (the phase at which stance starts) is  (lift - duty) mod 1.  Checked against CRAWL in tests/robot/test_body.py.
    """
    offsets = [0.0] * 4
    for k, leg in enumerate(order):
        offsets[leg] = (first_lift + k / 4.0 - duty) % 1.0
    return tuple(offsets)


def _planar_flow(time_offset, velocity, yaw_rate):
    """Heading angle and displacement accumulated over `time_offset` seconds of constant planar motion.

    The body moves with body-frame velocity (vx, vy) and yaw rate w, so after
    time T its heading is  angle = w T  and its velocity turns with it. The
    displacement is the integral of the rotated velocity:

        shift = integral from 0 to T of R(w s) (vx, vy) ds
              = ( sin(wT)/w * vx - (1 - cos(wT))/w * vy,
                  (1 - cos(wT))/w * vx + sin(wT)/w * vy )

    For w -> 0 the two integrals tend to T and 0 (straight-line motion), which
    is what the `small` branch uses to avoid dividing by zero.
    Returns (angle, shift) with shift shaped (..., 2). `time_offset` may be negative.
    """
    angle = yaw_rate * time_offset
    small = np.abs(yaw_rate) < 1e-9
    safe_rate = np.where(small, 1.0, yaw_rate)  # placeholder so the unused branch does not divide by zero
    int_cos = np.where(small, time_offset, np.sin(angle) / safe_rate)  # integral of cos(w s) ds
    int_sin = np.where(small, 0.0, (1.0 - np.cos(angle)) / safe_rate)  # integral of sin(w s) ds
    shift = np.stack(
        [int_cos * velocity[0] - int_sin * velocity[1], int_sin * velocity[0] + int_cos * velocity[1]], axis=-1
    )
    return angle, shift


def _stance_point(home, time_offset, velocity, yaw_rate):
    """Body-frame position of a grounded foot that sits at `home` when time_offset = 0.

    A planar rotation by the flow angle plus the flow displacement:
        [x, y] = R(angle) [home_x, home_y] + shift
    The height z is unchanged.
    """
    angle, shift = _planar_flow(time_offset, velocity, yaw_rate)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    x = cos_a * home[..., 0] - sin_a * home[..., 1] + shift[..., 0]
    y = sin_a * home[..., 0] + cos_a * home[..., 1] + shift[..., 1]
    return np.stack([x, y, np.broadcast_to(home[..., 2], x.shape)], axis=-1)


def foot_targets(gait: Gait, times, home, velocity=(0.0, 0.0), yaw_rate=0.0):
    """Foot targets in the body frame.

    times     scalar or (T,) times in seconds
    home      (4, 3) neutral foot positions (see Robot.stance)
    returns   feet (T, 4, 3), contact (T, 4) booleans, phase-in-swing (T, 4)
    """
    times = np.atleast_1d(np.asarray(times, float))[:, None]  # (T, 1)
    velocity = np.asarray(velocity, float)
    home = np.asarray(home, float)[None]  # (1, 4, 3)
    # Fraction of the cycle each leg has completed since its stance began, in [0, 1).
    phase = (times / gait.period - np.asarray(gait.offsets)[None]) % 1.0  # (T, 4)
    contact = phase < gait.duty
    stance_duration = gait.duty * gait.period  # seconds

    # Stance: progress runs 0 -> 1 and the foot passes `home` at progress = 0.5,
    # so the time until it reaches home is (0.5 - progress) * stance_duration.
    stance_progress = np.where(contact, phase / gait.duty, 0.0)
    stance = _stance_point(home, (0.5 - stance_progress) * stance_duration, velocity, yaw_rate)

    # Swing: from where stance ended to where the next one starts. swing_progress
    # runs 0 -> 1 over the swing. Stance ends half a stance after home and the
    # next begins half a stance before it.
    swing_progress = np.where(contact, 0.0, (phase - gait.duty) / (1.0 - gait.duty))
    swing_start = _stance_point(home, np.full_like(swing_progress, -0.5 * stance_duration), velocity, yaw_rate)
    swing_end = _stance_point(home, np.full_like(swing_progress, 0.5 * stance_duration), velocity, yaw_rate)
    # Blend u - sin(2 pi u)/(2 pi): its slope 1 - cos(2 pi u) is zero at u = 0 and 1,
    # so the foot leaves and lands with zero horizontal speed.
    blend = (swing_progress - np.sin(2 * np.pi * swing_progress) / (2 * np.pi))[..., None]
    swing = swing_start + (swing_end - swing_start) * blend
    # Lift: sin^2(pi u) is 0 at both ends with zero slope, peak `lift` mid-swing.
    swing[..., 2] += gait.lift * np.sin(np.pi * swing_progress) ** 2

    feet = np.where(contact[..., None], stance, swing)
    return feet, contact, swing_progress


def body_sway(gait: Gait, times, home):
    """Horizontal body shift (T, 3) that leans away from the leg in the air.

    Only for gaits that lift one foot at a time (gait.sway > 0 and no
    overlapping swings); zero otherwise. The lean is held for the whole of a
    leg's swing and blends smoothly to the next leg's lean while all four feet
    are down, so the body is already across before a foot leaves the ground.
    """
    times = np.atleast_1d(np.asarray(times, float))
    out = np.zeros((times.shape[0], 3))
    swing_len = 1.0 - gait.duty
    starts = (np.asarray(gait.offsets) + gait.duty) % 1.0  # phase at which each leg lifts
    order = np.argsort(starts)
    gaps = np.diff(np.append(starts[order], starts[order][0] + 1.0)) - swing_len
    if gait.sway == 0.0 or np.any(gaps < -1e-9):
        return out
    lean = -np.sign(np.asarray(home, float)[:, :2]) * gait.sway  # (4, 2) target per swinging leg
    phase = (times / gait.period) % 1.0
    for k, leg_index in enumerate(order):
        next_leg = order[(k + 1) % 4]
        since_lift = (phase - starts[leg_index]) % 1.0  # cycle fraction since this leg lifted
        holding = since_lift < swing_len
        moving = (~holding) & (since_lift < swing_len + gaps[k])
        out[holding, :2] = lean[leg_index]
        if gaps[k] > 1e-9:
            progress = (since_lift[moving] - swing_len) / gaps[k]  # 0 -> 1 across the gap
            # Smoothstep 3x^2 - 2x^3: zero slope at both ends, so the body starts and stops leaning gently.
            weight = (progress * progress * (3 - 2 * progress))[:, None]
            out[moving, :2] = lean[leg_index] * (1 - weight) + lean[next_leg] * weight
    return out


def body_path(times, velocity=(0.0, 0.0), yaw_rate=0.0):
    """World position (T, 3) and yaw (T,) of a body that starts at the origin."""
    times = np.atleast_1d(np.asarray(times, float))
    angle, shift = _planar_flow(times, np.asarray(velocity, float), yaw_rate)
    return np.concatenate([shift, np.zeros((times.shape[0], 1))], axis=1), angle


def stability_margin(feet_xy, contact, point_xy):
    """Distance from a point to the edge of the support polygon, per time step.

    Positive when the point is inside the polygon of grounded feet. With fewer
    than three feet down the support is a line, and the result is minus the
    distance to it. With no foot down there is no support at all and the result
    is -inf. Feet that sit on the same spot count once, so two coincident feet
    make a point and three feet with two coincident a line (no zero-length edge).
    Feet must be given in FL, FR, RL, RR order.
    """
    walk_order = [0, 1, 3, 2]  # FL, FR, RR, RL: around the body, clockwise seen from above
    margin = np.empty(feet_xy.shape[0])
    for i in range(feet_xy.shape[0]):
        grounded = np.array([feet_xy[i, k] for k in walk_order if contact[i, k]])
        point = point_xy[i]
        if len(grounded) == 0:
            margin[i] = -np.inf
            continue
        # Drop a foot that coincides with the one before it around the polygon: its edge has no length and no direction.
        distinct = [foot for j, foot in enumerate(grounded) if np.linalg.norm(foot - grounded[j - 1]) > 1e-12]
        grounded = np.array(distinct) if distinct else grounded[:1]
        if len(grounded) < 3:
            # Support is a point or a segment from a to b. Closest point on it is
            # start + f (end - start) with f = (point - start).(end - start) / |end - start|^2, clamped to [0, 1].
            start, end = grounded[0], grounded[-1]
            edge = end - start
            fraction = np.clip(np.dot(point - start, edge) / max(np.dot(edge, edge), 1e-12), 0, 1)
            margin[i] = -np.linalg.norm(point - (start + fraction * edge))
            continue
        signed_distances = []
        for j in range(len(grounded)):
            start, end = grounded[j], grounded[(j + 1) % len(grounded)]
            edge = end - start
            # 2-D cross product edge x (point - start), divided by |edge|, is the
            # signed distance from the edge's line to the point.
            cross = edge[0] * (point[1] - start[1]) - edge[1] * (point[0] - start[0])
            signed_distances.append(cross / np.linalg.norm(edge))
        # Feet are visited clockwise seen from above, so inside means every cross product is negative.
        margin[i] = (-np.array(signed_distances)).min()
    return margin
