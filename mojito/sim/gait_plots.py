"""Pictures of a gait replay: gait timing chart and an isometric animation (matplotlib; imported lazily).

Both take the dictionary returned by `gait_replay.run_gait`. The animation needs the run to have been made
with `record_every` set, because it is drawn from the recorded frames.
"""
from __future__ import annotations

import numpy as np

from ..robot import body
from ..palette import AQUA, BLUE, GREY, INK, ORANGE, PURPLE

LEG_COLOURS = {"FL": BLUE, "FR": ORANGE, "RL": AQUA, "RR": PURPLE}
BODY_LENGTH, BODY_WIDTH = 0.30, 0.15  # as drawn; the model's body is these placeholders (stand.py)


def _runs(flags, times):
    """(start, length) pairs of consecutive True values, for broken_barh."""
    out, start = [], None
    for i, flag in enumerate(flags):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            out.append((times[start], times[i] - times[start]))
            start = None
    if start is not None:
        out.append((times[start], times[-1] - times[start]))
    return out


def draw_timing(ax, result, contact_force=0.5, cursor=None):
    """One row per leg: the commanded stance (thin bar, from gaits.py) above the measured load (thick bar)."""
    t = result["time_s"]
    commanded = result["commanded_contact"] & (t >= result["walk_start_s"])[:, None]  # before that the feet are just standing
    ax.axvspan(t[0], result["walk_start_s"], color="#eeeeee", lw=0)
    loaded = result["foot_load_n"] > contact_force
    for row, name in enumerate(body.LEG_NAMES):
        y = len(body.LEG_NAMES) - 1 - row
        ax.broken_barh(_runs(loaded[:, row], t), (y - 0.18, 0.36), color=LEG_COLOURS[name], alpha=0.85, lw=0)
        ax.broken_barh(_runs(commanded[:, row], t), (y + 0.24, 0.12), color=INK, lw=0)
    ax.set_yticks(range(4))
    ax.set_yticklabels(body.LEG_NAMES[::-1], fontsize=8)
    ax.set_ylim(-0.5, 3.5)
    ax.set_xlim(t[0], t[-1])
    ax.set_xlabel("s   (black line: commanded stance, coloured bar: foot carrying more than 0.5 N)", fontsize=7)
    ax.tick_params(labelsize=7)
    if cursor is not None:
        ax.axvline(cursor, color=INK, lw=0.8)


def save_timing_chart(result, path, title=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 2.6))
    draw_timing(ax, result)
    ax.set_title(title, fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def _body_outline(frame):
    """Corners of the body rectangle in the world, from the recorded body pose."""
    corners = np.array([[1, 1, 0], [1, -1, 0], [-1, -1, 0], [-1, 1, 0], [1, 1, 0]], float)
    corners *= [BODY_LENGTH / 2, BODY_WIDTH / 2, 0]
    return frame["origin"] + corners @ frame["R"].T


def toward_camera(points, elev_deg, azim_deg):
    """How far toward the viewer each world point is, for a Matplotlib 3D axes at (elev, azim); bigger = nearer.

    Matplotlib orders separate artists by one depth number each, which puts a far leg's line above a
    nearer body outline. Painting by an explicit zorder from this number fixes that.
    """
    elev, azim = np.radians(elev_deg), np.radians(azim_deg)
    view = np.array([np.cos(elev) * np.cos(azim), np.cos(elev) * np.sin(azim), np.sin(elev)])
    return np.asarray(points, float) @ view


def save_isometric(result, path, title="", fps=25, follow=True, span=0.3, grid=0.1, contact_force=0.5):
    """Isometric (orthographic) animation of the four legs and the body, with the gait timing chart below.

    The view follows the body so the gait stays in frame; the floor grid is fixed in the world, so its
    lines slide past at the body's real speed. A filled foot carries more than `contact_force` newtons.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    frames = result["frames"]
    every = int(round((result["time_s"][1] - result["time_s"][0]) and (len(result["time_s"]) / len(frames))))
    t_frames = result["time_s"][::every][:len(frames)]
    fig = plt.figure(figsize=(6.4, 6.2))
    ax = fig.add_axes([0.0, 0.28, 1.0, 0.68], projection="3d")
    timing = fig.add_axes([0.1, 0.07, 0.86, 0.17])
    start_xy = frames[0]["origin"][:2]

    def draw(i):
        frame = frames[i]
        ax.clear()
        centre = frame["origin"][:2] if follow else start_xy
        ax.computed_zorder = False  # paint in the zorder given below: far legs under the body, near legs over it
        ax.set_proj_type("ortho")
        ax.view_init(elev=35.264, azim=-45)  # the isometric angles
        ax.set_xlim(centre[0] - span, centre[0] + span)
        ax.set_ylim(centre[1] - span, centre[1] + span)
        ax.set_zlim(0, span)
        ax.set_box_aspect((1, 1, 0.5), zoom=1.3)  # equal scale on all three axes
        ax.set_axis_off()
        # floor grid fixed in the world
        gx = np.arange(np.floor((centre[0] - span) / grid), np.ceil((centre[0] + span) / grid) + 1) * grid
        gy = np.arange(np.floor((centre[1] - span) / grid), np.ceil((centre[1] + span) / grid) + 1) * grid
        for x in gx:
            ax.plot([x, x], [gy[0], gy[-1]], [0, 0], color="#cccccc", lw=0.6, zorder=0)
        for y in gy:
            ax.plot([gx[0], gx[-1]], [y, y], [0, 0], color="#cccccc", lw=0.6, zorder=0)
        outline = _body_outline(frame)
        ax.plot(*outline.T, color=INK, lw=3, zorder=2)
        body_near = toward_camera(frame["origin"], 35.264, -45)
        for leg, name in enumerate(body.LEG_NAMES):
            points = frame["legs3d"][leg]
            z = 3 if toward_camera(points[0], 35.264, -45) >= body_near else 1  # shoulder nearer than the body centre
            ax.plot(*points.T, "-", color=LEG_COLOURS[name], lw=3, zorder=z)
            foot = points[2]
            down = frame["loads"][leg] > contact_force
            ax.plot([foot[0]], [foot[1]], [foot[2]], "o", ms=7, color=LEG_COLOURS[name],
                    mfc=LEG_COLOURS[name] if down else "white", zorder=z)
            ax.plot([foot[0]], [foot[1]], [0.0], ".", ms=3, color=GREY, zorder=0)  # shadow, to read the height
            ax.text(*(points[0] + [0, 0, 0.02]), name, fontsize=7, color=LEG_COLOURS[name], zorder=z)
        ax.set_title(f"{title}\nt = {t_frames[i]:.2f} s", fontsize=9, color=INK, y=0.93)
        timing.clear()
        draw_timing(timing, result, contact_force, cursor=t_frames[i])

    FuncAnimation(fig, draw, frames=range(len(frames))).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
