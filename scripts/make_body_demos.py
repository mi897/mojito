"""Whole-body demos driven by the trained leg network.

Every pose drawn here comes from the network's joint angles pushed through
forward kinematics. Nothing is drawn from the targets except the orange
markers, so any error the network makes is in the picture and in the numbers.

    results/posture_demo.gif   body twisting and turning with the feet planted
    results/trot_forward.gif   trot in a straight line
    results/trot_turn.gif      trot turning on the spot
    results/crawl_forward.gif  crawl with the body leaning away from the lifted foot
    results/body_metrics.json  tracking error, joint-limit margin, stability

    python scripts/make_body_demos.py [--weights weights/ik_leg.npz] [--only posture trot_forward ...]
"""
import argparse
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import IKNet, body, gaits  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e0"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 9,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlesize": 9.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 1.8,
})


def smooth_bump(t, start, length):
    """One full sine wave between start and start+length, zero elsewhere."""
    x = (t - start) / length
    return np.where((x >= 0) & (x <= 1), np.sin(2 * np.pi * x), 0.0)


def posture_script(robot, fps=15):
    """Feet planted; the body rolls, pitches, twists, bobs, then does all of it at once."""
    t = np.arange(0, 12.0, 1 / fps)
    roll = np.radians(18) * smooth_bump(t, 0.0, 2.0)
    pitch = np.radians(14) * smooth_bump(t, 2.0, 2.0)
    yaw = np.radians(22) * smooth_bump(t, 4.0, 2.0)
    z = 0.03 * smooth_bump(t, 6.0, 2.0)
    # Finale: the body circles like a hula hoop.
    x = (t - 8.0) / 4.0
    env = np.where((x >= 0) & (x <= 1), np.sin(np.pi * x) ** 2, 0.0)
    roll = roll + np.radians(12) * env * np.sin(2 * np.pi * (t - 8.0))
    pitch = pitch + np.radians(10) * env * np.cos(2 * np.pi * (t - 8.0))
    yaw = yaw + np.radians(14) * env * np.sin(np.pi * (t - 8.0))
    offset = np.stack([np.zeros_like(t), np.zeros_like(t), z], axis=1)
    home = robot.stance()
    feet_world = np.broadcast_to(home, (len(t), 4, 3)).copy()
    targets = body.apply_posture(feet_world, offset, roll, pitch, yaw)
    return dict(
        t=t, fps=fps, targets=targets, feet_world=feet_world, contact=np.ones((len(t), 4), bool),
        position=offset, rpy=np.stack([roll, pitch, yaw], axis=1),
        title="Twisting and turning in place: feet planted, body pose commanded",
        inputs=("Commanded body rotation (deg)", [("roll", np.degrees(roll)), ("pitch", np.degrees(pitch)), ("yaw", np.degrees(yaw))]),
    )


def gait_script(robot, gait, velocity, yaw_rate, cycles, fps, title):
    t = np.arange(0, cycles * gait.period, 1 / fps)
    home = robot.stance()
    feet, contact, _ = gaits.foot_targets(gait, t, home, velocity, yaw_rate)
    sway = gaits.body_sway(gait, t, home)
    targets = body.apply_posture(feet, offset=sway)
    pos, yaw = gaits.body_path(t, velocity, yaw_rate)
    position = pos + np.einsum("tij,tj->ti", body.rotation(yaw=yaw), sway)
    feet_world = body.to_world(feet, pos, yaw=yaw)
    rpy = np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], axis=1)
    return dict(t=t, fps=fps, targets=targets, feet_world=feet_world, contact=contact, position=position, rpy=rpy,
                title=title, inputs=("Feet on the ground", None), gait=gait, velocity=velocity, yaw_rate=yaw_rate)


def measure(robot, reference, s):
    """Run the network on every frame and record how well it did."""
    q = robot.solve(s["targets"])  # the model's outputs: (T, 4, 3)
    err = robot.tracking_error(s["targets"], q) * 1000
    s["q"], s["err_mm"] = q, err
    s["points_world"] = body.to_world(
        robot.joint_points(q).reshape(len(q), 16, 3), s["position"], *s["rpy"].T).reshape(len(q), 4, 4, 3)
    out = dict(
        frames=int(len(q)),
        all_targets_within_joint_limits=bool(robot.reachable(s["targets"]).all()),
        foot_error_mm=dict(median=float(np.median(err)), p95=float(np.percentile(err, 95)), max=float(err.max())),
        joint_limit_margin_deg=float(np.degrees(robot.limit_margin(q).min())),
        max_joint_difference_from_exact_deg=float(np.degrees(np.abs(q - reference.solve(s["targets"])).max())),
    )
    if "gait" in s:
        g = s["gait"]
        planted = s["contact"][1:] & s["contact"][:-1]
        actual_world = s["points_world"][:, :, 3]
        slip = np.linalg.norm(np.diff(actual_world, axis=0), axis=-1)[planted]
        margin = gaits.stability_margin(s["feet_world"][..., :2], s["contact"], s["position"][:, :2])
        out.update(
            gait=g.name, period_s=g.period, duty=g.duty, velocity_m_s=list(s["velocity"]), yaw_rate_rad_s=s["yaw_rate"],
            feet_down_min=int(s["contact"].sum(1).min()),
            stance_foot_drift_mm_per_frame_max=float(slip.max() * 1000),
            body_centre_inside_support_mm_min=float(margin.min() * 1000),
        )
    return out


def animate(robot, s, path, dpi=80):
    T = len(s["t"])
    fig = plt.figure(figsize=(10, 5.4))
    gs = fig.add_gridspec(3, 2, width_ratios=[1.65, 1], left=0.0, right=0.97, top=0.9, bottom=0.1, hspace=0.75, wspace=0.08)
    ax = fig.add_subplot(gs[:, 0], projection="3d")
    ax_in, ax_q, ax_e = (fig.add_subplot(gs[i, 1]) for i in range(3))
    fig.suptitle(s["title"], x=0.02, ha="left", fontsize=11, fontweight="bold")

    # --- side panels (static curves, moving cursor) ---
    label, series = s["inputs"]
    ax_in.set_title(label)
    if series:
        for (name, y), c in zip(series, (BLUE, ORANGE, AQUA)):
            ax_in.plot(s["t"], y, color=c, label=name)
        ax_in.legend(frameon=False, ncol=3, loc="upper right", fontsize=8, handlelength=1.2, borderaxespad=0)
        lo, hi = ax_in.get_ylim(); ax_in.set_ylim(lo, hi + 0.35 * (hi - lo))
    else:
        dt = s["t"][1] - s["t"][0]
        for k, name in enumerate(body.LEG_NAMES):
            on = s["contact"][:, k]
            edges = np.flatnonzero(np.diff(np.concatenate([[0], on.astype(int), [0]])))
            ax_in.broken_barh([(s["t"][a] if a < T else s["t"][-1] + dt, (b - a) * dt) for a, b in zip(edges[::2], edges[1::2])],
                              (3 - k - 0.3, 0.6), color=BLUE, lw=0)
        ax_in.set_yticks(range(4)); ax_in.set_yticklabels(body.LEG_NAMES[::-1]); ax_in.set_ylim(-0.6, 3.6)
        ax_in.grid(False)
    ax_q.set_title("Model output: front-left joint angles (deg)")
    for j, (name, c) in enumerate(zip(("abduction", "hip", "knee"), (BLUE, ORANGE, AQUA))):
        ax_q.plot(s["t"], np.degrees(s["q"][:, 0, j]), color=c, label=name)
    ax_q.legend(frameon=False, ncol=3, loc="upper right", fontsize=8, handlelength=1.2, borderaxespad=0)
    lo, hi = ax_q.get_ylim(); ax_q.set_ylim(lo, hi + 0.4 * (hi - lo))
    ax_e.set_title("Worst foot error across the four legs (mm)")
    ax_e.plot(s["t"], s["err_mm"].max(axis=1), color=BLUE)
    ax_e.set_ylim(0, max(0.5, s["err_mm"].max() * 1.25)); ax_e.set_xlabel("time (s)")
    cursors = [a.axvline(0, color=INK2, lw=1) for a in (ax_in, ax_q, ax_e)]
    for a in (ax_in, ax_q, ax_e):
        a.set_xlim(s["t"][0], s["t"][-1])

    # --- 3D scene ---
    ax.set_axis_off(); ax.set_proj_type("persp", focal_length=0.35)
    ax.view_init(elev=22, azim=-58); ax.set_box_aspect((1, 1, 0.62), zoom=1.45)
    half = 0.27
    gx = np.arange(-2.0, 2.0001, 0.05)
    grid_lines = [ax.plot([], [], [], color=GRID, lw=0.7)[0] for _ in range(2)]
    shadow = Poly3DCollection([np.zeros((4, 3))], facecolor="#e9e8e3", edgecolor="none")
    ax.add_collection3d(shadow)
    targets_on = ax.plot([], [], [], "o", color=ORANGE, ms=9, mfc="none", mew=1.8)[0]
    targets_off = ax.plot([], [], [], "o", color=ORANGE, ms=5, mfc="none", mew=1.2, alpha=0.6)[0]
    torso = Poly3DCollection([np.zeros((4, 3))], facecolor="#c9d6e6", edgecolor=INK2, linewidths=1.2, alpha=0.9)
    ax.add_collection3d(torso)
    front_label = ax.text(0, 0, 0, "front", color=INK2, fontsize=8, ha="center", va="bottom")
    legs = [ax.plot([], [], [], color=BLUE, lw=3.2, marker="o", ms=4.5, solid_capstyle="round")[0] for _ in range(4)]
    readout = ax.text2D(0.03, 0.02, "", transform=ax.transAxes, color=INK2, fontsize=9)
    ax.text2D(0.03, 0.065, "blue: where the model's angles put the legs    orange: requested foot positions",
              transform=ax.transAxes, color=INK2, fontsize=8)
    order = [0, 1, 3, 2]  # FL, FR, RR, RL around the body
    centre0 = s["position"][0].copy()

    def draw(i):
        P = s["points_world"][i]  # (4 legs, 4 points, 3)
        c = s["position"][i] if "gait" in s else centre0
        ax.set_xlim(c[0] - half, c[0] + half); ax.set_ylim(c[1] - half, c[1] + half); ax.set_zlim(-0.17, -0.17 + 2 * half * 0.62)
        # ground grid, clipped to the view
        xs = gx[(gx > c[0] - half) & (gx < c[0] + half)]; ys = gx[(gx > c[1] - half) & (gx < c[1] + half)]
        seg_x = np.concatenate([[[x, c[1] - half, -0.17], [x, c[1] + half, -0.17], [np.nan] * 3] for x in xs])
        seg_y = np.concatenate([[[c[0] - half, y, -0.17], [c[0] + half, y, -0.17], [np.nan] * 3] for y in ys])
        for line, seg in zip(grid_lines, (seg_x, seg_y)):
            line.set_data_3d(seg[:, 0], seg[:, 1], seg[:, 2])
        sh = P[order, 0].copy(); sh[:, 2] = -0.1695
        shadow.set_verts([sh])
        torso.set_verts([P[order, 0]])
        front = P[[0, 1], 0].mean(axis=0) + 0.6 * (P[[0, 1], 0].mean(axis=0) - P[[2, 3], 0].mean(axis=0)) * 0.12
        front_label.set_position_3d((front[0], front[1], front[2] + 0.012))
        for line, pts in zip(legs, P):
            line.set_data_3d(pts[:, 0], pts[:, 1], pts[:, 2])
        tw = body.to_world(s["targets"][i], s["position"][i], *s["rpy"][i])
        on = s["contact"][i]
        targets_on.set_data_3d(tw[on, 0], tw[on, 1], tw[on, 2])
        targets_off.set_data_3d(tw[~on, 0], tw[~on, 1], tw[~on, 2])
        readout.set_text(f"t = {s['t'][i]:5.2f} s    worst foot error {s['err_mm'][i].max():.2f} mm")
        for cur in cursors:
            cur.set_xdata([s["t"][i]] * 2)
        return []

    FuncAnimation(fig, draw, frames=T, blit=False).save(path, writer=PillowWriter(fps=s["fps"]), dpi=dpi)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/ik_leg.npz")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-gif", action="store_true", help="compute the metrics only")
    args = ap.parse_args()
    robot = body.Robot(IKNet.load(args.weights))
    reference = body.Robot(body.AnalyticSolver(), robot.lengths)

    demos = {
        "posture_demo": lambda: posture_script(robot),
        "trot_forward": lambda: gait_script(robot, gaits.TROT, (0.15, 0.0), 0.0, 4, 30, "Trot, straight ahead at 0.15 m/s"),
        "trot_turn": lambda: gait_script(robot, gaits.TROT, (0.0, 0.0), 1.2, 6, 30, "Trot, turning on the spot at 1.2 rad/s"),
        "crawl_forward": lambda: gait_script(robot, gaits.CRAWL, (0.04, 0.0), 0.0, 2, 20, "Crawl at 0.04 m/s, body leaning away from the lifted foot"),
    }
    os.makedirs("results", exist_ok=True)
    metrics_path = "results/body_metrics.json"
    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    for name, build in demos.items():
        if args.only and name not in args.only:
            continue
        s = build()
        metrics[name] = measure(robot, reference, s)
        if not args.no_gif:
            animate(robot, s, f"results/{name}.gif")
        print(name, json.dumps(metrics[name]), flush=True)
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=1)


if __name__ == "__main__":
    main()
