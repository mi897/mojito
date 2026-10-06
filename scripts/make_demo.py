"""Build the two demos from trained weights.

    results/step_demo.gif  three different-sized legs tracing the same step
    docs/demo.html         interactive page: drag the target, change the lengths

    python scripts/make_demo.py [--weights weights/ik_leg.npz]
"""
import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import IKNet, leg  # noqa: E402

BLUE, ORANGE, INK2, MUTED, SURFACE, GRID = "#2a78d6", "#eb6834", "#52514e", "#898781", "#fcfcfb", "#e6e5e0"


def step_path(n, la):
    """A walking step in the side plane: flat stance, raised swing."""
    u = np.linspace(0, 2 * np.pi, n, endpoint=False)
    z = -0.17 + np.where(np.sin(u) > 0, 0.05 * np.sin(u), 0.0)
    return np.stack([0.07 * np.cos(u), np.full(n, la + 0.01), z], axis=1)


def make_gif(net, path, frames=60):
    cfg = net.cfg
    scales = [-cfg.length_tolerance, 0.0, cfg.length_tolerance]
    target = step_path(frames, cfg.nominal[0])
    fig, axes = plt.subplots(1, 3, figsize=(9, 3.6), facecolor=SURFACE)
    artists = []
    for ax, s in zip(axes, scales):
        L = cfg.nominal * np.array([1.0, 1 + s, 1 + s])
        q = net.predict(target, L)
        pts = leg.joint_positions(q, L) * 1000
        err = np.linalg.norm(pts[:, 3] / 1000 - target, axis=1) * 1000
        ax.set_facecolor(SURFACE)
        ax.plot(target[:, 0] * 1000, target[:, 2] * 1000, color=ORANGE, lw=1.5, ls=(0, (4, 3)))
        (line,) = ax.plot([], [], color=BLUE, lw=4, marker="o", ms=6, solid_capstyle="round")
        (dot,) = ax.plot([], [], "o", color=ORANGE, ms=9, mfc="none", mew=2)
        ax.set_xlim(-170, 170); ax.set_ylim(-270, 40); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.set_title(f"links {s:+.0%}  ({L[1]*1000:.0f} / {L[2]*1000:.0f} mm)", fontsize=10, color=INK2, loc="left")
        ax.text(0.03, 0.04, f"worst error on this path {err.max():.2f} mm", transform=ax.transAxes, fontsize=9, color=INK2)
        artists.append((line, dot, pts))
    fig.suptitle("One network, three leg sizes, same foot path (side view; dashed = target)", fontsize=11, x=0.02, ha="left")
    fig.tight_layout()

    def draw(i):
        out = []
        for line, dot, pts in artists:
            line.set_data(pts[i, :, 0], pts[i, :, 2])
            dot.set_data([target[i, 0] * 1000], [target[i, 2] * 1000])
            out += [line, dot]
        return out

    FuncAnimation(fig, draw, frames=frames, blit=True).save(path, writer=PillowWriter(fps=24), dpi=90)
    plt.close(fig)


def make_html(net, template, path):
    with open(template) as f:
        html = f.read()
    with open(path, "w") as f:
        f.write(html.replace("/*WEIGHTS*/", net.to_json(digits=5)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/ik_leg.npz")
    args = ap.parse_args()
    net = IKNet.load(args.weights)
    os.makedirs("results", exist_ok=True)
    make_gif(net, "results/step_demo.gif")
    make_html(net, "docs/demo_template.html", "docs/demo.html")
    print("wrote results/step_demo.gif and docs/demo.html", os.path.getsize("docs/demo.html") // 1024, "KB")


if __name__ == "__main__":
    main()
