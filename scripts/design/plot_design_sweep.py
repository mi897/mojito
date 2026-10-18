"""Plots and tables for the design sweep. Reads results/design/design_sweep.json, writes design_sweep.png and
design_sweep_summary.json next to it, and prints the tables as Markdown.

A walk is "valid" when every commanded foot target stayed inside the joint limits, the body stayed upright
(tilt under 30 degrees, never below half the stance height) and the measured forward speed was positive.
The plot shows, for the leg designs at the placeholder body, the measured speed over the commanded speed at 0.10 m/s;
a cross marks a walk that is not valid. A valid walk can still be slow: read the number.

    python scripts/design/plot_design_sweep.py
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1fa6a0"
INK, GREY = "#333333", "#9a9a9a"
DIRECTORY = "results/design"
PLOT_SPEED = 0.10
LENGTHS = (0.10, 0.15, 0.20)


def valid(walk):
    return bool(walk["walked"] and walk["within_joint_limits"] and walk["velocity_ratio"] > 0.0)


def main():
    with open(os.path.join(DIRECTORY, "design_sweep.json")) as f:
        data = json.load(f)
    cells = data["cells"]
    scales = data["mass_scales"]

    def find(upper, lower, scale, body_length=0.30):
        for c in cells:
            d = c["design"]
            if (abs(d["upper_length_m"] - upper) < 1e-9 and abs(d["lower_length_m"] - lower) < 1e-9
                    and abs(d["body_length_m"] - body_length) < 1e-9 and c["mass_scale"] == scale):
                return c
        raise KeyError((upper, lower, scale, body_length))

    summary = {"valid_walks": {}, "stood": {}}
    for scale in scales:
        mass = find(0.15, 0.15, scale)["stand"]["total_mass_kg"]
        group = [c for c in cells if c["mass_scale"] == scale]
        summary["stood"][f"{mass:.2f} kg"] = f"{sum(c['stand']['stood'] for c in group)} of {len(group)}"
        for gait in ("trot", "crawl"):
            walks = [w for c in group for w in c["walk"] if w["gait"] == gait]
            summary["valid_walks"][f"{gait} {mass:.2f} kg"] = f"{sum(valid(w) for w in walks)} of {len(walks)}"
    with open(os.path.join(DIRECTORY, "design_sweep_summary.json"), "w") as f:
        json.dump(summary, f, indent=1)

    fig, axes = plt.subplots(2, len(scales), figsize=(3.3 * len(scales), 6.2), squeeze=False)
    for row, gait in enumerate(("trot", "crawl")):
        for col, scale in enumerate(scales):
            ax = axes[row][col]
            grid = np.full((3, 3), np.nan)
            marks = {}
            for i, upper in enumerate(LENGTHS):
                for j, lower in enumerate(LENGTHS):
                    walk = next(w for w in find(upper, lower, scale)["walk"]
                                if w["gait"] == gait and abs(w["speed_m_s"] - PLOT_SPEED) < 1e-9)
                    grid[i, j] = walk["velocity_ratio"]
                    marks[(i, j)] = valid(walk)
            ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0)
            for (i, j), ok in marks.items():
                ax.text(j, i, f"{grid[i, j]:.2f}" + ("" if ok else "\n×"), ha="center", va="center", fontsize=9,
                        color="white" if grid[i, j] > 0.6 else (INK if ok else ORANGE))
            ax.set_xticks(range(3), [f"{1000 * v:.0f}" for v in LENGTHS], color=INK)
            ax.set_yticks(range(3), [f"{1000 * v:.0f}" for v in LENGTHS], color=INK)
            ax.set_xlabel("lower link [mm]", color=INK, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{gait}\nupper link [mm]", color=INK, fontsize=9)
            if row == 0:
                ax.set_title(f"{find(0.15, 0.15, scale)['stand']['total_mass_kg']:.2f} kg", fontsize=10, color=INK)
    fig.suptitle(f"Measured speed over commanded ({PLOT_SPEED} m/s), × = not a valid walk", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(DIRECTORY, "design_sweep.png"), dpi=130)

    print("| design | mass kg | stood | peak torque / stall (stand) | static bound kg | trot valid (0.05, 0.10, 0.15) | crawl valid |")
    print("|---|---|---|---|---|---|---|")
    for c in cells:
        d, s = c["design"], c["stand"]
        label = (f"legs {1000 * d['upper_length_m']:.0f}/{1000 * d['lower_length_m']:.0f}, "
                 f"body {1000 * d['body_length_m']:.0f}x{1000 * d['body_width_m']:.0f}")
        cols = [", ".join(f"{w['velocity_ratio']:.2f}" + ("" if valid(w) else "×") for w in c["walk"] if w["gait"] == gait)
                for gait in ("trot", "crawl")]
        print(f"| {label} | {s['total_mass_kg']:.2f} | {'yes' if s['stood'] else 'NO'} | {s['run_peak_fraction_of_limit']:.2f} | "
              f"{s['static_bound_kg']:.1f} | {cols[0]} | {cols[1]} |")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
