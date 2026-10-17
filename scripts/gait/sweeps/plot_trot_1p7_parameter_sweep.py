"""Plot what each parameter did to the 1.7 s trot. Reads results/gait/sweeps/parameters/trot_1p7_parameter_sweep.json, writes
results/gait/sweeps/parameters/trot_1p7_parameter_effects.png.

One column per parameter. Top row: measured speed / commanded speed for every run that walked (dots) and their median
(line). Bottom row: largest body tilt of those runs and their median. Runs that fell are not drawn; the number that fell
at each value is printed under the column title. A speed ratio of 1 is the commanded speed.

    python scripts/gait/sweeps/plot_trot_1p7_parameter_sweep.py
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1fa6a0"
INK = "#333333"
PARAMETERS = (("duty", "duty (fraction on ground)", "{:g}"), ("speed_m_s", "commanded speed [m/s]", "{:g}"),
              ("height_m", "stance height [m]", "{:g}"), ("kp", "servo gain kp [N·m/rad]", "{:g}"),
              ("lift_m", "swing height [m]", "{:g}"))


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "gait", "sweeps", "parameters")
    runs = json.load(open(os.path.join(root, "trot_1p7_parameter_sweep.json")))["runs"]
    fig, axes = plt.subplots(2, len(PARAMETERS), figsize=(15, 6.2), sharey="row")
    for col, (key, label, fmt) in enumerate(PARAMETERS):
        values = sorted({r[key] for r in runs})
        walked = {v: [r for r in runs if r[key] == v and r["walked"]] for v in values}
        fell = [sum(1 for r in runs if r[key] == v and not r["walked"]) for v in values]
        total = [sum(1 for r in runs if r[key] == v) for v in values]
        xs = range(len(values))
        for row, (field, colour) in enumerate((("velocity_ratio", BLUE), ("max_tilt_deg", ORANGE))):
            ax = axes[row, col]
            for x, v in zip(xs, values):
                ys = [r[field] for r in walked[v]]
                step = 0.5 / max(1, len(ys))
                ax.scatter([x - 0.25 + step * i for i in range(len(ys))], sorted(ys), s=9, color=colour, alpha=0.35, linewidths=0)
            ax.plot(xs, [statistics.median(r[field] for r in walked[v]) for v in values], color=colour, lw=2, marker="o", ms=5)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([fmt.format(v) for v in values], color=INK)
            ax.grid(axis="y", color="#dddddd", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(label + "\nfell: " + ", ".join(f"{n}/{t}" for n, t in zip(fell, total)), fontsize=9, color=INK)
                ax.axhline(1.0, color=INK, ls="--", lw=0.8)
            else:
                ax.set_xlabel(label, color=INK, fontsize=9)
    axes[0, 0].set_ylabel("measured / commanded speed (walked runs)", color=INK, fontsize=9)
    axes[1, 0].set_ylabel("largest body tilt [deg] (walked runs)", color=INK, fontsize=9)
    fig.suptitle("1.7 s trot: effect of each parameter (192 runs; dots are runs, line is the median over the other parameters)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "trot_1p7_parameter_effects.png"), dpi=130)


if __name__ == "__main__":
    main()
