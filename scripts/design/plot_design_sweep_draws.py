"""Means, spreads and plots for the design sweep with draws. Reads results/design/design_sweep_draws.json, writes
design_sweep_draws_summary.json and design_sweep_draws.png next to it, and prints the table as Markdown.

Per (design, mass, gait, speed) cell, over its 10 draws: how many walks were valid (every commanded foot target inside the joint
limits, body upright, measured speed positive) and the mean and sample standard deviation (n - 1) of the speed ratio over all 10 runs,
valid or not. The spread is from the random mass and latency draws only. The plot shows the mean speed ratio at 0.10 m/s with the
valid count (k/10) under it.

    python scripts/design/plot_design_sweep_draws.py
"""
import json
import os
import statistics

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


def mean_sd(values):
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def key(design):
    return (round(design["upper_length_m"], 3), round(design["lower_length_m"], 3),
            round(design["body_length_m"], 3), round(design["body_width_m"], 3))


def main():
    with open(os.path.join(DIRECTORY, "design_sweep_draws.json")) as f:
        data = json.load(f)
    speeds, scales, draws = data["speeds_m_s"], data["mass_scales"], data["draws"]
    groups = {}
    for c in data["cells"]:
        groups.setdefault((key(c["design"]), c["mass_scale"]), []).append(c)

    rows = []
    for (design, scale), runs in groups.items():
        row = {"upper_mm": 1000 * design[0], "lower_mm": 1000 * design[1], "body_mm": [1000 * design[2], 1000 * design[3]],
               "mass_scale": scale, "n_draws": len(runs),
               "total_mass_kg_mean": statistics.fmean(r["stand"]["total_mass_kg"] for r in runs),
               "stood": sum(r["stand"]["stood"] for r in runs),
               "stand_peak_over_stall_mean": statistics.fmean(r["stand"]["run_peak_fraction_of_limit"] for r in runs),
               "static_bound_kg": runs[0]["stand"]["static_bound_kg"], "walks": {}}
        for gait in ("trot", "crawl"):
            for speed in speeds:
                walks = [w for r in runs for w in r["walk"] if w["gait"] == gait and abs(w["speed_m_s"] - speed) < 1e-9]
                mean, sd = mean_sd([w["velocity_ratio"] for w in walks])
                row["walks"][f"{gait} {speed:g}"] = {"valid": sum(valid(w) for w in walks), "ratio_mean": mean, "ratio_sd": sd}
        rows.append(row)
    with open(os.path.join(DIRECTORY, "design_sweep_draws_summary.json"), "w") as f:
        json.dump({"draws": draws, "rows": rows}, f, indent=1)

    def find(upper, lower, scale, body=(0.30, 0.15)):
        return next(r for r in rows if abs(r["upper_mm"] - 1000 * upper) < 1e-6 and abs(r["lower_mm"] - 1000 * lower) < 1e-6
                    and abs(r["body_mm"][0] - 1000 * body[0]) < 1e-6 and r["mass_scale"] == scale)

    fig, axes = plt.subplots(2, len(scales), figsize=(3.3 * len(scales), 6.2), squeeze=False)
    for row_i, gait in enumerate(("trot", "crawl")):
        for col, scale in enumerate(scales):
            ax = axes[row_i][col]
            grid = np.zeros((3, 3))
            for i, upper in enumerate(LENGTHS):
                for j, lower in enumerate(LENGTHS):
                    cell = find(upper, lower, scale)["walks"][f"{gait} {PLOT_SPEED:g}"]
                    grid[i, j] = cell["ratio_mean"]
                    ax.text(j, i, f"{cell['ratio_mean']:.2f}\n{cell['valid']}/{draws}", ha="center", va="center", fontsize=9,
                            color="white" if cell["ratio_mean"] > 0.6 else (INK if cell["valid"] == draws else ORANGE))
            ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(3), [f"{1000 * v:.0f}" for v in LENGTHS], color=INK)
            ax.set_yticks(range(3), [f"{1000 * v:.0f}" for v in LENGTHS], color=INK)
            ax.set_xlabel("lower link [mm]", color=INK, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{gait}\nupper link [mm]", color=INK, fontsize=9)
            if row_i == 0:
                ax.set_title(f"{find(0.15, 0.15, scale)['total_mass_kg_mean']:.2f} kg", fontsize=10, color=INK)
    fig.suptitle(f"Mean speed over commanded at {PLOT_SPEED} m/s; valid walks of {draws} draws below (orange: not all valid)",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(DIRECTORY, "design_sweep_draws.png"), dpi=130)

    header = ["design", "mass kg", "stood", "stand peak/stall"] + [f"{g} {s:g}" for g in ("trot", "crawl") for s in speeds]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for r in rows:
        label = f"legs {r['upper_mm']:.0f}/{r['lower_mm']:.0f}, body {r['body_mm'][0]:.0f}x{r['body_mm'][1]:.0f}"
        cols = [f"{r['walks'][k]['ratio_mean']:.2f} ± {r['walks'][k]['ratio_sd']:.2f} ({r['walks'][k]['valid']}/{draws})"
                for k in header[4:]]
        print(f"| {label} | {r['total_mass_kg_mean']:.2f} | {r['stood']}/{draws} | {r['stand_peak_over_stall_mean']:.2f} | "
              + " | ".join(cols) + " |")
    totals = {}
    for r in rows:
        for k, cell in r["walks"].items():
            totals.setdefault((k.split()[0], round(r["total_mass_kg_mean"], 1)), [0, 0])
            totals[(k.split()[0], round(r["total_mass_kg_mean"], 1))][0] += cell["valid"]
            totals[(k.split()[0], round(r["total_mass_kg_mean"], 1))][1] += draws
    print({f"{g} {m} kg": f"{v[0]} of {v[1]}" for (g, m), v in totals.items()})


if __name__ == "__main__":
    main()
