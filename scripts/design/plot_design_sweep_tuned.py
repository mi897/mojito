"""Tuned against untuned gaits, over 10 draws. Reads results/design/design_sweep_tuned.json, design_sweep_draws_summary.json and
design_tuning.json; writes design_sweep_tuned_summary.json and design_sweep_tuned.png next to them and prints the tables as Markdown.

Per (design, mass, gait, speed) cell, over its 10 draws: valid walks (every commanded foot target inside the joint limits, body
upright, measured speed positive) and the mean and sample standard deviation (n - 1) of the speed ratio over all 10 runs. The
untuned figures are the same cells from design_sweep_draws_summary.json. The plot shows the tuned mean speed ratio at 0.10 m/s
with the valid count (k/10) under it.

    uv run --extra sim python scripts/design/plot_design_sweep_tuned.py
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mojito.palette import INK, ORANGE  # noqa: E402
from mojito.results import results_path, write_result  # noqa: E402

DIRECTORY = results_path("design")
PLOT_SPEED = 0.10
LENGTHS = (0.10, 0.15, 0.20)


def valid(walk):
    return bool(walk["walked"] and walk["within_joint_limits"] and walk["velocity_ratio"] > 0.0)


def mean_sd(values):
    return statistics.fmean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def load(name):
    with open(os.path.join(DIRECTORY, name)) as f:
        return json.load(f)


def main():
    tuned, untuned, tuning = load("design_sweep_tuned.json"), load("design_sweep_draws_summary.json"), load("design_tuning.json")
    draws, speeds = tuned["draws"], tuned["speeds_m_s"]
    scales = sorted({r["mass_scale"] for r in tuned["runs"]})
    cells = {}
    for r in tuned["runs"]:
        d = r["design"]
        k = (round(1000 * d["upper_length"]), round(1000 * d["lower_length"]), round(1000 * d["body_length"]), r["mass_scale"],
             r["gait"], r["speed_m_s"])
        cells.setdefault(k, []).append(r)
    old = {}
    for row in untuned["rows"]:
        for key, cell in row["walks"].items():
            gait, speed = key.split()
            old[(round(row["upper_mm"]), round(row["lower_mm"]), round(row["body_mm"][0]), row["mass_scale"], gait, float(speed))] = cell
    best = {}
    for e in tuning["entries"]:
        d = e["design"]
        best[(round(1000 * d["upper_length"]), round(1000 * d["lower_length"]), round(1000 * d["body_length"]), e["mass_scale"], e["gait"])] = e

    rows = []
    for k, runs in sorted(cells.items()):
        mean, sd = mean_sd([r["velocity_ratio"] for r in runs])
        o = old[k]
        rows.append({"upper_mm": k[0], "lower_mm": k[1], "body_length_mm": k[2], "mass_scale": k[3], "gait": k[4], "speed_m_s": k[5],
                     "tuned_valid": sum(valid(r) for r in runs), "tuned_ratio_mean": mean, "tuned_ratio_sd": sd,
                     "untuned_valid": o["valid"], "untuned_ratio_mean": o["ratio_mean"], "untuned_ratio_sd": o["ratio_sd"],
                     "peak_torque_mean": statistics.fmean(r["peak_torque_fraction_of_limit"] for r in runs)})
    totals = {}
    for r in rows:
        key = f"{r['gait']} mass scale {r['mass_scale']}"
        t = totals.setdefault(key, {"tuned_valid": 0, "untuned_valid": 0, "walks": 0})
        t["tuned_valid"] += r["tuned_valid"]
        t["untuned_valid"] += r["untuned_valid"]
        t["walks"] += draws
    write_result(os.path.join(DIRECTORY, "design_sweep_tuned_summary.json"), {"draws": draws, "totals": totals, "rows": rows})

    fig, axes = plt.subplots(2, len(scales), figsize=(3.3 * len(scales), 6.2), squeeze=False)
    for row_i, gait in enumerate(("trot", "crawl")):
        for col, scale in enumerate(scales):
            ax = axes[row_i][col]
            grid = np.zeros((3, 3))
            for i, upper in enumerate(LENGTHS):
                for j, lower in enumerate(LENGTHS):
                    r = next(x for x in rows if x["upper_mm"] == round(1000 * upper) and x["lower_mm"] == round(1000 * lower)
                             and x["body_length_mm"] == 300 and x["mass_scale"] == scale and x["gait"] == gait
                             and abs(x["speed_m_s"] - PLOT_SPEED) < 1e-9)
                    grid[i, j] = r["tuned_ratio_mean"]
                    ax.text(j, i, f"{r['tuned_ratio_mean']:.2f}\n{r['tuned_valid']}/{draws}", ha="center", va="center", fontsize=9,
                            color="white" if r["tuned_ratio_mean"] > 0.6 else (INK if r["tuned_valid"] == draws else ORANGE))
            ax.imshow(grid, cmap="Blues", vmin=0.0, vmax=1.0)
            ax.set_xticks(range(3), [f"{1000 * v:.0f}" for v in LENGTHS], color=INK)
            ax.set_yticks(range(3), [f"{1000 * v:.0f}" for v in LENGTHS], color=INK)
            ax.set_xlabel("lower link [mm]", color=INK, fontsize=9)
            if col == 0:
                ax.set_ylabel(f"{gait}\nupper link [mm]", color=INK, fontsize=9)
            if row_i == 0:
                ax.set_title(f"mass scale {scale:g}", fontsize=10, color=INK)
    fig.suptitle(f"Tuned gaits: mean speed over commanded at {PLOT_SPEED} m/s; valid walks of {draws} draws below (orange: not all valid)",
                 color=INK, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(DIRECTORY, "design_sweep_tuned.png"), dpi=130)

    print("| design | mass scale | gait | tuned knee / lift mm / kp, kd | " + " | ".join(
        f"{s:g} m/s tuned (untuned)" for s in speeds) + " |")
    print("|" + "---|" * (4 + len(speeds)))
    for e in tuning["entries"]:
        d, b = e["design"], e["best"]
        key = (round(1000 * d["upper_length"]), round(1000 * d["lower_length"]), round(1000 * d["body_length"]), e["mass_scale"], e["gait"])
        cols = []
        for s in speeds:
            r = next(x for x in rows if (x["upper_mm"], x["lower_mm"], x["body_length_mm"], x["mass_scale"], x["gait"]) == key
                     and abs(x["speed_m_s"] - s) < 1e-9)
            cols.append(f"{r['tuned_ratio_mean']:.2f}±{r['tuned_ratio_sd']:.2f} ({r['tuned_valid']}/{draws}) "
                        f"({r['untuned_ratio_mean']:.2f} ({r['untuned_valid']}/{draws}))")
        print(f"| {key[0]}/{key[1]}, body {key[2]} | {key[3]:g} | {key[4]} | {b['knee']:.2f} / {1000 * b['lift']:.0f} / "
              f"{b['kp']:.0f}, {b['kd']:g} | " + " | ".join(cols) + " |")
    print(json.dumps(totals, indent=1))


if __name__ == "__main__":
    main()
