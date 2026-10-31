"""Means, standard deviations, paired differences and plots for the combined-values test. Reads
results/gait/sweeps/combined/trot_1p7_combined.json, writes trot_1p7_combined_summary.json and trot_1p7_combined.png next to it,
and prints the tables as Markdown.

Per (config, mass scale, speed) cell (10 runs): mean and sample standard deviation (n - 1) of the speed ratio over all 10 runs, mean tilt and
peak torque over the runs that stayed upright (tilt under 30 degrees), and the counts of runs inside the joint limits and moving forward.
"Valid" below means: every commanded foot target inside the joint limits and a positive measured speed. Paired differences subtract the
other config's ratio for the same draw (same mass and latency draw), then take the mean and sample standard deviation over the 10 draws.
The spread is from the random mass and latency draws only.

    uv run --extra sim python scripts/gait/sweeps/plot_trot_1p7_combined.py
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mojito.palette import AQUA, BLUE, GREY, INK, ORANGE  # noqa: E402
from mojito.results import results_path, write_result  # noqa: E402

PLOTTED = (("baseline", GREY, ""), ("best", BLUE, ""), ("best_stance_0.17", AQUA, ""),
           ("gains_only", ORANGE, "//"), ("swing_only", ORANGE, ".."), ("stance_only", ORANGE, "xx"))


def mean_sd(xs):
    return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (None, None)


def main():
    root = results_path("gait", "sweeps", "combined")
    data = json.load(open(os.path.join(root, "trot_1p7_combined.json")))
    runs, configs = data["runs"], list(data["configs"])
    speeds, scales = data["speeds_m_s"], data["mass_scale"]
    by = {}
    for r in runs:
        by.setdefault((r["config"], r["mass_scale"], r["speed_m_s"]), {})[r["iteration"]] = r
    cells = []
    for (c, m, s), d in by.items():
        rs = [d[i] for i in sorted(d)]
        up = [r for r in rs if r["max_tilt_deg"] < 30.0]
        rm, rsd = mean_sd([r["velocity_ratio"] for r in rs])
        tm, tsd = mean_sd([r["max_tilt_deg"] for r in up])
        qm, _ = mean_sd([r["peak_torque_fraction_of_limit"] for r in up])
        cells.append({"config": c, "mass_scale": m, "speed_m_s": s, "total_mass_kg": statistics.fmean(r["total_mass_kg"] for r in rs), "n": len(rs),
                      "walked": sum(r["walked"] for r in rs), "in_limits": sum(r["within_joint_limits"] for r in rs),
                      "forward": sum(r["velocity_ratio"] > 0 for r in rs), "ratio_mean": rm, "ratio_sd": rsd,
                      "tilt_mean_deg": tm, "tilt_sd_deg": tsd, "torque_mean": qm,
                      "saturated_mean": statistics.fmean(r["fraction_of_walking_saturated"] for r in rs)})

    def paired(a, b, m, s):
        da, db = by[(a, m, s)], by[(b, m, s)]
        return mean_sd([da[i]["velocity_ratio"] - db[i]["velocity_ratio"] for i in da])

    diffs = []
    for m in scales:
        for s in speeds:
            row = {"mass_scale": m, "speed_m_s": s}
            row["best_minus_baseline"] = paired("best", "baseline", m, s)
            for g in ("gains", "swing", "stance"):
                row[f"{g}_only_minus_baseline"] = paired(f"{g}_only", "baseline", m, s)
                row[f"best_minus_best_baseline_{g}"] = paired("best", f"best_baseline_{g}", m, s)
            row["sum_of_single_effects"] = sum(row[f"{g}_only_minus_baseline"][0] for g in ("gains", "swing", "stance"))
            diffs.append(row)
    write_result(os.path.join(root, "trot_1p7_combined_summary.json"), {"what": "mean and sample standard deviation per (config, mass, speed) cell, and paired differences, from trot_1p7_combined.json",
               "iterations": data["iterations"], "cells": cells, "paired_differences": diffs})

    cell = {(c["config"], c["mass_scale"], c["speed_m_s"]): c for c in cells}
    fig, axes = plt.subplots(2, len(scales), figsize=(15, 7.5))
    width = 0.8 / len(PLOTTED)
    for col, m in enumerate(scales):
        for row, (key, sdkey, label) in enumerate((("ratio_mean", "ratio_sd", "speed ratio (mean ± 1 sd)"), ("tilt_mean_deg", "tilt_sd_deg", "largest tilt [deg] (upright runs)"))):
            ax = axes[row, col]
            for k, (name, colour, hatch) in enumerate(PLOTTED):
                xs = np.arange(len(speeds)) + (k - (len(PLOTTED) - 1) / 2) * width
                ys = [cell[(name, m, s)][key] if cell[(name, m, s)][key] is not None else np.nan for s in speeds]
                es = [cell[(name, m, s)][sdkey] or 0.0 for s in speeds]
                ax.bar(xs, ys, width=width * 0.92, yerr=es, color=colour, hatch=hatch, edgecolor="white", linewidth=0.3, error_kw={"lw": 0.8, "capsize": 1.5},
                       label=name if (row == 0 and col == 0) else None)
            ax.set_xticks(range(len(speeds)))
            ax.set_xticklabels([f"{s:g}" for s in speeds], color=INK)
            ax.grid(axis="y", color="#dddddd", lw=0.6)
            ax.set_axisbelow(True)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(f"{cell[('baseline', m, speeds[0])]['total_mass_kg']:.2f} kg (mass scale {m:g})", fontsize=9, color=INK)
                ax.axhline(1.0, color=INK, ls="--", lw=0.8)
            else:
                ax.set_xlabel("commanded speed [m/s]", color=INK, fontsize=9)
            ax.set_ylabel(label if col == 0 else "", color=INK, fontsize=9)
    axes[0, 0].legend(fontsize=7, frameon=False, ncol=2, loc="lower left")
    fig.suptitle("1.7 s trot: do the best values add up? (10 runs per bar; baseline kp 30, kd 0.5, 30 mm, 0.19 m; best kp 50, kd 0.9, 22 mm, 0.16 m)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "trot_1p7_combined.png"), dpi=130)

    f = lambda x, d: "-" if x is None else f"{x:.{d}f}"
    for m in scales:
        print(f"\nmass {cell[('baseline', m, speeds[0])]['total_mass_kg']:.2f} kg: mean speed ratio ± sd (valid runs of 10: in limits and moving forward) \n| config | " + " | ".join(f"{s:g} m/s" for s in speeds) + " |\n|---|" + "---|" * len(speeds))
        for c in configs:
            print(f"| {c} | " + " | ".join(f"{f(cell[(c, m, s)]['ratio_mean'], 3)} ± {f(cell[(c, m, s)]['ratio_sd'], 3)} ({min(cell[(c, m, s)]['in_limits'], cell[(c, m, s)]['forward'])}/10)" for s in speeds) + " |")
    for m in scales:
        print(f"\nmass {cell[('baseline', m, speeds[0])]['total_mass_kg']:.2f} kg: mean tilt (deg, upright runs) and peak torque / stall\n| config | " + " | ".join(f"{s:g} m/s" for s in speeds) + " |\n|---|" + "---|" * len(speeds))
        for c in configs:
            print(f"| {c} | " + " | ".join(f"{f(cell[(c, m, s)]['tilt_mean_deg'], 1)} / {f(cell[(c, m, s)]['torque_mean'], 2)}" for s in speeds) + " |")
    print("\npaired differences in speed ratio (mean ± sd over the 10 draws)\n| mass kg | speed | best - baseline | sum of the three single-group effects | gains only | swing only | stance only | best - best with baseline gains | best - best with baseline swing | best - best with baseline stance |\n|---|---|---|---|---|---|---|---|---|---|")
    for d in diffs:
        g = lambda k: f"{f(d[k][0], 3)} ± {f(d[k][1], 3)}"
        print(f"| {cell[('baseline', d['mass_scale'], d['speed_m_s'])]['total_mass_kg']:.2f} | {d['speed_m_s']:g} | {g('best_minus_baseline')} | {f(d['sum_of_single_effects'], 3)} | "
              f"{g('gains_only_minus_baseline')} | {g('swing_only_minus_baseline')} | {g('stance_only_minus_baseline')} | "
              f"{g('best_minus_best_baseline_gains')} | {g('best_minus_best_baseline_swing')} | {g('best_minus_best_baseline_stance')} |")


if __name__ == "__main__":
    main()
