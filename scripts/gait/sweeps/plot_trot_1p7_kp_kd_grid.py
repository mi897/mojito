"""Means, standard deviations, best kd per kp, and plots for the kp x kd grid. Reads results/gait/sweeps/servo_gains/trot_1p7_kp_kd_grid.json,
writes trot_1p7_kp_kd_grid_summary.json, trot_1p7_kp_kd_lines.png and trot_1p7_kp_kd_maps.png next to it, and prints the tables.

Per cell (10 runs): mean and sample standard deviation (n - 1) of the speed ratio over all 10 runs (a run that fell counts with the
speed it made); mean tilt and mean peak torque over the runs that stayed upright (tilt under 30 degrees). Per kp: the kd with the highest
mean speed ratio, the range of kd whose mean is within 0.01 of that peak (the plateau), and the first kd above the peak whose mean falls
more than 0.05 below it (the cliff). The spread is from the random mass and latency draws only.

    uv run --extra sim python scripts/gait/sweeps/plot_trot_1p7_kp_kd_grid.py
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from mojito.palette import INK  # noqa: E402
from mojito.results import results_path, write_result  # noqa: E402



def mean_sd(xs):
    return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (None, None)


def summarise(data):
    cells = {}
    for kp in data["kp"]:
        for kd in data["kd"]:
            rs = [r for r in data["runs"] if r["kp"] == kp and r["kd"] == kd]
            up = [r for r in rs if r["max_tilt_deg"] < 30.0]
            rm, rsd = mean_sd([r["velocity_ratio"] for r in rs])
            tm, tsd = mean_sd([r["max_tilt_deg"] for r in up])
            qm, _ = mean_sd([r["peak_torque_fraction_of_limit"] for r in up])
            cells[(kp, kd)] = {"kp": kp, "kd": kd, "n": len(rs), "walked": sum(r["walked"] for r in rs), "upright": len(up),
                               "ratio_mean": rm, "ratio_sd": rsd, "tilt_mean_deg": tm, "tilt_sd_deg": tsd, "torque_mean": qm,
                               "saturated_mean": statistics.fmean(r["fraction_of_walking_saturated"] for r in rs)}
    best = []
    for kp in data["kp"]:
        row = [cells[(kp, kd)] for kd in data["kd"]]
        peak = max(row, key=lambda c: c["ratio_mean"])
        plateau = [c["kd"] for c in row if c["ratio_mean"] >= peak["ratio_mean"] - 0.01]
        cliff = next((c["kd"] for c in row if c["kd"] > peak["kd"] and c["ratio_mean"] < peak["ratio_mean"] - 0.05), None)
        low = next((c for c in row if c["ratio_mean"] >= peak["ratio_mean"] - 0.01), None)
        best.append({"kp": kp, "best_kd": peak["kd"], "ratio_mean": peak["ratio_mean"], "ratio_sd": peak["ratio_sd"],
                     "tilt_mean_deg": peak["tilt_mean_deg"], "torque_mean": peak["torque_mean"],
                     "plateau_kd_min": min(plateau), "plateau_kd_max": max(plateau), "cliff_kd": cliff,
                     "ratio_at_kd_0": cells[(kp, 0.0)]["ratio_mean"] if (kp, 0.0) in cells else None,
                     "ratio_at_lowest_plateau_edge": low["ratio_mean"]})
    return cells, best


def main():
    root = results_path("gait", "sweeps", "servo_gains")
    data = json.load(open(os.path.join(root, "trot_1p7_kp_kd_grid.json")))
    cells, best = summarise(data)
    kps, kds = data["kp"], data["kd"]
    write_result(os.path.join(root, "trot_1p7_kp_kd_grid_summary.json"), {"what": "mean and sample standard deviation per (kp, kd) cell and the best kd per kp, from trot_1p7_kp_kd_grid.json",
               "iterations": data["iterations"], "cells": list(cells.values()), "best_per_kp": best})

    # lines: speed ratio against kd, one line per kp, and the tilt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.2))
    colours = plt.cm.viridis(np.linspace(0, 0.92, len(kps)))
    for kp, col in zip(kps, colours):
        row = [cells[(kp, kd)] for kd in kds]
        axes[0].errorbar(kds, [c["ratio_mean"] for c in row], yerr=[c["ratio_sd"] for c in row], color=col, lw=1.4, marker="o", ms=3, capsize=2, label=f"kp {kp:g}")
        pts = [(c["kd"], c["tilt_mean_deg"]) for c in row if c["tilt_mean_deg"] is not None]
        axes[1].plot([p[0] for p in pts], [p[1] for p in pts], color=col, lw=1.4, marker="o", ms=3)
    axes[0].set_ylabel("speed ratio, measured / commanded (mean ± 1 sd, 10 runs)", color=INK, fontsize=9)
    axes[1].set_ylabel("largest body tilt [deg] (mean of upright runs)", color=INK, fontsize=9)
    for ax in axes:
        ax.set_xlabel("servo damping kd [N·m·s/rad]", color=INK, fontsize=9)
        ax.grid(color="#dddddd", lw=0.6)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes[0].axhline(1.0, color=INK, ls="--", lw=0.8)
    axes[0].legend(ncol=2, fontsize=7, frameon=False, title="servo gain", title_fontsize=8)
    fig.suptitle("1.7 s trot: kd swept at each kp (kp 20 to 150), everything else at the baseline", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "trot_1p7_kp_kd_lines.png"), dpi=130)

    # maps: kp rows x kd columns
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4))
    for ax, key, title, cmap in ((axes[0], "ratio_mean", "mean speed ratio", "viridis"), (axes[1], "tilt_mean_deg", "mean largest tilt [deg] (upright runs)", "magma_r")):
        grid = np.array([[np.nan if cells[(kp, kd)][key] is None else cells[(kp, kd)][key] for kd in kds] for kp in kps])
        im = ax.imshow(grid, origin="lower", aspect="auto", cmap=cmap, vmin=(0 if key == "ratio_mean" else None), vmax=(0.9 if key == "ratio_mean" else None))
        ax.set_xticks(range(len(kds)))
        ax.set_xticklabels([f"{k:g}" for k in kds], rotation=60, fontsize=7)
        ax.set_yticks(range(len(kps)))
        ax.set_yticklabels([f"{k:g}" for k in kps], fontsize=7)
        ax.set_xlabel("kd [N·m·s/rad] (not evenly spaced)", color=INK, fontsize=9)
        ax.set_ylabel("kp [N·m/rad]", color=INK, fontsize=9)
        ax.set_title(title, fontsize=9, color=INK)
        fig.colorbar(im, ax=ax, shrink=0.85)
        for i, kp in enumerate(kps):
            j = kds.index(next(b for b in best if b["kp"] == kp)["best_kd"])
            ax.plot(j, i, marker="*", color="white", ms=7, mec=INK, mew=0.5)
    fig.suptitle("1.7 s trot, kp x kd (star: kd with the highest mean speed ratio at that kp)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "trot_1p7_kp_kd_maps.png"), dpi=130)

    f = lambda x, d: "-" if x is None else f"{x:.{d}f}"
    print("| kp | best kd | speed ratio mean | sd | tilt (deg) | torque / stall | kd within 0.01 of the peak | first kd 0.05 below it (cliff) |\n|---|---|---|---|---|---|---|---|")
    for b in best:
        print(f"| {b['kp']:g} | {b['best_kd']:g} | {f(b['ratio_mean'], 3)} | {f(b['ratio_sd'], 3)} | {f(b['tilt_mean_deg'], 1)} | {f(b['torque_mean'], 2)} | "
              f"{b['plateau_kd_min']:g} to {b['plateau_kd_max']:g} | {'-' if b['cliff_kd'] is None else format(b['cliff_kd'], 'g')} |")
    print("\nmean speed ratio (rows kp, columns kd)\n| kp \\ kd | " + " | ".join(f"{k:g}" for k in kds) + " |\n|---|" + "---|" * len(kds))
    for kp in kps:
        print(f"| {kp:g} | " + " | ".join(f(cells[(kp, kd)]["ratio_mean"], 2) for kd in kds) + " |")
    print("\nsd of speed ratio\n| kp \\ kd | " + " | ".join(f"{k:g}" for k in kds) + " |\n|---|" + "---|" * len(kds))
    for kp in kps:
        print(f"| {kp:g} | " + " | ".join(f(cells[(kp, kd)]["ratio_sd"], 3) for kd in kds) + " |")
    print("\nmean tilt (deg, upright runs)\n| kp \\ kd | " + " | ".join(f"{k:g}" for k in kds) + " |\n|---|" + "---|" * len(kds))
    for kp in kps:
        print(f"| {kp:g} | " + " | ".join(f(cells[(kp, kd)]["tilt_mean_deg"], 1) for kd in kds) + " |")


if __name__ == "__main__":
    main()
