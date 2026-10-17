"""Means, standard deviations and plots for the one-at-a-time sweep. Reads results/gait/sweeps/trot_1p7_one_at_a_time.json,
writes trot_1p7_one_at_a_time_summary.json and trot_1p7_one_at_a_time.png next to it, and prints the tables as Markdown.

Per value (10 runs): mean, sample standard deviation (n - 1), minimum and maximum of the speed ratio over all 10 runs
(a run that fell counts with the speed it made); mean and standard deviation of the largest tilt and of the peak torque over the
runs that stayed upright (tilt under 30 degrees; a fall is near 180 degrees and would swamp the mean). "walked" is the replay's own flag: upright, not too low, and
every commanded foot target inside the joint limits; "in limits" counts the runs for which that last condition held. The spread comes only from the random
mass and latency draws described in trot_1p7_one_at_a_time.py.

    python scripts/gait/sweeps/plot_trot_1p7_one_at_a_time.py
"""
import json
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1fa6a0"
INK = "#333333"
LABELS = {"kp": "servo gain kp [N·m/rad] (kd tied)", "kd": "servo damping kd [N·m·s/rad] (kp 30)",
          "lift": "swing height [mm]", "height": "stance height [m]"}
SCALE = {"kp": 1, "kd": 1, "lift": 1000, "height": 1}


def mean_sd(xs):
    return (statistics.fmean(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0) if xs else (None, None)


def summarise(runs, sweeps):
    out = {}
    for sweep, values in sweeps.items():
        rows = []
        for v in values:
            rs = [r for r in runs if r["sweep"] == sweep and r["value"] == v]
            w = [r for r in rs if r["max_tilt_deg"] < 30.0]  # stayed upright
            ratio = [r["velocity_ratio"] for r in rs]
            rm, rsd = mean_sd(ratio)
            tm, tsd = mean_sd([r["max_tilt_deg"] for r in w])
            qm, qsd = mean_sd([r["peak_torque_fraction_of_limit"] for r in w])
            rows.append({"value": v, "n": len(rs), "walked": sum(r["walked"] for r in rs), "upright": len(w),
                         "in_limits": sum(r["within_joint_limits"] for r in rs), "ratio_mean": rm, "ratio_sd": rsd, "ratio_min": min(ratio),
                         "ratio_max": max(ratio), "velocity_mean_m_s": statistics.fmean(r["measured_velocity_m_s"] for r in rs),
                         "tilt_mean_deg": tm, "tilt_sd_deg": tsd, "torque_mean": qm, "torque_sd": qsd})
        out[sweep] = rows
    return out


def table(sweep, rows):
    f = lambda x, d: "-" if x is None else f"{x:.{d}f}"
    lines = [f"| {LABELS[sweep]} | walked | in limits | speed ratio mean | sd | min | max | tilt mean (deg) | sd | torque / stall mean | sd |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        v = r["value"] * SCALE[sweep]
        lines.append(f"| {v:g} | {r['walked']}/{r['n']} | {r['in_limits']}/{r['n']} | {f(r['ratio_mean'], 3)} | {f(r['ratio_sd'], 3)} | {f(r['ratio_min'], 2)} | "
                     f"{f(r['ratio_max'], 2)} | {f(r['tilt_mean_deg'], 1)} | {f(r['tilt_sd_deg'], 1)} | {f(r['torque_mean'], 2)} | {f(r['torque_sd'], 2)} |")
    return "\n".join(lines)


def main():
    root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "results", "gait", "sweeps")
    data = json.load(open(os.path.join(root, "trot_1p7_one_at_a_time.json")))
    runs, sweeps = data["runs"], data["sweeps"]
    summary = summarise(runs, sweeps)
    with open(os.path.join(root, "trot_1p7_one_at_a_time_summary.json"), "w") as fh:
        json.dump({"what": "mean and sample standard deviation per value from trot_1p7_one_at_a_time.json", "iterations": data["iterations"],
                   "summary": summary}, fh, indent=1)
    fig, axes = plt.subplots(3, len(sweeps), figsize=(15, 8.5))
    for col, (sweep, rows) in enumerate(summary.items()):
        xs = [r["value"] * SCALE[sweep] for r in rows]
        base = data["baseline"][{"kp": "kp", "kd": "kd", "lift": "lift_m", "height": "height_m"}[sweep]] * SCALE[sweep]
        for row, (title, key, sdkey, colour) in enumerate((("speed ratio (all 10 runs)", "ratio_mean", "ratio_sd", BLUE),
                                                           ("largest tilt [deg] (upright runs)", "tilt_mean_deg", "tilt_sd_deg", ORANGE))):
            ax = axes[row, col]
            if row == 0:
                for x, v in zip(xs, sweeps[sweep]):
                    ys = [r["velocity_ratio"] for r in runs if r["sweep"] == sweep and r["value"] == v]
                    ax.scatter([x] * len(ys), ys, s=8, color=colour, alpha=0.3, linewidths=0)
            pts = [(x, r[key], r[sdkey]) for x, r in zip(xs, rows) if r[key] is not None]
            ax.errorbar([p[0] for p in pts], [p[1] for p in pts], yerr=[p[2] for p in pts], color=colour, lw=1.8, marker="o", ms=4, capsize=3)
            ax.axvline(base, color=INK, ls=":", lw=0.8)
            ax.set_ylabel(title if col == 0 else "", color=INK, fontsize=9)
            ax.grid(axis="y", color="#dddddd", lw=0.6)
            for side in ("top", "right"):
                ax.spines[side].set_visible(False)
            if row == 0:
                ax.set_title(LABELS[sweep], fontsize=9, color=INK)
                ax.axhline(1.0, color=INK, ls="--", lw=0.8)
        ax = axes[2, col]
        ax.bar(xs, [r["walked"] for r in rows], width=(xs[1] - xs[0]) * 0.7, color=AQUA)
        ax.set_ylim(0, data["iterations"])
        ax.set_xlabel(LABELS[sweep], color=INK, fontsize=9)
        ax.set_ylabel("runs that walked (of 10)" if col == 0 else "", color=INK, fontsize=9)
        ax.axvline(base, color=INK, ls=":", lw=0.8)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    fig.suptitle("1.7 s trot, one parameter at a time, 10 runs per value (mean ± 1 sd; dots are runs; dotted line is the baseline value)",
                 fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(os.path.join(root, "trot_1p7_one_at_a_time.png"), dpi=130)
    for sweep, rows in summary.items():
        print(f"\n{table(sweep, rows)}")


if __name__ == "__main__":
    main()
