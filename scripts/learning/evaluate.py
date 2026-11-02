"""Measure the trained network against closed-form IK and draw the result plots.

    uv run --extra sim python scripts/learning/evaluate.py [--weights weights/ik_leg.npz] [--out results/learning]
"""
import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

import mojito  # noqa: E402
from mojito import palette  # noqa: E402
from mojito.learning.backend import add_backend_argument  # noqa: E402
from mojito.palette import BLUE, LABEL_INK as INK2, ORANGE, TITLE_INK as INK  # noqa: E402
from mojito.results import repo_path, results_path, write_result  # noqa: E402
from mojito.robot import leg  # noqa: E402

BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", BLUE, "#0d366b"])

plt.rcParams.update({**palette.panel_style(font_size=10, title_size=11, line_width=2), "figure.dpi": 150})


def errors_mm(net, targets, lengths):
    return np.linalg.norm(net.config.forward(net.predict(targets, lengths), lengths) - targets, axis=1) * 1000


def baseline_ik(config, targets, lengths):
    """Exact IK where a closed form exists (the 3-joint leg), else the numeric solver."""
    if leg.is_leg(config):
        return leg.analytic_ik(targets, lengths)
    return config.numeric_ik(targets, lengths, max_iterations=100)


def summary(errors):
    """Statistics of foot position errors in millimetres."""
    return dict(median_mm=float(np.median(errors)), mean_mm=float(errors.mean()), p95_mm=float(np.percentile(errors, 95)),
                p99_mm=float(np.percentile(errors, 99)), max_mm=float(errors.max()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=repo_path("weights", "ik_leg.npz"))
    parser.add_argument("--log", default=results_path("learning", "training_log.json"))
    parser.add_argument("--out", default=results_path("learning"))
    add_backend_argument(parser)
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = mojito.load_model(args.weights, backend=args.backend)
    config = net.config
    rng = np.random.default_rng(12345)  # never used in training
    n_targets = 200_000
    metrics = dict(nominal_lengths_m=config.nominal.tolist(), length_tolerance=config.length_tolerance,
                   reach_mm=config.reach * 1000, test_targets=n_targets)

    # 1. Held-out targets on held-out legs, lengths within the trained range.
    targets, lengths, joint_angles_true = leg.sample_batch(rng, n_targets, config)
    errors = errors_mm(net, targets, lengths)
    metrics["learned_varied_legs"] = summary(errors)
    joint_angles_net = net.predict(targets, lengths)
    metrics["joint_angle_error_deg"] = dict(
        median=float(np.degrees(np.median(np.abs(joint_angles_net - joint_angles_true)))),
        p95=float(np.degrees(np.percentile(np.abs(joint_angles_net - joint_angles_true), 95))))
    errors_exact = np.linalg.norm(config.forward(baseline_ik(config, targets, lengths), lengths) - targets, axis=1) * 1000
    metrics["analytic_varied_legs"] = summary(errors_exact)

    # 2. Same network told every leg is nominal: what ignoring length variation costs.
    joint_angles_blind = net.predict(targets, config.nominal)
    errors_blind = np.linalg.norm(config.forward(joint_angles_blind, lengths) - targets, axis=1) * 1000
    metrics["learned_assuming_nominal_lengths"] = summary(errors_blind)

    # 3. The nominal leg only.
    nominal_targets, nominal_lengths, _ = leg.sample_batch(rng, n_targets, config, tolerance=0.0)
    metrics["learned_nominal_leg"] = summary(errors_mm(net, nominal_targets, nominal_lengths))

    # 4. Speed.
    t0 = time.perf_counter(); net.predict(targets, lengths); t_net = time.perf_counter() - t0
    t0 = time.perf_counter(); baseline_ik(config, targets, lengths); t_exact = time.perf_counter() - t0
    one = targets[:1], lengths[:1]
    t0 = time.perf_counter()
    for _ in range(2000):
        net.predict(*one)
    metrics["speed"] = dict(batched_ns_per_target_learned=t_net / n_targets * 1e9,
                            batched_ns_per_target_analytic=t_exact / n_targets * 1e9,
                            single_call_us_learned=(time.perf_counter() - t0) / 2000 * 1e6)

    # 5. Error against how far the whole leg is scaled from nominal.
    scale_deviations = np.linspace(-0.20, 0.20, 21)
    sweep = []
    for deviation in scale_deviations:
        scaled_lengths = np.tile(config.nominal * (1 + deviation), (40_000, 1))
        scaled_targets = config.forward(leg.sample_joint_angles(rng, scaled_lengths, config), scaled_lengths)
        scaled_errors = errors_mm(net, scaled_targets, scaled_lengths)
        sweep.append((float(deviation), float(np.median(scaled_errors)), float(np.percentile(scaled_errors, 95))))
    metrics["length_scale_sweep"] = [dict(scale_deviation=deviation, median_mm=median, p95_mm=p95)
                                     for deviation, median, p95 in sweep]

    # ---------------- plots ----------------
    if os.path.exists(args.log):
        log = json.load(open(args.log))["log"]
        steps = [entry["step"] for entry in log]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(steps, [entry["median_mm"] for entry in log], color=BLUE, label="median")
        ax.plot(steps, [entry["p95_mm"] for entry in log], color=ORANGE, label="95th percentile")
        ax.set_yscale("log")
        ax.set_xlabel("training step")
        ax.set_ylabel("foot position error (mm)")
        ax.set_title("Validation error during training")
        ax.text(steps[-1], log[-1]["median_mm"], f"  {log[-1]['median_mm']:.2f} mm", va="center", color=INK2)
        ax.text(steps[-1], log[-1]["p95_mm"], f"  {log[-1]['p95_mm']:.2f} mm", va="center", color=INK2)
        ax.set_xlim(0, steps[-1] * 1.14)
        ax.legend(frameon=False)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "training_curve.png")); plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    percent_at_or_below = np.arange(1, n_targets + 1) / n_targets * 100
    ax.plot(np.sort(errors), percent_at_or_below, color=BLUE, label="told each leg's lengths")
    ax.plot(np.sort(errors_blind), percent_at_or_below, color=ORANGE, label="assuming nominal lengths")
    ax.set_xscale("log")
    ax.set_xlabel("foot position error (mm)")
    ax.set_ylabel("% of targets at or below")
    ax.set_title(f"Error on legs with lengths varied ±{config.length_tolerance:.0%}")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "error_distribution.png")); plt.close(fig)

    if leg.is_leg(config):  # the side-view slice is drawn for the 3-joint leg only
        # Side-view slice through the workspace of the nominal leg (foot directly below the hip).
        abduction_offset = config.nominal[0]
        grid_x, grid_z = np.meshgrid(np.linspace(-0.25, 0.25, 501), np.linspace(-0.25, 0.02, 271))
        grid = np.stack([grid_x.ravel(), np.full(grid_x.size, abduction_offset), grid_z.ravel()], axis=1)
        joint_angles = leg.analytic_ik(grid, config.nominal)
        reachable = (np.all((joint_angles >= config.limits[:, 0]) & (joint_angles <= config.limits[:, 1]), axis=1)
                     & (np.linalg.norm(leg.forward(joint_angles, config.nominal) - grid, axis=1) < 1e-9))
        grid_errors = errors_mm(net, grid, np.tile(config.nominal, (grid.shape[0], 1)))
        error_map = np.where(reachable, grid_errors, np.nan).reshape(grid_x.shape)
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        mesh = ax.pcolormesh(grid_x * 1000, grid_z * 1000, error_map, cmap=BLUES, vmin=0, vmax=np.nanpercentile(error_map, 99.5),
                             shading="auto")
        ax.plot(0, 0, "o", color=INK, ms=6)
        ax.annotate("shoulder", (0, 0), textcoords="offset points", xytext=(8, -3), color=INK2)
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlabel("forward x (mm)"); ax.set_ylabel("up z (mm)")
        ax.set_title("Where the error is: side-view slice, nominal leg")
        colorbar = fig.colorbar(mesh, ax=ax, shrink=0.85)
        colorbar.set_label("foot position error (mm)")
        colorbar.outline.set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "error_map.png")); plt.close(fig)
        metrics["slice_reachable_fraction"] = float(reachable.mean())

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    deviation_percent = np.array([row[0] for row in sweep]) * 100
    tolerance_percent = config.length_tolerance * 100
    ax.axvspan(-tolerance_percent, tolerance_percent, color="#f0efec", lw=0)
    ax.plot(deviation_percent, [row[1] for row in sweep], color=BLUE, marker="o", ms=4, label="median")
    ax.plot(deviation_percent, [row[2] for row in sweep], color=ORANGE, marker="o", ms=4, label="95th percentile")
    ax.set_yscale("log")
    ax.text(0, ax.get_ylim()[1], "trained range", ha="center", va="top", color=INK2)
    ax.set_xlabel("all three link lengths scaled by (%)")
    ax.set_ylabel("foot position error (mm)")
    ax.set_title("Error inside and beyond the trained range of leg sizes")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "error_vs_length.png")); plt.close(fig)

    # Written after the plots: the side-view slice adds slice_reachable_fraction to the metrics.
    write_result(os.path.join(args.out, "metrics.json"), metrics)
    show = {key: value for key, value in metrics.items() if key != "length_scale_sweep"}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
