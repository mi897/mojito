"""Measure the trained network against closed-form IK and draw the result plots.

    python scripts/evaluate.py [--weights weights/ik_leg.npz] [--out results]
"""
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mojito  # noqa: E402
from mojito import leg  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, SURFACE, GRID = "#0b0b0b", "#52514e", "#898781", "#fcfcfb", "#e6e5e0"
BLUES = LinearSegmentedColormap.from_list("blues", ["#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "lines.linewidth": 2, "figure.dpi": 150,
})


def errors_mm(net, p, lengths):
    return np.linalg.norm(net.cfg.forward(net.predict(p, lengths), lengths) - p, axis=1) * 1000


def baseline_ik(cfg, p, lengths):
    """Exact IK where a closed form exists (the 3-joint leg), else the numeric solver."""
    if leg.is_leg(cfg):
        return leg.analytic_ik(p, lengths)
    return cfg.numeric_ik(p, lengths, iters=100)


def summary(e):
    return dict(median_mm=float(np.median(e)), mean_mm=float(e.mean()), p95_mm=float(np.percentile(e, 95)),
                p99_mm=float(np.percentile(e, 99)), max_mm=float(e.max()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/ik_leg.npz")
    ap.add_argument("--log", default="results/training_log.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--backend", choices=mojito.backend.BACKENDS, default=None,
                    help="numpy or torch; default is $MOJITO_BACKEND, else numpy")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    net = mojito.load_model(args.weights, backend=args.backend)
    cfg = net.cfg
    rng = np.random.default_rng(12345)  # never used in training
    n = 200_000
    metrics = dict(nominal_lengths_m=cfg.nominal.tolist(), length_tolerance=cfg.length_tolerance,
                   reach_mm=cfg.reach * 1000, test_targets=n)

    # 1. Held-out targets on held-out legs, lengths within the trained range.
    p, lengths, q_true = leg.sample_batch(rng, n, cfg)
    e = errors_mm(net, p, lengths)
    metrics["learned_varied_legs"] = summary(e)
    q_net = net.predict(p, lengths)
    metrics["joint_angle_error_deg"] = dict(
        median=float(np.degrees(np.median(np.abs(q_net - q_true)))),
        p95=float(np.degrees(np.percentile(np.abs(q_net - q_true), 95))))
    e_exact = np.linalg.norm(cfg.forward(baseline_ik(cfg, p, lengths), lengths) - p, axis=1) * 1000
    metrics["analytic_varied_legs"] = summary(e_exact)

    # 2. Same network told every leg is nominal: what ignoring length variation costs.
    q_blind = net.predict(p, cfg.nominal)
    e_blind = np.linalg.norm(cfg.forward(q_blind, lengths) - p, axis=1) * 1000
    metrics["learned_assuming_nominal_lengths"] = summary(e_blind)

    # 3. The nominal leg only.
    pn, ln, _ = leg.sample_batch(rng, n, cfg, tolerance=0.0)
    metrics["learned_nominal_leg"] = summary(errors_mm(net, pn, ln))

    # 4. Speed.
    t0 = time.perf_counter(); net.predict(p, lengths); t_net = time.perf_counter() - t0
    t0 = time.perf_counter(); baseline_ik(cfg, p, lengths); t_exact = time.perf_counter() - t0
    one = p[:1], lengths[:1]
    t0 = time.perf_counter()
    for _ in range(2000):
        net.predict(*one)
    metrics["speed"] = dict(batched_ns_per_target_learned=t_net / n * 1e9,
                            batched_ns_per_target_analytic=t_exact / n * 1e9,
                            single_call_us_learned=(time.perf_counter() - t0) / 2000 * 1e6)

    # 5. Error against how far the whole leg is scaled from nominal.
    devs = np.linspace(-0.20, 0.20, 21)
    sweep = []
    for d in devs:
        L = np.tile(cfg.nominal * (1 + d), (40_000, 1))
        ps = cfg.forward(leg.sample_joint_angles(rng, L, cfg), L)
        es = errors_mm(net, ps, L)
        sweep.append((float(d), float(np.median(es)), float(np.percentile(es, 95))))
    metrics["length_scale_sweep"] = [dict(scale_deviation=d, median_mm=m, p95_mm=h) for d, m, h in sweep]

    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=1)

    # ---------------- plots ----------------
    if os.path.exists(args.log):
        log = json.load(open(args.log))["log"]
        steps = [r["step"] for r in log]
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.plot(steps, [r["median_mm"] for r in log], color=BLUE, label="median")
        ax.plot(steps, [r["p95_mm"] for r in log], color=ORANGE, label="95th percentile")
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
    frac = np.arange(1, n + 1) / n * 100
    ax.plot(np.sort(e), frac, color=BLUE, label="told each leg's lengths")
    ax.plot(np.sort(e_blind), frac, color=ORANGE, label="assuming nominal lengths")
    ax.set_xscale("log")
    ax.set_xlabel("foot position error (mm)")
    ax.set_ylabel("% of targets at or below")
    ax.set_title(f"Error on legs with lengths varied ±{cfg.length_tolerance:.0%}")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "error_distribution.png")); plt.close(fig)

    if leg.is_leg(cfg):  # the side-view slice is drawn for the 3-joint leg only
        # Side-view slice through the workspace of the nominal leg (foot directly below the hip).
        la = cfg.nominal[0]
        gx, gz = np.meshgrid(np.linspace(-0.25, 0.25, 501), np.linspace(-0.25, 0.02, 271))
        grid = np.stack([gx.ravel(), np.full(gx.size, la), gz.ravel()], axis=1)
        q = leg.analytic_ik(grid, cfg.nominal)
        ok = (np.all((q >= cfg.limits[:, 0]) & (q <= cfg.limits[:, 1]), axis=1)
              & (np.linalg.norm(leg.forward(q, cfg.nominal) - grid, axis=1) < 1e-9))
        em = np.where(ok, errors_mm(net, grid, np.tile(cfg.nominal, (grid.shape[0], 1))), np.nan).reshape(gx.shape)
        fig, ax = plt.subplots(figsize=(6.4, 3.9))
        im = ax.pcolormesh(gx * 1000, gz * 1000, em, cmap=BLUES, vmin=0, vmax=np.nanpercentile(em, 99.5), shading="auto")
        ax.plot(0, 0, "o", color=INK, ms=6)
        ax.annotate("shoulder", (0, 0), textcoords="offset points", xytext=(8, -3), color=INK2)
        ax.set_aspect("equal"); ax.grid(False)
        ax.set_xlabel("forward x (mm)"); ax.set_ylabel("up z (mm)")
        ax.set_title("Where the error is: side-view slice, nominal leg")
        cb = fig.colorbar(im, ax=ax, shrink=0.85); cb.set_label("foot position error (mm)"); cb.outline.set_visible(False)
        fig.tight_layout(); fig.savefig(os.path.join(args.out, "error_map.png")); plt.close(fig)
        metrics["slice_reachable_fraction"] = float(ok.mean())

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    d = np.array([s[0] for s in sweep]) * 100
    tol = cfg.length_tolerance * 100
    ax.axvspan(-tol, tol, color="#f0efec", lw=0)
    ax.plot(d, [s[1] for s in sweep], color=BLUE, marker="o", ms=4, label="median")
    ax.plot(d, [s[2] for s in sweep], color=ORANGE, marker="o", ms=4, label="95th percentile")
    ax.set_yscale("log")
    ax.text(0, ax.get_ylim()[1], "trained range", ha="center", va="top", color=INK2)
    ax.set_xlabel("all three link lengths scaled by (%)")
    ax.set_ylabel("foot position error (mm)")
    ax.set_title("Error inside and beyond the trained range of leg sizes")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout(); fig.savefig(os.path.join(args.out, "error_vs_length.png")); plt.close(fig)

    show = {k: v for k, v in metrics.items() if k != "length_scale_sweep"}
    print(json.dumps(show, indent=1))


if __name__ == "__main__":
    main()
