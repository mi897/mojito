"""Adapt to one particular leg from a handful of foot-position measurements.

A stand-in for the real robot: a leg whose true link lengths and servo zero
offsets differ from the drawing. We command some poses, "measure" where the
foot ended up (with noise), fit the six unknowns through forward kinematics,
and hand the fitted lengths to the same network. No retraining.

    python scripts/calibrate.py [--measurements 30] [--noise-mm 1.0]
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import IKNet, leg  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/ik_leg.npz")
    ap.add_argument("--measurements", type=int, default=30)
    ap.add_argument("--noise-mm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/calibration.json")
    args = ap.parse_args()

    net = IKNet.load(args.weights)
    cfg = net.cfg
    rng = np.random.default_rng(args.seed)

    # The "real" leg: unknown to the robot.
    true_lengths = cfg.nominal * np.array([1.06, 0.95, 1.08])
    true_offsets = np.radians([2.0, -3.0, 4.0])  # servo zero errors
    real_foot = lambda q_cmd: leg.forward(q_cmd + true_offsets, true_lengths)

    # Command poses, measure the foot.
    q_cmd = rng.uniform(cfg.limits[:, 0] + 0.1, cfg.limits[:, 1] - 0.1, size=(args.measurements, 3))
    measured = real_foot(q_cmd) + rng.normal(0, args.noise_mm / 1000, size=(args.measurements, 3))

    def residual(theta):
        return (leg.forward(q_cmd + theta[3:], theta[:3]) - measured).ravel() * 1000

    fit = least_squares(residual, np.concatenate([cfg.nominal, np.zeros(3)]))
    est_lengths, est_offsets = fit.x[:3], fit.x[3:]

    # Fresh targets this leg can reach; drive it with the network before and after.
    targets = real_foot(rng.uniform(cfg.limits[:, 0] + 0.15, cfg.limits[:, 1] - 0.15, size=(20000, 3)))
    err = lambda L, off: np.linalg.norm(real_foot(net.predict(targets, L) - off) - targets, axis=1) * 1000
    before, after = err(cfg.nominal, np.zeros(3)), err(est_lengths, est_offsets)

    out = dict(
        measurements=args.measurements, noise_mm=args.noise_mm,
        true_lengths_mm=(true_lengths * 1000).round(2).tolist(),
        estimated_lengths_mm=(est_lengths * 1000).round(2).tolist(),
        true_offsets_deg=np.degrees(true_offsets).round(2).tolist(),
        estimated_offsets_deg=np.degrees(est_offsets).round(2).tolist(),
        error_before_mm=dict(median=float(np.median(before)), p95=float(np.percentile(before, 95))),
        error_after_mm=dict(median=float(np.median(after)), p95=float(np.percentile(after, 95))),
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
