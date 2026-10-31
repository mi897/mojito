"""Adapt to one particular leg from a handful of foot-position measurements.

A stand-in for the real robot: a leg whose true link lengths and servo zero
offsets differ from the drawing. We command some poses, "measure" where the
foot ended up (with noise), fit the six unknowns through forward kinematics,
and hand the fitted lengths to the same network. No retraining.

    uv run --extra sim python scripts/learning/calibrate.py [--measurements 30] [--noise-mm 1.0]

With --urdf the network is checked against that URDF's limb, and the fitted
lengths are written back into a copy of the URDF (--out-urdf):

    uv run --extra sim python scripts/learning/calibrate.py --urdf robots/quadruped/quadruped.urdf --manifest robots/quadruped/quadruped.json --limb FL
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import least_squares

import mojito
from mojito.geometry import spec as specmod, urdf as urdf_mod
from mojito.learning.backend import add_backend_argument
from mojito.results import portable_path, repo_path, results_path, write_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=repo_path("weights", "ik_leg.npz"))
    ap.add_argument("--measurements", type=int, default=30)
    ap.add_argument("--noise-mm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=results_path("learning", "calibration.json"))
    ap.add_argument("--urdf", help="check the weights against this URDF and write the calibrated lengths back")
    ap.add_argument("--manifest")
    ap.add_argument("--limb", help="which limb of the URDF is the one being calibrated (default: the first)")
    ap.add_argument("--out-urdf", default=results_path("learning", "calibrated.urdf"))
    add_backend_argument(ap)
    args = ap.parse_args()

    robot = limb = None
    if args.urdf:
        robot = specmod.load_robot(args.urdf, args.manifest)
        limb = robot.limbs[args.limb or next(iter(robot.limbs))]
    # The network belongs to the limb's group, so check it against the group's canonical limb.
    # A mirrored limb (a right leg) has the same lengths in the same order, so the fit carries over.
    canonical = robot.group_of(limb.name).canonical if robot else None
    net = mojito.load_model(args.weights, backend=args.backend, spec=canonical)  # raises if URDF and weights disagree
    config = net.config
    n, dof = config.n_params, config.dof
    rng = np.random.default_rng(args.seed)

    # The "real" leg: unknown to the robot.
    true_lengths = config.nominal * np.resize([1.06, 0.95, 1.08], n)
    true_offsets = np.radians(np.resize([2.0, -3.0, 4.0], dof))  # servo zero errors
    real_foot = lambda q_cmd: config.forward(q_cmd + true_offsets, true_lengths)

    # Command poses, measure the foot.
    q_cmd = rng.uniform(config.limits[:, 0] + 0.1, config.limits[:, 1] - 0.1, size=(args.measurements, dof))
    measured = real_foot(q_cmd) + rng.normal(0, args.noise_mm / 1000, size=(args.measurements, 3))

    def residual(theta):
        return (config.forward(q_cmd + theta[n:], theta[:n]) - measured).ravel() * 1000

    fit = least_squares(residual, np.concatenate([config.nominal, np.zeros(dof)]))
    est_lengths, est_offsets = fit.x[:n], fit.x[n:]

    # Fresh targets this leg can reach; drive it with the network before and after.
    targets = real_foot(rng.uniform(config.limits[:, 0] + 0.15, config.limits[:, 1] - 0.15, size=(20000, dof)))
    err = lambda L, off: np.linalg.norm(real_foot(net.predict(targets, L) - off) - targets, axis=1) * 1000
    before, after = err(config.nominal, np.zeros(dof)), err(est_lengths, est_offsets)

    out = dict(
        measurements=args.measurements, noise_mm=args.noise_mm,
        true_lengths_mm=(true_lengths * 1000).round(2).tolist(),
        estimated_lengths_mm=(est_lengths * 1000).round(2).tolist(),
        true_offsets_deg=np.degrees(true_offsets).round(2).tolist(),
        estimated_offsets_deg=np.degrees(est_offsets).round(2).tolist(),
        error_before_mm=dict(median=float(np.median(before)), p95=float(np.percentile(before, 95))),
        error_after_mm=dict(median=float(np.median(after)), p95=float(np.percentile(after, 95))),
    )
    if robot is not None:
        # Corrected link lengths go back into a copy of the URDF, readable by any other tool.
        # Servo zero offsets have no place in a URDF, so they stay in the JSON report.
        warn = limb.check_params(est_lengths, config.length_tolerance)
        calibrated = robot.write_back({limb.name: est_lengths})
        os.makedirs(os.path.dirname(args.out_urdf) or ".", exist_ok=True)
        urdf_mod.write(calibrated, args.out_urdf)
        out.update(limb=limb.name, calibrated_urdf=portable_path(args.out_urdf), outside_trained_range=warn)
    write_result(args.out, out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
