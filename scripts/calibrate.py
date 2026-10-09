"""Adapt to one particular leg from a handful of foot-position measurements.

A stand-in for the real robot: a leg whose true link lengths and servo zero
offsets differ from the drawing. We command some poses, "measure" where the
foot ended up (with noise), fit the six unknowns through forward kinematics,
and hand the fitted lengths to the same network. No retraining.

    python scripts/calibrate.py [--measurements 30] [--noise-mm 1.0]

With --urdf the network is checked against that URDF's limb, and the fitted
lengths are written back into a copy of the URDF (--out-urdf):

    python scripts/calibrate.py --urdf robots/quadruped.urdf --manifest robots/quadruped.json --limb FL
"""
import argparse
import json
import os
import sys

import numpy as np
from scipy.optimize import least_squares

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mojito  # noqa: E402
from mojito import spec as specmod, urdf as urdf_mod  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/ik_leg.npz")
    ap.add_argument("--measurements", type=int, default=30)
    ap.add_argument("--noise-mm", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/calibration.json")
    ap.add_argument("--urdf", help="check the weights against this URDF and write the calibrated lengths back")
    ap.add_argument("--manifest")
    ap.add_argument("--limb", help="which limb of the URDF is the one being calibrated (default: the first)")
    ap.add_argument("--out-urdf", default="results/calibrated.urdf")
    ap.add_argument("--backend", choices=mojito.backend.BACKENDS, default=None,
                    help="numpy or torch; default is $MOJITO_BACKEND, else numpy")
    args = ap.parse_args()

    robot = limb = None
    if args.urdf:
        robot = specmod.load_robot(args.urdf, args.manifest)
        limb = robot.limbs[args.limb or next(iter(robot.limbs))]
    # The network belongs to the limb's group, so check it against the group's canonical limb.
    # A mirrored limb (a right leg) has the same lengths in the same order, so the fit carries over.
    canonical = robot.group_of(limb.name).canonical if robot else None
    net = mojito.load_model(args.weights, backend=args.backend, spec=canonical)  # raises if URDF and weights disagree
    cfg = net.cfg
    n, dof = cfg.n_params, cfg.dof
    rng = np.random.default_rng(args.seed)

    # The "real" leg: unknown to the robot.
    true_lengths = cfg.nominal * np.resize([1.06, 0.95, 1.08], n)
    true_offsets = np.radians(np.resize([2.0, -3.0, 4.0], dof))  # servo zero errors
    real_foot = lambda q_cmd: cfg.forward(q_cmd + true_offsets, true_lengths)

    # Command poses, measure the foot.
    q_cmd = rng.uniform(cfg.limits[:, 0] + 0.1, cfg.limits[:, 1] - 0.1, size=(args.measurements, dof))
    measured = real_foot(q_cmd) + rng.normal(0, args.noise_mm / 1000, size=(args.measurements, 3))

    def residual(theta):
        return (cfg.forward(q_cmd + theta[n:], theta[:n]) - measured).ravel() * 1000

    fit = least_squares(residual, np.concatenate([cfg.nominal, np.zeros(dof)]))
    est_lengths, est_offsets = fit.x[:n], fit.x[n:]

    # Fresh targets this leg can reach; drive it with the network before and after.
    targets = real_foot(rng.uniform(cfg.limits[:, 0] + 0.15, cfg.limits[:, 1] - 0.15, size=(20000, dof)))
    err = lambda L, off: np.linalg.norm(real_foot(net.predict(targets, L) - off) - targets, axis=1) * 1000
    before, after = err(cfg.nominal, np.zeros(dof)), err(est_lengths, est_offsets)

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
        warn = limb.check_params(est_lengths, cfg.length_tolerance)
        calibrated = robot.write_back({limb.name: est_lengths})
        os.makedirs(os.path.dirname(args.out_urdf) or ".", exist_ok=True)
        urdf_mod.write(calibrated, args.out_urdf)
        out.update(limb=limb.name, calibrated_urdf=args.out_urdf, outside_trained_range=warn)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
