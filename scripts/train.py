"""Train the leg IK network.

    python scripts/train.py                       # defaults
    python scripts/train.py --tolerance 0.2       # wider range of leg lengths
    python scripts/train.py --upper 0.15 --lower 0.14 --abduction 0.05
    python scripts/train.py --backend torch       # PyTorch instead of NumPy

    # Build the network from a URDF instead of the built-in leg:
    python scripts/train.py --urdf robots/quadruped.urdf --manifest robots/quadruped.json
    python scripts/train.py --urdf robots/planar.urdf --out weights/ik_planar.npz
"""
import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import mojito  # noqa: E402
from mojito import leg, spec as specmod  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abduction", type=float, default=leg.NOMINAL_LENGTHS[0], help="nominal abduction offset [m]")
    ap.add_argument("--upper", type=float, default=leg.NOMINAL_LENGTHS[1], help="nominal upper-leg length [m]")
    ap.add_argument("--lower", type=float, default=leg.NOMINAL_LENGTHS[2], help="nominal lower-leg length [m]")
    ap.add_argument("--tolerance", type=float, default=None, help="lengths vary by +/- this fraction in training (default 0.10, or the manifest's)")
    ap.add_argument("--urdf", help="build the network for a limb of this URDF (ignores --abduction/--upper/--lower)")
    ap.add_argument("--manifest", help="JSON manifest beside the URDF: limbs, mirrors, groups")
    ap.add_argument("--group", help="which model group of the URDF to train (default: the only one)")
    ap.add_argument("--hidden", type=int, nargs="+", default=[128, 128, 128])
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="weights/ik_leg.npz")
    ap.add_argument("--log", default="results/training_log.json")
    ap.add_argument("--device", default="cpu", help="torch backend only: cpu, cuda, mps")
    ap.add_argument("--backend", choices=mojito.backend.BACKENDS, default=None,
                    help="numpy or torch; default is $MOJITO_BACKEND, else numpy")
    args = ap.parse_args()

    if args.urdf:
        robot = specmod.load_robot(args.urdf, args.manifest)
        print(robot.summary(), flush=True)
        if args.group is None and len(robot.groups) != 1:
            ap.error(f"the URDF has {len(robot.groups)} model groups {list(robot.groups)}; choose one with --group")
        cfg = robot.groups[args.group or next(iter(robot.groups))].canonical
        if args.tolerance is not None:
            cfg = cfg.with_normalisation(cfg.nominal, args.tolerance, cfg.reach)
    else:
        cfg = leg.LegConfig(np.array([args.abduction, args.upper, args.lower]), None,
                            0.10 if args.tolerance is None else args.tolerance)
    kwargs = dict(device=args.device) if mojito.backend.resolve(args.backend) == "torch" else {}
    net = mojito.make_model(cfg, args.hidden, seed=args.seed, backend=args.backend, **kwargs)
    print(f"backend: {net.backend}", flush=True)
    rng = np.random.default_rng(args.seed)
    val_p, val_len, _ = leg.sample_batch(np.random.default_rng(10_000 + args.seed), 20000, cfg)

    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        lr = 1e-5 + 0.5 * (args.lr - 1e-5) * (1 + np.cos(np.pi * step / args.steps))
        p, lengths, _ = leg.sample_batch(rng, args.batch, cfg)
        net.train_step(p, lengths, lr)

        if step % 500 == 0 or step == 1:
            err = np.linalg.norm(cfg.forward(net.predict(val_p, val_len), val_len) - val_p, axis=1) * 1000
            row = dict(step=step, median_mm=float(np.median(err)), p95_mm=float(np.percentile(err, 95)),
                       max_mm=float(err.max()), seconds=time.time() - t0)
            log.append(row)
            if step % 5000 == 0 or step == 1:
                print("step {step:6d}  median {median_mm:8.4f} mm  p95 {p95_mm:8.4f} mm  max {max_mm:8.3f} mm  {seconds:6.0f}s".format(**row), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(args.log) or ".", exist_ok=True)
    net.save(args.out)
    with open(args.log, "w") as f:
        json.dump(dict(args=dict(vars(args), backend=net.backend), log=log), f, indent=1)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
