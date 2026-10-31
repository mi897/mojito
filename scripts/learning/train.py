"""Train the leg IK network.

    uv run --extra sim python scripts/learning/train.py                       # defaults
    uv run --extra sim python scripts/learning/train.py --tolerance 0.2       # wider range of leg lengths
    uv run --extra sim python scripts/learning/train.py --upper 0.15 --lower 0.14 --abduction 0.05
    uv run --extra sim python scripts/learning/train.py --backend torch       # PyTorch instead of NumPy

    # Build the network from a URDF instead of the built-in leg:
    uv run --extra sim python scripts/learning/train.py --urdf robots/quadruped/quadruped.urdf --manifest robots/quadruped/quadruped.json
    uv run --extra sim python scripts/learning/train.py --urdf robots/planar/planar.urdf --out weights/ik_planar.npz
"""
import argparse
import os
import time

import numpy as np

import mojito
from mojito.geometry import spec as specmod
from mojito.learning.backend import add_backend_argument
from mojito.results import portable_path, repo_path, results_path, write_result
from mojito.robot import leg


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
    ap.add_argument("--lr", "--learning-rate", dest="learning_rate", type=float, default=2e-3, help="peak learning rate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=repo_path("weights", "ik_leg.npz"))
    ap.add_argument("--log", default=results_path("learning", "training_log.json"))
    ap.add_argument("--device", default="cpu", help="torch backend only: cpu, cuda, mps")
    add_backend_argument(ap)
    args = ap.parse_args()

    if args.urdf:
        robot = specmod.load_robot(args.urdf, args.manifest)
        print(robot.summary(), flush=True)
        if args.group is None and len(robot.groups) != 1:
            ap.error(f"the URDF has {len(robot.groups)} model groups {list(robot.groups)}; choose one with --group")
        config = robot.groups[args.group or next(iter(robot.groups))].canonical
        if args.tolerance is not None:
            config = config.with_normalisation(config.nominal, args.tolerance, config.reach)
    else:
        config = leg.LegConfig(np.array([args.abduction, args.upper, args.lower]), None,
                               0.10 if args.tolerance is None else args.tolerance)
    kwargs = dict(device=args.device) if mojito.learning.backend.resolve(args.backend) == "torch" else {}
    net = mojito.make_model(config, args.hidden, seed=args.seed, backend=args.backend, **kwargs)
    print(f"backend: {net.backend}", flush=True)
    rng = np.random.default_rng(args.seed)
    val_p, val_len, _ = leg.sample_batch(np.random.default_rng(10_000 + args.seed), 20000, config)

    log, t0 = [], time.time()
    for step in range(1, args.steps + 1):
        learning_rate = 1e-5 + 0.5 * (args.learning_rate - 1e-5) * (1 + np.cos(np.pi * step / args.steps))
        p, lengths, _ = leg.sample_batch(rng, args.batch, config)
        net.train_step(p, lengths, learning_rate)

        if step % 500 == 0 or step == 1:
            err = np.linalg.norm(config.forward(net.predict(val_p, val_len), val_len) - val_p, axis=1) * 1000
            row = dict(step=step, median_mm=float(np.median(err)), p95_mm=float(np.percentile(err, 95)),
                       max_mm=float(err.max()), seconds=time.time() - t0)
            log.append(row)
            if step % 5000 == 0 or step == 1:
                print("step {step:6d}  median {median_mm:8.4f} mm  p95 {p95_mm:8.4f} mm  max {max_mm:8.3f} mm  {seconds:6.0f}s".format(**row), flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    net.save(args.out)
    recorded = dict({k: portable_path(v) for k, v in vars(args).items()}, backend=net.backend)  # paths without this machine's folders
    write_result(args.log, dict(args=recorded, log=log))
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
