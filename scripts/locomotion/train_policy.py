"""Work package 7: train the residual foot-offset policy with evolution strategies, then compare it with the scripted trot.

Writes weights/loco_policy.npz, results/locomotion/training_log.json, results/locomotion/loco_eval.json and
results/locomotion/loco_learning.png. Needs mujoco. Placeholder robot, servo model and friction throughout.

    uv run --extra sim python scripts/locomotion/train_policy.py [--iterations 150] [--workers 4]
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from mojito.locomotion.env import EnvConfig, LocomotionEnv, Randomisation
from mojito.locomotion.rollout import rollout
from mojito.locomotion.train import train
from mojito.palette import BLUE, GREY, INK, ORANGE
from mojito.results import repo_path, results_path, write_result

EVAL_SEEDS = range(10_000, 10_040)  # never used in training (which draws seeds below 2**30 from a different stream)
FAST_COMMAND = (0.30, 0.0, 0.0)     # faster than the training range (0.05 to 0.25 m/s): the scripted trot fails here


def summarise(env, policy, **kwargs):
    runs = [rollout(env, policy, seed, **kwargs) for seed in EVAL_SEEDS]
    return {"episodes": len(runs), "mean_return": float(np.mean([r["return"] for r in runs])),
            "fell_fraction": float(np.mean([r["fell"] for r in runs])),
            "mean_tracking_error_m_s": float(np.mean([r["tracking_error_m_s"] for r in runs])),
            "mean_peak_torque_fraction": float(np.mean([r["peak_torque_fraction"] for r in runs])),
            "mean_action_change": float(np.mean([r["action_change"] for r in runs]))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=150)
    parser.add_argument("--pairs", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = EnvConfig()
    policy, history = train(config, iterations=args.iterations, pairs=args.pairs, workers=args.workers, seed=args.seed)
    policy.save(repo_path("weights", "loco_policy.npz"))
    env = LocomotionEnv(config)
    nominal = LocomotionEnv(EnvConfig(randomisation=Randomisation.none()))
    evaluation = {"scripted_trot_randomised": summarise(env, None), "policy_randomised": summarise(env, policy),
                  "scripted_trot_nominal": summarise(nominal, None, randomise=False),
                  "policy_nominal": summarise(nominal, policy, randomise=False),
                  "scripted_trot_fast_command": summarise(env, None, command=FAST_COMMAND),
                  "policy_fast_command": summarise(env, policy, command=FAST_COMMAND)}
    for name, row in evaluation.items():
        print(f"{name:30s} return {row['mean_return']:7.1f}  fell {row['fell_fraction']:.2f}  "
              f"tracking error {row['mean_tracking_error_m_s']:.3f} m/s  peak torque {row['mean_peak_torque_fraction']:.2f}")
    write_result(results_path("locomotion", "training_log.json"), {
        "what": "Evolution-strategies training of a residual foot-offset policy (docs/STAGE2.md, work package 7)",
        "iterations": args.iterations, "pairs": args.pairs, "history": history})
    write_result(results_path("locomotion", "loco_eval.json"), {
        "what": "Learned policy against the scripted trot on held-out episodes (40 seeds each), placeholder robot and servo",
        "eval_seeds": [EVAL_SEEDS.start, EVAL_SEEDS.stop], "command_range": config.command_range, "fast_command": FAST_COMMAND,
        "randomisation": vars(config.randomisation), "reward_weights": vars(config.reward), "results": evaluation})
    figure, axis = plt.subplots(figsize=(6, 3.5))
    iterations = [row["iteration"] for row in history]
    axis.plot(iterations, [row["mean_return"] for row in history], color=BLUE, label="mean of candidates")
    axis.plot(iterations, [row["best_return"] for row in history], color=ORANGE, label="best candidate")
    axis.axhline(evaluation["scripted_trot_randomised"]["mean_return"], color=GREY, linestyle="--", label="scripted trot (held-out)")
    axis.set_xlabel("iteration", color=INK)
    axis.set_ylabel("episode return", color=INK)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(results_path("locomotion", "loco_learning.png"), dpi=150)


if __name__ == "__main__":
    main()
