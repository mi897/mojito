"""Run a policy in `LocomotionEnv` for one episode and score it."""
from __future__ import annotations

import numpy as np

from .env import LocomotionEnv


def rollout(env: LocomotionEnv, policy, seed: int, command=None, randomise: bool = True) -> dict:
    """One episode. `policy` is None for the scripted trot (zero action) or anything with `.act(observation)`.

    Returns return, steps, fell, mean tracking error (m/s, planar velocity against the command), peak torque fraction,
    mean action smoothness (|action change|^2) and the mean body velocity.
    """
    observation = env.reset(seed, command=command, randomise=randomise)
    total, errors, velocities, jerk, peak = 0.0, [], [], [], 0.0
    previous = np.zeros(12)
    done, info = False, {}
    while not done:
        action = np.zeros(12) if policy is None else policy.act(observation)
        observation, reward, done, info = env.step(action)
        total += reward
        errors.append(np.linalg.norm(info["velocity_body"][:2] - env.command[:2]))
        velocities.append(info["velocity_body"])
        jerk.append(float(np.sum((np.asarray(action) - previous) ** 2)))
        previous = np.asarray(action)
        peak = max(peak, info["peak_torque_fraction"])
    return {"return": total, "steps": env.step_count, "fell": info["fell"], "tracking_error_m_s": float(np.mean(errors)),
            "peak_torque_fraction": peak, "action_change": float(np.mean(jerk)), "mean_velocity_m_s": np.mean(velocities, axis=0).tolist(),
            "command": env.command.tolist(), "domain": dict(env.domain)}
