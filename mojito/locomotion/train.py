"""Train the residual foot-offset policy with evolution strategies (`es.py`) on `LocomotionEnv`, in worker processes."""
from __future__ import annotations

import multiprocessing as mp
import time

import numpy as np

from . import es
from .env import N_ACTIONS, N_OBSERVATIONS, EnvConfig, LocomotionEnv
from .policy import MLPPolicy
from .rollout import rollout

_env = None
_policy = None


def _init_worker(config):
    global _env, _policy
    _env = LocomotionEnv(config)
    _policy = MLPPolicy(N_OBSERVATIONS, N_ACTIONS)


def _score(job):
    """Mean episode return of the parameter vector over the given seeds (a worker function)."""
    parameters, seeds = job
    _policy.set_parameters(parameters)
    return float(np.mean([rollout(_env, _policy, seed)["return"] for seed in seeds]))


def train(config: EnvConfig | None = None, iterations: int = 60, pairs: int = 16, episodes_per_candidate: int = 2, sigma: float = 0.05,
          learning_rate: float = 0.03, workers: int = 4, seed: int = 0, log=print):
    """Returns (policy, history). history lists, per iteration, the mean and best candidate return and the wall time (s).

    Each iteration draws `pairs` noise vectors, scores theta +- sigma noise on the same `episodes_per_candidate` random
    episodes (new seeds every iteration), and takes an Adam step along `es.es_gradient`.
    """
    config = config or EnvConfig()
    rng = np.random.default_rng(seed)
    policy = MLPPolicy(N_OBSERVATIONS, N_ACTIONS, seed=seed)
    parameters = policy.parameters()
    optimiser = es.Adam(parameters.size, learning_rate)
    history = []
    context = mp.get_context("fork")
    with context.Pool(workers, initializer=_init_worker, initargs=(config,)) as pool:
        for iteration in range(iterations):
            started = time.time()
            noise = rng.normal(size=(pairs, parameters.size))
            seeds = [int(value) for value in rng.integers(1 << 30, size=episodes_per_candidate)]
            jobs = [(parameters + sign * sigma * epsilon, seeds) for sign in (1, -1) for epsilon in noise]
            returns = np.array(pool.map(_score, jobs, chunksize=1))
            gradient = es.es_gradient(returns[:pairs], returns[pairs:], noise, sigma)
            parameters = optimiser.step(parameters, gradient)
            history.append({"iteration": iteration, "mean_return": float(returns.mean()), "best_return": float(returns.max()),
                            "seconds": time.time() - started})
            log(f"iteration {iteration:3d}  mean return {returns.mean():8.2f}  best {returns.max():8.2f}  {time.time() - started:5.1f} s")
    policy.set_parameters(parameters)
    return policy, history
