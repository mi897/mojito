"""Evolution strategies (Salimans et al. 2017 style) for the locomotion policy: antithetic perturbations, centred-rank
fitness shaping, an Adam step on the estimated gradient. Pure NumPy, episodes spread over processes.

For parameters theta, noise epsilon_i ~ N(0, I) and returns F(theta +- sigma epsilon_i):

    gradient = 1 / (n sigma) sum_i (rank(F_plus_i) - rank(F_minus_i)) epsilon_i,   ranks centred to [-0.5, 0.5]

(a ranking makes the step insensitive to the scale of the return). The pair (+, -) is evaluated on the same episode seeds, so
the difference between them is the perturbation and not a different draw of mass, friction or command. A test checks the
estimator on a quadratic with a known gradient.
"""
from __future__ import annotations

import numpy as np


def centred_ranks(values) -> np.ndarray:
    """Ranks of `values` scaled to [-0.5, 0.5] (0 for the lowest, 1 for the highest, minus 0.5)."""
    values = np.asarray(values, float)
    ranks = np.empty(values.size)
    ranks[np.argsort(values.ravel())] = np.arange(values.size)
    return (ranks / max(values.size - 1, 1) - 0.5).reshape(values.shape)


def es_gradient(returns_plus, returns_minus, noise, sigma: float) -> np.ndarray:
    """Gradient estimate of the expected return from antithetic pairs: returns (n,), noise (n, n_parameters)."""
    shaped = centred_ranks(np.concatenate([returns_plus, returns_minus]))
    n_pairs = len(returns_plus)
    weights = shaped[:n_pairs] - shaped[n_pairs:]
    return weights @ noise / (n_pairs * sigma)


class Adam:
    """Adam ascent step (maximising), the same update as `mojito/learning/model.py` but for a flat vector."""

    def __init__(self, n_parameters: int, learning_rate: float = 0.02, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8):
        self.learning_rate, self.beta1, self.beta2, self.epsilon = learning_rate, beta1, beta2, epsilon
        self.first = np.zeros(n_parameters)
        self.second = np.zeros(n_parameters)
        self.count = 0

    def step(self, parameters, gradient) -> np.ndarray:
        self.count += 1
        self.first = self.beta1 * self.first + (1 - self.beta1) * gradient
        self.second = self.beta2 * self.second + (1 - self.beta2) * gradient ** 2
        first_hat = self.first / (1 - self.beta1 ** self.count)
        second_hat = self.second / (1 - self.beta2 ** self.count)
        return parameters + self.learning_rate * first_hat / (np.sqrt(second_hat) + self.epsilon)
