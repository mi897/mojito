"""A small feed-forward policy in NumPy (no PyTorch needed: work package 7's first learner is evolution strategies).

    hidden_1 = tanh(observation W0 + b0),  hidden_2 = tanh(hidden_1 W1 + b1),  action = tanh(hidden_2 W2 + b2)

with every weight matrix shaped (inputs, outputs) as in `mojito/learning`. The last layer starts at zero, so an untrained
policy outputs a zero action, which is exactly the scripted trot (`env.LocomotionEnv`): training starts from the baseline.
The parameters are also reachable as one flat vector (`parameters`, `set_parameters`), which is what the optimiser perturbs.
"""
from __future__ import annotations

import numpy as np


class MLPPolicy:
    def __init__(self, n_inputs: int, n_outputs: int, hidden=(64, 64), seed: int = 0):
        rng = np.random.default_rng(seed)
        sizes = [n_inputs, *hidden, n_outputs]
        # He-style scale for the hidden layers; zero for the last, so the first action is zero
        self.weights = [rng.normal(0.0, np.sqrt(1.0 / size_in), (size_in, size_out)) for size_in, size_out in zip(sizes[:-1], sizes[1:])]
        self.weights[-1] = np.zeros_like(self.weights[-1])
        self.biases = [np.zeros(size_out) for size_out in sizes[1:]]

    def act(self, observation) -> np.ndarray:
        """Action in [-1, 1] for one observation (n_inputs,)."""
        activations = np.asarray(observation, float)
        for weight, bias in zip(self.weights, self.biases):
            activations = np.tanh(activations @ weight + bias)
        return activations

    @property
    def n_parameters(self) -> int:
        return sum(weight.size + bias.size for weight, bias in zip(self.weights, self.biases))

    def parameters(self) -> np.ndarray:
        """All weights and biases as one vector, layer by layer (weights then biases), row-major."""
        return np.concatenate([np.concatenate([weight.ravel(), bias]) for weight, bias in zip(self.weights, self.biases)])

    def set_parameters(self, vector) -> None:
        """Inverse of `parameters`."""
        vector = np.asarray(vector, float)
        if vector.size != self.n_parameters:
            raise ValueError(f"expected {self.n_parameters} parameters, got {vector.size}")
        start = 0
        for layer, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            self.weights[layer] = vector[start:start + weight.size].reshape(weight.shape).copy()
            start += weight.size
            self.biases[layer] = vector[start:start + bias.size].copy()
            start += bias.size

    def save(self, path) -> None:
        np.savez(path, **{f"W{i}": weight for i, weight in enumerate(self.weights)}, **{f"b{i}": bias for i, bias in enumerate(self.biases)})

    @classmethod
    def load(cls, path) -> "MLPPolicy":
        with np.load(path) as archive:
            n_layers = len([key for key in archive.files if key.startswith("W")])
            weights = [archive[f"W{i}"] for i in range(n_layers)]
            biases = [archive[f"b{i}"] for i in range(n_layers)]
        policy = cls(weights[0].shape[0], weights[-1].shape[1], hidden=tuple(weight.shape[1] for weight in weights[:-1]))
        policy.weights, policy.biases = weights, biases
        return policy
