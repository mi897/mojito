"""Which numerical library runs the leg network: "numpy" (default) or "torch".

Three ways to choose, in order of priority:

1. Per call:      mojito.load_model(path, backend="torch")
2. Per process:   mojito.set_backend("torch")
3. Per shell:     export MOJITO_BACKEND=torch

Every script also takes --backend numpy|torch. Both backends read and write
the same weight files, so a network trained with one runs under the other.
PyTorch is imported only when it is asked for.
"""
from __future__ import annotations

import os

BACKENDS = ("numpy", "torch")
_current = None


def _check(name: str) -> str:
    name = str(name).lower()
    if name == "pytorch":
        name = "torch"
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; choose one of {BACKENDS}")
    return name


def get_backend() -> str:
    """The backend used when a call does not name one."""
    if _current is not None:
        return _current
    return _check(os.environ.get("MOJITO_BACKEND", "numpy"))


def set_backend(name: str) -> None:
    """Choose the default backend for the rest of this process."""
    global _current
    _current = _check(name)


def resolve(name: str | None) -> str:
    return get_backend() if name is None else _check(name)


def model_class(name: str | None = None):
    """The network class for a backend. Imports PyTorch only if needed."""
    name = resolve(name)
    if name == "numpy":
        from .model import IKNet

        return IKNet
    try:
        from .torch_model import TorchIKNet
    except ImportError as e:  # torch itself is missing
        raise ImportError(
            "The torch backend needs PyTorch, which is not installed. "
            "Install it (pip install torch) or use the numpy backend."
        ) from e
    return TorchIKNet


def make_model(cfg=None, hidden=(128, 128, 128), seed=0, backend=None, **kwargs):
    """A new, untrained network. Both backends start from identical weights for a given seed."""
    return model_class(backend)(cfg, hidden, seed=seed, **kwargs)


def load_model(path, backend=None, **kwargs):
    """Load trained weights into the chosen backend."""
    return model_class(backend).load(path, **kwargs)
