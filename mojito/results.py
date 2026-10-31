"""Where generated results go, and how they are written: JSON with a record of what produced it.

Every script that writes `results/**/*.json` goes through `write_result`, which adds a `provenance` block:

    "provenance": {"git_commit": ..., "git_dirty": ..., "script": "scripts/gait/gait_replay.py", "argv": [...],
                   "python": ..., "numpy": ..., "mujoco": ..., "utc": "YYYY-MM-DDTHH:MM:SSZ"}

`git_dirty` is true when tracked files differed from the commit at the time of the run, so the commit alone does not
reproduce the numbers. A version is null when that package is not installed (MuJoCo is optional).

Paths are built from the repository root found from this file, so scripts work from any working directory.
That needs the editable install `uv sync` makes (the package is used in place, not copied).
"""
from __future__ import annotations

import datetime
import importlib.metadata
import inspect
import json
import os
import platform
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROVENANCE_KEY = "provenance"


def repo_path(*parts: str) -> str:
    """Absolute path of `parts` under the repository root."""
    return os.path.join(ROOT, *parts)


def results_path(*parts: str) -> str:
    """Absolute path of `parts` under results/, e.g. results_path("gait", "gait_replay.json")."""
    return repo_path("results", *parts)


def relative_to_root(path: str) -> str:
    """`path` relative to the repository root when it lies inside it (so results do not record this machine's folders), else unchanged."""
    absolute = os.path.abspath(path)
    return os.path.relpath(absolute, ROOT) if absolute.startswith(ROOT + os.sep) else path


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _script() -> str | None:
    """The running script, relative to the repository root when it is inside it."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None) or (sys.argv[0] if sys.argv and sys.argv[0] else None)
    return relative_to_root(path) if path else None


def provenance() -> dict:
    """What produced a result: git commit and whether the tree was dirty, script and argv, versions, UTC time."""
    status = _git("status", "--porcelain", "--untracked-files=no")
    record = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "script": _script(),
        "argv": list(sys.argv[1:]),
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "mujoco": _version("mujoco"),
        "utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    if "torch" in sys.modules:  # only reported when the run used it
        record["torch"] = _version("torch")
    return record


def write_result(path: str, payload: dict, *, indent: int | None = 1, compact: bool = False) -> str:
    """Write `payload` as JSON to `path` with a `provenance` block added, creating the folder. Returns `path`.

    `payload` is not changed. `compact=True` writes without whitespace (large files such as physics_sweep.json);
    `indent=None` writes one line with the default separators.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"a result is a JSON object, got {type(payload).__name__}")
    if PROVENANCE_KEY in payload:
        payload = {k: v for k, v in payload.items() if k != PROVENANCE_KEY}  # a result read back and rewritten gets a fresh one
    record = {**payload, PROVENANCE_KEY: provenance()}
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        if compact:
            json.dump(record, f, separators=(",", ":"))
        else:
            json.dump(record, f, indent=indent)
    return path


def without_provenance(payload: dict) -> tuple[dict, dict | None]:
    """(`payload` without its provenance block, the block or None): for readers that only want the numbers."""
    if not isinstance(payload, dict) or PROVENANCE_KEY not in payload:
        return payload, None
    return {k: v for k, v in payload.items() if k != PROVENANCE_KEY}, payload[PROVENANCE_KEY]


def defaults_of(function) -> dict:
    """Default keyword values of `function`, so a script can record the settings a run actually used."""
    return {name: p.default for name, p in inspect.signature(function).parameters.items()
            if p.default is not inspect.Parameter.empty}
