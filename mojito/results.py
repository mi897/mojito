"""Where generated results go, and how they are written: JSON with a record of what produced it.

Every script that writes `results/**/*.json` goes through `write_result`, which adds a `provenance` block:

    "provenance": {"git_commit": ..., "git_dirty": ..., "script": "scripts/gait/gait_replay.py", "argv": [...],
                   "python": ..., "numpy": ..., "mujoco": ..., "utc": "YYYY-MM-DDTHH:MM:SSZ"}

`git_dirty` is true when, at the time of the run, a tracked file outside results/ differed from the commit or an
untracked file sat under mojito/ or scripts/: then the commit alone does not reproduce the numbers. results/ itself does
not count, so a run of `just results` does not call every file after the first dirty. `mujoco` and `torch` are recorded
only when the run imported them. Paths in `script` and `argv` are written relative to the repository root (or with the
home folder as ~), so no machine's folders reach committed files.

Paths are built from the repository root found from this file, so scripts work from any working directory.
That needs the editable install `uv sync` makes (the package is used in place, not copied); `repo_path` raises if the
package is not sitting in the repository.
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


class NotInRepository(RuntimeError):
    """mojito is installed as a copy (a non-editable install), so the repository root, and results/, cannot be found."""


def _check_root() -> None:
    if not os.path.exists(os.path.join(ROOT, "pyproject.toml")):
        raise NotInRepository(
            f"mojito is imported from {ROOT}, which is not the repository (no pyproject.toml there). The scripts need the "
            "editable install: run `uv sync --extra sim` (or `pip install -e .`) in the repository and use its .venv.")


def repo_path(*parts: str) -> str:
    """Absolute path of `parts` under the repository root. Raises NotInRepository for a non-editable install."""
    _check_root()
    return os.path.join(ROOT, *parts)


def results_path(*parts: str) -> str:
    """Absolute path of `parts` under results/, e.g. results_path("gait", "gait_replay.json")."""
    return repo_path("results", *parts)


def relative_to_root(path: str) -> str:
    """`path` relative to the repository root when it lies inside it (so results do not record this machine's folders), else unchanged."""
    absolute = os.path.abspath(path)
    return os.path.relpath(absolute, ROOT) if absolute.startswith(ROOT + os.sep) else path


def portable_path(value: str) -> str:
    """A path-like string as it should be recorded: relative to the repository root when inside it, with the home folder
    as ~ when under it, else unchanged. Strings that are not paths (no separator and no such file) pass through."""
    if not isinstance(value, str) or not value or (os.sep not in value and not os.path.exists(value)):
        return value
    absolute = os.path.abspath(value)
    if absolute.startswith(ROOT + os.sep):
        return os.path.relpath(absolute, ROOT)
    home = os.path.expanduser("~")
    if home not in ("", "/") and absolute.startswith(home + os.sep):
        return "~" + absolute[len(home):]
    return value


def _portable_arg(arg: str) -> str:
    """One command-line argument with any path in it made portable, including the value of --option=path."""
    if arg.startswith("-") and "=" in arg:
        option, value = arg.split("=", 1)
        return f"{option}={portable_path(value)}"
    return arg if arg.startswith("-") else portable_path(arg)


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
    return portable_path(path) if path else None


def provenance() -> dict:
    """What produced a result: git commit and whether the tree was dirty, script and argv, versions, UTC time."""
    changed = _git("status", "--porcelain", "--untracked-files=no", "--", ".", ":(exclude)results")
    untracked_code = _git("status", "--porcelain", "--untracked-files=all", "--", "mojito", "scripts")
    record = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if changed is None or untracked_code is None else bool(changed or untracked_code),
        "script": _script(),
        "argv": [_portable_arg(a) for a in sys.argv[1:]],
        "python": platform.python_version(),
        "numpy": _version("numpy"),
        "utc": datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    for package in ("mujoco", "torch"):  # only reported when the run used it
        if package in sys.modules:
            record[package] = _version(package)
    return record


def write_result(path: str, payload: dict, *, indent: int | None = 1, compact: bool = False, note: dict | None = None) -> str:
    """Write `payload` as JSON to `path` with a `provenance` block added, creating the folder. Returns `path`.

    `payload` is not changed. `compact=True` writes without whitespace (large files such as physics_sweep.json);
    `indent=None` writes one line with the default separators. `note` adds keys to the provenance block, for example
    which entries of a merged file this run wrote.
    """
    if not isinstance(payload, dict):
        raise TypeError(f"a result is a JSON object, got {type(payload).__name__}")
    if PROVENANCE_KEY in payload:
        payload = {k: v for k, v in payload.items() if k != PROVENANCE_KEY}  # a result read back and rewritten gets a fresh one
    record = {**payload, PROVENANCE_KEY: {**provenance(), **(note or {})}}
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
