#!/usr/bin/env bash
# One-shot setup on Linux/macOS: installs uv and just if missing, then the project environment.
#   ./scripts/setup.sh            core + MuJoCo, no PyTorch
#   ./scripts/setup.sh --torch    also PyTorch; CUDA build if an NVIDIA GPU is detected, else CPU
#   ./scripts/setup.sh --gpu      also PyTorch, CUDA build (no detection)
#   ./scripts/setup.sh --cpu      also PyTorch, CPU-only build (no detection)
set -euo pipefail
cd "$(dirname "$0")/.."

torch=""   # "", "auto", "cpu" or "gpu"
for arg in "$@"; do
    case "$arg" in
        --torch) [[ -z "$torch" ]] && torch=auto ;;
        --cpu)   torch=cpu ;;
        --gpu)   torch=gpu ;;
        *) echo "unknown option: $arg" >&2; exit 2 ;;
    esac
done

# Detection runs only when neither --cpu nor --gpu was given.
if [[ "$torch" == auto ]]; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        torch=gpu
        echo "NVIDIA GPU found ($(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)): CUDA build of PyTorch."
    else
        torch=cpu
        echo "No NVIDIA GPU found: CPU build of PyTorch."
    fi
fi

extras=(--extra sim)
[[ -n "$torch" ]] && extras+=(--extra "torch-$torch")

export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv >/dev/null; then
    echo "Installing uv (https://docs.astral.sh/uv/) ..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi

if ! command -v just >/dev/null; then
    echo "Installing just ..."
    uv tool install rust-just
fi

uv sync "${extras[@]}"

echo
echo "Done. If 'uv' or 'just' is not found in a new shell, add ~/.local/bin to PATH."
echo "Next: 'just test', then 'just' to list tasks."
