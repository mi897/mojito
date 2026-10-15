# Mojito tasks. `just` lists them; `just <name>` runs one. Setup: ./scripts/setup.sh

set shell := ["bash", "-euo", "pipefail", "-c"]

# Run everything through uv so the project's .venv is always used.
py := "uv run --extra sim python"

default:
    @just --list

# Create .venv and install dependencies; flags go to scripts/setup.sh: just setup --torch | --cpu | --gpu
setup *flags:
    ./scripts/setup.sh {{flags}}

# All unit tests (MuJoCo and PyTorch ones skip if absent)
test:
    {{py}} -m unittest discover tests

# Train the leg network (about 12 min on 2 CPU cores); extra flags pass through
train *args:
    {{py}} scripts/learning/train.py {{args}}

# Evaluate the network against closed-form IK, then calibrate
evaluate:
    {{py}} scripts/learning/evaluate.py
    {{py}} scripts/learning/calibrate.py

# Body animations and metrics (add --no-gif for numbers only)
body *args:
    {{py}} scripts/body/make_body_demos.py {{args}}

# Static torque envelope and mass bound
torque:
    {{py}} scripts/body/torque_envelope.py

# Stand and sit-to-stand in MuJoCo
stand:
    {{py}} scripts/sim/stand_test.py
    {{py}} scripts/sim/sit_to_stand.py

# Replay trot and crawl in physics
gait:
    {{py}} scripts/sim/gait_replay.py

# Try other gait timings: just try-gait --help
try-gait *args:
    {{py}} scripts/sim/try_gait.py {{args}}

# Live gait explorer at http://127.0.0.1:8765
live *args:
    {{py}} scripts/sim/live_gait.py {{args}}

# Build the website data from results/
site-build:
    {{py}} scripts/website/build_site.py

# Serve the built site statically at http://127.0.0.1:8000
site-serve: site-build
    {{py}} -m http.server -d site 8000 --bind 127.0.0.1

# Serve the site with live physics behind its playground (local only)
site-live:
    {{py}} scripts/website/serve_site.py

# 144 MuJoCo runs for the site's physics lab (slow)
site-sweep:
    {{py}} scripts/website/site_sweep.py
