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
    {{py}} scripts/stand/stand_test.py
    {{py}} scripts/stand/sit_to_stand.py

# Replay trot and crawl in physics
gait:
    {{py}} scripts/gait/gait_replay.py

# Try other gait timings: just try-gait --help
try-gait *args:
    {{py}} scripts/gait/try_gait.py {{args}}

# Live gait explorer at http://127.0.0.1:8765
live *args:
    {{py}} scripts/gait/live_gait.py {{args}}

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

# Order: weights, then what reads them, then the physics, then the site. The training is about 12 min, the kp x kd grid about
# two hours, the combined sweep about an hour, the design draws, tuning and tuned sweep about 80 min together (times from the
# scripts' docstrings, not re-timed). `just results-fast` skips the steps marked slow.
# SLOW, several hours: regenerate every committed result in dependency order
results: train results-from-weights results-physics results-sweeps-slow results-design-slow results-site

# Every result except the slow steps (training, kp x kd grid, combined sweep, design draws, tuning and tuned sweep)
results-fast: results-from-weights results-physics results-site

# Everything computed from weights/ik_leg.npz: evaluate, calibrate, the leg demos, the body demos
results-from-weights:
    {{py}} scripts/learning/evaluate.py
    {{py}} scripts/learning/calibrate.py
    {{py}} scripts/learning/make_demo.py
    {{py}} scripts/body/make_body_demos.py

# Static torque, stand, gait replays and the faster sweeps (minutes each)
results-physics:
    {{py}} scripts/body/torque_envelope.py
    {{py}} scripts/stand/stand_test.py
    {{py}} scripts/stand/sit_to_stand.py
    {{py}} scripts/stand/make_stand_demo.py
    {{py}} scripts/gait/gait_replay.py
    {{py}} scripts/gait/make_gait_replay_demo.py
    {{py}} scripts/gait/make_gait_iso_demos.py
    {{py}} scripts/gait/gait_speed_diagnosis.py
    {{py}} scripts/gait/pace_diagnosis.py
    {{py}} scripts/gait/sweeps/trot_period_sweep.py
    {{py}} scripts/gait/sweeps/trot_period_robustness.py
    {{py}} scripts/gait/sweeps/trot_1p7_parameter_sweep.py
    {{py}} scripts/gait/sweeps/plot_trot_1p7_parameter_sweep.py
    {{py}} scripts/gait/sweeps/trot_1p7_one_at_a_time.py
    {{py}} scripts/gait/sweeps/plot_trot_1p7_one_at_a_time.py
    {{py}} scripts/design/design_sweep.py
    {{py}} scripts/design/plot_design_sweep.py

# SLOW: the kp x kd grid (about two hours) and the combined sweep (about an hour), each with its plot
results-sweeps-slow:
    {{py}} scripts/gait/sweeps/trot_1p7_kp_kd_grid.py
    {{py}} scripts/gait/sweeps/plot_trot_1p7_kp_kd_grid.py
    {{py}} scripts/gait/sweeps/trot_1p7_combined.py
    {{py}} scripts/gait/sweeps/plot_trot_1p7_combined.py

# SLOW: design sweep with draws (about 25 min), per-design tuning (about 35 min), the tuned sweep (about 20 min)
results-design-slow:
    {{py}} scripts/design/design_sweep_draws.py
    {{py}} scripts/design/plot_design_sweep_draws.py
    {{py}} scripts/design/design_tune.py
    {{py}} scripts/design/design_sweep_tuned.py
    {{py}} scripts/design/plot_design_sweep_tuned.py

# The physics lab's 144 runs (about 2 min), then the website data from everything above
results-site:
    {{py}} scripts/website/site_sweep.py
    {{py}} scripts/website/build_site.py
