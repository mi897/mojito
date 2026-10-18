# Mojito

Robots that learn to move by themselves: a first idea of the task from
simulation, then refinement from real interaction. First robot: a quadruped
with 12 degrees of freedom, 3 per leg (abduction, hip pitch, knee), grippy
pads for feet.

Read `docs/NAMING.md` before naming anything (variables, arguments, functions) or
writing code that implements an equation. Read `docs/PLAN.md` for the staged roadmap and what is done, and
`docs/STAGE2.md` for the current stage (MuJoCo, servo model, work packages). Read `README.md`
for the overview, `docs/RESULTS.md` and `docs/PHYSICS.md` for results, `docs/USAGE.md` for usage. Keep both current when the work changes them.

## Working rules

- **Never commit to `main`.** Do all work on a separate branch named for what
  is being worked on, e.g. `stage-2-mujoco-model`, `fix/crawl-sway-speed`,
  `docs/claude-guides`. One topic per branch. Push the branch; open a pull
  request only when asked.
- If a branch builds on another unmerged branch, say so when reporting.
- Run the tests before every commit: `python -m unittest discover tests`.
- New maths gets a test that checks it against something independent:
  finite differences, the closed-form IK, or a conservation property (for
  example "grounded feet do not move in the world").
- Numbers in the README, the plan or a reply must come from a run, normally
  `results/*.json`. Say what was measured and on what. Never round a result
  in the flattering direction.
- State plainly what a result does not show. Stages 0 and 1 are pure
  geometry: no physics, so nothing about balance, torque or grip.
- All robot dimensions and joint limits are **placeholders** until the owner
  supplies real ones. Do not present them as the robot's.

## Layout

```
mojito/                the library, in four subpackages (each holds at most about a dozen files)
  geometry/            urdf.py (read and write URDF, no NumPy), spec.py (URDF + manifest -> LimbSpec / RobotSpec),
                       kinematics.py (FK, Jacobian, numeric IK for any serial chain), robots.py (generators for robots/)
  robot/               leg.py (3-joint leg, closed-form IK reference), body.py (legs on a body: frames, posture, per-leg solving),
                       gaits.py (trot, crawl, stability margin), statics.py (static joint torques, mass bound)
  learning/            model.py (IK network, NumPy, Adam), torch_model.py (PyTorch backend), backend.py (make_model, load_model)
  sim/                 mjcf.py (MuJoCo model from a URDF), actuators.py (servo model), sensors.py (joint sensor, IMU),
                       stand.py, gait_replay.py, design_sweep.py, gait_plots.py, live_sim.py (need mujoco, except actuators and sensors)
robots/                one folder per robot (quadruped/, hexapod/, planar/), each with its .urdf and .json manifest
site/                  website: pages, assets/mojito.js (JS port of leg, body, gaits, statics), charts, viewer, playground; data/ and media/ are generated
scripts/               entry points, one folder per topic
  learning/            train, evaluate, calibrate, make_demo
  body/                make_body_demos, torque_envelope
  stand/               stand_test, sit_to_stand, make_stand_demo
  design/              design_sweep, design_sweep_draws, design_tune, design_sweep_tuned and a plot_ for each sweep (stand and walk 11 designs at three masses: one run, 10 draws, or gaits re-tuned per design; results/design/)
  gait/                gait_replay, gait_speed_diagnosis, pace_diagnosis, make_gait_replay_demo, make_gait_iso_demos, try_gait, live_gait (+ live_gait.html)
  gait/sweeps/         trot_period_sweep, trot_period_robustness, trot_1p7_parameter_sweep, trot_1p7_one_at_a_time, trot_1p7_kp_kd_grid, trot_1p7_combined
                       and a plot_<name> for each (outputs in results/gait/sweeps/{period,parameters,servo_gains,combined}/)
  website/             build_site, serve_site, site_sweep, make_og_image, make_favicon
tests/                 unittest suites, mirroring the layout: geometry/, robot/, learning/, sim/, website/
weights/ik_leg.npz     the trained leg network
results/               generated, committed: learning/, body/, stand/, design/, gait/ (json metrics, png plots, gif animations); gait/sweeps/ has period/, parameters/, servo_gains/, combined/
docs/NAMING.md         naming and equation-commenting conventions
docs/USAGE.md  RESULTS.md  PHYSICS.md   usage, geometry results, physics results
docs/PLAN.md           roadmap;  docs/STAGE2.md  current stage;  docs/demo.html  interactive leg demo (generated)
```

## Keeping folders small

- **A folder holds at most about 12 files.** When one grows past that, split it by topic into subfolders
  *in the same change*: do not wait. Group by what the files are about, not by file type, and name the folder for the
  topic (`sim/`, `learning/`), not `misc/` or `utils/`. Never name a folder after a standard-library module (`site`, `test`).
- Mirror the grouping across `mojito/`, `scripts/`, `tests/` and `results/` so a topic is in the same-named folder in each.
- A script writes its outputs to the matching `results/<topic>/` folder. File names stay unique across `results/`
  (the website flattens them into `site/media/`).
- When you move a file, update every path that names it (imports, docs, `CLAUDE.md` layout, README, workflows, site pages) and run the tests.
- New packages need an `__init__.py`; new test folders too, or `unittest discover` skips them.

## Commands

Setup is `./scripts/setup.sh` (installs uv and just, then `uv sync`); dependencies are in `pyproject.toml`, pinned in
`uv.lock`. Tasks live in the `justfile` (`just` lists them). Run any script as `uv run --extra sim python <script>`.

```bash
python -m unittest discover tests          # all tests (just test); PyTorch ones skip if torch is absent
python scripts/learning/train.py                    # 40000 steps (about 12 min on 2 CPU cores, not re-timed), writes weights/ik_leg.npz
python scripts/learning/evaluate.py                 # results/learning/metrics.json and plots
python scripts/learning/calibrate.py                # results/learning/calibration.json
python scripts/learning/make_demo.py                # results/learning/step_demo.gif, docs/demo.html
python scripts/body/torque_envelope.py          # results/body/torque_envelope.json, static mass bound
python scripts/stand/stand_test.py               # results/stand/stand_test.json (needs mujoco)
python scripts/stand/sit_to_stand.py             # results/stand/sit_to_stand.json and .png (needs mujoco)
python scripts/design/design_sweep.py         # results/design/design_sweep.json (needs mujoco, about 2.5 min on 4 cores); then plot_design_sweep.py; design_sweep_draws.py is the same with 10 mass and latency draws per cell (about 25 min); design_tune.py (grid search of gait settings per design, about 35 min) then design_sweep_tuned.py (about 20 min)
python scripts/stand/make_stand_demo.py          # results/stand/stand_demo.gif, sit_to_stand_demo.gif (needs mujoco)
python scripts/gait/gait_replay.py              # results/gait/gait_replay.json (needs mujoco)
python scripts/gait/try_gait.py --help          # try other gait timings in physics, GIF in results/gait/try/
python scripts/gait/live_gait.py                # browser UI at 127.0.0.1:8765: sliders change the gait while it runs
python scripts/website/build_site.py               # website data from results/; then: python -m http.server -d site
python scripts/website/serve_site.py               # local-only server: the site plus live physics behind its playground (needs mujoco)
python scripts/website/site_sweep.py               # 144 MuJoCo runs for the site's physics lab -> results/gait/physics_sweep.json (needs mujoco)
python scripts/website/make_og_image.py           # site/assets/og-image.png, the 1200x630 social preview (needs google-chrome or chromium)
python scripts/website/make_favicon.py          # site/assets/favicon-32.png and apple-touch-icon.png from favicon.svg (needs google-chrome or chromium)
python scripts/body/make_body_demos.py          # body animations, results/body/body_metrics.json (add --no-gif for numbers only)
python scripts/learning/train.py --urdf robots/quadruped/quadruped.urdf --manifest robots/quadruped/quadruped.json   # network sized from a URDF
```

Every script takes `--backend numpy|torch`. The default is `$MOJITO_BACKEND`,
else NumPy.

## Conventions that are easy to get wrong

- **Units:** metres and radians in code. Millimetres and degrees only in
  printed output, plots and docs.
- **Leg frame:** origin at the shoulder, x forward, y lateral, z up. All leg
  maths is written for the left ("canonical") leg. A right leg is its mirror:
  flip y of the target going in, flip the abduction angle coming out
  (`leg.mirror_target`, `leg.mirror_angles`, `IKNet.predict_leg(right=True)`).
- **Body frame:** origin at the centre of the four shoulders, x forward,
  y left, z up. Rotation is `Rz(yaw) Ry(pitch) Rx(roll)`, body to world.
- **Leg order is always FL, FR, RL, RR** in `body.Robot` and `gaits`. Arrays
  are `(..., 4, 3)`. `body.URDFRobot` instead uses dicts keyed by limb name
  (the hexapod has FL, FR, ML, MR, RL, RR) and has no gaits.
- **The URDF is the source of truth for geometry.** Limb structure, axes,
  limits and the number of free lengths come from `robots/*/*.urdf` (plus the
  JSON manifest) via `mojito/geometry/spec.py`, not from Python constants. Weight files
  embed the limb spec; loading them against a different URDF raises
  `SpecMismatch`. The example URDFs in `robots/` come from the generators
  in `mojito/geometry/robots.py`, so change the generator and regenerate rather than
  hand-editing both. Joint-limit changes need retraining;
  link-length changes do not.
- **Link lengths are arguments, never module constants.** For the built-in
  3-joint leg, functions take `lengths` as `(..., 3)` = abduction offset,
  upper, lower (other URDF limbs have as many as the spec says), so a batch can
  hold different legs. `Robot.lengths` is `(4, 3)` and may be changed at any
  time. Keep it that way: training on varied lengths depends on it.
- **Joint angles:** `joint_angles = (abduction, hip pitch, knee)` (written `q` in
  maths and older docs). The knee bends one way
  only and stops short of straight (`leg.JOINT_LIMITS`), which makes IK
  unique. Changing the limits changes the problem; retrain and re-evaluate.
- **The training loss is foot position through forward kinematics**, never
  error against the sampled joint angles.
- **Closed-form IK (`leg.analytic_ik`) is the reference.** Judge the network
  against it and use `body.AnalyticSolver()` in tests of body and gait code so
  they do not depend on trained weights.
- **Backends:** both must keep the same public methods (`predict`,
  `predict_leg`, `train_step`, `loss_and_grads`, `state`, `save`, `load`,
  `to_json`) and the same `.npz` layout, with `W[i]` shaped (inputs, outputs).
  `predict` takes and returns NumPy arrays on both. A change to one backend
  needs the matching change to the other and a parity test in
  `tests/learning/test_backends.py`.

## Website rules

- The pages never hold a measured number: bind it with `data-bind` to `results/*.json` (checked by `tests/website/test_build_site.py`) or draw it from the JSON in a script.
- `site/assets/mojito.js` is a port of `mojito/robot/leg.py`, `body.py`, `gaits.py`, `statics.py`. Change one, change the other; `tests/website/test_site_js.py` compares them.
- `scripts/website/serve_site.py` is local only: it binds 127.0.0.1 and checks Host and Origin (tested in `tests/website/test_serve_site.py`). Do not add a public mode without asking the owner.
- Say in the page what a lab does not model. The kinematic lab is static (no dynamics); the physics lab replays stored runs, no interpolation.

## Generated files

`weights/ik_leg.npz`, everything in `results/` (including `physics_sweep.json`), and `docs/demo.html` are
outputs that are committed so the README renders. Regenerate them with the
scripts; do not edit them by hand. If the weights change, re-run evaluate,
calibrate, make_demo and make_body_demos and update every number quoted in
`README.md`, `docs/RESULTS.md`, `docs/PHYSICS.md` and `docs/PLAN.md` in the same branch. Edit
`docs/demo_template.html`, not `docs/demo.html`.

## Environment notes

- Dependencies: NumPy, SciPy, Matplotlib, Pillow. PyTorch and MuJoCo are optional (their tests skip if absent).
- Some cloud sessions cannot install packages. If PyTorch cannot be installed,
  do not claim the PyTorch path was run locally: push the branch and read the
  result of the `tests` GitHub Actions workflow, which runs both backends.
- Plots follow one palette (`BLUE`, `ORANGE`, `AQUA` at the top of the
  scripts), one y-axis per chart, text in neutral ink.

## Open questions for the owner

1. Real link lengths, abduction offset, body dimensions and joint limits. No
   robot exists; dimensions are to be explored in simulation (`docs/STAGE2.md`).
2. Actuators: 20 kg·cm hobby servos ("DSServo DS3218 20KG Digital Metal
   Servo"), supply adjustable 5–6 V, no position feedback, decided. Later: DC motors
   with position, velocity and current feedback. First robot: external joint-angle
   sensors, in line with the joint axis close to the motor. Sensors are custom Hall-effect boards. Learning is off board; control and logging go over the network to a laptop server. Open: datasheet figures (5–6 V is interpolated from listed 5 and 6.8 V points). Hall chip and magnet details are tabled. Servo figures in `docs/STAGE2.md`
   are unverified retailer numbers. Power and electronics: 2S 1500 mAh LiPo, regulator needed (battery is above the servo's range). Controller: ESP32 driving a PCA9685 over I2C, with an IMU wired directly to the ESP32 (model not chosen); no Raspberry Pi, no Blue Pill. One battery, separate servo and logic regulators. Open: PCA9685 board voltage rating against the 6 V rail (`docs/STAGE2.md`).
3. Measurement: simulation ground truth for now, external system later. Keep
   ground truth out of policy observations.
