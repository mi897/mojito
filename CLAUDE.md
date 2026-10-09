# Mojito

Robots that learn to move by themselves: a first idea of the task from
simulation, then refinement from real interaction. First robot: a quadruped
with 12 degrees of freedom, 3 per leg (abduction, hip pitch, knee), grippy
pads for feet.

Read `docs/PLAN.md` for the staged roadmap and what is done. Read `README.md`
for results and usage. Keep both current when the work changes them.

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
mojito/urdf.py         read and write URDF (no NumPy)
mojito/spec.py         URDF + manifest -> LimbSpec / RobotSpec; groups, mirrors, checks, write-back
mojito/kinematics.py   forward kinematics, Jacobian, numeric IK for any serial chain
mojito/robots.py       generators for the example URDFs in robots/
mojito/leg.py          the 3-joint leg: closed-form IK (reference), sampling
mojito/model.py        IK network, NumPy backend; shared base class; Adam
mojito/torch_model.py  IK network, PyTorch backend
mojito/backend.py      backend switch: make_model, load_model, set_backend
mojito/body.py         legs on a body: frames, posture, per-leg solving (Robot, URDFRobot)
mojito/gaits.py        trot and crawl foot trajectories, crawl lean, stability margin
robots/                example URDFs + manifests: quadruped, hexapod, planar
scripts/               train, evaluate, calibrate, make_demo, make_body_demos
tests/                 unittest suites (no pytest dependency)
weights/ik_leg.npz     the trained leg network
results/               metrics (json), plots (png), animations (gif)
docs/PLAN.md           roadmap;  docs/demo.html  interactive leg demo (generated)
```

## Commands

```bash
python -m unittest discover tests          # all tests; PyTorch ones skip if torch is absent
python scripts/train.py                    # about 12 min on 2 CPU cores, writes weights/ik_leg.npz
python scripts/evaluate.py                 # results/metrics.json and plots
python scripts/calibrate.py                # results/calibration.json
python scripts/make_demo.py                # results/step_demo.gif, docs/demo.html
python scripts/make_body_demos.py          # body animations, results/body_metrics.json (add --no-gif for numbers only)
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
- **Leg order is always FL, FR, RL, RR.** Arrays are `(..., 4, 3)`.
- **Link lengths are arguments, never module constants.** Functions take
  `lengths` as `(..., 3)` = abduction offset, upper, lower, so a batch can
  hold different legs. `Robot.lengths` is `(4, 3)` and may be changed at any
  time. Keep it that way: training on varied lengths depends on it.
- **Joint angles:** `q = (abduction, hip pitch, knee)`. The knee bends one way
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
  `tests/test_backends.py`.

## Generated files

`weights/ik_leg.npz`, everything in `results/`, and `docs/demo.html` are
outputs that are committed so the README renders. Regenerate them with the
scripts; do not edit them by hand. If the weights change, re-run evaluate,
calibrate, make_demo and make_body_demos and update every number quoted in
`README.md` and `docs/PLAN.md` in the same branch. Edit
`docs/demo_template.html`, not `docs/demo.html`.

## Environment notes

- Dependencies: NumPy, SciPy, Matplotlib, Pillow. PyTorch is optional.
- Some cloud sessions cannot install packages. If PyTorch cannot be installed,
  do not claim the PyTorch path was run locally: push the branch and read the
  result of the `tests` GitHub Actions workflow, which runs both backends.
- Plots follow one palette (`BLUE`, `ORANGE`, `AQUA` at the top of the
  scripts), one y-axis per chart, text in neutral ink.

## Open questions for the owner

1. Real link lengths, abduction offset, body dimensions and joint limits.
2. Actuators: hobby servos or torque-controlled motors.
3. How foot or body position will be measured on the real robot.
