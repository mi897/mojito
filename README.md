# Mojito

Robots that learn to move in their environment by themselves: a first idea of
the task from simulation, then refinement from real interaction.

The first robot is a quadruped with 12 degrees of freedom, 3 per leg. This
repository holds two stages so far. The full roadmap is in
[docs/PLAN.md](docs/PLAN.md).

- **Stage 0**: a neural network that learns inverse kinematics for one leg,
  with the link lengths as inputs so one network covers a family of slightly
  different legs.
- **Stage 1**: the whole body. Four legs driven by that one network, body
  posture control with the feet planted, and scripted trot and crawl gaits.

![Three leg sizes tracing the same step](results/step_demo.gif)

## Stage 0 results

Placeholder leg: 40 mm abduction offset, 120 mm upper, 120 mm lower (240 mm
reach). Each length varied independently by ±10% in training. Tested on
200,000 targets and leg sizes the network never saw.

| Foot position error | Median | 95th pct | 99th pct | Worst |
|---|---|---|---|---|
| Learned IK, told each leg's lengths | 0.17 mm | 0.84 mm | 1.57 mm | 7.4 mm |
| Learned IK, assuming every leg is nominal | 9.0 mm | 15.7 mm | 18.8 mm | 38 mm |
| Closed-form IK (ideal leg) | exact | exact | exact | exact |

- The network meets the Stage 0 targets (median under 1 mm, 95th percentile
  under 2 mm). Joint angles are within 0.04° of the exact solution at the
  median and 0.53° at the 95th percentile.
- Telling the network the lengths is what makes it work across legs: without
  that input the error is about 50 times larger.
- The worst cases sit on the edge of the workspace (see the error map below).
- Beyond the trained range accuracy falls off smoothly: at ±20% on all links
  the median is about 1.8 mm.
- The closed-form solution is exact and about 40 times faster in a batch. On an
  ideal leg there is no reason to prefer the network. Its value is that it can
  be adapted, which is the next point.

**Adapting to one specific leg.** `scripts/calibrate.py` simulates a leg whose
real lengths (+6%, −5%, +8%) and servo zero points (2°, −3°, 4°) are unknown.
From 30 foot measurements with 1 mm of noise it estimates all six numbers and
passes the lengths to the same network, with no retraining:

| | Median error | 95th pct |
|---|---|---|
| Before calibration | 16.1 mm | 18.6 mm |
| After calibration | 0.56 mm | 0.74 mm |

| | |
|---|---|
| ![](results/error_distribution.png) | ![](results/error_vs_length.png) |
| ![](results/error_map.png) | ![](results/training_curve.png) |

## Stage 1: whole body

Everything drawn in these animations comes from the network's joint angles
pushed through forward kinematics. Only the orange markers are the requested
foot positions. The side panels show the command, the network's output for one
leg, and the worst foot error at each instant.

**Twisting and turning in place.** The feet stay planted while the body is
commanded to roll, pitch, yaw and bob, then all at once.

![Body posture demo](results/posture_demo.gif)

**Trot** (diagonal pairs together, two feet down) and **crawl** (one foot up at
a time, body leaning away from it).

| | |
|---|---|
| ![](results/trot_forward.gif) | ![](results/trot_turn.gif) |
| ![](results/crawl_forward.gif) | |

| Demo | Worst foot error | Median | Closest joint to its limit | Feet down |
|---|---|---|---|---|
| Posture: roll ±18°, pitch ±14°, yaw ±22°, height ±30 mm | 0.22 mm | 0.11 mm | 15° | 4 |
| Trot forward, 0.15 m/s | 0.25 mm | 0.14 mm | 23° | 2 |
| Trot turning on the spot, 1.2 rad/s | 0.21 mm | 0.10 mm | 23° | 2 |
| Crawl forward, 0.04 m/s | 0.29 mm | 0.12 mm | 20° | 3 or 4 |

- Every requested foot position in all four demos is inside the joint limits,
  and the network's angles never differ from the exact solution by more than
  0.14°.
- Grounded feet do not slide: the gait generator holds each stance foot fixed
  in the world for any mix of forward, sideways and turning speed (tested to
  a nanometre with exact IK).
- In the crawl the body centre stays at least 38 mm inside the triangle of
  grounded feet.

**What this does not show.** These are geometric reference motions. There is
no physics, so nothing here says the robot would balance, that the motors are
strong enough, or that the feet would grip. The trot has only two feet down
and is balanced dynamically on a real robot, which geometry cannot check. The
crawl lean is also quick (the body shifts 50 mm in 0.2 s). Body dimensions are
placeholders: shoulders 240 mm apart front to back and 100 mm side to side,
standing 170 mm high.

```python
import numpy as np
import mojito
from mojito import body, gaits

robot = body.Robot(mojito.load_model("weights/ik_leg.npz"))  # or body.AnalyticSolver()
home = robot.stance()                                  # feet under the hips, body frame

# Posture: keep the feet where they are, tilt and twist the body.
targets = body.apply_posture(home, offset=(0, 0, 0.02), roll=0.2, yaw=0.3)
q = robot.solve(targets)                               # (4, 3) joint angles, FL FR RL RR
print(robot.tracking_error(targets, q) * 1000)         # mm per foot

# Gait: foot targets for a velocity command.
t = np.linspace(0, 1, 100)
feet, contact, _ = gaits.foot_targets(gaits.TROT, t, home, velocity=(0.15, 0.0), yaw_rate=0.3)
q = robot.solve(feet)                                  # (100, 4, 3)

robot.lengths[0] = [0.041, 0.118, 0.125]               # per-leg lengths, changeable at any time
```

## NumPy or PyTorch

The leg network has two interchangeable backends. **NumPy is the default** and
needs nothing extra. PyTorch is optional and is only imported when asked for.

| How | Example |
|---|---|
| One command | `python scripts/train.py --backend torch` (every script takes `--backend`) |
| Whole shell session | `export MOJITO_BACKEND=torch` |
| In code, for the process | `mojito.set_backend("torch")` |
| In code, for one call | `mojito.load_model("weights/ik_leg.npz", backend="torch")` |

```python
import mojito

net = mojito.load_model("weights/ik_leg.npz")                   # NumPy unless told otherwise
net = mojito.load_model("weights/ik_leg.npz", backend="torch")  # same file, PyTorch
new = mojito.make_model(backend="torch", device="cuda")         # untrained, on a GPU
```

- Both backends use the same weight files, so a network trained with one runs
  under the other, and both start from identical weights for a given seed.
- `predict` takes and returns NumPy arrays on both, so the body, gait and demo
  code does not care which is active.
- For building larger PyTorch models, `net.angles(p, lengths)` and
  `mojito.torch_model.forward_kinematics(q, lengths)` take tensors and are
  differentiable.
- PyTorch uses autograd where NumPy uses a hand-derived Jacobian. The tests
  check the two give the same predictions, gradients and training path.

## Run it

Needs Python 3.10+ with NumPy, SciPy, Matplotlib and Pillow
(`pip install -r requirements.txt`). No GPU needed. PyTorch is optional
(`pip install torch`).

```bash
python -m unittest discover tests     # kinematics and gradient checks
python scripts/train.py               # about 12 minutes on 2 CPU cores
python scripts/evaluate.py            # results/metrics.json and the plots
python scripts/calibrate.py           # adapt to a leg with unknown lengths
python scripts/make_demo.py           # results/step_demo.gif and docs/demo.html
python scripts/make_body_demos.py     # posture, trot and crawl animations and body_metrics.json
```

Open `docs/demo.html` in a browser to drag the foot target and change the link
lengths live.

### Changing the leg

Lengths are never hard-coded inside the maths. Every function takes them as an
argument, one row per leg:

```python
import numpy as np
import mojito
from mojito import leg

net = mojito.load_model("weights/ik_leg.npz")
my_leg = np.array([0.042, 0.115, 0.128])        # abduction offset, upper, lower [m]
target = np.array([0.03, 0.05, -0.18])          # foot position in the shoulder frame [m]

q = net.predict_leg(target, my_leg)             # left leg
q = net.predict_leg(target, my_leg, right=True) # right leg
print(leg.forward(q, my_leg))                   # where that actually puts a left foot
```

To train for a different nominal leg or a wider spread:

```bash
python scripts/train.py --upper 0.15 --lower 0.14 --abduction 0.05 --tolerance 0.2
```

Joint limits live in `JOINT_LIMITS` in `mojito/leg.py`.

## How it works

- **Frame.** Origin at the shoulder; x forward, y out to the side, z up.
- **Training data.** Random joint angles are pushed through forward kinematics
  to get foot positions, so every target is reachable.
- **Loss.** The predicted angles are pushed through forward kinematics and
  compared with the requested foot position. The network is never shown the
  angles that generated the target.
- **Inputs.** Foot position (3) and the leg's three lengths. **Outputs.**
  Three angles, squashed to lie inside the joint limits.
- **One model for four legs.** Right legs are mirror images of left legs.
- **Uniqueness.** The knee bends one way only and stops 0.3 rad short of
  straight, so each target has exactly one answer.

The network is 3 hidden layers of 128 units (about 34,000 weights). The NumPy
backend uses hand-derived gradients that the tests check against finite
differences.

## Layout

```
mojito/leg.py       forward kinematics, Jacobian, closed-form IK, sampling
mojito/model.py     the network (NumPy backend), shared base class, Adam
mojito/torch_model.py  the network (PyTorch backend)
mojito/backend.py   backend switch: make_model, load_model, set_backend
mojito/body.py      four legs on a body: frames, posture, per-leg solving
mojito/gaits.py     trot and crawl foot trajectories, crawl lean, stability
scripts/            train, evaluate, calibrate, make_demo, make_body_demos
tests/              unit tests
weights/ik_leg.npz  trained network
results/            metrics, plots, animation
docs/               plan and interactive demo
```
