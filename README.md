# Mojito

Robots that learn to move in their environment by themselves: a first idea of
the task from simulation, then refinement from real interaction.

The first robot is a quadruped with 12 degrees of freedom, 3 per leg. This
repository currently holds **Stage 0**: a neural network that learns inverse
kinematics for one leg, with the link lengths as inputs so one network covers
a family of slightly different legs. The full roadmap is in
[docs/PLAN.md](docs/PLAN.md).

![Three leg sizes tracing the same step](results/step_demo.gif)

## Results

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

## Run it

Needs Python 3.10+ with NumPy, SciPy, Matplotlib and Pillow
(`pip install -r requirements.txt`). No GPU, no PyTorch.

```bash
python -m unittest discover tests     # kinematics and gradient checks
python scripts/train.py               # about 12 minutes on 2 CPU cores
python scripts/evaluate.py            # results/metrics.json and the plots
python scripts/calibrate.py           # adapt to a leg with unknown lengths
python scripts/make_demo.py           # results/step_demo.gif and docs/demo.html
```

Open `docs/demo.html` in a browser to drag the foot target and change the link
lengths live.

### Changing the leg

Lengths are never hard-coded inside the maths. Every function takes them as an
argument, one row per leg:

```python
import numpy as np
from mojito import IKNet, leg

net = IKNet.load("weights/ik_leg.npz")
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

The network is 3 hidden layers of 128 units (about 34,000 weights), written in
NumPy with hand-derived gradients that the tests check against finite
differences.

## Layout

```
mojito/leg.py       forward kinematics, Jacobian, closed-form IK, sampling
mojito/model.py     the network, its gradients, Adam, save/load/export
scripts/            train, evaluate, calibrate, make_demo
tests/              unit tests
weights/ik_leg.npz  trained network
results/            metrics, plots, animation
docs/               plan and interactive demo
```
