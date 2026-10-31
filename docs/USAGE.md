[Back to the README](../README.md)

# Using Mojito

## Running the scripts

`./scripts/setup.sh` (or `uv sync --extra sim`) installs the `mojito` package editable into `.venv`. Run a script with
`uv run --extra sim python scripts/<topic>/<script>.py`, or with plain `python` after `source .venv/bin/activate`;
either works from any directory, because outputs go to paths built from the repository root.

Every results JSON a script writes goes through `mojito.results.write_result`, which adds a `provenance` block:

```json
"provenance": {"git_commit": "...", "git_dirty": false, "script": "scripts/gait/gait_replay.py", "argv": [],
               "python": "...", "numpy": "...", "mujoco": "...", "utc": "YYYY-MM-DDTHH:MM:SSZ"}
```

`git_dirty: true` means tracked files differed from that commit, so the commit alone does not reproduce the numbers.
Files committed before this was added have no block until their script is run again.

To regenerate everything in dependency order, `just results` (several hours; the steps and their times are in the
`justfile`): the weights, then evaluate, calibrate, make_demo and the body demos, then torque, stand and gait in
physics, the sweeps and the design sweeps, then `site_sweep.py` and `build_site.py`. `just results-fast` keeps the
current weights and skips the slow sweeps; `just results-from-weights` re-runs only what reads `weights/ik_leg.npz`.

## One leg

Link lengths are never constants inside the maths. Every function takes them
as an argument, so they can differ per leg and change at any time.

```python
import numpy as np
import mojito
from mojito import leg

net = mojito.load_model("weights/ik_leg.npz")
my_leg = np.array([0.042, 0.115, 0.128])          # abduction offset, upper, lower [m]
target = np.array([0.03, 0.05, -0.18])            # foot position in the shoulder frame [m]

q = net.predict_leg(target, my_leg)               # left leg: abduction, hip pitch, knee [rad]
q_right = net.predict_leg(target, my_leg, right=True)
print(leg.forward(q, my_leg))                     # where those angles put a left foot
print(leg.analytic_ik(target, my_leg))            # the exact answer, for comparison
```

## The whole body

```python
import numpy as np
import mojito
from mojito import body, gaits

robot = body.Robot(mojito.load_model("weights/ik_leg.npz"))   # or body.AnalyticSolver()
home = robot.stance()                                 # feet under the hips, body frame

# Posture: keep the feet where they are, move and tilt the body.
targets = body.apply_posture(home, offset=(0, 0, 0.02), roll=0.2, yaw=0.3)
q = robot.solve(targets)                              # (4, 3) joint angles, legs in FL FR RL RR order
print(robot.tracking_error(targets, q) * 1000)        # mm per foot
print(robot.reachable(targets))                       # inside the joint limits?

# Gait: foot targets for a velocity command.
t = np.linspace(0, 1, 100)
feet, contact, _ = gaits.foot_targets(gaits.TROT, t, home, velocity=(0.15, 0.0), yaw_rate=0.3)
q = robot.solve(feet)                                 # (100, 4, 3)

robot.lengths[0] = [0.041, 0.118, 0.125]              # give one leg different lengths
```

## NumPy or PyTorch

The leg network has two interchangeable backends. **NumPy is the default** and
needs nothing extra. PyTorch is optional (`./scripts/setup.sh --torch`, or `--cpu` / `--gpu`) and is imported
only when asked for.

| Scope | How |
|---|---|
| One command | `uv run --extra sim python scripts/learning/train.py --backend torch` |
| Whole shell session | `export MOJITO_BACKEND=torch` |
| The rest of a Python process | `mojito.set_backend("torch")` |
| One call | `mojito.load_model("weights/ik_leg.npz", backend="torch")` |

```python
net = mojito.load_model("weights/ik_leg.npz")                   # NumPy unless told otherwise
net = mojito.load_model("weights/ik_leg.npz", backend="torch")  # same file, PyTorch
new = mojito.make_model(backend="torch", device="cuda")         # untrained, on a GPU
```

- Both backends read and write the same weight files and start from identical
  weights for a given seed.
- `predict` takes and returns NumPy arrays on both, so the body, gait and demo
  code does not care which is active.
- Inside a larger PyTorch model, use `net.angles(target_position, lengths)` and
  `net.forward_kinematics(joint_angles, lengths)` (which is `net.config.forward`;
  `leg.forward` and `LimbSpec.forward` both accept tensors): tensors in, tensors
  out, differentiable.
- The tests check that the two backends agree on predictions, gradients and
  the path training takes. They run on both backends for every push.

## Training for a different leg

```bash
uv run --extra sim python scripts/learning/train.py --upper 0.15 --lower 0.14 --abduction 0.05 --tolerance 0.2
```

`--tolerance` is how far each length is varied in training (0.2 means ±20%).
Joint limits live in `JOINT_LIMITS` in `mojito/geometry/placeholders.py` (`leg.JOINT_LIMITS` re-exports
it; edit the placeholder, not leg.py, and regenerate `robots/` so the URDFs agree). After retraining,
re-run the other scripts so the results match the new weights.

## Robots from URDF

The robot is defined in a standard URDF, separate from the learning code, and
the network is generated from it. Nothing about the limb (joint count, axes,
limits, link lengths) is written in Python.

```
robots/quadruped/quadruped.urdf + .json   <- the definition (any tool can read the URDF)
        |  mojito.geometry.urdf     parse / write the URDF (pure Python, no NumPy)
        v
   RobotSpec / LimbSpec                  <- mojito.geometry.spec: the alignment layer
        |  mojito.kinematics  forward kinematics + Jacobian for any serial chain
        v                      (NumPy arrays or torch tensors, same code)
     IKNet(spec)                         <- inputs and outputs sized from the chain
```

```bash
uv run python -m mojito.geometry.spec robots/quadruped/quadruped.urdf --manifest robots/quadruped/quadruped.json   # what the model will see
uv run --extra sim python scripts/learning/train.py --urdf robots/quadruped/quadruped.urdf --manifest robots/quadruped/quadruped.json
uv run --extra sim python scripts/learning/calibrate.py --urdf robots/quadruped/quadruped.urdf --manifest robots/quadruped/quadruped.json --limb FR
```

```python
import mojito
from mojito import body, IKNet

robot = mojito.load_robot("robots/hexapod/hexapod.urdf", "robots/hexapod/hexapod.json")   # 6 legs, one network
net = IKNet.load("weights/ik_leg.npz", spec=robot.groups["leg"].canonical) # fails with a diff if they disagree
rig = body.URDFRobot(robot, net)
q = rig.solve(rig.stance())                                                # joint values per limb
```

**What the URDF decides.** Which joints each limb has and their axes and
limits (structure); how many free link lengths there are (one per link with a
non-zero origin translation, direction fixed); where each limb is mounted. The
free lengths are the network's "link length" inputs, so a longer leg in the
URDF needs no retraining, while a different joint limit does.

**What the manifest adds** (a small JSON file beside the URDF, so the URDF
stays a plain URDF): which links are limb tips, which limbs are mirror images
of which, group names, the length tolerance used in training, and an optional
`reach` that scales positions. With no manifest, every leaf link is a limb.

**Alignment.** Limbs with the same structure share one network (a "group").
A limb that is the mirror image of another joins its group, and the sign flips
that relate the two are found and verified numerically instead of assumed. The
limb description is saved inside the weight file with a signature, so loading
weights against a URDF raises `SpecMismatch` with a line for every difference
(joint limit, axis, number of lengths, kinematics). Weights from before this
layer load too, and are checked against the URDF the same way. Calibrated link
lengths go back into a copy of the URDF with `RobotSpec.write_back`.

**Limits.** Serial chains of revolute, continuous, prismatic and fixed joints
only: no branching limbs, closed loops, `<mimic>` or xacro. With more than 3
joints per limb the position-only IK is redundant, so the network learns one of
many valid solutions; resolving that is Stage 2. `evaluate.py` falls back to a
numeric solver where there is no closed form, and `make_demo.py` and the
side-view error map are for the 3-joint leg only. The gaits
(`gaits.py`) and `body.Robot` assume four legs in FL, FR, RL, RR order;
`body.URDFRobot` handles any limb layout (posture and solving, as in the
hexapod example) but has no gaits.
