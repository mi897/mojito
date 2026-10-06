# Plan

The goal is a quadruped that learns to move on its own: a first idea of the task
from simulation, then refinement from real interaction, with occasional return
trips to simulation to prepare for specific tasks.

The robot has 12 degrees of freedom, 3 per leg: abduction and hip pitch at the
shoulder, and a knee. Feet are grippy pads with no toes.

## Stage 0 — learned leg inverse kinematics (this proof of concept)

**Question it answers:** can one small network turn "put the foot here" into
joint angles, for a whole family of slightly different legs, accurately enough
to be useful?

| Decision | Choice | Why |
|---|---|---|
| Simulator | Pure geometry in NumPy | IK has no dynamics. A physics engine adds nothing yet. |
| Training targets | Random joint angles pushed through forward kinematics | Every target is reachable by construction. |
| Loss | Distance between the requested foot position and where the predicted angles put the foot | Never regresses on angles, so it cannot average two valid poses into an invalid one. |
| Leg lengths | Network inputs, randomised ±10% per link in training | One network covers a family of legs. Adapting to a real leg means estimating a few numbers, not retraining. |
| Models | One, shared by all four legs | Legs are nominally identical. Right legs are mirror images: flip y in, flip abduction out. Per-leg differences enter through the length inputs. |
| Outputs | Squashed into the joint limits | Every prediction is a legal pose. |
| Ambiguity | Knee bends one way only, stops short of straight | Makes the solution unique and keeps clear of the full-extension singularity. |
| Sampling | Poses resampled in proportion to the volume the foot sweeps | Evens out coverage of the workspace. The effect is mild with the current joint limits. |
| Baseline | Closed-form IK | Exact for an ideal leg, so it is the yardstick. |

**Left out on purpose.** Feeding the current joint angles in, with a penalty on
joint movement, was in the earlier sketch. With the knee restricted the
solution is already unique, so that input would carry no information here. It
comes back in Stage 2, where there is real redundancy to resolve.

**Done when:**

- median foot error under 1 mm and 95th percentile under 2 mm on unseen targets
  and unseen leg lengths (the leg reaches 240 mm);
- error stays at that level across the whole ±10% range of lengths;
- a simulated "real" leg with unknown lengths and servo offsets can be brought
  back to that accuracy from a few dozen noisy foot measurements, with no
  retraining.

**Deliverables:** tested kinematics, training and evaluation scripts, result
plots, an animated demo and an interactive page.

## Stage 1 — whole body, still geometry

- Body-frame targets: place four feet given a body pose (height, roll, pitch, yaw).
- Scripted gaits (trot, crawl) driven through the learned IK, to produce
  reference motions and a sanity check on workspace and limits.
- Port the model to PyTorch so it can sit inside larger learned controllers.

## Stage 2 — physics simulation

- MuJoCo model built from the same length parameters, so a leg-length change
  propagates to both the simulator and the IK input.
- Reinforcement learning for locomotion, with the policy acting in foot space
  through the IK layer. Randomise masses, friction, motor strength and latency.
- Reintroduce current-angle input and smooth-motion penalties here.

## Stage 3 — the real robot

- **Decide the measurement source first.** Everything learned from real
  interaction needs to know where the feet or body actually went. Options, in
  rising cost: stance feet constrained to a flat floor plus an IMU; a single
  camera with markers; motion capture.
- Calibrate each leg (lengths, servo zero offsets) with the Stage 0 procedure.
- If calibration leaves a systematic error, add a small learned correction per
  leg on top of the shared model.

## Stage 4 — the loop

- Log real experience, fit the simulator's parameters to it, retrain or
  fine-tune in the updated simulator, redeploy.
- Generate new simulated environments aimed at whatever the real robot is
  currently bad at.

## Open questions

1. Real link lengths, abduction offset and joint limits (all placeholders now).
2. Actuators: hobby servos or torque-controlled motors? This decides how much
   of Stage 2 is about position tracking versus force.
3. How will foot position be measured on the real robot?
