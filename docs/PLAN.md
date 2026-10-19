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

**Status: built.**

- Body-frame foot targets for a commanded body pose (height, shift, roll,
  pitch, yaw) with the feet planted. One shared leg network drives all four
  legs; each leg can have its own lengths.
- Scripted trot and crawl from a velocity command (forward, sideways, yaw
  rate). Stance feet are held exactly fixed in the world. The crawl leans the
  body away from the lifted foot and stays statically stable.
- Across the posture, trot and crawl demos the network's worst foot error is
  0.29 mm and every target is inside the joint limits.
- The leg network runs on NumPy (default) or PyTorch behind one switch, with
  shared weight files.

**Known gaps carried forward:** no physics, so balance, motor effort and grip
are unchecked; body dimensions are placeholders; the crawl lean is fast.

## Robot description — URDF as the single source of truth (done)

The robot's geometry lives in a URDF, not in Python, so other tools and languages
can read it. `mojito/geometry/urdf.py` parses it, `mojito/geometry/spec.py` derives the network's
description from it (and checks saved weights against it), `mojito/geometry/kinematics.py`
runs any serial chain with NumPy or torch, and `IKNet` is sized from the chain.
Any number of limbs, any number of joints per limb; limbs with the same structure
share a network, mirrored limbs reuse it with verified sign flips.

Not done yet: gaits for robots other than the four-legged one (`gaits.py` still
assumes FL/FR/RL/RR), a JavaScript port of the spec layer for the browser demo, and
resolving redundancy for limbs with more than 3 joints (Stage 2).

## Stage 2 — physics simulation

**Status: in progress, in detail in `docs/STAGE2.md`.** Work packages 0 (static mass bound) and 1 (MuJoCo model generated from the URDF, `mojito/sim/mjcf.py`, foot positions tested against `mojito/geometry/kinematics.py`-based `body.Robot` to 1e-9 m) and 2 (servo model in `mojito/sim/actuators.py`, joint-sensor and IMU models in `mojito/sim/sensors.py`, tested against closed-form holding deflection) and 3 (stand test and sit-to-stand test: `mojito/sim/stand.py`, `results/stand/stand_test.json`, `results/stand/sit_to_stand.json`) and 4 (trot and crawl replayed in physics: `mojito/sim/gait_replay.py`, `results/gait/gait_replay.json`) and 5 (design sweep over leg lengths, body size and mass: `mojito/sim/design_sweep.py`, `results/design/design_sweep.json`) are done; the mechanical fit check is next. Work package 4 now uses a 1.7 s trot period: the Stage 1 0.5 s trot was not optimal in physics, and 1.7 s was chosen from a period sweep and checked against other gains, masses and swing heights (`docs/STAGE2.md`, "Longer trot periods"). The trot default is now tuned further: 22 mm swing in `gaits.TROT`, and for the physics trot a 0.16 m stance with kp 50, kd 0.9 (`gait_replay.settings_for`), from one-at-a-time, kp x kd and combined sweeps (`docs/STAGE2.md`); the crawl keeps its original settings.
No physical robot exists yet, so the stage opens as a design-exploration tool
(try dimensions in simulation before fixing the design), with learning after it.
Actuators are 20 kg·cm hobby servos ("DSServo DS3218 20KG", supply adjustable 5–6 V,
listed figures unverified, no position feedback), with external joint-angle sensors on
the first robot and DC motors with position, velocity and current feedback later (see
`docs/STAGE2.md`). A first static bound on total mass is done: with the Stage 2
design placeholders (150 mm links, 1.25× the Stage 0–1 ones so the servos are not
oversized), about 1.8 kg (trot) to 2.2 kg (crawl) at most at 5 V and half of stall
torque, before leg and servo mass and gait dynamics; 1.5 kg is the working target, set below that bound.
Stage 0–1 keeps its 120 mm placeholders until a design is chosen. Simulation ground truth (foot and base world positions) is used
for evaluation and rewards, not as policy input.

- MuJoCo model built from the same length parameters (done, placeholder masses; torque motors optional via `actuated=True`), so a leg-length change
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
   No robot exists; dimensions will be chosen by trying designs in simulation.
2. Actuators: decided as 20 kg·cm hobby servos (DSServo DS3218 20KG), supply
   adjustable 5–6 V, no position feedback; external joint-angle sensors for the
   first robot; DC motors with position, velocity and current feedback later. Still
   open: datasheet figures (5–6 V is interpolated from listed 5 V and 6.8 V points). The Hall chip and magnet details are tabled, and the controller is an ESP32 driving a PCA9685 (no Raspberry Pi), on one battery with separate servo and logic regulators. Sensors are custom Hall-effect boards; learning is off board, with control and logging over the network to a laptop.
3. Foot position on the real robot: simulation ground truth for now; an external
   measuring system can be built later if needed.
