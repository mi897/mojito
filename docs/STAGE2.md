# Stage 2 — physics simulation (MuJoCo)

Branch: `stage-2-mujoco-model`. Status: **planned, nothing built yet.**
`docs/PLAN.md` has the roadmap; this file holds the detail for this stage.

## Why this stage exists

No physical robot exists yet. The owner wants to try different dimensions in
simulation before finalising a design. So the first product of this stage is a
**design-exploration tool**: given a set of dimensions, does the robot stand,
walk, and stay inside the servos' limits? Learning locomotion (RL) comes after
the model is trusted.

## Decisions already made

| Question | Answer | Consequence |
|---|---|---|
| Real dimensions | None yet; to be chosen from simulation | Dimensions are sweep parameters, never constants. The URDF generator in `mojito/robots.py` stays the single source. |
| Actuators | DS3218-class 20 kg·cm digital hobby servos | Model each joint as a torque-limited, speed-limited position servo with a delay. See "Servo model". |
| Measurement | Simulation ground truth for now; an external measuring system may come later | Policies must not observe anything the real robot could not measure. |

## Foot and base positions in simulation

Put a `site` at each foot tip and at the body-frame origin (the centre of the
four shoulders, not the base body's own origin). MuJoCo gives world positions
in `data.site_xpos`; the base orientation is `data.xmat`. The relative foot
position is `R_baseᵀ · (foot_world − body_origin_world)`, in the same body
frame as `mojito/body.py`.

Use this ground truth for rewards, evaluation and foot-slip measurement. Do
**not** put it in the policy's observation: the real robot will only have
joint readings (if its servos report them) and an IMU. Anything the policy
sees in simulation must exist on hardware.

## Servo model

The owner's servo is listed as "DSServo DS3218 20KG Digital Metal Servo", which
reads as the standard DS3218 (the "MG" rows below), not the PRO. The PRO rows
are kept for comparison only. The figures below come from
retailer listings and a third-party database found by web search; **no
manufacturer datasheet was found, the retailer page the owner linked
(store.roboticsbd.com) could not be opened from the build sandbox, and the
listings disagree.** Treat every number as unverified until the owner checks the
datasheet that shipped with the servo or measures it.

| Quantity | Listed value | Notes |
|---|---|---|
| Operating voltage | 4.8–6.8 V | Typical listing. |
| Stall torque, DS3218 (MG) | 19 kg·cm at 5 V, 21.5 kg·cm at 6.8 V | The "20 kg" in the name is a rounded marketing figure. |
| Stall torque, DS3218 PRO | 21 kg·cm at 5 V, 23.5 kg·cm at 6.8 V | A different variant. Which one is being bought is unknown. |
| No-load speed (MG) | 0.16 s/60° at 5 V, 0.14 s/60° at 6.8 V | About 6.5 and 7.5 rad/s. |
| Stall current | 1.8 A at 5 V, 2.6 A at 6.8 V (MG listing) | A third-party database gives 1.8–2.2 A for a different brand's DS3218. |
| Pulse range | 500–2500 µs, neutral 1500 µs, dead band 3 µs | Maps to the 270° version only when set up for it; 180° is the common default. |
| Size, weight | 40 × 20 × 40.5 mm, 60 g | |
| Position feedback | **Not stated in any listing; assume none** | Standard hobby servos accept commands only. |

Conversion used: 1 kg·cm = 0.0980665 N·m. So 19 kg·cm = 1.86 N·m and
21.5 kg·cm = 2.11 N·m.

**Model for the first version:**

- MuJoCo `position` actuator per joint, with `forcerange` set to a
  **parameter**; default 1.86 N·m (the lowest listed stall torque), since
  sustained loads should not assume the best case.
- Joint velocity limit from the no-load speed, also a parameter.
- Command latency and a servo position resolution (dead band) as parameters.
- The supply voltage, and whether the real part is the MG or the PRO, change
  these numbers. They are inputs, not constants, and a sweep over them is part
  of the plan.

**What this means for the design.** The position the servo is commanded to is
not necessarily the position it reaches. If the servo reports nothing back,
the real robot knows its pose only by assuming commanded equals actual, which
fails under load. That matters for Stage 3, and the simulation should be able
to show how large the commanded-versus-actual gap becomes at the torque limit.

## Work packages

Each is one branch-sized step. Tick them off here as they land.

1. [ ] **MJCF from the URDF.** Generate a MuJoCo model from the same
   parameters as `robots/quadruped.urdf`, with foot and body-frame sites, a
   floor with friction, and placeholder inertias derived from the dimensions.
   Test: with the model in a fixed pose, foot positions from MuJoCo equal
   `mojito/kinematics.py` to numerical precision, for random lengths and angles
   (left and right legs, and several body poses).
2. [ ] **Servo model.** Torque limit, speed limit, latency, resolution, as
   parameters. Test against a closed form: a servo holding a known mass at a
   known lever arm should settle at the expected deflection when the torque is
   below the limit, and saturate when it is above.
3. [ ] **Stand test.** Hold the Stage 1 standing posture from the IK layer
   under gravity. Report per joint torque against the limit. Test the torque
   estimate against the static calculation for a symmetric stance.
4. [ ] **Replay Stage 1 gaits in physics.** Feed the trot and crawl foot
   trajectories through the IK layer into the servo model. Report whether the
   body stays up, foot slip, and peak torque. This tests the known weaknesses
   from Stage 1: the trot's reliance on dynamic balance and the quick crawl
   lean.
5. [ ] **Design sweep.** Vary link lengths, body size and mass; report which
   designs stand, which gaits survive, and the torque margin. Output to
   `results/` with the sweep ranges and mass assumptions stated.
6. [ ] **Locomotion learning.** Policy acts in foot space through the IK layer.
   Randomise mass, friction, servo strength and latency. Reintroduce the
   current-angle input and smoothness penalties. Only after 1–5.

## What this stage will and will not show

- Masses and inertias are guesses until a real design exists, so a torque
  margin from the sweep is a comparison between designs, not a prediction of
  a built robot.
- Foot friction ("grippy pads") needs a measured coefficient before slip
  results mean anything. Until then it is a swept parameter.
- Simulated servos are an idealisation. Gear backlash, voltage sag under load
  and the servo's own control loop are not modelled at first.

## Environment

MuJoCo (`pip install mujoco`) installs in the cloud session; version 3.15.0
was installed when this plan was written. It is optional for the rest of the
repo, so tests that need it must skip when it is absent, as the PyTorch tests
do.

## Open items for the owner

1. Supply voltage, and the datasheet figures for the DS3218 20KG (listing title
   suggests the standard variant, not the PRO). Please check the datasheet or the box.
2. Does the servo report its position back? If not, say so; it changes what the
   policy may observe.
3. A rough target for total mass and body size, so the sweep has a sensible
   range.
