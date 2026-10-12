# Stage 2 — physics simulation (MuJoCo)

Status: **planned; the first number (a static mass bound, below) is done.**
Earlier branch `stage-2-mujoco-model` (PR #9) added this plan; the mass bound is on
`stage-2-design-envelope`.
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
| Actuators, first robot | "DSServo DS3218 20KG Digital Metal Servo", supply adjustable 5–6.5 V | Torque-limited, speed-limited position servos whose limits depend on voltage. See "Servo model". |
| Actuators, later robots | DC motors reporting position, velocity and current | The actuator is an interface: command in, torque out, plus a list of what it reports. Swapping servos for motors must not change the policy code. |
| Joint feedback | The servo reports nothing. The first robot gets external joint-angle sensors fed back to the system | Simulation reports a **measured** angle (noise, quantisation, delay, offset), never the true one, as the policy's joint observation. |
| Measurement | Simulation ground truth for evaluation and rewards; external measuring system later | Policies must not observe anything the real robot could not measure. |
| Body shape | Longer than wide, about 2:1 | Read as 2:1 shoulder spacing (the body frame's definition), so the overall frame is a little larger. Treated as a design-sweep parameter. |

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
- Torque and speed are functions of supply voltage. The owner keeps the supply
  within 5–6.5 V, which sits inside the listed 5–6.8 V points, so no
  extrapolation is needed. The listings give two points only (5 V and 6.8 V);
  between them the model uses a straight line, which is an assumption (it gives
  21.1 kg·cm, 2.07 N·m, at 6.5 V). Outside 5–6.5 V the model refuses to run.
- Variant: the title "DS3218 20KG Digital Metal Servo" reads as the standard
  part, so the PRO figures are not used.

**What this means for the design.** The position the servo is commanded to is
not necessarily the position it reaches. The servo reports nothing back, so
without extra sensors the robot would assume commanded equals actual, which
fails under load. The simulation should show how large the commanded-versus-
actual gap becomes at the torque limit.

## Joint sensing and the actuator interface

- **First robot:** an external angle sensor on each joint (type not chosen;
  encoders or potentiometers are the usual options). Model it as true angle
  plus offset, noise, quantisation and delay, all parameters. The policy sees
  this measured angle. **Placement (decided):** packaged close to the motor, in
  line with the joint axis, so it does not interfere with anything else on the
  leg. Consequence for the model: a sensor on the joint axis reads the same
  angle the link has, so there is no linkage or belt between sensor and joint to
  model. One thing to check on the real part: whether it reads the servo's
  output shaft (after the gearbox, the link's true angle) or something before the
  gearbox (which would hide gear play). The DS3218's shaft is only accessible on
  the horn side, so this must be settled when the mount is designed. Sensor
  type (magnetic or optical, resolution) is not chosen; resolution is a model
  parameter until then.
- **Later robots:** DC motors with position, velocity and current feedback.
  Current is a torque estimate, so the policy could observe effort. Keep the
  policy's observation a named list built from whatever the actuator reports,
  so a motor model can replace the servo model without edits elsewhere.
- A servo with a built-in position loop and a motor under torque control are
  different control problems (position tracking versus force). The interface
  should allow both; only the servo one is built first.

## How heavy can it be? (static bound)

`scripts/torque_envelope.py` (maths in `mojito/statics.py`, test in
`tests/test_statics.py`) computes the joint torque needed to hold the body's
weight in three stances, with the placeholder leg (40, 120, 120 mm) standing
170 mm high, feet under the hips, and the placeholder body (240 × 100 mm
shoulder spacing). The output is `results/torque_envelope.json`.

What it computes: vertical foot loads only, from weight. What it leaves out:
the mass of the legs and of the servos hanging on them, gait dynamics and
landing impacts, friction limits, and heat. So it is an **upper bound on the
total mass a stance can hold, not a mass you can walk with.**

Largest total mass whose weight keeps every joint under the torque shown, using
the unverified listed stall torques (1.86 N·m at 5 V, and 2.07 N·m at 6.5 V by
straight-line interpolation between the listed 5 V and 6.8 V points):

| Stance | 5 V, full stall torque | 5 V, half of stall | 6.5 V, half of stall |
|---|---|---|---|
| Standing, 4 feet | 8.97 kg | 4.48 kg | 4.98 kg |
| Crawl, 3 feet, body leaning 25 mm | 5.66 kg | 2.83 kg | 3.14 kg |
| Trot, 2 diagonal feet | 4.48 kg | 2.24 kg | 2.49 kg |

**How to read the table.** Each cell answers one question: if the robot froze in
this pose, how heavy could it be before the weakest servo reached the torque
shown? The cell is that total mass.

*Rows are how many feet carry the weight.* Standing: four legs, about a quarter
of the weight each (the easiest case). Crawl: three legs, with the body leaning
away from the lifted foot, so the load is uneven; the busiest leg carries about
40% of the weight (derived from `results/torque_envelope.json`, not a separate
output). Trot: two diagonal legs, half each (the hardest case).

*The knee decides.* The foot sits under the hip, so the hip-pitch joint holds
almost nothing. The knee is offset horizontally from the foot by about 85 mm,
and it must resist the leg's share of the weight across that lever arm:
knee torque = share of weight × 9.81 m/s² × mass × 0.085 m.

*Columns are how much torque the servo is allowed to give.* Full stall is the
listed 1.86 N·m at 5 V, which a servo can hold only briefly. "Half of stall" is
my assumption for a torque it could hold continuously. 6.5 V gives more torque
than 5 V, so the allowed mass rises.

*Worked example, trot at 5 V, half of stall.* Knee torque per kg of robot
= 0.5 × 9.81 × 0.085 ≈ 0.42 N·m. The servo allows 0.93 N·m. 0.93 ÷ 0.42 ≈ 2.2 kg,
matching the 2.24 kg in the table (the small difference is rounding of the lever
arm here).

*Reading the result.* These are ceilings from weight on the feet alone, not
targets. The trot row is the one to watch, because a walking robot passes through
its harshest poses. Everything listed as left out above (leg and servo mass,
gait dynamics, friction, heat) lowers the usable mass, and the servo torques are
unverified listing figures on placeholder leg lengths.

- "Half of stall" is an assumption, not a datasheet value: stall torque is the
  most the servo can give for an instant, and the listings give no continuous
  rating.
- In every stance the knee is the limiting joint; with the foot under the hip
  the hip-pitch joint carries no static torque. This is specific to that
  stance, and walking moves the foot off the hip.
- The mass scales inversely with leg length: scaling every link and the stand
  height by 1.5 cuts the 5 V, half-stall limits to 2.99, 1.89 and 1.49 kg
  (standing, crawl, trot). Longer legs mean a lighter robot.
- **Body ratio:** at the same width, changing length over width from 1 to 2.4
  moves the crawl limit between 2.83 and 3.10 kg, within about 10%, and not in
  one direction. The ratio does not limit the static torque. It matters for
  tipping and pitch, which this calculation does not cover.
- **Servo mass counts.** Twelve servos at 60 g (listed) are 0.72 kg before the
  frame, battery, electronics and sensors.

**Working target, a recommendation rather than a result:** aim for a total mass
of about 1.5 kg for the first robot with these placeholder legs. At that mass
the trot knee needs about 0.62 N·m, one third of the 5 V stall figure. That
leaves room for gait peaks, which the static bound ignores. The stand test and
gait replay in the work packages below are what turn this into a real answer.

## Work packages

Each is one branch-sized step. Tick them off here as they land.

0. [x] **Static mass bound.** `scripts/torque_envelope.py`, as above.
1. [ ] **MJCF from the URDF.** Generate a MuJoCo model from the same
   parameters as `robots/quadruped.urdf`, with foot and body-frame sites, a
   floor with friction, and placeholder inertias derived from the dimensions.
   Test: with the model in a fixed pose, foot positions from MuJoCo equal
   `mojito/kinematics.py` to numerical precision, for random lengths and angles
   (left and right legs, and several body poses).
2. [ ] **Servo and sensor model.** Torque and speed limits as functions of
   voltage (5–6.5 V, straight line between the listed 5 V and 6.8 V points), latency,
   resolution, and a joint-angle sensor with noise, quantisation, delay and
   offset, behind an actuator interface that a motor model can fill later. Test against a closed form: a servo holding a known mass at a
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

1. Datasheet figures for the DS3218 20KG. The model interpolates torque and speed
   between the listed 5 V and 6.8 V points to cover 5–6.5 V; the datasheet would
   confirm or replace that.
2. Joint-angle sensor type and resolution (placement is decided: in line with
   the joint axis, close to the motor).
3. Whether the 1.5 kg working target is acceptable, or whether the robot must
   carry a heavier battery or payload (then the legs must be shorter).
