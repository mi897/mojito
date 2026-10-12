# Stage 2 — physics simulation (MuJoCo)

Status: **in progress; work packages 0 to 3 are done.**
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
| Actuators, first robot | "DSServo DS3218 20KG Digital Metal Servo", supply adjustable 5–6 V | Torque-limited, speed-limited position servos whose limits depend on voltage. See "Servo model". |
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
  within 5–6 V, which sits inside the listed 5–6.8 V points, so no
  extrapolation is needed. The listings give two points only (5 V and 6.8 V);
  between them the model uses a straight line, which is an assumption (it gives
  20.4 kg·cm, 2.00 N·m, at 6 V). Outside 5–6 V the model refuses to run.
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
  the horn side, so this must be settled when the mount is designed. **Sensor type
  (decided):** a custom Hall-effect board, a magnetic sensor reading a magnet on
  the joint axis. Risks to test early, because the sensor sits next to the
  motor: the motor's own magnets, a steel horn or screws, and high-current wires
  can distort the field and bias the reading; the magnet must be centred and at a
  fixed gap, or the error is a smooth nonlinearity over the turn. Each joint gets
  a calibration table against known angles. The model's sensor parameters are
  therefore: resolution, noise, a per-joint offset, a smooth nonlinearity, delay
  (bus read time), and a bias that may follow motor current. Resolution is open
  until the chip is chosen (12-bit is 0.088°, 14-bit is 0.022°). Twelve sensors
  also set the bus: many magnetic encoder chips share one fixed I2C address, so
  plan SPI with a select line per sensor or an I2C multiplexer; the Blue Pill
  has two SPI ports and ten ADC channels.
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
weight in three stances, with the Stage 2 design placeholders (see "Sizing
against the servo"): leg 50, 150, 150 mm standing 212.5 mm high, feet under the
hips, and a body of 300 × 150 mm shoulder spacing. The output is
`results/torque_envelope.json`.

What it computes: vertical foot loads only, from weight. What it leaves out:
the mass of the legs and of the servos hanging on them, gait dynamics and
landing impacts, friction limits, and heat. So it is an **upper bound on the
total mass a stance can hold, not a mass you can walk with.**

Largest total mass whose weight keeps every joint under the torque shown, using
the unverified listed stall torques (1.86 N·m at 5 V, and 2.00 N·m at 6 V by
straight-line interpolation between the listed 5 V and 6.8 V points):

| Stance | 5 V, full stall torque | 5 V, half of stall | 6 V, half of stall |
|---|---|---|---|
| Standing, 4 feet | 7.18 kg | 3.59 kg | 3.85 kg |
| Crawl, 3 feet, body leaning 25 mm | 4.31 kg | 2.15 kg | 2.31 kg |
| Trot, 2 diagonal feet | 3.59 kg | 1.79 kg | 1.92 kg |

**How to read the table.** Each cell answers one question: if the robot froze in
this pose, how heavy could it be before the weakest servo reached the torque
shown? The cell is that total mass.

*Rows are how many feet carry the weight.* Standing: four legs, about a quarter
of the weight each (the easiest case). Crawl: three legs, with the body leaning
away from the lifted foot, so the load is uneven; the busiest leg carries about
42% of the weight (derived from `results/torque_envelope.json`, not a separate
output). Trot: two diagonal legs, half each (the hardest case).

*The knee decides.* The foot sits under the hip, so the hip-pitch joint holds
almost nothing. The knee is offset horizontally from the foot by about 106 mm,
and it must resist the leg's share of the weight across that lever arm:
knee torque = share of weight × 9.81 m/s² × mass × 0.106 m.

*Columns are how much torque the servo is allowed to give.* Full stall is the
listed 1.86 N·m at 5 V, which a servo can hold only briefly. "Half of stall" is
my assumption for a torque it could hold continuously. 6 V gives more torque
than 5 V, so the allowed mass rises.

*Worked example, trot at 5 V, half of stall.* Knee torque per kg of robot
= 0.5 × 9.81 × 0.106 ≈ 0.52 N·m. The servo allows 0.93 N·m. 0.93 ÷ 0.52 ≈ 1.8 kg,
matching the 1.79 kg in the table (the small difference is rounding of the lever
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
- The mass scales inversely with leg length. Scaling the 150 mm design by 1.25
  (links 188 mm) cuts the 5 V, half-stall limits from 3.59, 2.15, 1.79 kg to
  2.87, 1.72, 1.44 kg (standing, crawl, trot). Longer legs mean a lighter robot.
  Going from the 120 mm links of the Stage 0–1 placeholders to 150 mm already
  cost about a fifth of the limit (trot, 5 V, half of stall: 2.24 to 1.79 kg).
- **Body ratio:** at the same width (150 mm), changing length over width from 1
  to 2.4 moves the crawl limit between 2.24 and 2.08 kg, about 8%, slightly
  lower for longer bodies. The ratio does not limit the static torque much. It
  matters for tipping and pitch, which this calculation does not cover.
- **Servo mass counts.** Twelve servos at 60 g (listed) are 0.72 kg before the
  frame, battery, electronics and sensors.

**Working target, a recommendation rather than a result:** treat 1.5 kg as an
upper limit for the first robot with the design placeholders, and aim below it.
At 1.5 kg the trot knee needs about 0.78 N·m, 42% of the 5 V stall figure and
84% of the half-of-stall ceiling at 5 V (78% at 6 V). That is thinner than it
looked with the 120 mm legs, and the static bound ignores gait peaks. Running
nearer 6 V helps a little (about 7% more torque than at 5 V). The stand test and gait replay in the work packages below are
what turn this into a real answer.

## Sizing against the servo

The listings give the servo as about 40 × 20 × 40.5 mm and 60 g (retailer
numbers, unverified). They do not give mounting-hole spacing, shaft position or
horn size, which decide the real fit. Against the Stage 0–1 placeholders (40 mm
abduction offset, 120 mm links, 240 × 100 mm shoulder spacing) the servo was
about as big as the abduction offset and a third of a link, so the legs and body
looked small for it.

Decision (owner asked for parts that are not smaller than the motors): the
**Stage 2 design placeholders are 1.25× the Stage 0–1 ones.**

| | Stage 0–1 placeholder | Stage 2 design placeholder |
|---|---|---|
| Abduction offset | 40 mm | 50 mm |
| Upper and lower leg | 120 mm each | 150 mm each |
| Stand height | 170 mm | 212.5 mm |
| Shoulder spacing, length × width | 240 × 100 mm | 300 × 150 mm (2:1) |

The Stage 0–1 network, results and demos keep the 120 mm placeholders. Changing
them would mean retraining and regenerating every result, so it is deferred; the
two sets differ until a design is chosen. The MuJoCo model and the design sweep
use the new sizes (`scripts/torque_envelope.py` already does).

What this does not settle: whether the structure is sound. The frame and links
must carry the servos, battery and electronics, with enough stiffness and
strength at the peak gait loads. That needs a mass budget (servos 0.72 kg listed;
battery, electronics, frame and sensors are not known yet) and a volume check
that the battery and electronics fit between the hip servos. See work package 6.

## Power and electronics (first robot)

Chosen by the owner: XW Power Eagle LiPo, 2S, 7.4 V nominal, 1500 mAh, 25C
continuous (25 A as listed), 90C burst, T-plug, 76 g, 26 × 34 × 72 mm. A PCA9685
16-channel 12-bit I2C servo driver and an STM32F103C8T6 "Blue Pill" board were
the first picks. **Update (decided):** the controller is an ESP32. The Raspberry Pi
and the Blue Pill are dropped (the Pi: too much hassle with the SD card and power).
The PCA9685 stays: the ESP32 talks to it over I2C so that more servos are
controlled with fewer pins. Hardware specifics such as the Hall chip and magnet are **tabled** and
will be revisited later.

Consequences of keeping the PCA9685 (to check when the hardware is designed):

- **Voltage rating of the board.** Several PCA9685 board listings cap the servo
  supply pin at 6 V (the chip's own supply is limited to 5.5 V). The servo rail
  is now capped at 6 V, which matches those listings exactly, with no margin:
  regulator tolerance and load transients add on top, so set the regulator a
  little below 6 V. The actual board's rating is unknown. Either keep the rail at
  or below it, or wire the servos' power directly and give the
  board only signal and logic supply. The rating of the actual board is not
  known; read its schematic.
- **Servo power should bypass the board.** Board listings give small per-channel
  current figures (tens of mA, for signal), and no rating for the power terminal.
  Servo current (about 2.5 A each at stall) should come straight from the servo
  regulators, with only the signal wire through the PCA9685.
- **Resolution** is set by the PCA9685, about 0.66° per count over 270° (see
  above), coarser than the servo's listed 3 µs dead band.
- **Update time.** Writing all 12 channels in one I2C transfer is about 50 bytes,
  which takes roughly 1 ms at 400 kHz (arithmetic, not measured).
- **Hardware stop.** The board's output-enable pin can be wired to an ESP32 pin
  to switch all outputs off. What the DS3218 does with no pulses (go limp or
  hold) is not known and must be tested before it is relied on.

What follows from those numbers (arithmetic and reasoning, not measurements):

- **The battery cannot feed the servos directly.** A 2S pack reads about 8.4 V
  full, above the listed 6.8 V servo maximum and the 5–6 V range chosen
  above. The servo rail needs a regulator set to 5–6 V. As the pack drops, a
  plain buck regulator stops holding 6 V near the bottom of the pack, so the
  low-voltage cut-off must be set with that in mind (roughly 6.5–7 V for a 6 V
  rail; check the regulator's dropout).
- **Servo current is the sizing problem.** The listings give 2.6 A stall at
  6.8 V and 1.8 A at 5 V per servo. Twelve at once, interpolated to 6 V (2.2 A each), is
  about 27 A. Servos will not all stall together, so the real peak is lower, but
  the real figure is unknown until measured. Plan several regulators (for
  example one per group of legs) rated well above the typical draw, and measure
  current per rail. A single small regulator will brown out.
- **Runtime is short.** 1.5 Ah is 11.1 Wh. Taking 80% as usable, 1.2 Ah lasts
  about 14 minutes at an average 5 A from the battery and about 7 minutes at
  10 A. The average draw is not known.
- **Mass and volume are not the problem.** The battery is 76 g and fits easily in
  the 300 × 150 mm body. Servos (0.72 kg listed) and the frame dominate.
- **The PCA9685 limits servo resolution.** At 50 Hz one count is 20 ms / 4096 =
  4.9 µs. The listed 500–2500 µs range is about 410 counts, or about 0.66° per
  count over 270° (0.44° over 180°). The servo's listed dead band is 3 µs, so the
  driver is coarser than the servo. The Blue Pill's hardware timers can produce
  1 µs steps directly. The PCA9685's internal clock accuracy was not in the
  sources I read; plan a frequency calibration.
- **IMU (added, model not chosen):** connects directly to the ESP32. It is the
  only way to measure body orientation on the real robot. Points to settle when
  the part is chosen: I2C or SPI (the PCA9685 shares an I2C bus, and a bus write to
  all 12 servos takes about 1 ms, so a fast IMU may prefer SPI or its own bus),
  its sample rate (a few hundred Hz is plenty for the policy), and mounting
  (servo vibration reaches it, so mount it rigidly near the body centre or
  damped, and keep it clear of the Hall sensors' magnets). Simulation side: the
  policy may observe body angular rate and the gravity direction in the body
  frame, which an IMU can supply; give them gyro noise, a bias that drifts, and
  accelerometer noise as parameters. The sensor's mass is negligible against the
  budget.
- **Still not in the list:** current sensing on the servo rails.
- Note on the older plan: the Blue Pill's STM32F103 had no floating-point unit and
  20 KB of RAM. The ESP32 has a floating-point unit and Wi-Fi, so it can also run
  the closed-form inverse kinematics and short gait loops.

**Raspberry Pi: dropped.** Learning is off board, control is over the network, and
logs go to a laptop, so onboard Linux is not needed. An ESP32 has Wi-Fi and a
floating-point unit. It gives up local storage and an easy route to ROS onboard;
ROS can run on the laptop, or micro-ROS on the ESP32 later. The ESP32 is kept on
its own regulated supply (see the power layout below), joined to the servo rail
by a common ground only.

**Power layout (decided):** one battery,
two separate regulators. A servo-rail regulator (or several) set to 5–6 V and a
small logic regulator for the ESP32 and sensors. A separate logic battery adds
mass, a second charger and a second pack to monitor, and still shares a ground, so
it isolates less than it seems. Details: the logic regulator input gets its own
filtering so servo current surges do not disturb it, with bulk capacitance close
to the ESP32 and close to the servos; a single ground point; a fuse on the servo
rail; a switch or e-stop on the servo rail alone so the ESP32 can stay on and
connected with the servos unpowered; and the ESP32 reads battery voltage through a
divider to trigger a safe stop and protect the pack.

## Network control and logging

Decided: control over the network, learning off board, logs saved on a server
hosted on a laptop, to be used for training and for updating a world model that
improves sim-to-real transfer. This is a standard arrangement. Design points:

- **Do not close the control loop over Wi-Fi.** Wi-Fi delay varies by tens of
  milliseconds. Joint-level timing stays on the microcontroller. The laptop sends
  slow commands (a gait, a velocity, or short buffered joint trajectories). Closed-form
  inverse kinematics is small enough that it probably can run on the
  microcontroller or the Pi; the learned network need not.
- **Fail safe.** If commands stop arriving for a set time (a few hundred
  milliseconds, to be tuned), the robot holds or lowers itself, and a low
  battery voltage does the same.
- **Log at the source.** Stamp every sample with the microcontroller's own clock,
  not the arrival time, and number the samples so gaps are visible. Over UDP,
  packets can be lost, so use TCP for logs, or buffer on the robot and send after.
- **What to log:** commanded servo positions, measured joint angles, IMU, battery
  voltage and rail currents, the gait command and state, and the configuration (firmware
  version, link lengths, supply voltage). Log raw readings, not only derived values.
  Rough size: 100 bytes per sample at 200 Hz is about 20 kB/s, which Wi-Fi handles
  easily.
- **Why this matters for the simulation:** a world model is fitted by comparing
  logged commanded-versus-measured motion with the simulator's. That is only
  possible with excitation data (varied commands) and consistent timestamps. It
  also needs the same joint-sensor model as the simulation, so the sensor
  parameters above double as what the logs will be used to estimate.
- Not yet covered: body position in the world. The robot's own sensors give joint
  angles and body orientation (with an IMU), not where the body is. An external
  tracking system, if built later, would supply it.

## Work packages

Each is one branch-sized step. Tick them off here as they land.

0. [x] **Static mass bound.** `scripts/torque_envelope.py`, as above.
1. [x] **MJCF from the URDF.** `mojito/mjcf.py` generates a MuJoCo model from a
   `URDFModel` (so a leg-length change reaches both the IK input and the
   simulator), with a floating base, joint limits from the URDF, a site at each
   foot tip and at the body-frame origin, a floor with friction, and capsule
   geometry with placeholder masses (`DEFAULT_MASSES`, about 1.5 kg total, not the robot's).
   Only the foot pads and the floor collide. No actuators yet (work package 2).
   Tests (`tests/test_mjcf.py`): foot positions from MuJoCo equal
   `body.Robot.feet` to 1e-9 m for 25 random draws of per-leg lengths, body size,
   joint angles (left and right legs) and base pose, both in the body frame and in
   the world; joint ranges equal the URDF's; a longer shank lowers the foot; and
   a stiffly held stance dropped on the floor ends on four contacts whose normal
   force sums to the weight within 3%.
2. [x] **Servo and sensor model** (`mojito/actuators.py` and `mojito/sensors.py`, with `tests/test_actuators.py` and `tests/test_sensors.py`; `mjcf_from_urdf(..., actuated=True)` adds torque motors and a gyro and accelerometer).
   Defaults are the unverified listed figures; kp 10 N·m/rad, kd 0.2, latency 20 ms and the
   sensor and IMU noise values are my placeholders, not measurements. Tests: a servo holding a mass on a
   horizontal arm settles at the root of kp·q = m g L cos q (below the limit), and above the limit sags to
   cos q = stall/(m g L), where the torque equals the stall value. Spec as planned: Torque and speed limits as functions of
   voltage (5–6 V, straight line between the listed 5 V and 6.8 V points), latency,
   resolution, a joint-angle sensor with noise, quantisation, delay and
   offset, and an IMU (gyro and accelerometer noise, drifting bias), behind an actuator interface that a motor model can fill later. Test against a closed form: a servo holding a known mass at a
   known lever arm should settle at the expected deflection when the torque is
   below the limit, and saturate when it is above.
3. [x] **Stand test** (`mojito/stand.py`, `scripts/stand_test.py`, `tests/test_stand.py`, `results/stand_test.json`).
   The IK standing posture (design placeholders: 50, 150, 150 mm legs, 212.5 mm, 300 × 150 mm) is commanded
   through the servo model into the actuated MuJoCo model; the last 0.5 s of a 3 s run is reported.
   What was measured, all at 5 V with the unverified listed stall torque (1.86 N·m) and placeholder
   servo gains (kp 10 N·m/rad, kd 0.2):
   - *Against the static calculation (1.5 kg, leg mass near zero).* Foot loads sum to the weight. The servo
     torques match `statics.joint_torques` evaluated at the measured foot loads and joint angles to within
     0.5% (abduction and knee; hip pitch differs by about 0.002 N·m, which was not explained). The
     feet do **not** share the weight equally: 3.55 N front, 3.81 N rear per foot, because the knees bend one way. The
     equal-quarter prediction of `torque_envelope.py` is therefore off by up to 8%: knee 0.390 N·m
     (front) and 0.421 N·m (rear) measured, 0.3895 N·m predicted.
   - *Default placeholder masses (1.72 kg total: servos, links, body).* Knee torque peaks at 0.52 N·m, 28% of the 5 V
     limit (26% of the 6 V limit, 2.00 N·m); front feet carry 3.4 N each and rear feet 5.0 N each. Largest
     knee gap between commanded and actual: 2.7°. The body ends 6 mm below the height the feet touch the floor at the start.
   - *Mass sweep, leg mass near zero, 5 V.* With kp 10 the robot stands up to 4 kg (whole-run peak 88% of the
     limit, knee gap 8°, body 32 mm lower) and falls over at 4.5 kg, with a peak torque
     that reached the limit. With kp 100 (kd 1) it stands up to 5.5 kg and collapses at 6 kg. The static bound
     from `torque_envelope.py` is 7.18 kg at full stall torque. Both runs fail below it.
   - *Why it fails early.* At 6 kg (kp 100) the knees saturate in the first
     0.2 s and the legs buckle. The settling overshoot and the uneven foot loads put the rear knees above the equal-quarter
     figure. I looked at this one run; the other collapse masses are measured but their mechanism was not traced.
   Not shown: how heavy a real robot can be (kp, kd, latency, masses and the servo figures are all placeholders, and the
   collapse mass moved from 4.5 to 6 kg between two gain settings); anything about walking; friction limits and foot slip (not
   measured here). After a collapse the body passes through the floor, because only
   the foot pads collide, so the "height drop" of about 450 mm in the failed runs is not a physical height.
4. [ ] **Replay Stage 1 gaits in physics.** Feed the trot and crawl foot
   trajectories through the IK layer into the servo model. Report whether the
   body stays up, foot slip, and peak torque. This tests the known weaknesses
   from Stage 1: the trot's reliance on dynamic balance and the quick crawl
   lean.
5. [ ] **Design sweep.** Vary link lengths, body size and mass; report which
   designs stand, which gaits survive, and the torque margin. Output to
   `results/` with the sweep ranges and mass assumptions stated.
6. [ ] **Mechanical fit and structure check.** Servo-sized boxes in the model
   (listed 40 × 20 × 40.5 mm), a mass budget (servos, battery, electronics, frame,
   sensors), battery and electronics volume against the body, and link stiffness
   and strength against the peak joint torque from the stand test and gait
   replay. Needs the battery and electronics choices from the owner.
7. [ ] **Locomotion learning.** Policy acts in foot space through the IK layer.
   Randomise mass, friction, servo strength and latency. Reintroduce the
   current-angle input and smoothness penalties. Only after 1–6.

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
   between the listed 5 V and 6.8 V points to cover 5–6 V; the datasheet would
   confirm or replace that.
2. Tabled for later: the Hall-effect chip, resolution, and magnet and horn
   details (type and placement are decided: custom Hall board in line with the
   joint axis, close to the motor).
3. The servo-rail regulators and their current rating, the PCA9685 board's actual
   voltage rating against the 6 V rail, and the IMU model and bus. Battery, ESP32, PCA9685 and
   the one-battery power layout are decided. If the robot must carry more than about 1.5 kg, the legs must be
   shorter.
