# Stage 2 — physics simulation (MuJoCo)

Status: **in progress; work packages 0 to 4 are done.**
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
| Real dimensions | None yet; to be chosen from simulation | Dimensions are sweep parameters, never constants. The URDF generator in `mojito/geometry/robots.py` stays the single source. |
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
frame as `mojito/robot/body.py`.

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

`scripts/body/torque_envelope.py` (maths in `mojito/robot/statics.py`, test in
`tests/robot/test_statics.py`) computes the joint torque needed to hold the body's
weight in three stances, with the Stage 2 design placeholders (see "Sizing
against the servo"): leg 50, 150, 150 mm standing 212.5 mm high, feet under the
hips, and a body of 300 × 150 mm shoulder spacing. The output is
`results/body/torque_envelope.json`.

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
42% of the weight (derived from `results/body/torque_envelope.json`, not a separate
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
use the new sizes (`scripts/body/torque_envelope.py` already does).

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

0. [x] **Static mass bound.** `scripts/body/torque_envelope.py`, as above.
1. [x] **MJCF from the URDF.** `mojito/sim/mjcf.py` generates a MuJoCo model from a
   `URDFModel` (so a leg-length change reaches both the IK input and the
   simulator), with a floating base, joint limits from the URDF, a site at each
   foot tip and at the body-frame origin, a floor with friction, and capsule
   geometry with placeholder masses (`DEFAULT_MASSES`, about 1.5 kg total, not the robot's).
   Only the foot pads and the floor collide. No actuators yet (work package 2).
   Tests (`tests/sim/test_mjcf.py`): foot positions from MuJoCo equal
   `body.Robot.feet` to 1e-9 m for 25 random draws of per-leg lengths, body size,
   joint angles (left and right legs) and base pose, both in the body frame and in
   the world; joint ranges equal the URDF's; a longer shank lowers the foot; and
   a stiffly held stance dropped on the floor ends on four contacts whose normal
   force sums to the weight within 3%.
2. [x] **Servo and sensor model** (`mojito/sim/actuators.py` and `mojito/sim/sensors.py`, with `tests/sim/test_actuators.py` and `tests/sim/test_sensors.py`; `mjcf_from_urdf(..., actuated=True)` adds torque motors and a gyro and accelerometer).
   Defaults are the unverified listed figures; kp 10 N·m/rad, kd 0.2, latency 20 ms and the
   sensor and IMU noise values are my placeholders, not measurements. Tests: a servo holding a mass on a
   horizontal arm settles at the root of kp·q = m g L cos q (below the limit), and above the limit sags to
   cos q = stall/(m g L), where the torque equals the stall value. Spec as planned: Torque and speed limits as functions of
   voltage (5–6 V, straight line between the listed 5 V and 6.8 V points), latency,
   resolution, a joint-angle sensor with noise, quantisation, delay and
   offset, and an IMU (gyro and accelerometer noise, drifting bias), behind an actuator interface that a motor model can fill later. Test against a closed form: a servo holding a known mass at a
   known lever arm should settle at the expected deflection when the torque is
   below the limit, and saturate when it is above.
3. [x] **Stand test** (`mojito/sim/stand.py`, `scripts/stand/stand_test.py`, `tests/sim/test_stand.py`, `results/stand/stand_test.json`).
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
   **Sit to stand** (`stand.run_sit_to_stand`, `scripts/stand/sit_to_stand.py`, `results/stand/sit_to_stand.json` and `.png`,
   `scripts/stand/make_stand_demo.py` for the animations). Same model, 5 V, near-weightless legs. Start crouched
   at 140 mm (the placeholder knee limit of -2.3 rad stops the 150 mm legs at about 130 mm, so this is a deep
   crouch and not sitting down), hold 1 s, rise to 212.5 mm over 0.5 to 4 s along a minimum-jerk profile, hold 1 s.
   - Per newton of foot load the crouch needs 25% more knee torque than standing (0.325 against 0.260 N·m per kg of
     robot at an equal quarter share each, from `statics`), so the crouch is the hardest pose of the rise.
   - With the command resolution made fine (0.006°), a slow rise peaks at the crouch hold: for 3 kg the 2 s rise
     peaks at 0.54 of the stall torque and the hold at 0.53; for 4 kg, 0.72 and 0.72. A 0.5 s rise is
     higher (3 kg: 0.67; 4 kg: 0.87, with the servo at its limit for 67% of the rise, and it still stood).
   - Servo gains and command resolution matter. With the PCA9685's 0.66° steps and kp 100 N·m/rad, every
     command step is a 1.15 N·m torque step (62% of stall), so even a 1.5 kg robot shows a rise peak of 0.88 whatever the rise
     time, against 0.28 with fine resolution. That is a property of the model as built (stiff gain plus coarse
     command steps), not a measurement of the servo; a real servo's own loop and dead band are not modelled.
   - Where it stops: with kp 10 it rises with 4 kg in 1 s or longer, but not with 5 kg at any rise time (peak torque at the limit); with kp 100
     it rises with 4 kg and not with 5 kg (at the limit even in the crouch hold). The static bound for standing is
     7.18 kg, and these crouch results are below it. I did not trace why kp 10 holds 5 kg in the crouch (0.75 of stall)
     while kp 100 saturates.
   - Not shown: a real robot's limit (masses, gains and the servo figures are placeholders), rising from lying down,
     slip, or the effect of leg mass.

   **Weight distribution** (`results/stand/stand_test.json`). The feet do not share the weight equally. Standing with
   near-weightless legs: 24.1% on each front foot and 25.9% on each rear foot (front pair 48.3%). With the default
   placeholder masses: 20.4% front and 29.6% rear (front pair 40.9%); in the 140 mm crouch 19.7% and 30.3%
   (front pair 39.5%). Left and right match. The mechanism was not traced. Likely contributors: the knees all bend
   backwards, so the shank and foot masses sit behind the hips and move the centre of mass rearward, and the
   knee-bend direction makes front and rear legs not mirror images. Only the second of these applies to the near-weightless legs.

4. [x] **Replay Stage 1 gaits in physics** (`mojito/sim/gait_replay.py`, `scripts/gait/gait_replay.py`, `tests/sim/test_gait_replay.py`, `results/gait/gait_replay.json`).
   The trot and crawl foot targets (`gaits.foot_targets`, plus `gaits.body_sway` for the crawl) go through the
   closed-form IK and the servo model into the actuated MuJoCo model. The body is free: 1 s standing, 0.5 s blend,
   then 3 trot cycles (1.7 s period, `gaits.TROT`; the first version of this package used the original 0.5 s trot, see "Longer trot periods") or 2 crawl cycles. Default placeholder masses (1.72 kg), 5 V, kp 10, kd 0.2,
   PCA9685 command resolution, floor and pad friction as in `mjcf.py`. Measured velocity is the body displacement over
   the walking phase. A foot is planted if it carries more than 0.5 N in two consecutive 2 ms steps; slip is its
   horizontal movement per step.
   - *Stays up.* All 9 runs finished upright: tilt at most 12.1° (the turning trot; 3.5° to 9.8° in the others), body origin no lower than 198.7 mm, against 218.6 mm after the initial stand.
   - *It does not walk at the commanded speed.* Trot (1.7 s period), commanded 0 / 0.05 / 0.15 / 0.30 m/s: measured -0.008 / +0.011 / +0.100 /
     +0.248 m/s. Crawl, commanded 0 / 0.02 / 0.04 / 0.08: measured -0.011 / +0.009 / +0.028 / +0.063. Marching in
     place drifts backwards. Trot turning at 0.6 rad/s measured 0.38 rad/s, with 0.02 m/s of backward drift (the 1.2 rad/s turn of the 0.5 s trot measured 0.82 rad/s but falls over at periods of 1.6 s and above). Cause: see "Why the body falls short" below.
   - *Slip.* The foot tip (sphere centre) of a foot carrying more than 0.5 N moved at most 2.13 mm in a 2 ms step (1.1 m/s; trot
     0.30 m/s), 0.41 to 1.11 mm at the other trot speeds and 0.62 to 1.6 mm in the crawl; means 0.03 to 0.24 mm per step (0.015 to
     0.12 m/s). I did not separate sliding from the sphere rolling or from touchdown and lift-off, so this does **not**
     show that friction was no limit.
   - *Torque.* Peak |torque| as a fraction of the 1.86 N·m stall limit: trot 0.61 to 0.93 (0.85 turning); crawl 0.94 to 0.99, with
     the servos at their limit for 1.3% to 4.9% of the walking time. Standing with these masses peaks at 0.28 (knee). The crawl
     peaks higher on the servos than the trot; the reason was not traced.
   - *Joint tracking.* Largest gap between commanded and actual joint angle during walking: 6.5 to 16°.
   - *Feet down.* Fewest feet carrying more than 0.5 N: 2 in every run (1 in the trot at 0.30 m/s), including the crawl, which Stage 1 designed to keep 3 down.
     Whether this is a brief unloading or a sustained lift was not examined. Mean feet down: trot 2.9 to 3.3, crawl 3.3.
   - *Heavy robot.* At 8 kg (legs near zero) the trot saturates the servos for more than 20% of the walking time and the run does not count as walked (test).
   Animations: `scripts/gait/make_gait_replay_demo.py` writes `results/gait/trot_replay.gif` and `results/gait/crawl_replay.gif` (side view of the
   left legs, filled foot = loaded, dashed line = where the body would be at the commanded speed).

   **Why the body falls short** (`scripts/gait/gait_speed_diagnosis.py`, `results/gait/gait_speed_diagnosis.json`; same model, one run
   per entry). A planted foot is fixed in the world, so one whose commanded body-frame velocity is v asks the body to move at -v.
   `ask` is that, averaged over the loaded feet weighted by normal force. **The table below is the first version, at the Stage 1 trot period of 0.5 s;
   `results/gait/gait_speed_diagnosis.json` now holds the same variants at 1.7 s (table after the two causes).** Trot at 0.15 m/s, 0.5 s period:

   | variant | measured m/s | ask m/s | load on feet commanded to swing |
   |---|---|---|---|
   | baseline (kp 10, 20 ms latency, PCA9685 steps) | 0.041 | 0.103 | 24% |
   | latency 0 | 0.042 | 0.101 | 23% |
   | fine command resolution | 0.038 | 0.104 | 23% |
   | kp 100, kd 1 | 0.095 | 0.116 | 17% |
   | ideal servo (kp 100, kd 1, no latency, fine, 10x torque) | 0.088 | 0.095 | 23% |
   | swing height 60 mm | 0.079 | 0.120 | 17% |

   Two causes, measured separately:
   1. *The servos are soft (kp 10).* The ask is 0.103 and the body makes 0.041. With kp 100 the body makes 0.095 and with the
      ideal servo 0.088 (ask 0.095). Latency and command resolution made no difference (0.042, 0.038). Marching in place
      (command 0) drifts back at 0.038 m/s at kp 10 and 0.005 at kp 100, so that drift is softness too. In the crawl, which keeps nearly all
      load on the feet it commands to stand, the ask is 0.040 for a 0.04 command; the body makes 0.028 at kp 10 and
      0.040 at kp 100. So the whole crawl shortfall is the servo not following.
   2. *Feet that should be in the air still carry weight, and they push the body backward.* In the trot, 24% of the load is on feet commanded to
      swing, so the ask is 0.103 instead of 0.15, and no servo setting fixes that (ideal servo: ask 0.095). The rear
      feet are the cause: with the commanded foot less than 8 mm above its stance height the rear foot was loaded in every step I looked at (172 of 172
      steps; front foot 40% of 172), and 88% between 8 and 15 mm. The loaded rear feet sit 12 mm higher in the body frame than commanded (front 6 mm): the legs sag with the weight, so a 30 mm swing (19 mm actually achieved)
      does not clear the ground for the first and last part of the swing.
   A taller swing (60 mm) raised the trot to 0.079 but also raised the tilt to 8.3° (4.1° at 30 mm) and with the ideal servo it did not help (0.079), so I did not establish a
   fix. 100 mm gave 0.186 (ideal servo) with 21° tilt: not a result I trust. Not tested: friction, body mass distribution, gait timing, or how the
   weight split (20% front, 30% rear for the default masses) feeds into this. The sag figures are the measured *mean* over loaded stance steps in one trot run.

   **The same variants at the 1.7 s trot** (`results/gait/gait_speed_diagnosis.json`, regenerated; one run per entry, 3 cycles):

   | variant | measured m/s | ask m/s | load on feet commanded to swing |
   |---|---|---|---|
   | baseline (kp 10, 20 ms latency, PCA9685 steps) | 0.100 | 0.064 | 32% |
   | latency 0 | 0.101 | 0.064 | 33% |
   | fine command resolution | 0.100 | 0.064 | 32% |
   | kp 100, kd 1 | 0.120 | 0.075 | 27% |
   | ideal servo | 0.128 | 0.074 | 27% |
   | swing height 60 mm | 0.049 | 0.050 | 35% |
   | ideal servo, swing height 60 mm | 0.082 | 0.042 | 35% |

   What changed: the body now makes *more* than `ask`, the opposite of the 0.5 s trot, and the load on feet commanded to swing rose from 24% to 32% even though
   the speed rose. So "load on swing feet pushes the body backward" no longer accounts for the speed: the `ask` figure is a rigid-foot estimate and a longer
   stance (850 ms) lets feet slide and the body lurch in ways it does not model. I did not trace this. What the table does support: servo softness still costs
   speed (0.100 at kp 10, 0.120 at kp 100, 0.128 ideal), latency and command resolution still do not matter, and a 60 mm swing still does not help (0.049; ideal servo 0.082).

   **How the gait timing is set** (`mojito/robot/gaits.py`, `Gait`). Four numbers, all per cycle:
   - `period`: seconds per cycle. Trot 1.7 s (`gaits.TROT`; the original 0.5 s trot is `gaits.TROT_STAGE1`), crawl 2.0 s.
   - `duty`: the fraction of the cycle a foot is on the ground. Trot 0.5, crawl 0.85. So stance lasts `duty * period` (`Gait.stance_time`)
     and swing `(1 - duty) * period` (`Gait.swing_time`): trot 0.25 s / 0.25 s (1.7 s trot: 0.85 s / 0.85 s), crawl 1.7 s / 0.3 s.
   - `offsets` (FL, FR, RL, RR): the cycle phase at which each leg *starts its stance*. A leg's phase is `(t / period - offset) mod 1`;
     it is on the ground while the phase is below `duty` and swinging after that, so it lifts at `(offset + duty) mod 1` (`Gait.swing_starts()`).
     Trot (0, 0.5, 0.5, 0): FL and RR together, FR and RL half a cycle later. Crawl (0.75, 0.25, 0.5, 0): lifts at phases FR 0.10, RL 0.35,
     FL 0.60, RR 0.85, spaced a quarter cycle apart; each swing is 0.15 of the cycle, so there is a 0.10 gap with all four down.
     `gaits.sequence_offsets(order, duty)` makes the offsets for a one-leg-at-a-time order.
   - `lift` (swing height, m) and `sway` (crawl body lean, m). The lean only acts when no two swings overlap.
   Where each foot goes follows from these and the commanded velocity: a foot on the ground moves backward in the body frame at the body's
   velocity, so one stance covers `velocity * stance_time` (trot at 0.15 m/s: 37.5 mm at 0.5 s, 128 mm at 1.7 s; crawl at 0.04 m/s: 68 mm), and the swing carries the
   foot forward to where the next stance starts, with a sin² lift. The commanded stance and the measured foot load are drawn together in
   the timing chart at the bottom of each isometric GIF (`results/gait/trot_iso.gif`, `results/gait/trot_turn_iso.gif`, `results/gait/crawl_iso.gif`, from
   `scripts/gait/make_gait_iso_demos.py`).

   **Longer trot periods: the 0.5 s trot was not optimal, and the period was found by sweeping** (`scripts/gait/sweeps/trot_period_sweep.py`, `results/gait/sweeps/period/trot_period_sweep.json`; `scripts/gait/sweeps/trot_period_robustness.py`, `results/gait/sweeps/period/trot_period_robustness.json`; finding).
   The original Stage 1 trot (0.5 s, now `gaits.TROT_STAGE1`) was set for the geometry demos and never tuned in physics. The first physics replay used it and the body made 0.041 m/s of a 0.15 m/s command; a first try at 1 s was worse (0.032), so the period was swept instead of guessed. Only `period` changes
   (duty 0.5, diagonal pairs, 30 mm swing); same physics as work package 4; one run per cell, 4 cycles at 0.5 s and 3 otherwise. Measured forward speed, m/s:

   | period | cmd 0.05 | cmd 0.15 | cmd 0.30 | tilt marching | turn 0.6 rad/s | turn 1.2 rad/s |
   |---|---|---|---|---|---|---|
   | 0.5 s | -0.011 | 0.041 | 0.111 | 5.0° | walked | walked |
   | 1.0 s | 0.004 | 0.032 | 0.176 | 4.5° | walked | walked |
   | 1.5 s | 0.007 | 0.089 | 0.245 | 8.7° | walked | walked (19° tilt) |
   | 1.7 s | 0.011 | 0.100 | 0.248 | 9.1° | walked | fell |
   | 1.9 s | 0.015 | 0.106 | 0.245 (not counted) | 9.0° | walked | fell |
   | 2.0 s | 0.017 | 0.108 | 0.219 (not counted) | 9.0° | walked | fell |

   (1.6 and 1.8 s are in the JSON: the same trend.) At 0.15 m/s the body makes 27% of the command at 0.5 s, 21% at 1 s, 59% at 1.5 s, 67% at 1.7 s and 72% at 2 s.
   Marching in place drifts back less (-0.038 m/s at 0.5 s, -0.006 at 2 s). Peak servo torque barely changes (0.66 to 0.78 of stall at 0.15 m/s).
   - *Costs.* The body tilts more: 8.7° to 9.1° marching in place at 1.5 to 2 s against 5.0° at 0.5 s. The 1.2 rad/s turn tips over from 1.6 s up
     (the stance, 0.8 s or more, sweeps the feet through a large angle). At 0.30 m/s the servos saturate for 15% to 25% of the walk from 1.0 s up (0% at 0.5 s), and from 1.9 s that passes the 20% at which a run is not counted as walked (`walked` in the JSON); tilt there is 13.6° and 14.3°, not a fall.
   - *Why 1.7 s.* The rule, written after seeing the runs: the longest period for which every forward run up to 0.30 m/s counts as walked, with tilt under 10° and at least two
     feet loaded at all times except at 0.30 m/s. 1.8 s fails it (tilt 11.7° at 0.30 m/s, one foot loaded at some moment at 0.15 m/s); 1.9 s and 2.0 s fail it at 0.30 m/s. It is not an optimum:
     1.5 to 2 s differ by only 0.02 m/s at 0.15 m/s, and I have no measure of how much a small change to the setup moves one run (the simulation is deterministic, so a repeat gives the same number).
   - *Does the choice hold up?* `trot_period_robustness.py` repeats the period sweep over the physics lab's grid: 3 servo gains x 3 masses x 2 swing heights, 0.15 m/s, periods 0.5 to 4 s, 144 runs, one run per cell.
     With the 30 mm swing, the 1.7 s trot made more speed than the 0.5 s trot in 8 of the 9 gain-and-mass cells (the exception: kp 30 with 1.75x mass, 0.075 against 0.091 m/s) and tilted more in all 9. Median speed over the nine cells: 0.090 m/s at 0.5 s, 0.073 at 1 s,
     0.109 at 1.5 s, 0.116 at 1.7 s, 0.124 at 2 s, 0.118 at 2.5 s, 0.121 at 3 s; median tilt 5.7°, 7.2°, 7.6°, 8.0°, 8.5°, 9.2°, 9.7°. So 1.7 s sits on a plateau from about 1.5 s to 3 s: it is not a peak, and a longer period buys a little speed for more tilt. At 4 s no run counts as walked.
     **With the 60 mm swing the result reverses**: every period tilts 17° or more, the 1.7 s trot is faster than the 0.5 s trot in only 1 of 9 cells and walks in 6 of 9 (the 0.5 s trot walks in all 9), and 0.5 s makes the most speed in 6 of the 9 cells. The case for 1.7 s is therefore for the default 30 mm swing only.
     (An earlier version of this section said the 1.7 s trot was faster in all nine cells of `physics_sweep.json`; that compared against a 0.5 s run that was not kept in `results/`, and the committed grid shows 8 of 9.)
   - *Not shown.* Why a longer period helps (not traced); behaviour on a real robot; duty or offsets other than the defaults; speeds other than 0.15 m/s in the gain, mass and swing grid; periods between the ones listed beyond what the JSONs hold.
   - *Tuned default (22 mm swing, 0.16 m stance, kp 50, kd 0.9).* After the sweeps below, `gaits.TROT` has a 22 mm swing (it was 30 mm; `TROT_STAGE1` keeps 0.5 s and 30 mm), and the physics trot runs with a 0.16 m stance, kp 50 and kd 0.9 through `gait_replay.settings_for(gait)`, which gives every other gait (crawl, pace) the original 0.2125 m stance and kp 10, kd 0.2. The live explorer presets, the JavaScript port, the website's trot presets and the physics demos follow; the stand and sit-to-stand tests, the crawl, the pace and the physics lab grid (`physics_sweep.json`, which varies kp and swing itself) are unchanged. `pace_diagnosis.py`, `gait_speed_diagnosis.py` and `trot_period_sweep.py` are pinned to the 30 mm swing, 0.2125 m stance and kp 10 they were measured with, and were rerun after the change: their JSON files came out identical.
     What changed in `gait_replay.json` (one run per case, 1.72 kg, old values in brackets, the original 30 mm / 0.2125 m / kp 10 trot): at 0.15 m/s the body made 0.132 m/s (0.100), at 0.05 m/s 0.038 (0.011), at 0.30 m/s 0.249 (0.248); marching in place 0.000 m/s (-0.008); the 0.6 rad/s turn 0.535 rad/s (0.38) with tilt 6.5 degrees (12.1); tilt at 0.15 m/s 5.9 degrees (5.2); peak torque at 0.15 m/s 0.89 of stall (0.76). Two things got worse: tilt at 0.15 m/s rose from 5.2 to 5.9 degrees, and the 0.30 m/s run no longer counts as walked, because the servos are saturated for 41% of the walk (21% before; the limit is 20%). The crawl rows are identical. The Stage 1 geometry changed slightly (forward-trot joint margin 20.15 to 20.27 degrees, turn 10.79 to 10.89, median foot error 0.1126 to 0.1129 mm). Not tested at the tuned settings: turning at 1.2 rad/s, speeds above 0.30 m/s, and the demos' turn at 0.6 rad/s beyond this one run. The best settings came from a placeholder model; at 3 kg the 22 mm swing was worse than 30 mm ("Do the best values add up?").
   - *Use.* `gaits.TROT` is now 1.7 s everywhere: the physics replays, the physics demos (`trot_replay.gif`, `trot_iso.gif`, `trot_turn_iso.gif`, the last turning at 0.6 rad/s because 1.2 rad/s falls over at 1.7 s), the physics lab, the live explorer, the website's trot presets, and the Stage 1 geometry trot (JavaScript port, `body_metrics.json`, `trot_forward.gif`, `trot_turn.gif`).
     The geometry demos were first made at 0.5 s and regenerated at 1.7 s after this finding. That changed their numbers: worst forward-trot foot error 0.25 to 0.29 mm, closest joint approach in the 1.2 rad/s turn 22° to 10.8° (forward trot 23° to 20°), body centre up to 38 mm instead of 11 mm off the line between grounded feet. The original is kept as `gaits.TROT_STAGE1`. The pace (the trot with other offsets) moved to 1.7 s with it and its diagnosis was rerun (below).

   **Live explorer** (`scripts/gait/live_gait.py`, `mojito/sim/live_sim.py`, `scripts/gait/live_gait.html`): a local browser page with sliders for every timing and servo parameter, a draggable isometric view and the timing strips, driven by the same physics. Setup is in the README. With constant parameters it matches `run_gait` to 2 mm after 3 s (`tests/sim/test_live_sim.py`); changing parameters live is eased (phase accumulated, velocity and lift low-passed, foot targets speed-limited), which `run_gait` does not do.

   **Drive mode** (the Drive card of the same page, and the same controls on the website playground's Live physics tab via `scripts/website/serve_site.py`; both use `site/assets/drive.js`, whose key and stick mapping is tested in `tests/website/test_drive_js.py`): keyboard (W S forward and back, A D sidestep, Q E turn, Space keep walking, 1 2 trot or crawl, R reset, C chase camera) or any standard gamepad through the browser Gamepad API (left stick forward and sidestep, right stick turn, A, B, X, LB, RB for the same actions; 15% dead zone). It only sends `vx`, `vy`, `yaw_rate` and `walk` to the existing `/params`, so the physics and the easing above are unchanged; it walks while there is input and for about 0.5 s after, then blends to standing. Full-stick values are `live_sim.DRIVE`: trot 0.25 m/s forward, 0.12 sideways, 0.8 rad/s; crawl 0.06, 0.03, 0.3. Each motion alone, and the trot with all three together, ran 6 s without falling, but the robot is slower than commanded: in those runs the trot measured 0.237 m/s for a 0.25 command forward, the crawl 0.044 for 0.06, and the trot with all three at full stick (0.25 forward, 0.12 left, 0.8 rad/s) 0.10 forward and 0.18 left while turning, so the speed readout in the view is measured, not commanded. The pace is not offered because it barely moves in physics. `tests/sim/test_live_sim.py` checks the limits against the slider limits and that the full command does not fall in 4 s. The keyboard and a mocked gamepad were driven in headless Chromium on both pages (holding W for 4 s measured 0.24 m/s on the standalone page and 0.23 on the playground); a real controller has not been tried yet.

   **Pose mode** (M, or Y on a gamepad, switches between walking and posing; both pages): the robot stands, the feet stay planted and the same keys or sticks tilt the body: forward (W, left stick up) pitches the nose down, right (D, left stick right) rolls the right side down, Q or right stick left yaws the body left; released, the body springs back to level. It adds three smoothed parameters to `LiveSim`, `body_roll`, `body_pitch`, `body_yaw` (rad, limit ±0.4, ±0.4, ±0.5), applied with `body.apply_posture` to the foot targets after the gait and sway, so with zero posture the live run is unchanged (the match to `run_gait` above still passes). Full stick is `live_sim.POSE`: roll 0.15 rad (8.6°), pitch 0.15 rad (8.6°), yaw 0.2 rad (11.5°). Probed standing for 3 s with the trot settings: roll or pitch alone at 0.25 rad (14.3°) measured 14.1 to 15.5° and yaw 0.35 rad (20.1°) measured 17.1° (trot) and 19.3° (crawl); at all 27 corners of the stick box with the full-stick values no servo saturated and all four feet stayed down, with the worst corner 5.6° off the commanded angle in one of roll, pitch or yaw. The first values tried (0.2, 0.2, 0.3) saturated the servos at six corners of the trot, which is why they are smaller. Combined tilts at 0.25 rad or more saturate the servos and lift a foot (3 feet down at 0.35, 0.35, 0.5). In pose mode the camera stays still instead of following the yaw. `tests/sim/test_live_sim.py` checks that the feet do not move in the world while the body tilts (under 5 mm over 1.5 s), that it reaches the commanded roll and pitch within 2°, and that all eight corners hold; `tests/website/test_drive_js.py` checks the key and stick signs. Driven with the keyboard and a mocked gamepad in headless Chromium on both pages (8.4° roll, 8.5° pitch and 12.1° yaw measured for 8.6°, 8.6° and 11.5° commanded); not tried with a real controller. This is posing on flat ground in a placeholder model: it shows what range the legs and servos allow, not what a built robot could do. Keys are all-or-nothing, and the 0.3 s smoothing is what eases them in. Placeholder dimensions, servo figures and friction: this says nothing about the real robot.

   **Trying variations** (`scripts/gait/try_gait.py`). Starts from the trot or crawl and overrides any timing, runs it through the same physics, prints the timing table
   and the measured result, and writes `results/gait/try/<name>_timing.png`, `_iso.gif` and `.json` (not committed). For example
   `--base trot --offsets 0,0.5,0,0.5` (pace), `--duty 0.4` (a flight phase), `--base crawl --order FL,RR,FR,RL`, `--period 1.2 --duty 0.8 --kp 100 --kd 1`.
   Two runs so far, only to check the tool, one run each (3 cycles at most, see output of the script): crawl lifting FL, RR, FR, RL at 0.04 m/s stayed up and
   measured 0.028 m/s (same as the built-in order); a pace at 0.1 m/s (at the then-current 0.5 s period) stayed up with 6.9° tilt and measured -0.032 m/s (backward); at 1.7 s it measures -0.012 m/s (kp 10, 1.72 kg). See below for why the pace goes backward.

   **Why the pace goes backward** (`scripts/gait/pace_diagnosis.py`, `results/gait/pace_diagnosis.json`; pace = FL and RL together, then FR and RR; 0.1 m/s commanded; one run per entry).
   The pace was first diagnosed at the original 0.5 s period and redone at 1.7 s, the trot period used everywhere now (see "Longer trot periods"); both are given where they differ.
   - *Established for the base pace.* Only two feet on one side are commanded down, so the support is a line along that side, 125 mm from the
     centre of mass. The body is not held up sideways: it rolls toward the lifted side until the lifted feet touch the floor. Evidence: the roll range is 12.8°
     at 1.7 s (13.5° at 0.5 s), against 13.5° for `2 atan(lift / track)` with the 30 mm lift and the 0.25 m foot track; with a 60 mm lift it is 28.1°, against 26.9°.
     Load sits on the side commanded to swing: at 1.7 s the left feet carry 46% while the left legs are commanded down and 54% while the right legs are
     (51% / 50% for the trot; at 0.5 s: 39% / 61%). So 54% of the load is on feet commanded to swing (trot 31%; at 0.5 s: 65%, trot 23%).
     A loaded swing foot moves forward in the body frame, so it pushes the body back: the load-weighted ask is -0.011 m/s and the body made -0.012 (at 0.5 s: ask -0.032, measured -0.030).
     The same holds with kp 100 (ask -0.013, measured +0.001) and the ideal servo (ask -0.013, measured -0.017), so it is not the servo softness of the trot.
     The effect shrinks with the period: measured -0.030 m/s at 0.5 s, -0.016 at 1 s, -0.012 at 1.7 s and -0.012 at 2 s, because the swing foot then moves forward more slowly.
     The pace is much less backward at 1.7 s than at 0.5 s, but it still does not go forward: 15 of the 18 pace cells of the physics lab (0.1 m/s) end with the body moving backward (-0.006 to -0.060 m/s; one of them, kp 10 with 1.75x mass and a 60 mm swing, fell); the three that do not are kp 100 at 0.001 to 0.006 m/s.
   - *Variants at 1.7 s.* Swing height 60 mm: ask -0.025, measured -0.022 (59% of the load on swing feet, roll 28.1°). Duty 0.6: ask -0.013, measured -0.015; duty 0.7: ask -0.015, measured -0.017
     (swing-feet load 43% and 33%). In all of these the ask and the measured speed agree to within 0.003 m/s. At 0.5 s the explanation failed for three of these variants
     (60 mm: ask +0.059, measured -0.058; duty 0.7: ask +0.013, measured -0.039; a lateral shift: ask +0.084, measured -0.036), and "something else also pushes the body back" was
     the finding; at 1.7 s that is not seen in these runs, and why it differed at 0.5 s was not traced.
   - *Not established.* The lateral body shift toward the stance side (the crawl's lean idea; sign and shape found by trial at 0.5 s) did not unload the swing feet at 1.7 s: they still carry 66% (30 mm shift, measured -0.032, roll 14.3°),
     and the 60 mm shift tipped the robot over. So this quick test does not test the idea at 1.7 s; it would need tuning for that period. I did not find a pace that goes forward.

   Not shown: any real robot's behaviour (dimensions, masses, gains, latency, servo figures and friction are placeholders);
   other speeds, turn rates, terrain or disturbances; whether higher gains or a tuned gait close the speed gap; one run
   of each case, no randomisation.
   - *Other parameters at 1.7 s* (`scripts/gait/sweeps/trot_1p7_parameter_sweep.py`, `results/gait/sweeps/parameters/trot_1p7_parameter_sweep.json`; plot `results/gait/sweeps/parameters/trot_1p7_parameter_effects.png` from `scripts/gait/sweeps/plot_trot_1p7_parameter_sweep.py`; website: Results, "What each parameter does to the 1.7 s trot"): 192 runs, 3 cycles each, 1.72 kg, one run per cell. Grid: duty 0.5, 0.55, 0.6, 0.7; commanded speed 0.05, 0.1, 0.15, 0.2 m/s; stance height 0.19, 0.2125, 0.23 m; kp 10 or 30; swing height 30 or 60 mm. 183 of 192 walked ("walked" means upright, not too low, run finished; it does not mean the speed was met). All 9 falls had a 60 mm swing and duty 0.55 or more, at 0.15 or 0.2 m/s. Below, "speed ratio" is measured over commanded forward speed and each figure is the median over the runs that walked, with every other parameter varied across its grid; a ratio of 1 would be the commanded speed.
     - *Swing height* mattered most. 30 mm: speed ratio 0.64, tilt 7.3°, peak torque 0.78 of stall, no falls. 60 mm: ratio 0.42, tilt 18.7°, torque 0.98, 9 of 96 fell.
     - *Servo gain.* kp 30 against kp 10: ratio 0.55 against 0.38, 3 against 6 falls out of 96, but a larger tilt (9.3° against 6.9°).
     - *Commanded speed.* The ratio rises with it: 0.32 at 0.05 m/s, 0.42 at 0.1, 0.54 at 0.15, 0.58 at 0.2. Absolute speed rises too, so the shortfall looks like a roughly fixed amount rather than a fixed fraction; I did not fit or trace it. Falls only at 0.15 and 0.2 m/s (3 and 6 of 48).
     - *Stance height.* Lower gave more speed and fewer falls: ratio 0.55, 0.48, 0.42 and falls 0, 3, 6 of 64 for 0.19, 0.2125, 0.23 m. The walked runs tilted more at 0.19 m (11.4°) than at 0.23 m (9.1°). A lower body also changes joint angles, and I did not separate that from the height itself.
     - *Duty.* No effect on speed (median ratio 0.47 to 0.50). Falls rose with it (0, 1, 2, 6 of 48 at 0.5, 0.55, 0.6, 0.7). Median tilt was 13.6° at 0.5 and 8.4 to 9.3° above it; I did not trace that. Duty 0.5 has moments with only one foot down in some runs (`feet_down_min` 1).
     - *Best settings in this grid* (highest speed ratio among the runs that walked, per commanded speed; all four were at 30 mm swing, kp 30 and 0.19 m stance, and the other duties were within 0.03 of them). At 0.05 m/s: duty 0.5, 0.034 m/s measured, ratio 0.68, tilt 8.8°, peak torque 0.75 of stall. At 0.1 m/s: duty 0.55, 0.080 m/s, ratio 0.80, 7.8°, 0.80. At 0.15 m/s: duty 0.55, 0.128 m/s, ratio 0.86, 7.7°, 0.85. At 0.2 m/s: duty 0.55, 0.178 m/s, ratio 0.89, 8.1°, 0.84. So within the grid the best is a 30 mm swing, kp 30, a 0.19 m stance and a duty of 0.5 to 0.55. Duty 0.5 at 0.19 m, kp 30, 30 mm made ratios 0.68, 0.78, 0.85, 0.88 at the four speeds with at least two feet down in every run. Duty 0.7 at 0.2 m/s with the same settings made ratio 0.5 with servos saturated 36% of the walk. Three of the four winning values (30 mm, kp 30, 0.19 m) are the edge of the range tested, so the optimum may lie beyond them; the next sweep tests that. Not shown: that these are optima (one run per cell, no spread), or that they hold at other masses.
     Not shown: interactions beyond the grid, mass (see the robustness sweep), friction, turning, terrain, or why these effects occur (not traced). The medians mix other parameters, so they describe this grid, not each parameter alone, and one run per cell gives no spread.
   - *Pushing the best settings further, one parameter at a time, with repeats* (`scripts/gait/sweeps/trot_1p7_one_at_a_time.py`, `results/gait/sweeps/parameters/trot_1p7_one_at_a_time.json`; means and standard deviations in `trot_1p7_one_at_a_time_summary.json` and the plot `trot_1p7_one_at_a_time.png`, both from `scripts/gait/sweeps/plot_trot_1p7_one_at_a_time.py`; website: Results, "Pushing the best settings further"). Baseline: period 1.7 s, duty 0.5, diagonal offsets, 0.15 m/s commanded, 0.19 m stance, 30 mm swing, kp 30, kd 0.5, 1.72 kg. Four sweeps, each changing one thing: kp 15 to 150 in 10 steps with kd tied as 0.5 x sqrt(kp / 30); kd 0.1 to 2.0 in 10 values at kp 30; swing 10 to 30 mm in 11 values; stance height 0.09 to 0.19 m in 11 values. 420 runs, 3 cycles each.
     *What "10 iterations" means here.* The simulation is deterministic, so repeating an identical run gives an identical number. Each value was run 10 times with a random draw of two small perturbations: every body mass times a uniform factor in 0.95 to 1.05, and the servo latency uniform in 18 to 22 ms. Draw i uses seed i, and the same 10 draws are used at every value. So the standard deviations below are the spread from those perturbations only. They are small because the perturbations are small, they say nothing about a real robot, and they do not include model error, so a difference of one or two standard deviations between neighbouring values is not evidence of a real difference. Speed ratio is measured over commanded forward speed, over all 10 runs (a run that fell counts with the speed it made); tilt and torque are over the runs that stayed upright (tilt under 30 degrees). "walked" is the replay's flag: upright, not too low, and every commanded foot target inside the joint limits; "in limits" counts that last condition. sd is the sample standard deviation (n - 1). 350 of 420 runs walked; the 70 that did not are the stance heights 0.09 to 0.15 m, below.
     **Servo gain kp** (kd tied):
     | servo gain kp [N·m/rad] (kd tied) | walked | in limits | speed ratio mean | sd | min | max | tilt mean (deg) | sd | torque / stall mean | sd |
     |---|---|---|---|---|---|---|---|---|---|---|
     | 15 | 10/10 | 10/10 | 0.789 | 0.003 | 0.78 | 0.79 | 6.4 | 0.1 | 0.79 | 0.02 |
     | 30 | 10/10 | 10/10 | 0.848 | 0.004 | 0.84 | 0.85 | 7.9 | 0.0 | 0.85 | 0.01 |
     | 45 | 10/10 | 10/10 | 0.860 | 0.003 | 0.85 | 0.86 | 8.2 | 0.0 | 0.87 | 0.02 |
     | 60 | 10/10 | 10/10 | 0.856 | 0.003 | 0.85 | 0.86 | 8.5 | 0.0 | 0.94 | 0.01 |
     | 75 | 10/10 | 10/10 | 0.845 | 0.003 | 0.84 | 0.85 | 8.6 | 0.0 | 1.00 | 0.00 |
     | 90 | 10/10 | 10/10 | 0.830 | 0.003 | 0.82 | 0.83 | 8.7 | 0.0 | 1.00 | 0.00 |
     | 105 | 10/10 | 10/10 | 0.812 | 0.003 | 0.81 | 0.82 | 8.8 | 0.0 | 1.00 | 0.00 |
     | 120 | 10/10 | 10/10 | 0.792 | 0.004 | 0.79 | 0.80 | 8.9 | 0.0 | 1.00 | 0.00 |
     | 135 | 10/10 | 10/10 | 0.779 | 0.003 | 0.78 | 0.79 | 9.0 | 0.0 | 1.00 | 0.00 |
     | 150 | 10/10 | 10/10 | 0.768 | 0.003 | 0.76 | 0.77 | 9.0 | 0.0 | 1.00 | 0.00 |
     The ratio peaks at kp 45 (0.860, sd 0.003) and falls to 0.768 at kp 150. kp 45 against kp 60 (0.860 against 0.856) is about one sd, so I would not call the peak sharper than "somewhere from 30 to 60". Peak torque reaches the stall limit (1.00) from kp 75 up, and tilt rises steadily (6.4° at 15, 9.0° at 150). So a stiffer servo beyond about 45 made the trot slower here, not faster. Why was not traced; saturation from the torque limit is one candidate I did not test.
     **Servo damping kd** (kp 30):
     | servo damping kd [N·m·s/rad] (kp 30) | walked | in limits | speed ratio mean | sd | min | max | tilt mean (deg) | sd | torque / stall mean | sd |
     |---|---|---|---|---|---|---|---|---|---|---|
     | 0.1 | 10/10 | 10/10 | 0.859 | 0.002 | 0.85 | 0.86 | 8.1 | 0.0 | 0.96 | 0.02 |
     | 0.25 | 10/10 | 10/10 | 0.853 | 0.003 | 0.85 | 0.86 | 8.0 | 0.0 | 0.89 | 0.01 |
     | 0.4 | 10/10 | 10/10 | 0.849 | 0.004 | 0.84 | 0.85 | 7.9 | 0.0 | 0.86 | 0.01 |
     | 0.55 | 10/10 | 10/10 | 0.848 | 0.004 | 0.84 | 0.85 | 7.8 | 0.0 | 0.85 | 0.01 |
     | 0.7 | 10/10 | 10/10 | 0.847 | 0.004 | 0.84 | 0.85 | 7.7 | 0.0 | 0.84 | 0.01 |
     | 0.85 | 10/10 | 10/10 | 0.845 | 0.003 | 0.84 | 0.85 | 7.6 | 0.0 | 0.83 | 0.01 |
     | 1 | 10/10 | 10/10 | 0.845 | 0.003 | 0.84 | 0.85 | 7.5 | 0.0 | 0.80 | 0.00 |
     | 1.25 | 10/10 | 10/10 | 0.846 | 0.003 | 0.84 | 0.85 | 7.3 | 0.0 | 0.79 | 0.02 |
     | 1.5 | 10/10 | 10/10 | 0.812 | 0.027 | 0.77 | 0.84 | 7.5 | 0.1 | 1.00 | 0.00 |
     | 2 | 10/10 | 10/10 | -0.180 | 0.041 | -0.24 | -0.13 | 7.1 | 0.1 | 1.00 | 0.00 |
     Flat from 0.1 to 1.25 (0.845 to 0.859, a spread of 0.014; tilt falls from 8.1° to 7.3° as kd rises). At 1.5 the mean drops to 0.812 with a larger sd (0.027) and the torque saturates. At 2.0 the body goes backward (mean speed ratio -0.180): all 10 of those runs still count as "walked", because the flag checks uprightness, not direction. I did not trace why heavy damping reverses the motion.
     **Swing height**:
     | swing height [mm] | walked | in limits | speed ratio mean | sd | min | max | tilt mean (deg) | sd | torque / stall mean | sd |
     |---|---|---|---|---|---|---|---|---|---|---|
     | 10 | 10/10 | 10/10 | 0.748 | 0.003 | 0.74 | 0.75 | 1.3 | 0.0 | 0.81 | 0.01 |
     | 12 | 10/10 | 10/10 | 0.800 | 0.003 | 0.79 | 0.80 | 2.0 | 0.0 | 0.79 | 0.01 |
     | 14 | 10/10 | 10/10 | 0.823 | 0.003 | 0.82 | 0.83 | 2.6 | 0.0 | 0.77 | 0.01 |
     | 16 | 10/10 | 10/10 | 0.837 | 0.003 | 0.83 | 0.84 | 3.2 | 0.0 | 0.80 | 0.01 |
     | 18 | 10/10 | 10/10 | 0.844 | 0.003 | 0.84 | 0.85 | 3.9 | 0.0 | 0.83 | 0.01 |
     | 20 | 10/10 | 10/10 | 0.858 | 0.003 | 0.85 | 0.86 | 4.6 | 0.0 | 0.80 | 0.02 |
     | 22 | 10/10 | 10/10 | 0.859 | 0.003 | 0.85 | 0.86 | 5.2 | 0.0 | 0.81 | 0.02 |
     | 24 | 10/10 | 10/10 | 0.859 | 0.003 | 0.85 | 0.86 | 5.9 | 0.0 | 0.82 | 0.01 |
     | 26 | 10/10 | 10/10 | 0.858 | 0.003 | 0.85 | 0.86 | 6.5 | 0.0 | 0.81 | 0.01 |
     | 28 | 10/10 | 10/10 | 0.854 | 0.003 | 0.85 | 0.86 | 7.2 | 0.0 | 0.82 | 0.01 |
     | 30 | 10/10 | 10/10 | 0.848 | 0.004 | 0.84 | 0.85 | 7.9 | 0.0 | 0.85 | 0.01 |
     The speed ratio is highest from 20 to 26 mm (0.858 to 0.859, within about one sd of each other) and lower at both ends: 0.748 at 10 mm and 0.848 at 30 mm. Tilt falls almost linearly as the swing drops, from 7.9° at 30 mm to 1.3° at 10 mm. So 20 mm gives about the speed of 30 mm (0.858 against 0.848) with 4.6° of tilt instead of 7.9°.
     **Stance height**:
     | stance height [m] | walked | in limits | speed ratio mean | sd | min | max | tilt mean (deg) | sd | torque / stall mean | sd |
     |---|---|---|---|---|---|---|---|---|---|---|
     | 0.09 | 0/10 | 0/10 | 0.172 | 0.007 | 0.16 | 0.18 | 2.5 | 0.0 | 1.00 | 0.00 |
     | 0.1 | 0/10 | 0/10 | 0.229 | 0.008 | 0.21 | 0.24 | 2.0 | 0.0 | 1.00 | 0.00 |
     | 0.11 | 0/10 | 0/10 | 0.434 | 0.003 | 0.43 | 0.44 | 2.2 | 0.0 | 1.00 | 0.00 |
     | 0.12 | 0/10 | 0/10 | 0.684 | 0.001 | 0.68 | 0.69 | 2.8 | 0.1 | 1.00 | 0.00 |
     | 0.13 | 0/10 | 0/10 | 0.785 | 0.002 | 0.78 | 0.79 | 4.7 | 0.0 | 1.00 | 0.00 |
     | 0.14 | 0/10 | 0/10 | 0.875 | 0.002 | 0.87 | 0.88 | 6.3 | 0.0 | 0.99 | 0.00 |
     | 0.15 | 0/10 | 0/10 | 0.879 | 0.002 | 0.88 | 0.88 | 7.5 | 0.0 | 0.84 | 0.01 |
     | 0.16 | 10/10 | 10/10 | 0.870 | 0.003 | 0.87 | 0.87 | 8.1 | 0.0 | 0.86 | 0.02 |
     | 0.17 | 10/10 | 10/10 | 0.864 | 0.003 | 0.86 | 0.87 | 8.0 | 0.0 | 0.84 | 0.01 |
     | 0.18 | 10/10 | 10/10 | 0.855 | 0.003 | 0.85 | 0.86 | 8.0 | 0.0 | 0.83 | 0.01 |
     | 0.19 | 10/10 | 10/10 | 0.848 | 0.004 | 0.84 | 0.85 | 7.9 | 0.0 | 0.85 | 0.01 |
     At 0.15 m and below, none of the 70 runs counts as walked, and the reason is that the commanded foot path leaves the joint limits (0 of 10 "in limits"), not that the robot fell: the largest tilt is 2 to 7.5°. At 0.09 to 0.11 m the servos are saturated almost the whole walk (torque 1.00) and the speed ratio collapses to 0.17 to 0.43. The ratio at 0.14 and 0.15 m (0.875, 0.879) is higher than at 0.19 m (0.848), but those runs asked for foot positions the joints cannot reach, so I do not count them as a valid test of the gait. Among heights where every target is reachable (0.16 m and above) the ratio rises as the body gets lower: 0.870 at 0.16 m against 0.848 at 0.19 m, with 8.1° against 7.9° tilt.
     *What this does and does not show.* One parameter at a time from one baseline: the best value of each is not necessarily the best together (for example kp 45, 20 to 24 mm swing and 0.16 m stance were not run together). The baseline's duty 0.5, 0.15 m/s command and mass were not changed. The perturbation spread is small, and the model's own uncertainty (friction, damping, servo figures) is not in it. Everything is placeholder.
   - *kp and kd varied independently, 14 values of kp times 17 values of kd, 10 runs per cell* (`scripts/gait/sweeps/trot_1p7_kp_kd_grid.py`, `results/gait/sweeps/servo_gains/trot_1p7_kp_kd_grid.json`; means, standard deviations and the best kd per kp in `trot_1p7_kp_kd_grid_summary.json`; plots `trot_1p7_kp_kd_lines.png` and `trot_1p7_kp_kd_maps.png` from `scripts/gait/sweeps/plot_trot_1p7_kp_kd_grid.py`; website: Results, "kd at each kp"). Why: the kp sweep above tied kd to kp, which could have hidden a better kd at high kp (tilt rose with kp), so kp 20 to 150 in steps of 10 was crossed with kd 0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.1, 1.2, 1.3, 1.4, 1.5, 1.7, 2.0, 3.0 (denser where the speed falls away). Everything else is the baseline (1.7 s, duty 0.5, 0.15 m/s, 0.19 m stance, 30 mm swing, 1.72 kg). 2380 runs, 3 cycles each, the same 10 random mass (+-5%) and latency (18 to 22 ms) draws in every cell as before; the spread is from those draws only. Speed ratio is over all 10 runs; tilt and torque are over runs with tilt under 30 degrees (all 2380 stayed upright).
     *The curve is not bell-shaped.* At each kp, speed rises with kd from a low value, sits on a plateau, then falls off a cliff above kd 1.4. At kd 1.5 the mean ratio is 0.02 (kp 20) to 0.26 (kp 150) below its peak, at 1.7 it is 0.30 to 0.34 at every kp, at 2.0 it is negative (the body goes backward) for kp up to 90 and 0.01 to 0.09 above that, and at 3.0 it is negative for every kp (-0.19 to -0.88). The cliff is at about the same kd for every kp from 20 to 150 (the ratio first falls more than 0.05 below the peak at 1.5, or at 1.7 for kp 20 and 30, where it is gentler at 1.5), so it does not scale with kp as the tied rule assumed. Why it is there was not traced (the servo model has a 20 ms latency and the simulation is stepped at 2 ms; I did not test whether it is a stability limit of those). The standard deviation grows at and beyond it (0.013 to 0.046 at kd 1.5 and above, against mostly 0.002 to 0.006 on the plateau).
     *Best kd per kp* (highest mean speed ratio; the plateau is the kd range within 0.01 of that peak; the cliff is the first kd above the peak that is more than 0.05 below it):
     | kp | best kd | speed ratio mean | sd | tilt (deg) | torque / stall | kd within 0.01 of the peak | first kd 0.05 below it (cliff) |
     |---|---|---|---|---|---|---|---|
     | 20 | 0.06 | 0.830 | 0.003 | 7.9 | 0.89 | 0.03 to 0.3 | 1.7 |
     | 30 | 0.1 | 0.859 | 0.002 | 8.1 | 0.96 | 0.1 to 1.4 | 1.7 |
     | 40 | 0.3 | 0.863 | 0.002 | 8.2 | 0.88 | 0.1 to 1.4 | 1.5 |
     | 50 | 1.4 | 0.865 | 0.003 | 8.1 | 0.99 | 0.2 to 1.4 | 1.5 |
     | 60 | 1.4 | 0.865 | 0.003 | 8.3 | 1.00 | 0.7 to 1.4 | 1.5 |
     | 70 | 1.4 | 0.862 | 0.002 | 8.4 | 1.00 | 1.2 to 1.4 | 1.5 |
     | 80 | 1.4 | 0.855 | 0.002 | 8.6 | 1.00 | 1.3 to 1.4 | 1.5 |
     | 90 | 1.4 | 0.846 | 0.001 | 8.7 | 1.00 | 1.3 to 1.4 | 1.5 |
     | 100 | 1.4 | 0.837 | 0.002 | 8.7 | 1.00 | 1.3 to 1.4 | 1.5 |
     | 110 | 1.4 | 0.826 | 0.005 | 8.8 | 1.00 | 1.3 to 1.4 | 1.5 |
     | 120 | 1.4 | 0.813 | 0.003 | 8.9 | 1.00 | 1.3 to 1.4 | 1.5 |
     | 130 | 1.4 | 0.802 | 0.004 | 8.9 | 1.00 | 1.3 to 1.4 | 1.5 |
     | 140 | 1.4 | 0.788 | 0.009 | 9.0 | 1.00 | 1.2 to 1.4 | 1.5 |
     | 150 | 1.3 | 0.778 | 0.004 | 9.0 | 1.00 | 1.2 to 1.3 | 1.5 |
     *Mean speed ratio* (rows kp, columns kd):
     mean speed ratio (rows kp, columns kd)
     | kp \ kd | 0 | 0.03 | 0.06 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.7 | 2 | 3 |
     |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
     | 20 | 0.80 | 0.82 | 0.83 | 0.83 | 0.83 | 0.82 | 0.82 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.81 | 0.30 | -0.32 | -0.88 |
     | 30 | 0.79 | 0.83 | 0.85 | 0.86 | 0.85 | 0.85 | 0.85 | 0.85 | 0.84 | 0.84 | 0.85 | 0.85 | 0.85 | 0.81 | 0.33 | -0.18 | -0.61 |
     | 40 | 0.78 | 0.81 | 0.83 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.78 | 0.32 | -0.13 | -0.50 |
     | 50 | 0.76 | 0.80 | 0.82 | 0.85 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.87 | 0.74 | 0.32 | -0.10 | -0.43 |
     | 60 | 0.76 | 0.78 | 0.79 | 0.81 | 0.84 | 0.85 | 0.85 | 0.86 | 0.86 | 0.86 | 0.86 | 0.86 | 0.87 | 0.71 | 0.32 | -0.08 | -0.39 |
     | 70 | 0.71 | 0.75 | 0.78 | 0.80 | 0.83 | 0.84 | 0.85 | 0.85 | 0.85 | 0.85 | 0.85 | 0.86 | 0.86 | 0.69 | 0.32 | -0.06 | -0.36 |
     | 80 | 0.70 | 0.74 | 0.78 | 0.80 | 0.82 | 0.83 | 0.84 | 0.84 | 0.84 | 0.84 | 0.84 | 0.85 | 0.85 | 0.64 | 0.32 | -0.04 | -0.33 |
     | 90 | 0.70 | 0.74 | 0.76 | 0.78 | 0.80 | 0.82 | 0.82 | 0.83 | 0.83 | 0.83 | 0.83 | 0.84 | 0.85 | 0.60 | 0.32 | -0.01 | -0.31 |
     | 100 | 0.67 | 0.71 | 0.73 | 0.75 | 0.78 | 0.80 | 0.81 | 0.82 | 0.82 | 0.82 | 0.82 | 0.83 | 0.84 | 0.58 | 0.33 | 0.01 | -0.28 |
     | 110 | 0.68 | 0.67 | 0.70 | 0.73 | 0.76 | 0.79 | 0.79 | 0.80 | 0.80 | 0.81 | 0.81 | 0.82 | 0.83 | 0.57 | 0.32 | 0.02 | -0.26 |
     | 120 | 0.67 | 0.68 | 0.70 | 0.72 | 0.74 | 0.77 | 0.78 | 0.79 | 0.79 | 0.80 | 0.80 | 0.81 | 0.81 | 0.55 | 0.33 | 0.04 | -0.24 |
     | 130 | 0.61 | 0.63 | 0.68 | 0.69 | 0.73 | 0.76 | 0.77 | 0.77 | 0.78 | 0.79 | 0.79 | 0.80 | 0.80 | 0.54 | 0.34 | 0.06 | -0.23 |
     | 140 | 0.57 | 0.58 | 0.62 | 0.64 | 0.71 | 0.74 | 0.75 | 0.76 | 0.77 | 0.78 | 0.78 | 0.79 | 0.79 | 0.53 | 0.34 | 0.08 | -0.20 |
     | 150 | 0.54 | 0.57 | 0.58 | 0.61 | 0.69 | 0.72 | 0.74 | 0.74 | 0.75 | 0.77 | 0.77 | 0.78 | 0.76 | 0.52 | 0.34 | 0.09 | -0.19 |
     *Standard deviation of the speed ratio* (10 runs per cell):
     sd of speed ratio
     | kp \ kd | 0 | 0.03 | 0.06 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.7 | 2 | 3 |
     |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
     | 20 | 0.004 | 0.003 | 0.003 | 0.003 | 0.003 | 0.004 | 0.004 | 0.004 | 0.003 | 0.003 | 0.003 | 0.004 | 0.004 | 0.013 | 0.029 | 0.043 | 0.031 |
     | 30 | 0.011 | 0.005 | 0.004 | 0.002 | 0.004 | 0.004 | 0.004 | 0.004 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.027 | 0.025 | 0.041 | 0.025 |
     | 40 | 0.006 | 0.004 | 0.004 | 0.004 | 0.002 | 0.002 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.002 | 0.034 | 0.025 | 0.038 | 0.034 |
     | 50 | 0.004 | 0.003 | 0.002 | 0.003 | 0.003 | 0.003 | 0.002 | 0.003 | 0.002 | 0.003 | 0.003 | 0.003 | 0.003 | 0.033 | 0.023 | 0.038 | 0.033 |
     | 60 | 0.007 | 0.006 | 0.005 | 0.005 | 0.003 | 0.003 | 0.002 | 0.003 | 0.003 | 0.002 | 0.002 | 0.002 | 0.003 | 0.035 | 0.025 | 0.036 | 0.028 |
     | 70 | 0.016 | 0.004 | 0.004 | 0.006 | 0.003 | 0.003 | 0.004 | 0.003 | 0.002 | 0.003 | 0.003 | 0.003 | 0.002 | 0.034 | 0.023 | 0.031 | 0.030 |
     | 80 | 0.009 | 0.008 | 0.006 | 0.004 | 0.003 | 0.004 | 0.002 | 0.003 | 0.003 | 0.002 | 0.002 | 0.003 | 0.002 | 0.043 | 0.025 | 0.033 | 0.030 |
     | 90 | 0.014 | 0.008 | 0.005 | 0.005 | 0.004 | 0.003 | 0.005 | 0.002 | 0.003 | 0.002 | 0.002 | 0.002 | 0.001 | 0.026 | 0.027 | 0.046 | 0.025 |
     | 100 | 0.016 | 0.012 | 0.006 | 0.007 | 0.004 | 0.005 | 0.004 | 0.003 | 0.003 | 0.003 | 0.002 | 0.001 | 0.002 | 0.022 | 0.023 | 0.037 | 0.027 |
     | 110 | 0.029 | 0.016 | 0.012 | 0.006 | 0.003 | 0.006 | 0.003 | 0.003 | 0.003 | 0.003 | 0.002 | 0.002 | 0.005 | 0.030 | 0.020 | 0.031 | 0.028 |
     | 120 | 0.026 | 0.021 | 0.010 | 0.007 | 0.004 | 0.006 | 0.004 | 0.004 | 0.002 | 0.003 | 0.003 | 0.002 | 0.003 | 0.024 | 0.024 | 0.034 | 0.022 |
     | 130 | 0.029 | 0.025 | 0.012 | 0.012 | 0.005 | 0.005 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.004 | 0.020 | 0.022 | 0.028 | 0.036 |
     | 140 | 0.017 | 0.019 | 0.016 | 0.019 | 0.009 | 0.008 | 0.005 | 0.004 | 0.003 | 0.002 | 0.002 | 0.002 | 0.009 | 0.018 | 0.024 | 0.026 | 0.030 |
     | 150 | 0.019 | 0.013 | 0.014 | 0.011 | 0.010 | 0.007 | 0.003 | 0.003 | 0.003 | 0.003 | 0.003 | 0.004 | 0.019 | 0.021 | 0.027 | 0.033 | 0.030 |
     *Mean largest tilt in degrees:*
     mean tilt (deg, upright runs)
     | kp \ kd | 0 | 0.03 | 0.06 | 0.1 | 0.2 | 0.3 | 0.5 | 0.7 | 0.9 | 1.1 | 1.2 | 1.3 | 1.4 | 1.5 | 1.7 | 2 | 3 |
     |---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
     | 20 | 8.2 | 7.9 | 7.9 | 7.8 | 7.6 | 7.4 | 7.0 | 6.8 | 6.5 | 6.3 | 6.2 | 6.1 | 6.0 | 6.2 | 6.5 | 5.9 | 5.4 |
     | 30 | 8.5 | 8.4 | 8.4 | 8.1 | 8.0 | 8.0 | 7.9 | 7.7 | 7.5 | 7.4 | 7.3 | 7.2 | 7.2 | 7.5 | 7.7 | 7.1 | 6.7 |
     | 40 | 8.8 | 8.8 | 8.5 | 8.4 | 8.3 | 8.2 | 8.2 | 8.1 | 8.0 | 7.9 | 7.9 | 7.8 | 7.7 | 8.1 | 8.2 | 7.8 | 7.4 |
     | 50 | 8.8 | 8.7 | 8.5 | 8.7 | 8.4 | 8.4 | 8.4 | 8.3 | 8.3 | 8.2 | 8.2 | 8.1 | 8.1 | 8.5 | 8.5 | 8.2 | 7.9 |
     | 60 | 9.0 | 9.0 | 9.0 | 8.9 | 8.6 | 8.6 | 8.5 | 8.5 | 8.4 | 8.4 | 8.3 | 8.3 | 8.3 | 8.7 | 8.7 | 8.4 | 8.3 |
     | 70 | 8.9 | 8.8 | 8.9 | 8.8 | 8.8 | 8.7 | 8.7 | 8.6 | 8.5 | 8.5 | 8.5 | 8.5 | 8.4 | 8.8 | 8.8 | 8.6 | 8.4 |
     | 80 | 9.0 | 9.1 | 8.9 | 8.9 | 9.0 | 8.9 | 8.8 | 8.7 | 8.6 | 8.6 | 8.6 | 8.6 | 8.6 | 8.9 | 8.9 | 8.7 | 8.7 |
     | 90 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 9.0 | 8.9 | 8.8 | 8.7 | 8.7 | 8.7 | 8.7 | 8.7 | 9.0 | 9.0 | 8.8 | 8.7 |
     | 100 | 9.1 | 9.1 | 9.1 | 9.1 | 9.1 | 9.0 | 8.9 | 8.9 | 8.8 | 8.8 | 8.8 | 8.7 | 8.7 | 9.0 | 9.0 | 8.9 | 8.8 |
     | 110 | 9.2 | 9.2 | 9.1 | 9.2 | 9.2 | 9.2 | 9.0 | 8.9 | 8.9 | 8.8 | 8.8 | 8.8 | 8.8 | 9.0 | 9.0 | 9.0 | 8.9 |
     | 120 | 9.1 | 9.2 | 9.2 | 9.2 | 9.2 | 9.2 | 9.1 | 9.0 | 8.9 | 8.9 | 8.9 | 8.8 | 8.9 | 9.1 | 9.1 | 9.0 | 8.9 |
     | 130 | 9.2 | 9.2 | 9.2 | 9.2 | 9.2 | 9.2 | 9.2 | 9.0 | 9.0 | 8.9 | 8.9 | 8.9 | 8.9 | 9.1 | 9.1 | 9.0 | 9.0 |
     | 140 | 9.3 | 9.2 | 9.2 | 9.2 | 9.2 | 9.3 | 9.2 | 9.1 | 9.0 | 9.0 | 9.0 | 8.9 | 9.0 | 9.2 | 9.1 | 9.1 | 9.1 |
     | 150 | 9.3 | 9.2 | 9.2 | 9.2 | 9.2 | 9.3 | 9.3 | 9.1 | 9.1 | 9.0 | 9.0 | 9.0 | 9.1 | 9.2 | 9.1 | 9.1 | 9.1 |
     *What it shows.* (1) At kp 50 to 60 the speed ratio is 0.865 on the plateau (kd 0.2 to 1.4 at kp 50, 0.7 to 1.4 at kp 60), the best in this grid; kp 40 to 70 are within 0.003 of it (0.862 to 0.865), so I would not call one of them the optimum. (2) The highest kd below the cliff (1.4) (1.3 at kp 150) is the best at every kp from 50 up, so for high kp more damping helps up to the cliff and not beyond. But it helps little: at kp 150 the best is 0.778 (kd 1.3), against 0.768 for the tied rule (kd 1.12) and 0.865 at kp 50 to 60. So the tied rule was not what made high kp slow; no kd rescues it. (3) Tilt depends mostly on kp, not kd: at kp 150 it is 9.0 to 9.3 degrees for every kd from 0 to 3, while at kp 20 it falls from 8.2 (kd 0) to 6.0 (kd 1.4) and at kp 40 from 8.8 to 7.7. Raising kd from 0 to 1.4 lowers the tilt by 2.2 degrees at kp 20 and by 0.2 at kp 150; that is the whole effect. So the tilt at high kp is not a damping problem in this model. (4) The torque is at its stall limit (peak 1.00 of stall) at the best kd for every kp from 60 up, consistent with a stiff servo saturating, but I did not test that as the cause of the slowdown. (5) The low-kd side matters more at high kp: kd 0 costs 0.03 (kp 20), 0.07 (kp 30) and 0.24 (kp 150) of the mean ratio against the peak. For use: a mid-plateau kd (about 0.7 to 1.1) at kp 50 to 60 gives 0.86 with margin from the cliff; kd 1.4 gives the same or a hair more but sits one grid step from a sharp drop.
     *Not shown.* The cliff and plateau edges are located to the grid (0.1 steps near the cliff), not exactly. One baseline (0.15 m/s, 0.19 m, 30 mm, duty 0.5, 1.72 kg); other speeds, masses and swing heights may move the plateau. The spread is from the mass and latency draws only. "Walked" was true for all 2380 runs, including those that went backward, so it is not a useful filter here. Placeholder dimensions, masses, servo figures and friction.
   - *Do the best values add up when used together?* (`scripts/gait/sweeps/trot_1p7_combined.py`, `results/gait/sweeps/combined/trot_1p7_combined.json`; means, standard deviations and paired differences in `trot_1p7_combined_summary.json`; plot `trot_1p7_combined.png`, both from `scripts/gait/sweeps/plot_trot_1p7_combined.py`; website: Results, "Do the best values add up?"). Nine settings of (kp, kd, swing, stance), each at four commanded speeds (0.05, 0.1, 0.15, 0.2 m/s) and three masses (the placeholder masses times 0.6, 1.0, 1.75, which are 1.04, 1.73 and 3.02 kg as the means over the 10 random draws; the nominal figures are 1.03, 1.72 and 3.01), 10 runs per cell, 1080 runs, 1.7 s period, duty 0.5. The settings: *baseline* (kp 30, kd 0.5, 30 mm, 0.19 m); *best* (kp 50, kd 0.9, 22 mm, 0.16 m: the best value of each from the sweeps above, with kd in the middle of the plateau and not at the cliff); *best_stance_0.17* (best with a 0.17 m stance, for margin from the joint limits); *best_baseline_gains / _swing / _stance* (best with one group put back to the baseline value: gains means kp and kd together); *gains_only / swing_only / stance_only* (baseline with one group changed to the best). The same 10 random mass (+-5% per body) and latency (18 to 22 ms) draws are used in every cell, so differences are paired (the other config's ratio subtracted for the same draw, then the mean and sample standard deviation over the 10 draws). The spread is from those draws only. All 1080 runs counted as walked, stayed inside the joint limits (even 0.16 m at 0.2 m/s) and moved forward.
     *Mean speed ratio ± sd (valid runs of 10 in brackets):*
     mass 1.04 kg: mean speed ratio ± sd (valid runs of 10: in limits and moving forward) 
     | config | 0.05 m/s | 0.1 m/s | 0.15 m/s | 0.2 m/s |
     |---|---|---|---|---|
     | baseline | 0.750 ± 0.002 (10/10) | 0.829 ± 0.003 (10/10) | 0.871 ± 0.002 (10/10) | 0.895 ± 0.002 (10/10) |
     | best | 0.787 ± 0.002 (10/10) | 0.855 ± 0.002 (10/10) | 0.883 ± 0.001 (10/10) | 0.895 ± 0.001 (10/10) |
     | best_stance_0.17 | 0.778 ± 0.002 (10/10) | 0.852 ± 0.002 (10/10) | 0.876 ± 0.001 (10/10) | 0.887 ± 0.001 (10/10) |
     | best_baseline_gains | 0.770 ± 0.003 (10/10) | 0.861 ± 0.002 (10/10) | 0.892 ± 0.002 (10/10) | 0.907 ± 0.002 (10/10) |
     | best_baseline_swing | 0.778 ± 0.002 (10/10) | 0.853 ± 0.003 (10/10) | 0.875 ± 0.002 (10/10) | 0.893 ± 0.002 (10/10) |
     | best_baseline_stance | 0.802 ± 0.003 (10/10) | 0.855 ± 0.002 (10/10) | 0.875 ± 0.001 (10/10) | 0.894 ± 0.001 (10/10) |
     | gains_only | 0.770 ± 0.004 (10/10) | 0.831 ± 0.002 (10/10) | 0.857 ± 0.002 (10/10) | 0.883 ± 0.002 (10/10) |
     | swing_only | 0.773 ± 0.005 (10/10) | 0.849 ± 0.003 (10/10) | 0.884 ± 0.002 (10/10) | 0.902 ± 0.002 (10/10) |
     | stance_only | 0.761 ± 0.004 (10/10) | 0.852 ± 0.003 (10/10) | 0.888 ± 0.002 (10/10) | 0.909 ± 0.002 (10/10) |

     mass 1.73 kg: mean speed ratio ± sd (valid runs of 10: in limits and moving forward) 
     | config | 0.05 m/s | 0.1 m/s | 0.15 m/s | 0.2 m/s |
     |---|---|---|---|---|
     | baseline | 0.677 ± 0.005 (10/10) | 0.783 ± 0.004 (10/10) | 0.848 ± 0.004 (10/10) | 0.880 ± 0.002 (10/10) |
     | best | 0.756 ± 0.003 (10/10) | 0.846 ± 0.002 (10/10) | 0.880 ± 0.002 (10/10) | 0.895 ± 0.002 (10/10) |
     | best_stance_0.17 | 0.755 ± 0.002 (10/10) | 0.837 ± 0.003 (10/10) | 0.876 ± 0.002 (10/10) | 0.891 ± 0.002 (10/10) |
     | best_baseline_gains | 0.712 ± 0.004 (10/10) | 0.825 ± 0.004 (10/10) | 0.867 ± 0.003 (10/10) | 0.891 ± 0.002 (10/10) |
     | best_baseline_swing | 0.745 ± 0.005 (10/10) | 0.837 ± 0.003 (10/10) | 0.875 ± 0.002 (10/10) | 0.897 ± 0.001 (10/10) |
     | best_baseline_stance | 0.755 ± 0.005 (10/10) | 0.836 ± 0.003 (10/10) | 0.872 ± 0.003 (10/10) | 0.892 ± 0.001 (10/10) |
     | gains_only | 0.736 ± 0.004 (10/10) | 0.816 ± 0.003 (10/10) | 0.859 ± 0.002 (10/10) | 0.885 ± 0.003 (10/10) |
     | swing_only | 0.687 ± 0.004 (10/10) | 0.811 ± 0.004 (10/10) | 0.859 ± 0.003 (10/10) | 0.883 ± 0.002 (10/10) |
     | stance_only | 0.700 ± 0.005 (10/10) | 0.819 ± 0.003 (10/10) | 0.870 ± 0.003 (10/10) | 0.897 ± 0.002 (10/10) |

     mass 3.02 kg: mean speed ratio ± sd (valid runs of 10: in limits and moving forward) 
     | config | 0.05 m/s | 0.1 m/s | 0.15 m/s | 0.2 m/s |
     |---|---|---|---|---|
     | baseline | 0.253 ± 0.166 (10/10) | 0.285 ± 0.042 (10/10) | 0.432 ± 0.016 (10/10) | 0.529 ± 0.036 (10/10) |
     | best | 0.239 ± 0.085 (10/10) | 0.296 ± 0.033 (10/10) | 0.430 ± 0.020 (10/10) | 0.556 ± 0.014 (10/10) |
     | best_stance_0.17 | 0.365 ± 0.146 (10/10) | 0.279 ± 0.070 (10/10) | 0.382 ± 0.047 (10/10) | 0.451 ± 0.043 (10/10) |
     | best_baseline_gains | 0.125 ± 0.066 (10/10) | 0.254 ± 0.027 (10/10) | 0.407 ± 0.014 (10/10) | 0.537 ± 0.011 (10/10) |
     | best_baseline_swing | 0.325 ± 0.011 (10/10) | 0.502 ± 0.002 (10/10) | 0.623 ± 0.005 (10/10) | 0.711 ± 0.013 (10/10) |
     | best_baseline_stance | 0.627 ± 0.073 (10/10) | 0.484 ± 0.099 (10/10) | 0.536 ± 0.061 (10/10) | 0.571 ± 0.070 (10/10) |
     | gains_only | 0.591 ± 0.106 (10/10) | 0.522 ± 0.110 (10/10) | 0.517 ± 0.060 (10/10) | 0.548 ± 0.037 (10/10) |
     | swing_only | 0.358 ± 0.150 (10/10) | 0.283 ± 0.055 (10/10) | 0.433 ± 0.035 (10/10) | 0.523 ± 0.034 (10/10) |
     | stance_only | 0.291 ± 0.010 (10/10) | 0.496 ± 0.008 (10/10) | 0.625 ± 0.013 (10/10) | 0.712 ± 0.015 (10/10) |
     *Mean largest tilt in degrees / mean peak torque over stall (runs that stayed upright):*
     mass 1.04 kg: mean tilt (deg, upright runs) and peak torque / stall
     | config | 0.05 m/s | 0.1 m/s | 0.15 m/s | 0.2 m/s |
     |---|---|---|---|---|
     | baseline | 9.0 / 0.50 | 8.2 / 0.51 | 8.3 / 0.52 | 8.7 / 0.55 |
     | best | 6.1 / 0.55 | 6.1 / 0.64 | 6.2 / 0.69 | 6.4 / 0.68 |
     | best_stance_0.17 | 5.9 / 0.61 | 6.0 / 0.64 | 6.2 / 0.64 | 6.4 / 0.80 |
     | best_baseline_gains | 5.9 / 0.44 | 5.8 / 0.51 | 5.9 / 0.53 | 6.1 / 0.59 |
     | best_baseline_swing | 8.6 / 0.54 | 8.6 / 0.67 | 8.8 / 0.74 | 9.0 / 0.81 |
     | best_baseline_stance | 6.1 / 0.60 | 5.9 / 0.64 | 6.1 / 0.64 | 6.3 / 0.73 |
     | gains_only | 9.1 / 0.61 | 8.4 / 0.64 | 8.6 / 0.66 | 9.0 / 0.67 |
     | swing_only | 5.9 / 0.48 | 5.7 / 0.51 | 5.7 / 0.51 | 6.0 / 0.55 |
     | stance_only | 8.6 / 0.44 | 8.3 / 0.50 | 8.5 / 0.56 | 8.8 / 0.62 |

     mass 1.73 kg: mean tilt (deg, upright runs) and peak torque / stall
     | config | 0.05 m/s | 0.1 m/s | 0.15 m/s | 0.2 m/s |
     |---|---|---|---|---|
     | baseline | 8.8 / 0.75 | 7.9 / 0.76 | 7.9 / 0.85 | 8.2 / 0.87 |
     | best | 5.9 / 0.74 | 5.8 / 0.84 | 5.9 / 0.88 | 6.0 / 0.98 |
     | best_stance_0.17 | 5.6 / 0.81 | 5.7 / 0.87 | 5.9 / 0.86 | 6.0 / 0.91 |
     | best_baseline_gains | 5.7 / 0.73 | 5.4 / 0.80 | 5.4 / 0.85 | 5.6 / 0.89 |
     | best_baseline_swing | 8.6 / 0.74 | 8.3 / 0.86 | 8.5 / 0.93 | 8.7 / 0.99 |
     | best_baseline_stance | 5.9 / 0.80 | 5.6 / 0.85 | 5.7 / 0.85 | 6.0 / 0.89 |
     | gains_only | 9.0 / 0.84 | 8.2 / 0.84 | 8.3 / 0.88 | 8.7 / 0.89 |
     | swing_only | 5.7 / 0.69 | 5.2 / 0.78 | 5.2 / 0.81 | 5.5 / 0.84 |
     | stance_only | 8.6 / 0.74 | 8.0 / 0.84 | 8.1 / 0.86 | 8.4 / 0.85 |

     mass 3.02 kg: mean tilt (deg, upright runs) and peak torque / stall
     | config | 0.05 m/s | 0.1 m/s | 0.15 m/s | 0.2 m/s |
     |---|---|---|---|---|
     | baseline | 6.1 / 1.00 | 6.5 / 1.00 | 9.0 / 1.00 | 12.0 / 1.00 |
     | best | 5.4 / 1.00 | 5.9 / 1.00 | 5.8 / 1.00 | 5.9 / 1.00 |
     | best_stance_0.17 | 4.5 / 1.00 | 5.9 / 1.00 | 6.2 / 1.00 | 7.0 / 1.00 |
     | best_baseline_gains | 5.3 / 1.00 | 5.5 / 1.00 | 5.6 / 1.00 | 5.7 / 1.00 |
     | best_baseline_swing | 6.0 / 1.00 | 5.9 / 1.00 | 5.9 / 1.00 | 6.0 / 1.00 |
     | best_baseline_stance | 4.4 / 1.00 | 4.3 / 1.00 | 4.8 / 1.00 | 6.2 / 1.00 |
     | gains_only | 7.1 / 1.00 | 5.7 / 1.00 | 7.9 / 1.00 | 10.5 / 1.00 |
     | swing_only | 3.7 / 1.00 | 4.6 / 1.00 | 5.6 / 1.00 | 9.1 / 1.00 |
     | stance_only | 5.5 / 1.00 | 5.5 / 1.00 | 5.6 / 1.00 | 5.6 / 1.00 |
     *Paired differences in speed ratio:*
     paired differences in speed ratio (mean ± sd over the 10 draws)
     | mass kg | speed | best - baseline | sum of the three single-group effects | gains only | swing only | stance only | best - best with baseline gains | best - best with baseline swing | best - best with baseline stance |
     |---|---|---|---|---|---|---|---|---|---|
     | 1.04 | 0.05 | 0.037 ± 0.001 | 0.054 | 0.020 ± 0.002 | 0.023 ± 0.003 | 0.011 ± 0.002 | 0.017 ± 0.003 | 0.009 ± 0.001 | -0.015 ± 0.002 |
     | 1.04 | 0.1 | 0.026 ± 0.002 | 0.045 | 0.002 ± 0.003 | 0.020 ± 0.001 | 0.023 ± 0.001 | -0.005 ± 0.001 | 0.002 ± 0.002 | 0.001 ± 0.001 |
     | 1.04 | 0.15 | 0.012 ± 0.002 | 0.015 | -0.014 ± 0.001 | 0.013 ± 0.001 | 0.017 ± 0.001 | -0.008 ± 0.001 | 0.009 ± 0.002 | 0.009 ± 0.001 |
     | 1.04 | 0.2 | -0.000 ± 0.001 | 0.008 | -0.013 ± 0.001 | 0.006 ± 0.001 | 0.014 ± 0.001 | -0.012 ± 0.001 | 0.002 ± 0.001 | 0.001 ± 0.001 |
     | 1.73 | 0.05 | 0.079 ± 0.002 | 0.094 | 0.059 ± 0.002 | 0.011 ± 0.001 | 0.024 ± 0.001 | 0.044 ± 0.002 | 0.011 ± 0.002 | 0.001 ± 0.003 |
     | 1.73 | 0.1 | 0.063 ± 0.002 | 0.097 | 0.033 ± 0.002 | 0.028 ± 0.001 | 0.036 ± 0.001 | 0.021 ± 0.001 | 0.009 ± 0.002 | 0.010 ± 0.001 |
     | 1.73 | 0.15 | 0.031 ± 0.002 | 0.043 | 0.010 ± 0.002 | 0.011 ± 0.001 | 0.022 ± 0.001 | 0.013 ± 0.002 | 0.005 ± 0.001 | 0.008 ± 0.001 |
     | 1.73 | 0.2 | 0.015 ± 0.001 | 0.025 | 0.005 ± 0.001 | 0.003 ± 0.001 | 0.017 ± 0.001 | 0.004 ± 0.001 | -0.002 ± 0.001 | 0.003 ± 0.001 |
     | 3.02 | 0.05 | -0.014 ± 0.082 | 0.481 | 0.338 ± 0.086 | 0.105 ± 0.028 | 0.039 ± 0.157 | 0.114 ± 0.019 | -0.086 ± 0.082 | -0.388 ± 0.048 |
     | 3.02 | 0.1 | 0.011 ± 0.019 | 0.445 | 0.237 ± 0.075 | -0.003 ± 0.020 | 0.211 ± 0.037 | 0.041 ± 0.007 | -0.206 ± 0.033 | -0.188 ± 0.066 |
     | 3.02 | 0.15 | -0.001 ± 0.011 | 0.280 | 0.086 ± 0.047 | 0.001 ± 0.024 | 0.193 ± 0.010 | 0.024 ± 0.006 | -0.192 ± 0.015 | -0.106 ± 0.042 |
     | 3.02 | 0.2 | 0.028 ± 0.023 | 0.197 | 0.019 ± 0.005 | -0.006 ± 0.006 | 0.183 ± 0.021 | 0.019 ± 0.003 | -0.155 ± 0.004 | -0.014 ± 0.057 |
     *What it shows.* (1) At the nominal and light masses (1.04 and 1.73 kg) the best settings together beat the baseline at every speed up to 0.15 m/s, by a small amount: at 1.73 kg +0.079, +0.063, +0.031 at 0.05, 0.1, 0.15 m/s (standard deviation of the paired difference 0.002), at 1.04 kg +0.037, +0.026, +0.012. At 0.2 m/s the gain is +0.015 (1.73 kg) and 0.000 (1.04 kg), because the baseline already makes 0.88 to 0.90 of the command there. The speed ratio reaches 0.88 to 0.90 at 0.15 and 0.2 m/s. (2) The tilt falls more clearly: 5.8 to 6.4 degrees against 7.9 to 9.0 at 1.04 and 1.73 kg, 2 to 3 degrees lower at every speed. (3) The gains are less than the sum of the three single-group gains in all 8 of those cells (for example at 1.73 kg and 0.1 m/s +0.063 against 0.097, at 1.04 kg and 0.2 m/s 0.000 against 0.008): the three overlap, so they do not add up fully. (4) With one group put back, at 1.73 kg all three groups still contribute in most cells (the gains +0.044, +0.021, +0.013, +0.004 across the four speeds; the swing +0.011, +0.009, +0.005, -0.002; the stance +0.001, +0.010, +0.008, +0.003). At 1.04 kg the new gains are slightly worse than the baseline gains from 0.1 m/s up (-0.005, -0.008, -0.012), and the lower stance is worse at 0.05 m/s (-0.015), so the best settings are not best everywhere. (5) The 0.17 m stance is 0.001 to 0.009 lower in speed ratio than 0.16 m in all 8 cells at 1.04 and 1.73 kg, so it buys joint-limit margin for little. (6) At 3.02 kg nothing is clear. The servos are at their stall limit (peak torque 1.00) in every run, the speed ratio is 0.24 to 0.56 with the best settings (against 0.25 to 0.53 for the baseline), the paired differences are within about 1.2 standard deviations of zero (for example -0.014 +- 0.082 at 0.05 m/s and +0.028 +- 0.023 at 0.2 m/s), and the three single-group gains (sum 0.20 to 0.48) vanish when combined (0.03 or less). The 22 mm swing is clearly worse at this mass: putting the swing back to 30 mm raises the ratio by 0.086, 0.206, 0.192, 0.155 at the four speeds. The standard deviation is large (up to 0.17), so the results at 3.02 kg are noisy as well as different.
     *Use.* For a robot near the placeholder mass (about 1 to 1.7 kg, the working target is 1.5 kg) the best settings, or the same with a 0.17 m stance, are a small improvement in speed and a clear one in tilt at 0.15 m/s and below; a heavier robot would need its own settings, and a 30 mm swing did better at 3 kg. This is a placeholder-model result: servo figures, masses and friction are unverified.
     *Not shown.* Other duties, periods or commanded speeds above 0.2 m/s; turning; terrain; friction; the cliff in kd moving with mass (all three kd values used, 0.5 and 0.9, were below it, but I did not sweep kd at the other masses); why the single gains overlap or why the 22 mm swing fails at 3 kg. The spread is from the mass and latency draws only, so it is small and does not cover model error. All dimensions, masses, servo figures and friction are placeholders.
5. [x] **Design sweep** (`mojito/sim/design_sweep.py`, `scripts/design/design_sweep.py` and `plot_design_sweep.py`, `tests/sim/test_design_sweep.py`, `results/design/design_sweep.json`, `design_sweep_summary.json`, `design_sweep.png`).
   Eleven designs: upper and lower link each 100, 150, 200 mm (nine, at the 300 × 150 mm body), plus 240 × 120 and 360 × 180 mm bodies at the 150 / 150 mm legs; abduction offset 50 mm throughout.
   Each at three total masses (1.03, 1.72, 3.01 kg: the placeholder masses times 0.6, 1.0, 1.75, every role scaled alike). Per design and mass: one stand test and the trot and crawl at 0.05, 0.10, 0.15 m/s,
   **each run 10 s of simulated time** (the walks stand 1 s, blend 0.5 s, then walk 8.5 s: 5 trot cycles, 4.25 crawl cycles). One run per cell, no random draws, 5 V, the placeholder servo figures.
   Gait timings and servo gains are **not re-tuned** (those tuned on the 150 mm legs: trot kp 50, kd 0.9; crawl the originals). Only the standing height changes: each leg stands with the same knee bend as the
   150 / 150 mm leg does at the tuned height (`design_sweep.scaled_height`). My first version scaled the height with upper + lower length; it bent the knee to its limit on unequal legs, so the failures were mine, and I replaced it.
   A walk is "valid" when every commanded foot target stayed inside the joint limits, the body stayed upright and the measured speed was positive; valid does not mean fast, so the speed ratio (measured over commanded) is reported too.
   *What was measured* (`python scripts/design/plot_design_sweep.py` prints the full table):
   - Standing: 11 of 11 designs stand at 1.03 and 1.72 kg, 10 of 11 at 3.01 kg. The one that does not is 200 / 200 mm (peak torque 1.00 of stall, body low; its static bound is 5.4 kg, so it fails below the bound, as the
     150 / 150 mm leg did in the stand test). Peak torque over stall in the whole stand run at 1.72 kg runs from 0.19 (100 / 100 mm) to 0.52 (200 / 200 mm); the longer the legs, the closer to stall. The static bound for four feet runs from 10.8 kg (100 / 100) to 5.4 kg (200 / 200).
   - Trot (valid walks, of 33 = 11 designs × 3 speeds): 26 at 1.03 kg, 26 at 1.72 kg, 22 at 3.01 kg. At 1.03 and 1.72 kg every design with the lower link no longer than the upper walks at all three speeds. At 3.01 kg the 150 / 150 and 200 / 150 mm legs reach only 0.18 to 0.44 of the commanded speed, the 150 / 200 mm leg goes backward or leaves the limits, and the 200 / 200 mm leg cannot stand; peak torque is at stall in all of them. The 100 / 100, 150 / 100 and 200 / 100 mm legs still make 0.80 to 0.95 (peak torque 0.90 to 1.00).
   - Trot, short upper with long lower link (100 / 150 and 100 / 200 mm): not valid at any speed or mass, because the commanded feet leave the joint limits. At rest the hip pitch is already 1.32 and 1.49 rad at the trot's stance height (placeholder limit 1.5), which fits, but I did not trace the failing joint in each walk.
   - Trot speed ratio at 0.10 m/s and 1.72 kg: 100 / 100 0.94, 150 / 100 0.92, 150 / 150 0.85, 200 / 200 0.73. A longer lower link costs speed.
   - Crawl: 21, 20, 7 valid of 33 at 1.03, 1.72, 3.01 kg. At 0.15 m/s it is invalid at the placeholder 150 / 150 mm legs at every mass (targets outside the joint limits). At 0.10 m/s it is invalid for the three designs with a 100 mm lower link (100 / 100, 150 / 100, 200 / 100). Only the 150 / 200 and 200 / 200 mm legs crawl validly at 0.15 m/s, at 1.03 kg. At 3.01 kg it mostly fails.
   - Body size at 150 / 150 mm legs (1.72 kg, trot ratio at 0.05, 0.10, 0.15 m/s): 240 × 120: 0.70, 0.79, 0.84; 300 × 150: 0.76, 0.85, 0.88; 360 × 180: 0.78, 0.87, 0.90. A longer body is slightly faster; at 3.01 kg only the 360 × 180 mm body crawls validly at 0.05 and 0.10 m/s (0.67, 0.29).
   - Peak torque while walking at 1.72 kg runs from 0.57 to 1.00 of stall depending on design, speed and gait; at 3.01 kg nearly every walk reaches 0.9 or more.
   **Repeated with 10 random draws** (`scripts/design/design_sweep_draws.py`, `plot_design_sweep_draws.py`, `results/design/design_sweep_draws.json`, `design_sweep_draws_summary.json`, `design_sweep_draws.png`). All 33 design-and-mass cells, not a chosen subset: 330 sets of 7 runs (one stand and six walks each, 2310 runs).
   Each draw scales every body mass by a uniform 0.95 to 1.05 and draws the servo latency uniformly from 18 to 22 ms (seed = draw number, the same 10 draws in every cell, as in the trot sweeps). The spread is from those draws only.
   - Almost every cell gives the same verdict in all 10 draws (10/10 valid or 0/10), and the speed-ratio standard deviation is 0.00 to 0.03 for most of them. So at 1.03 and 1.72 kg the single-run table above was not a lucky draw, with that much mass and latency variation. Valid walks over all designs and speeds, of 330: trot 260, 260, 220 and crawl 210, 198, 70 at 1.04, 1.73, 3.02 kg (the single runs gave 26, 26, 22 and 21, 20, 7, times 10, except the crawl at 1.73 kg).
   - Stand: 320 of 330 stand; the 10 that do not are the 200 / 200 mm legs at 3.02 kg (0 of 10).
   - Cells that are not settled by the draws: crawl at 1.73 kg with 200 / 200 mm legs was valid in the single run at 0.05 m/s (0.50) but in 7 of 10 draws with a mean ratio of 0.08 and a standard deviation of 0.83, and at 0.15 m/s in 1 of 10; the 150 / 150 mm crawl at 3.02 kg and 0.05 m/s is valid in 2 of 10 (mean -0.60, sd 0.65); the 100 / 150 mm crawl at 3.02 kg and 0.10 m/s is valid in 8 of 10 but barely moves (0.03 ± 0.22). These are on the edge: a few percent of mass or a millisecond of latency changes the outcome. The single-run figures above for those cells should not be read as results.
   - At 3.02 kg the trot with 150 / 150 mm legs makes 0.23 ± 0.08, 0.29 ± 0.03, 0.43 ± 0.02 of the commanded speed at 0.05, 0.10, 0.15 m/s (all valid): consistently slow, not marginal.
   - What this does not cover: model error (servo figures, friction, masses and joint limits are placeholders); draws of friction, kp or kd, or of the leg dimensions themselves. A draw spread this narrow says the verdicts are not fragile to mass and latency; it does not say they would hold on the real robot.
   **Re-tuned per design** (`scripts/design/design_tune.py`, `design_sweep_tuned.py`, `plot_design_sweep_tuned.py`; `results/design/design_tuning.json`, `design_sweep_tuned.json`, `design_sweep_tuned_summary.json`, `design_sweep_tuned.png`).
   For each of the 11 designs, three masses and both gaits, a grid search of 45 settings: knee bend with the foot below the hip (-74°, -90°, -103°, -116°, -129°, which sets the standing height), swing height (15, 22, 30 mm) and servo gains ((kp, kd) = (10, 0.2), (30, 0.5), (50, 0.9)).
   Scored at 0.10 m/s on the nominal masses and latency, 10 s per run, 2970 runs: valid first, then speed ratio closest to 1, then lowest peak torque. The best setting per design, mass and gait was then run at 0.05, 0.10, 0.15 m/s for the same 10 draws as above (1980 runs). The search saw one speed and no draws, so the other two speeds and the draws are a check on it. Period, duty, sway and stride are **not** tuned. The best settings differ per design and per mass; the table of them is in `design_sweep_tuned.json`.
   - Valid walks over all designs and speeds, of 330 (tuned, with the untuned figure above in brackets), at 1.04, 1.73, 3.02 kg: trot 320 (260), 310 (260), 299 (220); crawl 200 (210), 210 (198), 191 (70). The trot gain is mostly the 100 / 150 and 100 / 200 mm legs and the heavy cases; the crawl gain is almost all at 3.02 kg.
   - 3.02 kg, trot, speed ratio at 0.05, 0.10, 0.15 m/s: 150 / 150 mm 0.66, 0.79, 0.83 (untuned 0.23, 0.29, 0.43); 200 / 150 mm 0.59, 0.74, 0.66 (0.17, 0.32, 0.44); 100 / 150 mm 0.73, 0.84, 0.89 with 0.89 not valid at 0.15 (untuned not valid at any speed). Of the 11 trot winners at 3.02 kg with a positive speed, 5 chose the least bent knee in the grid (-74°) and 4 chose -116°, the untuned bend.
   - Still poor: the 200 / 200 mm trot at 3.02 kg (-0.09, 0.05, 0.23; the leg did not stand untuned either); the 150 / 200 mm trot at 3.02 kg (0.48, 0.38 ± 0.18 with 9 of 10 valid, 0.13 not valid at 0.15 m/s).
   - The crawl with a 100 mm lower link (100 / 100, 150 / 100, 200 / 100 mm) has no valid setting among the 45 at any mass: every candidate left the joint limits or fell. 13, 12, 13 of the 33 crawl cells (1.04, 1.73, 3.02 kg) are never valid, against 1, 2, 3 of 33 trot cells. A different crawl period, duty or sway might still work; I did not try them.
   - One cell got worse: the 150 / 200 mm crawl at 1.04 kg and 0.15 m/s was valid untuned (0.49, 10 of 10) and 0 of 10 tuned, because the settings chosen at 0.10 m/s send the feet outside the limits at 0.15. Tuning at one speed does not carry to all speeds.
   - Not settled by the draws (1 to 9 of 10 valid): 150 / 200 mm crawl at 3.02 kg at 0.05 m/s (-0.22 ± 0.94, 4 of 10) and 0.10 m/s (0.68 ± 0.36, 8 of 10); 200 / 200 mm crawl at 3.02 kg at 0.15 m/s (0.60 ± 0.41, 9 of 10); the 150 / 200 mm trot above.
   - 12 of the 198 tuned cells have a mean speed ratio above 1.05 (faster than commanded); the score was closeness to 1, so these count as valid.
   - What this does and does not say: the tuned numbers are each design at its own best of a coarse grid, so they compare designs fairly, but a real robot has one set of settings per load and a search of 45 settings is not an optimum. The least bent knee in the grid (-74°) was chosen in 25 of the 66 entries (-129°, the most bent, in none), and the highest gains (kp 50) or lowest (kp 10) in 53 of 66, so a wider grid with straighter legs and other gains might do better. Everything remains a placeholder (servo figures, masses, friction, joint limits); the draws cover mass and latency only.
   *Not shown:* a different gait period, duty or stride might rescue a design that failed (only knee bend, swing height and gains were re-tuned, above); the table above is one run per cell (see the 10-draw repeat for how stable each is), and differences of a few hundredths in speed ratio between designs are within what one run can show; turning, terrain, friction; any of this for a real robot (all dimensions, masses, servo figures, friction and joint limits are placeholders; the joint limits decide several of the failures); leg mass scaling with leg length (masses are one factor on all roles); why 3.01 kg collapses the walks.
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
