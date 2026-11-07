# Comparisons with other papers, and things to try

Companion to `docs/LITERATURE.md` (the survey). This file is the working list: how Mojito's choices differ from what
a paper did, and experiments that follow. It is a draft that grows as papers are read. Nothing here is a Mojito result;
every "to try" item is untested, and each says what it would and would not show.

How much of each paper was read is in `docs/LITERATURE.md`. Papers are cited by arXiv number.

## 1. Comparisons

### Action space and control layer

| | Paper | Mojito |
|---|---|---|
| Gait in Eight, 2503.08375 | Real HoneyBadger quadruped. Two policies on the same learner (CrossQ): joint-target offsets (JTP) clipped to a maximum deviation from a nominal stance, and a CPG whose output is a Cartesian offset per foot, added to the CPG foot position and converted by analytic IK before PD. Reports JTP for agile high-speed locomotion, CPG-with-foot-offsets for stable natural gaits. Actions filtered with a One-Euro filter; action delays randomized in simulation. | Plan: policy acts in foot space through the IK layer (`docs/PLAN.md`, Stage 2). Scripted trot and crawl exist (`mojito/robot/gaits.py`); a learned residual on them would be the analogue of the CPG arm. No learned policy yet. **Comparison planned as work package 8 in `docs/STAGE2.md`, idea 1 below.** |
| CPG-RL, 2211.00458 | Policy sets oscillator amplitude, frequency and phase; CPG gives foot positions; analytic IK; PD. Reports transfer to a Unitree A1 without domain randomization or added noise. | Same foot-space-then-IK pipeline. Mojito randomizes latency and masses in its design sweeps (`docs/STAGE2.md`), not yet in a learning loop. |
| A Walk in the Park, 2208.07860 | Joint PD targets at 20 Hz, each limited to a box [default - o, default + o] with o = (0.2, 0.4, 0.4) rad; says the limit is crucial; lower stance used on hardware because it failed less. | Foot-space residual would limit the foot offset in metres, which limits joint motion indirectly. The right size is not known. |
| Solo12, 2309.16683 | Joint impedance references at 100 Hz over PD at 10 kHz; randomizing PD gains was enough for their transfer; identified joint friction and used a Joule-plus-friction energy penalty. | Mojito's PD is the servo model (kp, kd, latency, torque limit). Sensitivity to kp and kd is already swept (`docs/STAGE2.md`, gait sweeps). |
| Learning Torque Control, 2203.05194 | A1; policy outputs torques at 200 Hz, no PD; reports a higher converged return than position control in one comparison with a simplified reward, no extra sim-to-real difficulty with ordinary domain randomization. | Not applicable to the DS3218 (internal position loop, no torque command). Applicable to the later DC-motor robot. |

### Learned IK

| | Paper | Mojito |
|---|---|---|
| MorphIK, 2609.29908 | Flow matching conditioned on a transformer encoding of the URDF chain, trained on procedurally generated 6-9 DoF arms; FK-based Cartesian auxiliary loss; best of 16 samples; about 5 cm raw on unseen real arms; DLS polishing brings it under 1 cm in one step and under 1 mm in about three. | One small network over a family of 3-joint legs (lengths as inputs), FK loss only, one deterministic output. Median 0.17 mm, 95th percentile 0.84 mm on unseen legs with varied lengths (`results/learning/metrics.json`). Closed-form IK is exact on this leg. |
| IKDiffuser, 2506.13087 | Diffusion IK for kinematic trees (several end-effectors in one model), one model per robot, used to seed optimization solvers and to maximize manipulability at inference; seven platforms including humanoids. | Mojito solves each leg separately (`mojito/robot/body.py`). A joint solve over all four feet is not part of the plan. |

### Actuators and simulation fidelity

| | Paper | Mojito |
|---|---|---|
| SPI-Active, 2505.14266 | Identifies mass, inertia and actuator parameters of a Unitree Go2 (and G1) by sampling-based search (CMA-ES) over a simulator, minimizing state prediction error, then optimizes exploration commands for Fisher information. Per-joint tanh torque saturation (tau = kappa_i tanh(tau_PD / kappa_i)) fitted better than a plain PD or a single shared kappa on two jump tasks. **The base-state cost term uses motion-capture trajectories.** Offline. | Servo model clamps force at a fixed range (`forcerange`) with a joint velocity limit, latency and command resolution (`docs/STAGE2.md`, "Servo model"). No fitting to data (none exists). No motion capture on the robot. |
| Proprioceptive distribution matching, 2604.11090 | Compares hardware and simulation rollouts as distributions of joint observations and actions (average of 1D Wasserstein distances), so no motion capture or time alignment; Go2; reports under five minutes of hardware data suffice. | Stage 4 plans to fit the simulator to logged experience; Mojito will have joint-angle sensors and an IMU but no motion capture. |
| Actuator Reality Shaping, 2607.02205 | Shapes the real actuator to track the ideal second-order PD response the simulator assumes, using a disturbance observer; needs motor current and torque constant; shown on a Dynamixel YM070, an arm, a wheeled-legged robot and a humanoid. | The simulator already assumes an ideal-ish PD servo with limits. The DS3218 has no current access, so not applicable to the first robot. |

### Learning from real interaction

| | Paper | Mojito |
|---|---|---|
| A Walk in the Park, 2208.07860 | A1; 20 minutes; model-free; learned resets; velocity from a Kalman filter on IMU and leg kinematics plus binary foot contacts. | No policy and no hardware yet. |
| Gait in Eight, 2503.08375 | 8 minutes; proprioceptive-only reward; says the velocity estimate is biased low without foot contact sensors (assumes all four feet are on the ground). | The robot has grippy pads and no contact sensors. |
| 3 minutes of real data, 2410.09163 | Go1; 18 000 steps (3.0 min); model-based RL with a rigid-body Lagrangian model (mass matrix, Coriolis and gravity terms from the differentiable simulator Brax) plus a learned residual; **base velocity from Vicon**. | A MuJoCo model of the robot exists. The paper's method needs a differentiable simulator and proprioception that carries the rigid-body state. |
| DayDreamer, 2206.14176 | A1; about 1 hour from scratch without resets; recovers from pushes in about 10 minutes; world model. | No world model planned. |

### Design exploration and multi-embodiment

| | Paper | Mojito |
|---|---|---|
| Boekel et al., 2611.02462 | Isaac Lab, 4096 environments; one policy trained across designs with the design vector as part of the observation (leg scale factors 0.70 to 1.30 among the parameters), then Bayesian optimization over designs with the policy held fixed; hardware shows lower cost of transport (18.6% inspection, 35.7% payload) than a nominal design. Compliant four-bar legs. | Design sweeps over leg lengths, body size and mass with a scripted gait re-tuned per design (`results/design/`). Lengths are already inputs to the IK network. |
| URMA, 2409.06366 | One policy, attention encoders over per-joint and per-foot descriptions, trained on 16 simulated robots; zero-shot to a held-out Unitree A1 and a MAB Silver Badger; three real robots deployed zero-shot. | Not planned. |
| DrEureka, 2406.01967 | LLM writes reward and domain-randomization ranges; reports 34% higher forward velocity than human-designed settings on a Go1. | Rewards and randomization ranges are not yet written. |

## 2. Things to try

Each item: what, source, why, what it needs, what it would not show. Marked **now** (possible in the current simulator),
**WP7** (needs a trained policy, `docs/STAGE2.md` work package 7), or **hardware**. The order within each group is not a ranking.

### Possible now

1. **Joint-target versus foot-offset action spaces** (Gait in Eight). Scheduled as `docs/STAGE2.md` work package 8; it needs WP7
   first. *Why:* the paper found the two trade off (joint targets for agile high speed; foot offsets with a CPG for stable
   natural gaits), and Mojito's plan assumes foot space without having shown it is better. *Design:* same learner, same simulator,
   same reward and servo model, several seeds; arms are (a) joint-target offsets clipped around the nominal stance, (b) foot-offset
   residual on the scripted trot through analytic IK, (c) the same through the learned IK network. *Measure:* speed tracking, share
   of valid runs, falls, peak joint torque, action smoothness, learning curve. *Would not show:* anything about hardware, or
   that the paper's finding holds for hobby servos; arm (c) only shows whether the network's 0.17 mm median error matters here.
2. **Error budget of the learned IK in a loop.** Before (1c), check that the network's error does not interact with
   the servo's positioning error by running the existing trot with analytic versus learned IK and comparing foot slip and
   speed. *Would not show:* effects in a learning loop.
3. **DLS polish from a network seed** (MorphIK). On a limb with no closed form, count damped-least-squares steps to reach a
   tolerance from the network output versus from the previous solution or the nominal pose, using `mojito/geometry/kinematics.py`.
   *Why:* MorphIK reports that the learned seed makes a few steps enough. *Would not show:* whether the network is needed for
   the 3-joint leg, where the closed form is exact.
4. **Does the network earn its place?** Use a limb with more than 3 joints (redundant) in `robots/`, where there is no closed form,
   and compare the network with numeric IK on accuracy, run time and consistency between consecutive targets. *Why:* the review
   found no accuracy argument for the network on the 3-joint leg. *Would not show:* anything for the quadruped.
5. **Per-joint tanh torque saturation versus hard clamp** (SPI-Active). Add the smooth saturation as a servo option and run the stand
   test, sit-to-stand and trot to see whether peak torque, deflection or validity change. *Why:* in their jump tasks it fitted better
   than a plain PD. *Would not show:* which is closer to a DS3218; that needs measured holding-torque data.
6. **Sim-to-sim recovery of servo parameters** (proprioceptive distribution matching). Treat one simulator with hidden kp, kd, latency
   and torque limit as "the real robot", log joint angles and commands, and see how well a Wasserstein distribution match
   recovers the hidden values and how much data it needs. *Why:* tests the Stage 4 idea before hardware exists. *Would not show:*
   success on real servos, which have effects the simulator lacks. **Step-by-step plan in section 3.**
7. **Base-velocity estimator without contact sensors.** In simulation with the IMU and joint-sensor models (`mojito/sim/sensors.py`),
   estimate body velocity from the IMU and leg forward kinematics assuming stance feet are fixed, and compare with ground truth
   across trot and crawl. *Why:* Gait in Eight reports a bias without foot contact sensors, and the robot has none. It informs open
   question 3. *Would not show:* bias on real terrain or real sensors. This is the no-camera baseline for section 4.
8. **Action filtering for servo jitter** (Gait in Eight's One-Euro filter). With the PCA9685 step size and servo latency in the model,
   test whether filtering the commanded targets reduces jitter and torque spikes in the trot, and what it costs in tracking.

### Needs a trained policy (WP7)

9. **Action range around the nominal pose** (Walk in the Park). Sweep the foot-offset limit and the default stance height and
   see how learning speed and falls change.
10. **Randomization of PD gains, mass, friction, latency and servo strength** (Solo12, DrEureka). Which ones are needed for a policy to
    survive a held-out simulator with different values? *Would not show:* hardware transfer.
11. **Policy conditioned on leg lengths** (Boekel, URMA). Train on varied lengths with the lengths in the observation and test on
    held-out designs; then re-run a subset of the design sweep with the learned policy and check whether the ranking of designs
    changes compared with the scripted-gait sweep.
12. **Cost-of-transport and energy terms** from the servo model (Solo12's Joule-plus-friction penalty, Boekel's energy model) as a
    reward or evaluation metric.

### Needs hardware

13. **Learned resets / self-righting** (Walk in the Park, DayDreamer) so real training does not need a person to reset the robot.
14. **Servo wear and battery budget for real training.** The papers train for 3 to 60 minutes on robots with robust motors; the
    2S 1500 mAh battery and 20 kg·cm hobby servos may limit this. Log cumulative servo travel and current from the start. (My inference,
    not from the papers.)
15. **Velocity source for rewards on hardware:** compare the IMU-plus-kinematics estimate (7) against an external measurement when one
    exists.

### Later robot (DC motors with current feedback)

16. **Actuator Reality Shaping** and **torque-output policies** (2607.02205, 2203.05194): both need motor current and a torque command.

### Camera plus IMU (section 4)

18. **Camera-plus-IMU velocity estimate in simulation** against the IMU-plus-kinematics baseline (item 7): render a base-mounted camera in
    MuJoCo, estimate velocity and yaw rate, compare with ground truth across gait speeds. Possible now; best case only (no motion
    blur, rolling shutter or lighting).
19. **Camera estimate on the robot,** checked against an external camera with markers. Needs hardware and a camera choice.

### Low priority

17. LLM-written rewards and randomization ranges (DrEureka). **World-model or differentiable-simulator learning** (DayDreamer,
    3-minute paper): large additions to the plan; revisit only if sample count on hardware turns out to be the limit.

## 3. Proprioceptive distribution matching: step by step

Source: Dao et al. 2026 (2604.11090). Goal: adjust simulator parameters until the simulator, driven by the same commands, produces
the same *distribution* of joint positions, velocities and commands as the hardware, without motion capture and without lining the
two up in time. In the labels below, **[paper]** is what the paper did and **[Mojito]** is my adaptation, which is untested.

**What the paper's results do and do not support.** Sim-to-sim, on a Go2 with six known friction and armature values, the
distribution cost recovered them to within rounding with no noise. With torque noise plus a random 0.02 to 0.1 s time shift in the
target logs, the mean parameter error was 8.0%, 8.3% and 18.7% at noise 1.0, 2.5 and 5.0 N·m, against 15.2%, 40.7% and 86.3% for direct
comparison of aligned observations, and 7.3%, 15.6% and 37.8% for a state-matching cost that is given the true base state (Table II
of the paper; for the Wasserstein cost individual parameters ranged from 1.0% to 46.6% error, and for the privileged state cost up to 179%). On hardware it used 64 trajectories of 4 s each (about 4.3 minutes) on a Go2
with a spring tied to a calf joint, and on a bipedal task, with a residual actuator network, and reports less drift after fine-tuning the policy. Failure
regimes (the robot falling) make the distributions ambiguous, and the authors list that as a limitation. Hardware success was
measured on one robot type (a Go2) with the final position measured by hand.

### Step 0. Choose what is identified
- **[paper]** Six static parameters (joint friction and armature for hip, thigh and calf), bounded to [0, 0.5]; or small neural residuals
  (action delta or residual torque) for effects that static parameters cannot represent.
- **[Mojito]** Start with the four servo-model numbers in `ServoParams` (`mojito/sim/actuators.py`): `kp`, `kd`, `latency`,
  `torque_scale`. Voltage is measured, not fitted, and `resolution` follows from the PCA9685 counts. Joint friction and armature, as in the paper,
  would be added if `mjcf.py` exposes them (not checked). Latency is applied in whole simulator steps (`Servo`, `ceil(latency / dt)`), so the cost may be
  stepwise in it; the identifiability check (step 4) will show whether that matters.
- Record the search ranges in the result JSON. They should come from the spread of the listed servo figures in `docs/STAGE2.md`, not from the value being recovered.

### Step 1. Build the "hardware" side in simulation (sim-to-sim test)
- **[paper]** The target simulator has the hidden values; run the policy for 64 trajectories of 4 s with random velocity commands; add
  zero-mean Gaussian torque noise and a random 0.02 to 0.1 s shift of the logged trajectory to mimic hardware.
- **[Mojito]** Hidden `ServoParams` values differ from the nominal ones. Run the gait through `run_gait`
  (`mojito/sim/gait_replay.py`) at several commanded speeds and yaw rates. Torque noise has to be scaled to the servo: the paper's 1.0 to
  5.0 N·m is large next to the 1.86 N·m lowest listed stall torque, so use fractions of the torque limit (for example 2%, 5%, 10%; my choice).
  The logged side goes through `JointSensor` (noise, quantisation, delay), and so does the simulated side being fitted, because the real logs will.
  Vary the supply voltage in the hidden model and see whether fitting at the wrong voltage biases the other parameters.
- Excite every parameter: static deflection under load shows `kp` (the stand test and `sit_to_stand` give this); fast joint moves in the trot at
  several speeds show `kd` and `latency`; the heaviest loads (sit-to-stand, 3 kg designs) show `torque_scale`. That mapping is a hypothesis; step 4 tests it.

### Step 2. Features, windowing and normalisation
- **[paper]** Three features per joint: position, velocity and policy action; accelerations were too noisy. Each trajectory is treated as unordered
  samples. All values scaled to [-1, 1] using shift and scale from the *hardware* data, applied unchanged to simulated samples and fixed during the whole fit.
- **[Mojito]** Joint position = measured angle. Action = the commanded joint target (the output of the IK layer). Velocity has to be
  differenced from measured angles, since the planned sensors report angle only. With the default 0.1° noise, differencing at
  200 Hz gives about 28°/s standard deviation of velocity noise (0.1° x sqrt 2 x 200, computed from the default `JointSensorParams.noise`), so
  filter the angle the same way on both sides before differencing, or fit on a lower-rate velocity. Drop the stand-up and ramp-in
  (`stand_time`, `ramp_time` in `run_gait`) so only the steady gait is compared, since the method assumes a stationary distribution. The number of samples on the two sides
  can differ; use a quantile-based distance (step 3) instead of requiring equal counts.

### Step 3. The cost, and its tests
- **[paper]** For each feature, the mean over joints of the one-dimensional Wasserstein distance (L1 distance between sorted samples), summed over the three features;
  chosen because it needs no time alignment and is cheap, though it ignores correlations between joints.
- **[Mojito]** `scipy.stats.wasserstein_distance` (SciPy is already a dependency) per joint and feature, on the normalised data. Add a penalty for runs
  in which the robot falls or leaves the joint limits, as `design_sweep` already counts validity; the paper has no mechanism for this and calls it a limitation, so how to do it is open.
- **Tests (independent checks, per `CLAUDE.md`):** the distance of a set to itself is 0; shifting every sample by c gives exactly |c|; permuting or time-shifting
  the samples changes nothing; the sorted-sample formula agrees with the SciPy function on unequal sample counts.

### Step 4. Check identifiability before optimising
- With the other three parameters held at their hidden values, sweep each one over its range and plot the cost. Each curve should have a clear minimum at the hidden value.
- Sweep pairs (kp with kd, kd with latency, kp with torque_scale) and look for long valleys. A flat curve or valley means the data cannot tell those parameters apart, so change the excitation (step 1) or fix one of them.
- This step is mine, not the paper's; it is cheap and tells whether the optimiser's answer can be trusted.

### Step 5. Optimise
- **[paper]** CMA-ES (the `cmaes` Python package), initial mean 0 in the normalised space, initial covariance 0.1 (friction) and 0.05 (armature), bounds [0, 0.5], 200 iterations for the six static parameters; 10 000 iterations for the neural residual models. Each cost
  evaluation runs a fresh set of rollouts.
- **[Mojito]** Same optimiser. Parameters in a normalised or log space over the step 0 ranges. `cmaes` is not currently a dependency; adding one is a decision for the owner (`scipy.optimize.differential_evolution` is a fallback already installed).
  The default CMA-ES population for 4 parameters is 8, so 200 iterations is about 1 600 candidate evaluations. If each evaluates 12 ten-second runs, that is about 19 000 runs. At the rate of `design_tune.py` (2 970 runs in about 10 minutes, `docs/STAGE2.md` and `CLAUDE.md`), that is on the order of an hour; this is an estimate to be replaced by a
  measurement (the core count behind the 10-minute figure is not stated). Run several seeds and report the spread, not the best.

### Step 6. Evaluate
- Recovered versus hidden values, as percent error per parameter, at each noise level and time shift (the paper's Tables I and II).
- Baselines on the same data, to compare with the paper: direct L2 on aligned observations (the paper's MatchO), and L2 on joint states plus ground-truth base state with resets (MatchS, which uses what the real robot will not have).
- Held-out behaviour: with the recovered values, run commands that were not used in the fit and compare against the hidden-parameter run on speed ratio, validity and peak torque (the metrics `design_sweep` reports), alongside the nominal parameters. The paper's version fine-tunes a policy; Mojito has no policy before work package 7, so this is the substitute.
- Write the numbers with `mojito.results.write_result` into `results/` like the other studies.

### Step 7. Break the model on purpose
- **[paper]** A virtual spring on one calf joint (a PD term, P = 75, D = 0) made static parameters insufficient; a residual actuator model was needed, and the same spring was then fitted on hardware.
  The more expressive residual models did worse than static parameters when the true change *was* static.
- **[Mojito]** Add an effect that `ServoParams` cannot express (a joint spring, a dead band different from the PCA9685 step, or stiction) to the hidden side and see what the static fit does. This shows how far the servo model can be wrong before the fit misleads, and whether a residual term is worth adding.

### Step 8. Hardware protocol (when a robot exists)
- Log what `docs/STAGE2.md` ("Network control and logging") lists: commanded positions, measured joint angles, IMU, battery voltage and rail currents, with the microcontroller's clock and sample numbers. About 4 minutes of walking is the paper's amount; Mojito's trot is slow, so expect more runs for the same number of gait cycles.
- Calibrate the joint sensors first (offset and once-per-turn nonlinearity, using known angles) and apply that correction in the logs, otherwise the fit can absorb a sensor bias into the servo parameters (my inference).
- Hold the supply voltage fixed and log it; servo torque and speed depend on it. The paper names actuator heating as a source of mismatch, so note the order of runs and check for drift between early and late runs.
- Run the same commands in the fitted simulator and on the robot, then repeat the step 6 comparison on held-out commands. No motion capture is required, but then velocity and distance on hardware are only as good as the estimator in section 4 or a hand measurement.

### What this plan will not show
- It does not show the servo model is right, only that a method recovers values inside it, until step 8.
- Sim-to-sim recovery is a best case: the "hardware" is made from the same code as the model.
- The paper's hardware evidence is one robot and hand-measured drift; treat the 18.7% and similar figures as indications about that setting, not as expected Mojito accuracy.

## 4. Camera plus IMU for velocity and motion

The aim is to estimate body velocity, yaw rate and related quantities (slip, distance travelled, heading drift) from a camera and the IMU, as a better
alternative to the IMU-plus-leg-kinematics estimate. Papers below are 2021 to 2026, read at the abstract level plus selected lines (see `docs/LITERATURE.md`).

### Which camera
- **Onboard camera:** what the robot itself could measure, so its estimate may be used for rewards, logging and, if timing allows, observations.
- **External fixed camera with markers:** the Stage 3 option already in `docs/PLAN.md`. It measures where the body went, so it is for evaluation and for
  checking the onboard estimate, not for policy observations (`CLAUDE.md`: keep ground truth out of policy observations).
Both are useful and they answer different questions. Which is wanted first is an open question below.

### What the papers say
| Paper | What it does | Relevance |
|---|---|---|
| Cerberus, 2209.07654 | Visual-inertial-leg odometry with stereo cameras, IMU, joint encoders and contact sensors; online kinematic calibration and contact outlier rejection; reports under 1% position drift on an A1 over long, high-speed runs. | The online calibration of kinematic parameters echoes Mojito's calibration step. It uses contact sensors, which the robot lacks. |
| Visual-Inertial and Leg Odometry Fusion, 2210.02127 | Stereo VIO fused with leg odometry in an EKF on a Solo12; VIO has latency and a low rate, so its estimate is predicted forward at the IMU rate. | Matches the off-board architecture and the rule not to close the loop over Wi-Fi: latency compensation is the key piece. |
| WALK-VIO, 2111.15164 | Visual-inertial odometry with leg-kinematic constraints that adapt to the walking motion; walking wobble lowers VIO accuracy; evaluated on simulation datasets the authors created and published. | Gait motion hurts camera tracking; relevant to a small, jerky servo robot. |
| Learning Inertial Odometry, 2111.00789 | A network learns a displacement from IMU data, fused with leg odometry; 37% lower relative pose error in difficult cases (slipping, compressible ground) on an ANYmal, 22% lower when combined with vision in visually degraded conditions. | An IMU-only improvement that needs no camera; needs training data from the robot. |
| Sensor Configuration Matters, 2606.19067 | Benchmark on an ANYmal D: global-shutter cameras reduced motion-induced tracking failures compared with rolling shutter; tightly coupled inertial integration could make tracking worse. | Camera choice matters. The shutter type of low-cost camera modules is not known here and should be checked (my assumption is that many are rolling shutter). |
| Robust RL for resource-constrained quadrupeds, 2505.12537 | A small quadruped that trains a policy and a state estimator together to provide odometry, optionally fused with VIO; time-of-flight sensor placement matters. | A small-robot example of a learned state estimator in the loop. |

The robots in these papers carry stereo cameras or depth sensors and a computer on board (A1, Solo12, ANYmal). None is a hobby-servo robot with an ESP32. The controller here
has no computer for visual processing, so frames would be streamed to the laptop, adding latency and timing uncertainty.

### Plan
1. **Baseline without a camera.** Item 7: IMU plus leg kinematics in simulation using `mojito/sim/sensors.py`, error against ground truth for trot and crawl at several speeds. Everything below is judged against this.
2. **Render a camera in simulation.** Mount a camera on the base in MuJoCo (the repo already renders GIFs) and render frames during `run_gait` at the camera rate. Check frame timing against the simulation clock.
3. **Estimator ladder, cheapest first:** (a) optical-flow velocity from a downward or forward camera, scaled by camera height known from leg kinematics (my idea, not from the papers); (b) visual-inertial odometry run on the laptop; (c) fuse (a) or (b) with leg odometry in an EKF, with the camera delayed measurement and prediction forward at IMU rate (as in the Dhédin paper); (d) optionally a learned inertial displacement (as in the Buchanan paper).
4. **Measure** velocity and yaw-rate error, drift per metre travelled, tracking failures, and added latency against ground truth, across gait speeds, camera frame rates and added vibration, and compare each rung to the baseline from step 1.
5. **Add realism to the rendered camera** (blur, rolling-shutter skew, noise, dropped frames, timestamp jitter) and see which rung survives. Without this, step 4 is a best case.
6. **Hardware:** choose the camera (global shutter if possible), calibrate the camera-IMU extrinsics and time offset, mount an external camera with markers as the reference, and repeat step 4 on the robot. Slip can be detected by comparing the visual velocity with the kinematic velocity.
7. **Use of the result:** the estimate feeds reward computation, logs and evaluation, and slow commands; joint-level control stays on the microcontroller (`docs/STAGE2.md`, network design points). It could also be a distribution-matching feature in section 3 once calibrated, since it adds body motion that joint logs alone lack.

### What this plan will not show
- Simulated frames are cleaner than real ones; a good result there says nothing about blur, lighting or servo vibration.
- The papers' accuracy figures come from robots with better cameras and computers.

## 5. Open questions raised

- Which camera is wanted first, the onboard one (estimator and possibly policy input) or an external one with markers (evaluation)?


- Gait in Eight and CPG-RL both use analytic IK. If Mojito's policy also goes through analytic IK, the learned IK network is not in the
  locomotion loop, and the case for it rests on Stage 3 calibration and non-closed-form limbs. Is that the intended role?
- Which of arms (a) to (c) in item 1 should be the main one for the first robot?
