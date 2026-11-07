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
   and torque limit as "the real robot", log joint angles and commands under a random exploration policy, and see how well a Wasserstein
   distribution match recovers the hidden values, and how much data it needs. *Why:* tests the Stage 4 idea before hardware exists.
   *Would not show:* success on real servos, which have effects the simulator lacks.
7. **Base-velocity estimator without contact sensors.** In simulation with the IMU and joint-sensor models (`mojito/sim/sensors.py`),
   estimate body velocity from the IMU and leg forward kinematics assuming stance feet are fixed, and compare with ground truth
   across trot and crawl. *Why:* Gait in Eight reports a bias without foot contact sensors, and the robot has none. It informs open
   question 3. *Would not show:* bias on real terrain or real sensors.
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

### Low priority

17. LLM-written rewards and randomization ranges (DrEureka). **World-model or differentiable-simulator learning** (DayDreamer,
    3-minute paper): large additions to the plan; revisit only if sample count on hardware turns out to be the limit.

## 3. Open questions raised

- Gait in Eight and CPG-RL both use analytic IK. If Mojito's policy also goes through analytic IK, the learned IK network is not in the
  locomotion loop, and the case for it rests on Stage 3 calibration and non-closed-form limbs. Is that the intended role?
- Which of arms (a) to (c) in item 1 should be the main one for the first robot?
