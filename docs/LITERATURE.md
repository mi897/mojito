# Literature review (arXiv, November 2021 to November 2026)

What the recent literature says about the parts of Mojito's plan, and where Mojito differs. Written 2026-11-07.

## How this was made, and its limits

- About 20 keyword queries on the arXiv API (relevance-sorted, restricted to submissions from 2021-11-01 to 2026-11-08),
  then a second round of targeted queries. About 190 papers were screened by title and abstract.
- Citation counts are from the Semantic Scholar batch API on 2026-11-07. They favour older papers; a 2026 paper with 3
  citations is not worse than a 2022 paper with 150.
- **Read in more detail (key sections, found by searching the full text, not every line):** MorphIK, A Walk in the Park,
  Actuator Reality Shaping, Gait in Eight, CPG-RL, Controlling the Solo12, Simulator Adaptation via Proprioceptive Distribution
  Matching, SPI-Active, Learning Torque Control, Learning to Walk from Three Minutes (setup, results, limitations).
  **Method and setup lines only:** URMA, DrEureka, Boekel co-design, IKDiffuser, DayDreamer.
What is said about a paper below only covers what was read.
- The search is keyword-based. "Nothing found" means this search found nothing, not that nothing exists.
- Google Scholar was not reachable from the session. Semantic Scholar was, but rate-limited.
- Nothing here is a result of Mojito. Mojito figures quoted come from `results/learning/metrics.json`.

## 1. Learned inverse kinematics (Stage 0)

| Paper | What it does | Relation to Mojito |
|---|---|---|
| MorphIK, Clasmeier et al. 2026 ([2609.29908](https://arxiv.org/abs/2609.29908)) | One flow-matching model, conditioned on a transformer encoding of the robot read from its URDF, solves IK for unseen revolute serial arms with 6 to 9 DoF. Trained only on procedurally generated robots with forward kinematics giving the targets. Raw error about 5 cm on unseen real arms; under 1 cm after one damped-least-squares step and under 1 mm after three. | Same recipe as Mojito at a different scale: morphology as input, targets from forward kinematics, a Cartesian (FK) loss added so that the network commits to one solution rather than averaging several. Mojito's problem is much smaller (a 3-joint leg, 3 length inputs). Its use of a classical solver to polish a learned seed is the idea behind Mojito's calibration step, in the other direction. |
| IKDiffuser 2025 ([2506.13087](https://arxiv.org/abs/2506.13087)) | Diffusion IK for kinematic trees with several end-effectors; one model per robot; used to seed optimization solvers; seven platforms including humanoids. | Not leg-specific. |

Mojito measured on unseen legs with varied lengths: median foot error 0.17 mm, 95th percentile 0.84 mm
(`results/learning/metrics.json`, key `learned_varied_legs`). The closed-form IK on the same legs is exact to rounding
(about 1e-14 mm).

**What this means.** For a 3-joint leg the closed form is exact and cheap, and the legged-locomotion papers below use
analytic IK (Gait in Eight says so explicitly). So the learned network is not justified by accuracy on this leg. Its
justification has to be the cases without a closed form (limbs with more joints, other URDFs, a real leg with unknown
lengths and offsets). I found no paper on learned IK specifically for legs with link lengths as inputs, which is
consistent with this being a thin niche, not necessarily an open one.

## 2. Acting in foot space through IK (Stage 2 reinforcement learning)

Mojito plans a policy that acts in foot space through the IK layer.

- **Gait in Eight**, Bohlinger et al. 2025 ([2503.08375](https://arxiv.org/abs/2503.08375)). Compares two action spaces on a
  real quadruped (MAB HoneyBadger): offsets to joint targets (JTP), and a central pattern generator (CPG) whose policy output
  is a Cartesian offset per foot, added to the CPG foot position and converted with analytical IK before a PD controller.
  Their summary: joint targets for agile high-speed locomotion, CPG with foot offsets for stable natural gaits. Learning
  takes about 8 minutes of real-time training.
- **CPG-RL**, Bellegarda and Ijspeert 2022 ([2211.00458](https://arxiv.org/abs/2211.00458)). The policy sets oscillator
  parameters; CPG states give desired foot positions; IK gives joint angles; PD tracks them. Reported to transfer to a
  Unitree A1 without domain randomization or added noise.
- **A Walk in the Park**, Smith et al. 2022 ([2208.07860](https://arxiv.org/abs/2208.07860)). Joint PD targets at 20 Hz,
  constrained to a box around a default pose. They state that constraining the action space is crucial and that the
  default pose matters for exploration: a lower stance was used on the real robot because it failed less.
- **Solo12**, Aractingi et al. 2023 ([2309.16683](https://arxiv.org/abs/2309.16683)). Joint impedance references from a
  policy running at 100 Hz over PD loops at 10 kHz; randomizing the PD gains was enough for their transfer; direct torque
  control was found harder to learn than position control.

**For Mojito.** A residual in foot space around a scripted gait through IK is an established design, and it is what the
plan describes. The evidence that it beats joint-target actions is mixed (they trade off), so it is cheap and informative to
implement both action spaces in Stage 2 and compare them in the Mojito simulator. Constraining the action range around a
nominal pose and picking a safe default stance are both supported.

## 3. Servo and actuator models, sim-to-real (Stage 2)

- **Actuator Reality Shaping**, Yamamori et al. 2026 ([2607.02205](https://arxiv.org/abs/2607.02205)). Instead of making
  the simulator match the actuator, a two-degree-of-freedom controller with a disturbance observer makes the real motor follow
  the ideal second-order PD response the simulator assumes. It needs the motor's torque constant, rotor inertia and a
  current command, and was shown on a Dynamixel YM070 high-gear-ratio servo, an arm, a wheeled-legged robot and a humanoid.
  **Does not apply to the first Mojito robot:** the DS3218 has its own internal loop, no current access and no position
  feedback. It may apply to the later DC motors with position, velocity and current feedback.
- **Solo12** (above): identified actuator friction and added an energy-loss penalty from Joule and friction losses.
- **Learning Torque Control**, Chen et al. 2022 ([2203.05194](https://arxiv.org/abs/2203.05194)): A1 policy outputs torques at
  200 Hz with no PD loop (position policies typically run at 20 to 100 Hz); reports a higher converged return in one comparison
  with a simplified reward and no extra sim-to-real difficulty. Needs a torque command, so not applicable to the DS3218.
- **DrEureka**, Ma et al. 2024 ([2406.01967](https://arxiv.org/abs/2406.01967)): a language model writes the reward and the
  domain-randomization ranges; on a Go1 it reports 34% higher forward velocity than human-designed settings.
- I found no paper that models hobby servos (torque-limited, internal loop, latency) for legged sim-to-real. Mojito's servo
  model (`mojito/sim/actuators.py`) therefore rests on its own assumptions, which `docs/STAGE2.md` already flags as unverified.

## 4. Fitting the simulator to real data (Stage 4)

- **Simulator Adaptation via Proprioceptive Distribution Matching**, Dao et al. 2026
  ([2604.11090](https://arxiv.org/abs/2604.11090)). Compares hardware and simulation rollouts as distributions of joint
  observations and actions (a Wasserstein-based cost), so it needs neither motion capture nor time-aligned trajectories.
  Per its abstract, less than five minutes of hardware data is sufficient; evaluated on a Go2 quadruped in simulation
  ablations plus real-world experiments.
- **SPI-Active**, Sobanbabu et al. 2025 ([2505.14266](https://arxiv.org/abs/2505.14266)): sampling-based (CMA-ES)
  identification of mass, inertia and actuator parameters on a Unitree Go2 and G1, then active exploration that optimizes
  commands for Fisher information. It uses a per-joint tanh torque-saturation actuator model, which beat a plain PD and a
  single shared saturation on two jump tasks. **Its base-state cost term uses motion-capture trajectories**, and it runs offline.

**For Mojito.** Mojito will have joint-angle sensors and an IMU but no motion capture, so the proprioceptive-only matching
of Dao et al. is the closer fit than methods that need ground-truth base trajectories.

## 5. Learning from real interaction (Stages 3 and 4)

| Paper | Robot | Real-world data | Notes |
|---|---|---|---|
| A Walk in the Park 2022 | Unitree A1 | 20 minutes | model-free, learned resets, PD target actions |
| Learning to Walk from Three Minutes 2024 ([2410.09163](https://arxiv.org/abs/2410.09163)) | Unitree Go1 | 3.0 minutes (18 000 steps) | model-based; rigid-body model from the differentiable simulator Brax plus a learned residual; base velocity from a Vicon system; about 0.98 m/s on hard ground and 0.53 m/s on memory foam |
| Gait in Eight 2025 | MAB HoneyBadger | 8 minutes | CrossQ; proprioceptive-only reward |
| DayDreamer 2022 ([2206.14176](https://arxiv.org/abs/2206.14176)) | Unitree A1 | about 1 hour | world model; no resets; recovers from pushes in about 10 minutes |

**Measurement is the common difficulty**, and it bears directly on Mojito's open question 3 (where do real foot and base
positions come from). Walk in the Park estimates body velocity with a Kalman filter fusing integrated acceleration and leg
forward kinematics, and uses binary foot contacts. Gait in Eight limits the reward to proprioceptive quantities and reports
that its velocity estimate is biased low without foot contact sensors, because it must assume all four feet are on the
ground. The three-minute paper uses motion capture. So the IMU-plus-leg-kinematics route is used in practice, with a known
bias; a Mojito robot with grippy pads and no contact sensors would be in the situation Gait in Eight describes.

All of these use commercial or research-grade robots with joint torque or high-quality position control. None is a
hobby-servo robot, so the sample counts do not transfer directly. In addition (my inference, not from the papers), real
training on a robot with 20 kg·cm hobby servos adds wear and a battery limit that the papers do not face.

## 6. Design exploration (the design sweeps)

- **URMA**, Bohlinger et al. 2024 ([2409.06366](https://arxiv.org/abs/2409.06366)): one policy for many morphologies via
  attention encoders over per-joint and per-foot descriptions; trained on 16 simulated robots, zero-shot to a held-out A1 and
  a MAB Silver Badger, and deployed zero-shot on three real robots. Related: ManyQuadrupeds ([2310.10486](https://arxiv.org/abs/2310.10486)),
  Multi-Loco ([2506.11470](https://arxiv.org/abs/2506.11470)).
- **Boekel et al. 2026** ([2611.02462](https://arxiv.org/abs/2611.02462)): morphology-conditioned policy (design vector in the
  observation, leg scale factors 0.70 to 1.30 among the parameters) trained across designs in Isaac Lab, then Bayesian optimization
  of the design with the policy held fixed; reports 18.6% and 35.7% lower cost of transport on hardware for two tasks. The robot has
  compliant four-bar legs, unlike Mojito's.

**For Mojito.** The design sweeps tune a scripted gait per design, which is a weaker controller than a learned one. A
conclusion like "design A beats design B" may change once a learned policy is used. A morphology-conditioned policy (the
URMA/Boekel route) would match how Mojito already treats lengths as inputs, but that is a larger step than the plan has.

## 7. Low-cost platforms

Solo12 ([2309.16683](https://arxiv.org/abs/2309.16683)), HyperDog ([2209.09171](https://arxiv.org/abs/2209.09171)),
Q8bot ([2508.01149](https://arxiv.org/abs/2508.01149)), MEVIUS2 ([2603.22031](https://arxiv.org/abs/2603.22031)) and
additively manufactured quadrupeds ([2410.00122](https://arxiv.org/abs/2410.00122)) are the nearest comparable platforms
(titles and abstracts only, except Solo12). I found none using an ESP32 with a PCA9685 and learned control.

## Summary of how Mojito compares

| Mojito choice | Literature |
|---|---|
| Learned IK over leg lengths, FK loss | Same recipe as MorphIK at a larger scale; not needed for accuracy on a 3-joint leg, where the closed form is exact. |
| Policy acts in foot space through IK | Used by CPG-RL and Gait in Eight; joint-target actions are the alternative and trade off. Compare both. |
| Ground truth kept out of policy observations | Consistent with the real-robot papers; reward on hardware then needs a state estimate, with known bias. |
| Servo model with limits, latency, resolution | No hobby-servo precedent found. Actuator Reality Shaping needs current-controlled motors, so it is a candidate for the later DC-motor robot, not the DS3218. |
| Fit the simulator to real data (Stage 4) | Proprioceptive distribution matching and SPI-Active are the closest published methods. |
| Scripted gaits in design sweeps | Recent work optimizes design with a learned controller instead. |

## What this review does not show

- It does not show that any of these methods works on a hobby-servo robot, or on Mojito's.
- It does not compare the papers' results with each other: they use different robots, rewards and metrics.
- It covers arXiv preprints found by keyword. Peer review status was not checked, and journal-only work is missing.

See `docs/PAPER_COMPARISONS.md` for the side-by-side comparisons with Mojito and the list of experiments these suggest.
