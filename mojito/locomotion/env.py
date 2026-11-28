"""Locomotion environment for work package 7 (docs/STAGE2.md): the actuated quadruped, a scripted trot, and a policy that
adds foot offsets to it.

The policy acts in foot space through the IK layer:

    foot_target = scripted_trot_target + action_scale * action      (body frame, m; action in [-1, 1], shape (4, 3))
    joint_targets = closed-form IK (foot_target)  ->  servo model  ->  torques  ->  MuJoCo

so the action is a residual on `gaits.foot_targets`, not a joint command (the comparison with joint targets is work package 8).
A zero action is exactly the scripted trot, which is the baseline every learned policy is judged against.

What the policy sees (`observation`) is only what the hardware would measure: joint angles from `sensors.JointSensor`
(noise, quantisation, delay), their finite difference, the IMU's angular rate and specific force, the previous action, the gait
phase (a clock, as the scripted trot has one) and the command. Ground truth (body velocity, tilt, height) is used for the reward
and termination only, never in the observation (CLAUDE.md, open question 3).

Reward per control step (all in `reward_terms`; weights are `RewardWeights`):
    tracking   exp(-|v_body - command|^2 / tracking_width^2) for the forward and lateral velocity in the body frame,
               plus the same for the yaw rate
    upright    -(1 - R[2, 2]) : zero when level, -1 when the body is on its side
    torque     -|torque / stall|^2 summed over joints
    smooth     -|action - previous_action|^2
    alive      constant bonus per step survived
An episode ends when the body tilts past `max_tilt_deg` or drops under half the stance height.

Randomised per episode (`Randomisation`): total mass, floor friction, servo strength and latency. Everything is a placeholder
(dimensions, masses, servo figures, friction), see docs/STAGE2.md; this is a first environment, not a tuned one.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..geometry.placeholders import SIM_QUADRUPED
from ..robot import gaits, statics
from ..sim import actuators, mjcf, sensors
from ..sim.gait_replay import TROT_SERVO, TROT_STANCE_HEIGHT, commanded_feet
from ..sim.robot import SimRobot

CONTROL_PERIOD = 0.02  # s between policy actions (50 Hz); equals the servo latency default, a multiple of the physics step
ACTION_SCALE = 0.02    # m of foot offset at action = 1
STAND_TIME = 0.5       # s standing before the gait blends in
RAMP_TIME = 0.5        # s to blend from the standing feet to the trot
N_OBSERVATIONS = 12 + 12 + 3 + 3 + 12 + 2 + 3
N_ACTIONS = 12


@dataclass
class RewardWeights:
    tracking_width: float = 0.1   # m/s (and rad/s for the yaw rate): the velocity error at which tracking falls to exp(-1)
    tracking: float = 1.0
    yaw_tracking: float = 0.5
    upright: float = 2.0
    torque: float = 0.02
    smooth: float = 0.05
    alive: float = 0.2


@dataclass
class Randomisation:
    """Ranges (low, high) sampled uniformly per episode; (1, 1)-style equal ends switch a term off."""
    mass_scale: tuple = (0.8, 1.2)
    friction: tuple = (0.6, 1.2)         # floor friction coefficient (the placeholder is 1.0)
    torque_scale: tuple = (0.8, 1.0)     # fraction of the listed servo stall torque
    latency: tuple = (0.01, 0.04)        # s

    @classmethod
    def none(cls):
        return cls(mass_scale=(1.0, 1.0), friction=(mjcf.FLOOR_FRICTION,) * 2, torque_scale=(1.0, 1.0), latency=(0.02, 0.02))

    def sample(self, rng) -> dict:
        return {name: float(rng.uniform(*getattr(self, name))) for name in ("mass_scale", "friction", "torque_scale", "latency")}


@dataclass
class EnvConfig:
    episode_time: float = 6.0                   # s of policy control after the stand
    command_range: tuple = ((0.05, 0.25), (-0.03, 0.03), (-0.3, 0.3))  # vx, vy m/s, yaw rate rad/s
    max_tilt_deg: float = 30.0
    gait: gaits.Gait = field(default_factory=lambda: gaits.TROT)
    reward: RewardWeights = field(default_factory=RewardWeights)
    randomisation: Randomisation = field(default_factory=Randomisation)
    timestep: float = 0.002


class LocomotionEnv:
    """One robot, reset and stepped like a gym environment, with `reset(seed)` -> observation, `step(action)` ->
    (observation, reward, done, info). Not thread-safe; one per process."""

    def __init__(self, config: EnvConfig | None = None):
        self.config = config or EnvConfig()
        self.n_physics_steps = int(round(CONTROL_PERIOD / self.config.timestep))
        self.n_episode_steps = int(round(self.config.episode_time / CONTROL_PERIOD))

    # ----- episode -----
    def reset(self, seed: int = 0, command=None, randomise: bool = True) -> np.ndarray:
        """Build a fresh robot (so the sampled mass applies), stand it, and return the first observation.

        `command` (vx, vy, yaw rate) overrides the sampled one; `randomise=False` uses the nominal placeholder values.
        """
        config = self.config
        rng = np.random.default_rng(seed)
        self.rng = rng
        randomisation = config.randomisation if randomise else Randomisation.none()
        self.domain = randomisation.sample(rng)
        masses = {role: mass * self.domain["mass_scale"] for role, mass in SIM_QUADRUPED.masses.items()}
        self.physics = SimRobot(SIM_QUADRUPED, TROT_STANCE_HEIGHT, masses, config.timestep)
        floor_id = self.physics.model.geom("floor").id
        self.physics.model.geom_friction[floor_id, 0] = self.domain["friction"]
        self.kinematics = self.physics.kinematics
        self.home = self.kinematics.stance(TROT_STANCE_HEIGHT)
        self.servo = actuators.Servo(actuators.ServoParams(**TROT_SERVO, torque_scale=self.domain["torque_scale"],
                                                          latency=self.domain["latency"]))
        self.joint_sensor = sensors.JointSensor(seed=int(rng.integers(1 << 31)))
        self.imu = sensors.Imu(seed=int(rng.integers(1 << 31)))
        if command is None:
            command = [rng.uniform(*bounds) for bounds in config.command_range]
        self.command = np.asarray(command, float)
        self.time = 0.0
        self.step_count = 0
        self.previous_action = np.zeros(N_ACTIONS)
        self.previous_measured = self.joint_sensor.read(self.physics.joint_angles(), config.timestep)
        self.joint_velocity_estimate = np.zeros(12)
        self.gyro = np.zeros(3)
        self.specific_force = np.array([0.0, 0.0, statics.G])
        self._velocity_world = np.zeros(3)
        self._stand_hold()
        return self.observation()

    def _stand_hold(self) -> None:
        """Run the stand phase with the scripted command (no policy yet), as the gait replays do."""
        n_stand = int(round(STAND_TIME / self.config.timestep))
        joint_targets = self.kinematics.solve(self.home).ravel()
        for _ in range(n_stand):
            self._physics_step(joint_targets)

    # ----- stepping -----
    def scripted_feet(self, time: float) -> np.ndarray:
        """(4, 3) body-frame trot foot targets at policy time `time` s (the baseline), as `gait_replay.commanded_feet`."""
        targets, _ = commanded_feet(self.config.gait, self.home, [time], 0.0, RAMP_TIME, tuple(self.command[:2]), float(self.command[2]))
        return targets[0]

    def _physics_step(self, joint_targets) -> None:
        dt = self.config.timestep
        torques = self.servo.step(joint_targets, self.physics.joint_angles(), self.physics.joint_velocities(), dt)
        self.physics.step(torques)
        self.last_torques = torques
        self.measured = self.joint_sensor.read(self.physics.joint_angles(), dt)

    def step(self, action):
        """Apply `action` (12,), clipped to [-1, 1], for one control period. Returns (observation, reward, done, info)."""
        config = self.config
        action = np.clip(np.asarray(action, float), -1.0, 1.0)
        feet = self.scripted_feet(self.time) + ACTION_SCALE * action.reshape(4, 3)
        # foot_target = scripted_trot_target + ACTION_SCALE action; unreachable targets are not solved for (the IK clamps),
        # they are counted in `info` so a run that leans on them shows it
        reachable = bool(self.kinematics.reachable(feet).all())
        joint_targets = self.kinematics.solve(feet).ravel()
        torque_sum = np.zeros(12)
        for _ in range(self.n_physics_steps):
            self._physics_step(joint_targets)
            torque_sum += np.abs(self.last_torques)
        self.time += CONTROL_PERIOD
        self.step_count += 1
        self._read_imu()
        reward, terms = self.reward_terms(action, torque_sum / self.n_physics_steps)
        self.previous_action = action
        tilt = self.physics.tilt_deg()
        height = self.physics.origin()[2]
        fell = bool(tilt > config.max_tilt_deg or height < 0.5 * (TROT_STANCE_HEIGHT + mjcf.FOOT_RADIUS))
        done = fell or self.step_count >= self.n_episode_steps
        info = {"fell": fell, "reachable": reachable, "tilt_deg": tilt, "terms": terms,
                "velocity_body": self.body_velocity(), "peak_torque_fraction": float(np.abs(self.last_torques).max() / self.servo.params.stall_torque)}
        return self.observation(), reward, done, info

    # ----- sensing -----
    def _read_imu(self) -> None:
        """Gyro (body frame) and specific force from the simulation, through the IMU model."""
        data = self.physics.data
        rotation = self.physics.body_rotation()
        # free-joint angular velocity is already in the body frame; specific force = R^T (a - g) with gravity -G on z
        gyro = data.qvel[3:6].copy()
        specific_force = rotation.T @ (data.qacc[0:3] + np.array([0.0, 0.0, statics.G]))
        self.gyro, self.specific_force = self.imu.read(gyro, specific_force, CONTROL_PERIOD)
        self.joint_velocity_estimate = (self.measured - self.previous_measured) / CONTROL_PERIOD
        self.previous_measured = self.measured.copy()
        self._velocity_world = data.qvel[0:3].copy()

    def body_velocity(self) -> np.ndarray:
        """(3,) ground-truth body velocity in the body frame, m/s (reward and evaluation only)."""
        return self.physics.body_rotation().T @ self._velocity_world

    def observation(self) -> np.ndarray:
        """(N_OBSERVATIONS,) what the hardware could measure; see the module docstring for the layout."""
        home_angles = self.kinematics.solve(self.home).ravel()
        phase = 2 * np.pi * self.time / self.config.gait.period
        gravity_direction = self.specific_force / max(np.linalg.norm(self.specific_force), 1e-6)
        return np.concatenate([
            self.measured - home_angles,                    # 12 joint angles relative to standing
            0.1 * self.joint_velocity_estimate,             # 12 finite-difference velocities, scaled to about unit size
            0.2 * self.gyro,                                # 3
            gravity_direction,                              # 3
            self.previous_action,                           # 12
            [np.sin(phase), np.cos(phase)],                 # 2 gait clock
            self.command * [4.0, 4.0, 2.0],                 # 3 command, scaled
        ])

    # ----- reward -----
    def reward_terms(self, action, mean_abs_torque):
        """Weighted reward and its terms. tracking = exp(-|v - v_cmd|^2 / w^2) etc. (module docstring)."""
        weights = self.config.reward
        velocity = self.body_velocity()
        yaw_rate = self.true_yaw_rate()
        planar_error = np.sum((velocity[:2] - self.command[:2]) ** 2)
        terms = {
            "tracking": weights.tracking * np.exp(-planar_error / weights.tracking_width ** 2),
            "yaw_tracking": weights.yaw_tracking * np.exp(-(yaw_rate - self.command[2]) ** 2 / weights.tracking_width ** 2),
            "upright": -weights.upright * (1.0 - self.physics.body_rotation()[2, 2]),
            "torque": -weights.torque * float(np.sum((mean_abs_torque / self.servo.params.stall_torque) ** 2)),
            "smooth": -weights.smooth * float(np.sum((action - self.previous_action) ** 2)),
            "alive": weights.alive,
        }
        return float(sum(terms.values())), {key: float(value) for key, value in terms.items()}

    def true_yaw_rate(self) -> float:
        """Ground-truth yaw rate about the world vertical, rad/s: the body-frame angular velocity rotated to the world, z."""
        return float((self.physics.body_rotation() @ self.physics.data.qvel[3:6])[2])
