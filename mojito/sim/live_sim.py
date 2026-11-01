"""A gait replay you can steer while it runs (used by scripts/gait/live_gait.py).

Same model as `gait_replay.run_gait` (closed-form IK, `actuators.Servo`, actuated MuJoCo model), but advanced a
few steps at a time, with the gait and servo parameters changeable between steps.

Changing parameters while walking needs care, because a gait is a function of time:
- the cycle position is accumulated, `cycles += dt / period`, and the gait is evaluated at `cycles * period`, so
  changing the period does not jump the phase;
- velocity, yaw rate, swing height and sway are low-passed (time constant `SMOOTHING_TIME`) so the feet do not jump;
- the commanded foot positions are limited to `MAX_TARGET_SPEED` so that changing duty or offsets, which can move a
  foot target instantly, cannot ask for an instant jump.
With constant parameters this reproduces `run_gait` (tests/sim/test_live_sim.py).

Everything is a placeholder (dimensions, masses, servo figures, friction), see docs/STAGE2.md.
"""
from __future__ import annotations

import dataclasses
from collections import deque

import numpy as np

from ..geometry.placeholders import SIM_QUADRUPED
from ..robot import body, gaits, leg
from . import actuators, gait_replay, mjcf, stand
from .robot import SimRobot, roll_pitch_yaw

SMOOTHING_TIME = 0.3       # s, low-pass on velocity, yaw rate, lift and sway
MAX_TARGET_SPEED = 1.5     # m/s, limit on how fast a commanded foot position may move
BLEND_TIME = 0.5           # s, to blend between standing and walking
HISTORY = 2.0              # s, window for the measured velocity
CONTACT_FORCE = 0.5        # N, a foot carrying more is "down"


def preset_for(gait, forward_speed) -> dict:
    """Live parameters for `gait` commanded at `forward_speed` (m/s): timing and swing from the Gait, servo gains and stance
    height from `gait_replay.settings_for`, which are what its results in docs/STAGE2.md were measured with."""
    settings = gait_replay.settings_for(gait)
    return dict(period=gait.period, duty=gait.duty, offsets=list(gait.offsets), lift=gait.lift, sway=gait.sway,
                vx=forward_speed, vy=0.0, yaw_rate=0.0, kp=settings["servo"].kp, kd=settings["servo"].kd, height=settings["height"])


# The trot gets the tuned settings of gait_replay (kp 50, kd 0.9, 0.16 m stance, 22 mm swing); the crawl and the pace the original
# servo (kp 10, kd 0.2), the 0.2125 m stance and a 30 mm swing. The speed is the one each preset starts at, m/s.
PRESETS = {"trot": preset_for(gaits.TROT, 0.15), "crawl": preset_for(gaits.CRAWL, 0.04), "pace": preset_for(gaits.PACE, 0.1)}
DEFAULTS = dict(
    PRESETS["trot"],
    # body posture while the feet stay planted (rad): the body rolls, pitches and yaws about its own centre
    body_roll=0.0, body_pitch=0.0, body_yaw=0.0,
    # body height change (m) relative to the stance height, feet planted; positive raises the body
    body_height=0.0,
    # total mass, kg: the placeholder's 1.72 kg, rounded to 1 g so that it reads 1.72 and not the float sum 1.7200000000000002
    mass=round(SIM_QUADRUPED.total_mass, 3), speed=1.0, walk=False,
    # robot design: these rebuild the model (like mass). Metres: abduction offset (la), upper (lu) and lower (ll) leg, body size.
    # The stance height, `height`, comes from the trot preset and needs no rebuild (see SMOOTHED).
    la=SIM_QUADRUPED.abduction_offset, lu=SIM_QUADRUPED.upper_length, ll=SIM_QUADRUPED.lower_length,
    body_length=SIM_QUADRUPED.body_length, body_width=SIM_QUADRUPED.body_width,
    voltage=actuators.ServoParams().voltage,
)
# Full-stick command (forward m/s, left m/s, yaw rad/s) for drive mode (keyboard or gamepad), per preset. Probed in 6 s runs
# (forward, back, sideways and turning each alone, and the trot with all three at once): none fell. The pace is left out,
# it barely moves in physics. The measured speed is lower than the command, see docs/STAGE2.md.
# height_rate is how fast a held raise or lower input (X and Z, the right stick up and down) moves the stance height in walk mode, m/s.
DRIVE = {"trot": dict(vx=0.25, vy=0.12, yaw_rate=0.8, height_rate=0.04), "crawl": dict(vx=0.06, vy=0.03, yaw_rate=0.3, height_rate=0.04)}
# (low, high) for each numeric parameter; anything outside is clamped
LIMITS = dict(period=(0.2, 4.0), duty=(0.1, 0.95), lift=(0.0, 0.1), sway=(0.0, 0.06), vx=(-0.5, 0.5), vy=(-0.3, 0.3),
              yaw_rate=(-3.0, 3.0), body_roll=(-0.4, 0.4), body_pitch=(-0.4, 0.4), body_yaw=(-0.5, 0.5), body_height=(-0.04, 0.04), kp=(1.0, 300.0), kd=(0.0, 5.0), mass=(0.5, 8.0), speed=(0.1, 2.0),
              la=(0.02, 0.09), lu=(0.08, 0.24), ll=(0.08, 0.24), body_length=(0.15, 0.5), body_width=(0.08, 0.3),
              height=(0.1, 0.32), voltage=(actuators.V_MIN, actuators.V_MAX))
REBUILD = ("mass", "la", "lu", "ll", "body_length", "body_width")  # changing any of these needs a new model (the stance height does not: see `height`)
# `height` is the stance height (m): the feet are placed that far below the hips. It is eased like a command, so it can be changed while walking
# and the body rises or sinks over the planted feet; the model is only built at that height when it is reset.
SMOOTHED = ("vx", "vy", "yaw_rate", "lift", "sway", "body_roll", "body_pitch", "body_yaw", "body_height", "height")
# Full-stick body posture for the drive mode that tilts the body instead of walking (rad, and m for the height): the whole slider
# range, so a full stick goes as far as the page's sliders. Corners of the stick box beyond what the servos hold are in docs/STAGE2.md.
POSE = {key: LIMITS[key][1] for key in ("body_roll", "body_pitch", "body_yaw", "body_height")}


DEFAULT_TOTAL_MASS = SIM_QUADRUPED.total_mass  # 1.72 kg: body and four legs of mjcf.DEFAULT_MASSES


def masses_for_total(total_mass):
    """The placeholder masses of mjcf.DEFAULT_MASSES, all multiplied by total_mass / 1.72 kg, so the robot weighs `total_mass` kg."""
    return {role: mass * total_mass / DEFAULT_TOTAL_MASS for role, mass in mjcf.DEFAULT_MASSES.items()}


def design_of(params):
    """The placeholders.QuadrupedDesign that live parameters describe: leg (la, lu, ll), body size, stance height, total mass."""
    return dataclasses.replace(SIM_QUADRUPED, abduction_offset=params["la"], upper_length=params["lu"], lower_length=params["ll"],
                               body_length=params["body_length"], body_width=params["body_width"],
                               stance_height=params["height"], masses=masses_for_total(params["mass"]))


def browser_constants() -> dict:
    """Everything the JavaScript port of this simulation (site/assets/browsersim.js) needs that is a number in Python: written to
    site/data/livemeta.js by scripts/website/build_site.py, so no figure is copied by hand. `meta` is what the local server's /meta serves."""
    servo = {k: float(v) for k, v in dataclasses.asdict(actuators.ServoParams()).items()}
    return {
        "meta": {"limits": LIMITS, "presets": PRESETS, "defaults": DEFAULTS, "drive": DRIVE, "pose": POSE},
        "constants": {
            "smoothing_time": SMOOTHING_TIME, "max_target_speed": MAX_TARGET_SPEED, "blend_time": BLEND_TIME, "history": HISTORY,
            "contact_force": CONTACT_FORCE, "timestep": 0.002, "smoothed": list(SMOOTHED), "rebuild": list(REBUILD),
            "default_total_mass": DEFAULT_TOTAL_MASS, "masses": mjcf.DEFAULT_MASSES,
            "servo": servo, "servo_voltage_range": [actuators.V_MIN, actuators.V_MAX],
            "joint_limits": np.asarray(leg.JOINT_LIMITS, float).tolist(),
            "model": {"foot_radius": mjcf.FOOT_RADIUS, "body_thickness": mjcf.BODY_THICKNESS, "link_radius": mjcf.LINK_RADIUS,
                      "floor_friction": mjcf.FLOOR_FRICTION, "foot_friction": mjcf.FOOT_FRICTION, "gravity": mjcf.GRAVITY,
                      "joint_damping": mjcf.JOINT_DAMPING, "joint_armature": mjcf.JOINT_ARMATURE, "floor_half_size": mjcf.FLOOR_HALF_SIZE,
                      "friction_torsion": mjcf.FRICTION_TORSION, "friction_rolling": mjcf.FRICTION_ROLLING},
        },
    }


class LiveSim:
    """One simulated robot. `set_params` any time; `step(n)` advances n physics steps (2 ms each by default)."""

    def __init__(self, params=None, timestep=0.002):
        self.timestep = timestep
        self.params = {k: (list(v) if isinstance(v, list) else v) for k, v in DEFAULTS.items()}
        if params:
            self.set_params(**params)
        self.reset()

    # ----- parameters -----
    def set_params(self, **changes) -> bool:
        """Apply changes (clamped). Returns True if the model must be rebuilt (the mass or a design dimension changed)."""
        rebuild = False
        for key, value in changes.items():
            if key not in self.params:
                continue
            if key == "offsets":
                value = [float(x) % 1.0 for x in value]
                if len(value) != 4:
                    continue
            elif key == "walk":
                value = bool(value)
            else:
                low, high = LIMITS.get(key, (-np.inf, np.inf))
                value = float(np.clip(float(value), low, high))
            if key in REBUILD and value != self.params[key]:
                rebuild = True
            self.params[key] = value
        if hasattr(self, "servo"):
            self.servo.params = dataclasses.replace(self.servo.params, kp=self.params["kp"], kd=self.params["kd"],
                                                   voltage=self.params["voltage"])
        return rebuild

    def gait(self) -> gaits.Gait:
        params = self.params
        return gaits.Gait("live", period=params["period"], duty=params["duty"], offsets=tuple(params["offsets"]),
                          lift=self._smooth["lift"], sway=self._smooth["sway"])

    # ----- state -----
    def reset(self):
        """Rebuild the model with the current parameters and stand still."""
        params = self.params
        self.physics = SimRobot(design_of(params), timestep=self.timestep)
        self.robot = self.physics.kinematics  # a body.Robot with the closed-form IK, for the targets
        self.model, self.data = self.physics.model, self.physics.data
        self.servo = actuators.Servo(actuators.ServoParams(kp=params["kp"], kd=params["kd"], voltage=params["voltage"]))
        self.home = self.robot.stance(params["height"])
        self.time = 0.0
        self.cycles = 0.0
        self.blend_progress = 0.0
        self._smooth = {k: float(self.params[k]) for k in SMOOTHED}
        self._previous_targets = self.home.copy()
        self.loads = np.zeros(4)
        self.commanded_contact = np.ones(4, bool)
        self.torque_fraction = 0.0
        self.saturated = False
        self._history = deque()  # (time, origin xy, yaw)

    def step(self, n: int = 1):
        dt = self.timestep
        params = self.params
        for _ in range(n):
            alpha = 1.0 - np.exp(-dt / SMOOTHING_TIME)
            for key in SMOOTHED:
                self._smooth[key] += alpha * (params[key] - self._smooth[key])
            gait = self.gait()
            self.home = self.robot.stance(self._smooth["height"])   # the feet follow the eased stance height
            self.blend_progress = float(np.clip(self.blend_progress + (dt if params["walk"] else -dt) / BLEND_TIME, 0.0, 1.0))
            velocity = (self._smooth["vx"], self._smooth["vy"])
            gait_time = self.cycles * gait.period
            feet, contact, _ = gaits.foot_targets(gait, gait_time, self.home, velocity, self._smooth["yaw_rate"])
            self.commanded_contact = contact[0] if self.blend_progress > 0 else np.ones(4, bool)
            blend = float(stand.smoothstep(self.blend_progress))
            # feet in the unposed body frame, then the body is moved over them: shifted by the sway and rotated by the posture
            # angles (with zero posture this is home + blend (walking - home), what the replays do)
            if blend > 0.0:
                sway = blend * gaits.body_sway(gait, gait_time, self.home)[0]
                unposed = self.home + blend * (feet[0] - self.home)
            else:
                sway, unposed = np.zeros(3), self.home
            targets = body.apply_posture(unposed, offset=sway + np.array([0.0, 0.0, self._smooth["body_height"]]), roll=self._smooth["body_roll"], pitch=self._smooth["body_pitch"],
                                         yaw=self._smooth["body_yaw"])
            limit = MAX_TARGET_SPEED * dt
            targets = self._previous_targets + np.clip(targets - self._previous_targets, -limit, limit)
            self._previous_targets = targets
            if params["walk"]:
                self.cycles += dt / gait.period
            joint_targets = self.robot.solve(targets).ravel()
            torque = self.servo.step(joint_targets, self.physics.joint_angles(), self.physics.joint_velocities(), dt)
            self.physics.step(torque)
            self.time += dt
            self.torque_fraction = float(np.abs(torque).max() / self.servo.params.stall_torque)
            self.saturated = bool(self.servo.last_saturated.any())
            self._record()

    def _record(self):
        self.loads = self.physics.foot_loads()
        self._history.append((self.time, self.physics.origin()[:2], self.physics.yaw()))
        while self._history and self._history[0][0] < self.time - HISTORY:
            self._history.popleft()

    def measured_velocity(self):
        """Body velocity (forward, left) in m/s over the last HISTORY s, in the body frame at the start of the window."""
        if len(self._history) < 2:
            return [0.0, 0.0]
        (start_time, start_xy, start_yaw), (end_time, end_xy, _) = self._history[0], self._history[-1]
        if end_time - start_time < 0.2:
            return [0.0, 0.0]
        # the world displacement in the body frame at the start: forward = cos(yaw) dx + sin(yaw) dy, left = -sin(yaw) dx + cos(yaw) dy
        displacement = end_xy - start_xy
        cos_yaw, sin_yaw = np.cos(start_yaw), np.sin(start_yaw)
        return [float((cos_yaw * displacement[0] + sin_yaw * displacement[1]) / (end_time - start_time)),
                float((-sin_yaw * displacement[0] + cos_yaw * displacement[1]) / (end_time - start_time))]

    def snapshot(self) -> dict:
        """JSON-ready state for drawing and readouts."""
        origin, rotation = self.physics.origin(), self.physics.body_rotation()
        tilt = self.physics.tilt_deg()
        legs = self.physics.legs_world()
        roll, pitch, yaw = (float(np.degrees(angle)) for angle in roll_pitch_yaw(rotation))  # R = Rz(yaw) Ry(pitch) Rx(roll), body to world
        fell = bool(tilt > 60.0 or origin[2] < 0.5 * self._smooth["height"])
        return {
            "time": self.time, "legs": legs.round(5).tolist(), "origin": origin.round(5).tolist(),
            "rotation": rotation.round(5).tolist(), "loads": self.loads.round(3).tolist(),
            "down": (self.loads > CONTACT_FORCE).tolist(), "commanded_down": self.commanded_contact.tolist(),
            "torque_fraction": round(self.torque_fraction, 3), "saturated": self.saturated,
            "tilt_deg": round(tilt, 1), "yaw_deg": round(yaw, 1), "pitch_deg": round(pitch, 1), "roll_deg": round(roll, 1), "measured_velocity": [round(v, 3) for v in self.measured_velocity()],
            "feet_down": int((self.loads > CONTACT_FORCE).sum()), "fell": fell, "total_mass_kg": round(float(self.model.body_subtreemass[1]), 3),
            "blend": round(float(stand.smoothstep(self.blend_progress)), 2), "cycle_phase": round(self.cycles % 1.0, 3),
            "body_size": [self.params["body_length"], self.params["body_width"]],
            "stall_torque_nm": round(self.servo.params.stall_torque, 3),
            "stance_reachable": bool(self.robot.reachable(self.home).all()),
            "params": self.params,
        }
