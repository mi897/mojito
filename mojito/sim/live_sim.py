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

from ..robot import body, gaits
from . import actuators, mjcf, stand

SMOOTHING_TIME = 0.3       # s, low-pass on velocity, yaw rate, lift and sway
MAX_TARGET_SPEED = 1.5     # m/s, limit on how fast a commanded foot position may move
BLEND_TIME = 0.5           # s, to blend between standing and walking
HISTORY = 2.0              # s, window for the measured velocity
CONTACT_FORCE = 0.5        # N, a foot carrying more is "down"

DEFAULTS = dict(
    period=1.7, duty=0.5, offsets=[0.0, 0.5, 0.5, 0.0], lift=0.022, sway=0.0,
    vx=0.15, vy=0.0, yaw_rate=0.0,
    kp=50.0, kd=0.9, mass=1.72, speed=1.0, walk=False,
    # robot design: these rebuild the model (like mass). Metres.
    la=0.05, lu=0.15, ll=0.15, body_length=0.30, body_width=0.15, height=0.16,
    voltage=5.0,
)
# The trot uses the tuned settings of gait_replay (kp 50, kd 0.9, 0.16 m stance, 22 mm swing); the crawl and the pace keep the original
# servo (kp 10, kd 0.2), 0.2125 m stance and 30 mm swing, which is what their results in docs/STAGE2.md were measured with.
PRESETS = {
    "trot": dict(period=1.7, duty=0.5, offsets=[0.0, 0.5, 0.5, 0.0], lift=0.022, sway=0.0, vx=0.15, vy=0.0, yaw_rate=0.0,
                 kp=50.0, kd=0.9, height=0.16),
    "crawl": dict(period=2.0, duty=0.85, offsets=[0.75, 0.25, 0.5, 0.0], lift=0.03, sway=0.025, vx=0.04, vy=0.0, yaw_rate=0.0,
                  kp=10.0, kd=0.2, height=0.2125),
    "pace": dict(period=1.7, duty=0.5, offsets=[0.0, 0.5, 0.0, 0.5], lift=0.03, sway=0.0, vx=0.1, vy=0.0, yaw_rate=0.0,
                 kp=10.0, kd=0.2, height=0.2125),
}
# (low, high) for each numeric parameter; anything outside is clamped
LIMITS = dict(period=(0.2, 4.0), duty=(0.1, 0.95), lift=(0.0, 0.1), sway=(0.0, 0.06), vx=(-0.5, 0.5), vy=(-0.3, 0.3),
              yaw_rate=(-3.0, 3.0), kp=(1.0, 300.0), kd=(0.0, 5.0), mass=(0.5, 8.0), speed=(0.1, 2.0),
              la=(0.02, 0.09), lu=(0.08, 0.24), ll=(0.08, 0.24), body_length=(0.15, 0.5), body_width=(0.08, 0.3),
              height=(0.1, 0.32), voltage=(actuators.V_MIN, actuators.V_MAX))
REBUILD = ("mass", "la", "lu", "ll", "body_length", "body_width", "height")  # changing any of these needs a new model
SMOOTHED = ("vx", "vy", "yaw_rate", "lift", "sway")


DEFAULT_TOTAL_MASS = mjcf.DEFAULT_MASSES["body"] + 4 * sum(v for k, v in mjcf.DEFAULT_MASSES.items() if k != "body")  # 1.72 kg


def scaled_masses(total):
    """The placeholder masses of mjcf.DEFAULT_MASSES, all multiplied by total / 1.72 kg."""
    return {k: v * total / DEFAULT_TOTAL_MASS for k, v in mjcf.DEFAULT_MASSES.items()}


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
        """Apply changes (clamped). Returns True if the model must be rebuilt (the mass changed)."""
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
        p = self.params
        return gaits.Gait("live", period=p["period"], duty=p["duty"], offsets=tuple(p["offsets"]),
                          lift=self._smooth["lift"], sway=self._smooth["sway"])

    # ----- state -----
    def reset(self):
        """Rebuild the model with the current parameters and stand still."""
        import mujoco

        self._mujoco = mujoco
        p = self.params
        masses = scaled_masses(p["mass"])
        self.robot, self.model, self.data, self.qadr, self.dadr, self.aid, self.pad, self.start_z = stand._open(
            np.array([p["la"], p["lu"], p["ll"]]), p["height"], p["body_length"], p["body_width"], masses, self.timestep)
        self.servo = actuators.Servo(actuators.ServoParams(kp=p["kp"], kd=p["kd"], voltage=p["voltage"]))
        self.servo.reset()
        self.home = self.robot.stance(p["height"])
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
        mujoco = self._mujoco
        dt = self.timestep
        p = self.params
        for _ in range(n):
            alpha = 1.0 - np.exp(-dt / SMOOTHING_TIME)
            for key in SMOOTHED:
                self._smooth[key] += alpha * (p[key] - self._smooth[key])
            gait = self.gait()
            self.blend_progress = float(np.clip(self.blend_progress + (dt if p["walk"] else -dt) / BLEND_TIME, 0.0, 1.0))
            velocity = (self._smooth["vx"], self._smooth["vy"])
            gait_time = self.cycles * gait.period
            feet, contact, _ = gaits.foot_targets(gait, gait_time, self.home, velocity, self._smooth["yaw_rate"])
            self.commanded_contact = contact[0] if self.blend_progress > 0 else np.ones(4, bool)
            blend = float(stand.smoothstep(self.blend_progress))
            if blend > 0.0:
                sway = gaits.body_sway(gait, gait_time, self.home)
                walking = body.apply_posture(feet, offset=sway)[0]
                targets = self.home + blend * (walking - self.home)
            else:
                targets = self.home
            limit = MAX_TARGET_SPEED * dt
            targets = self._previous_targets + np.clip(targets - self._previous_targets, -limit, limit)
            self._previous_targets = targets
            if p["walk"]:
                self.cycles += dt / gait.period
            joint_targets = self.robot.solve(targets).ravel()
            torque = self.servo.step(joint_targets, self.data.qpos[self.qadr], self.data.qvel[self.dadr], dt)
            self.data.ctrl[self.aid] = torque
            mujoco.mj_step(self.model, self.data)
            self.time += dt
            self.torque_fraction = float(np.abs(torque).max() / self.servo.params.stall_torque)
            self.saturated = bool(self.servo.last_saturated.any())
            self._record()

    def _foot_loads(self):
        mujoco = self._mujoco
        loads = np.zeros(4)
        wrench = np.zeros(6)
        for c in range(self.data.ncon):
            contact = self.data.contact[c]
            for geom in (contact.geom1, contact.geom2):
                if geom in self.pad:
                    mujoco.mj_contactForce(self.model, self.data, c, wrench)
                    loads[self.pad[geom]] += wrench[0]
        return loads

    def _record(self):
        self.loads = self._foot_loads()
        origin = self.data.site("body_origin")
        rotation = origin.xmat.reshape(3, 3)
        self._history.append((self.time, origin.xpos[:2].copy(), float(np.arctan2(rotation[1, 0], rotation[0, 0]))))
        while self._history and self._history[0][0] < self.time - HISTORY:
            self._history.popleft()

    def measured_velocity(self):
        """Body velocity (forward, left) in m/s over the last HISTORY s, in the body frame at the start of the window."""
        if len(self._history) < 2:
            return [0.0, 0.0]
        (t0, xy0, yaw0), (t1, xy1, _) = self._history[0], self._history[-1]
        if t1 - t0 < 0.2:
            return [0.0, 0.0]
        d = xy1 - xy0
        c, s = np.cos(yaw0), np.sin(yaw0)
        return [float((c * d[0] + s * d[1]) / (t1 - t0)), float((-s * d[0] + c * d[1]) / (t1 - t0))]

    def snapshot(self) -> dict:
        """JSON-ready state for drawing and readouts."""
        d = self.data
        origin = d.site("body_origin")
        rotation = origin.xmat.reshape(3, 3)
        tilt = float(np.degrees(np.arccos(np.clip(rotation[2, 2], -1.0, 1.0))))
        legs = np.array([[d.body(f"{n}_hip").xpos, d.body(f"{n}_shank").xpos, d.site(f"{n}_foot").xpos]
                         for n in body.LEG_NAMES])
        yaw = float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))
        fell = bool(tilt > 60.0 or origin.xpos[2] < 0.5 * self.params["height"])
        return {
            "time": self.time, "legs": legs.round(5).tolist(), "origin": origin.xpos.round(5).tolist(),
            "rotation": rotation.round(5).tolist(), "loads": self.loads.round(3).tolist(),
            "down": (self.loads > CONTACT_FORCE).tolist(), "commanded_down": self.commanded_contact.tolist(),
            "torque_fraction": round(self.torque_fraction, 3), "saturated": self.saturated,
            "tilt_deg": round(tilt, 1), "yaw_deg": round(yaw, 1), "measured_velocity": [round(v, 3) for v in self.measured_velocity()],
            "feet_down": int((self.loads > CONTACT_FORCE).sum()), "fell": fell, "total_mass_kg": round(float(self.model.body_subtreemass[1]), 3),
            "blend": round(float(stand.smoothstep(self.blend_progress)), 2), "cycle_phase": round(self.cycles % 1.0, 3),
            "body_size": [self.params["body_length"], self.params["body_width"]],
            "stall_torque_nm": round(self.servo.params.stall_torque, 3),
            "stance_reachable": bool(self.robot.reachable(self.home).all()),
            "params": self.params,
        }
