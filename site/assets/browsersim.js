/* The live simulation of mojito/sim/live_sim.py, running in the browser on MuJoCo's WebAssembly build (@mujoco/mujoco).
 *
 * A line-by-line port of live_sim.LiveSim, actuators.Servo and mjcf.mjcf_from_urdf for the quadruped; the gait, posture and inverse
 * kinematics are mojito.js (tested against the Python). Every number that is a constant in Python comes from `live`, which is
 * site/data/livemeta.js written by scripts/website/build_site.py from live_sim.browser_constants(): nothing is copied by hand.
 * tests/website/test_browsersim.py runs it against the Python on the same model.
 *
 * Placeholder dimensions, masses, servo figures and friction, like the Python version (docs/STAGE2.md). */
(function (root) {
  'use strict';
  const Mojito = (typeof require !== 'undefined' && typeof module !== 'undefined') ? require('./mojito.js') : root.Mojito;
  const LEGS = ['FL', 'FR', 'RL', 'RR'], JOINTS = ['abduction', 'hip_pitch', 'knee'];
  const clamp = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const lerp = (x, x0, y0, x1, y1) => y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  const rint = x => { const r = Math.round(x); return Math.abs(x % 1) === 0.5 && r % 2 !== 0 ? r - 1 : r; };   // round half to even, like numpy
  const fmt = a => a.map(v => String(+Number(v).toPrecision(9))).join(' ');
  const smoothstep = x => { x = clamp(x, 0, 1); return x ** 3 * (10 - 15 * x + 6 * x * x); };

  /* ---------- the MuJoCo model (mjcf.mjcf_from_urdf of robots.quadruped_urdf, actuated) ---------- */
  // design: {la, lu, ll, body_length, body_width, start_z, masses, timestep}; c = live.constants
  function mjcfQuadruped(c, design) {
    const m = c.model, M = design.masses, lim = c.joint_limits;
    const x = design.body_length / 2, y = design.body_width / 2, la = design.la, lu = design.lu, ll = design.ll;
    const body = [design.body_length, design.body_width, m.body_thickness].map(v => v / 2);
    const friction = `${m.foot_friction} ${m.friction_torsion} ${m.friction_rolling}`;
    const capsule = (end, mass) => `<geom type="capsule" fromto="${fmt([0, 0, 0, ...end])}" size="${m.link_radius}" mass="${mass}" rgba="0.8 0.8 0.85 1" />`;
    let legs = '', motors = '';
    const mounts = { FL: [x, y], FR: [x, -y], RL: [-x, y], RR: [-x, -y] };
    for (const name of LEGS) {
      const [mx, my] = mounts[name], right = my < 0, side = right ? -1 : 1;
      const [a, h, k] = lim, abd = right ? [-a[1], -a[0]] : a;   // only the abduction range is mirrored
      legs += `<body name="${name}_hip" pos="${fmt([mx, my, 0])}"><joint name="${name}_abduction" type="hinge" axis="1 0 0" range="${fmt(abd)}" />`
        + capsule([0, side * la, 0], M.hip)
        + `<body name="${name}_thigh" pos="${fmt([0, side * la, 0])}"><joint name="${name}_hip_pitch" type="hinge" axis="0 1 0" range="${fmt(h)}" />`
        + capsule([0, 0, -lu], M.thigh)
        + `<body name="${name}_shank" pos="${fmt([0, 0, -lu])}"><joint name="${name}_knee" type="hinge" axis="0 1 0" range="${fmt(k)}" />`
        + capsule([0, 0, -ll], M.shank)
        + `<body name="${name}_foot" pos="${fmt([0, 0, -ll])}"><site name="${name}_foot" size="0.004" rgba="0 0.6 1 1" />`
        + `<geom name="${name}_foot_pad" type="sphere" size="${m.foot_radius}" mass="${M.foot}" contype="1" conaffinity="1" friction="${friction}" rgba="0.1 0.1 0.1 1" />`
        + '</body></body></body></body>';
      for (const j of JOINTS) motors += `<motor name="${name}_${j}" joint="${name}_${j}" gear="1" />`;
    }
    const half = m.floor_half_size;
    return `<mujoco model="mojito_quadruped"><compiler angle="radian" autolimits="true" /><option timestep="${design.timestep}" gravity="0 0 ${m.gravity}" />`
      + `<default><joint damping="${m.joint_damping}" armature="${m.joint_armature}" /><geom contype="0" conaffinity="0" /></default>`
      + `<worldbody><geom name="floor" type="plane" size="${half} ${half} 0.1" contype="1" conaffinity="1" friction="${m.floor_friction} ${m.friction_torsion} ${m.friction_rolling}" />`
      + `<body name="body" pos="0 0 ${design.start_z}"><freejoint name="base" /><site name="body_origin" size="0.004" rgba="1 0 0 1" />`
      + `<geom type="box" size="${fmt(body)}" mass="${M.body}" rgba="0.5 0.5 0.55 1" />${legs}</body></worldbody>`
      + `<actuator>${motors}</actuator><sensor><gyro name="gyro" site="body_origin" /><accelerometer name="accel" site="body_origin" /></sensor></mujoco>`;
  }

  /* ---------- actuators.Servo ---------- */
  class Servo {
    constructor(params) { this.params = Object.assign({}, params); this.reset(); }
    reset() { this.queue = []; this.time = 0; this.held = null; this.lastSaturated = false; }
    get stallTorque() { const p = this.params; return p.torque_scale * lerp(p.voltage, 5, p.stall_5v, 6.8, p.stall_68v); }
    get noloadSpeed() { const p = this.params; return lerp(p.voltage, 5, p.noload_5v, 6.8, p.noload_68v); }
    quantise(targets) { const p = this.params; return targets.map(t => p.centre + rint((t - p.centre) / p.resolution) * p.resolution); }
    // commanded joint targets and true state -> torques; `lastSaturated` says whether any joint was at its limit
    step(targets, angles, velocities, dt) {
      const p = this.params;
      this.queue.push([this.time + p.latency, this.quantise(targets)]);
      let seen = null;
      while (this.queue.length && this.queue[0][0] <= this.time + 1e-12) seen = this.queue.shift()[1];
      if (seen !== null) this.held = seen; else if (this.held === null) this.held = this.quantise(angles);
      this.time += dt;
      const stall = this.stallTorque, noload = this.noloadSpeed;
      let saturated = false;
      const out = angles.map((angle, i) => {
        // torque = kp (target - angle) - kd velocity, limited by the torque-speed line along the motion (braking is not reduced)
        const velocity = velocities[i], torque = p.kp * (this.held[i] - angle) - p.kd * velocity;
        const alongMotion = Math.sign(torque) === Math.sign(velocity);
        const limit = alongMotion ? stall * clamp(1 - Math.abs(velocity) / noload, 0, 1) : stall;
        if (Math.abs(torque) >= limit) saturated = true;
        return clamp(torque, -limit, limit);
      });
      this.lastSaturated = saturated;
      return out;
    }
  }

  /* ---------- live_sim.LiveSim ---------- */
  class BrowserSim {
    // mujoco: the loaded @mujoco/mujoco module; live: MOJITO_LIVE; params: initial overrides (a preset, say)
    constructor(mujoco, live, params) {
      this.mj = mujoco; this.meta = live.meta; this.c = live.constants; this.timestep = this.c.timestep;
      this.params = JSON.parse(JSON.stringify(this.meta.defaults));
      this.model = this.data = null;
      if (params) this.setParams(params);
      this.reset();
    }

    // Apply changes (clamped). Returns true if the model must be rebuilt (a design value or the mass changed).
    setParams(changes) {
      let rebuild = false;
      for (const [key, raw] of Object.entries(changes)) {
        if (!(key in this.params)) continue;
        let value = raw;
        if (key === 'offsets') {
          if (!Array.isArray(raw) || raw.length !== 4) continue;
          value = raw.map(v => ((Number(v) % 1) + 1) % 1);
        } else if (key === 'walk') value = !!raw;
        else { const [lo, hi] = this.meta.limits[key] || [-Infinity, Infinity]; value = clamp(Number(raw), lo, hi); }
        if (this.c.rebuild.includes(key) && value !== this.params[key]) rebuild = true;
        this.params[key] = value;
      }
      if (this.servo) Object.assign(this.servo.params, { kp: this.params.kp, kd: this.params.kd, voltage: this.params.voltage });
      return rebuild;
    }

    gait() {
      const p = this.params;
      return { period: p.period, duty: p.duty, offsets: p.offsets, lift: this.smooth.lift, sway: this.smooth.sway };
    }

    // Rebuild the model with the current parameters and stand still.
    reset() {
      const mj = this.mj, p = this.params, c = this.c, scale = p.mass / c.default_total_mass;
      if (this.data) this.data.delete();
      if (this.model) this.model.delete();
      const masses = Object.fromEntries(Object.entries(c.masses).map(([k, v]) => [k, v * scale]));
      this.robot = new Mojito.Robot({ lengths: [p.la, p.lu, p.ll], bodyLength: p.body_length, bodyWidth: p.body_width, limits: c.joint_limits });
      const initial = this.robot.solve(this.robot.stance(p.height));
      this.startZ = p.height + c.model.foot_radius;   // the pad centre is the foot tip; its lowest point is one radius lower
      const xml = mjcfQuadruped(c, { la: p.la, lu: p.lu, ll: p.ll, body_length: p.body_length, body_width: p.body_width,
        start_z: this.startZ, masses, timestep: this.timestep });
      this.model = mj.MjModel.from_xml_string(xml);
      this.data = new mj.MjData(this.model);
      const id = (type, name) => mj.mj_name2id(this.model, mj.mjtObj[type].value, name);
      const names = LEGS.flatMap(l => JOINTS.map(j => `${l}_${j}`));
      this.qadr = names.map(n => this.model.jnt_qposadr[id('mjOBJ_JOINT', n)]);
      this.dadr = names.map(n => this.model.jnt_dofadr[id('mjOBJ_JOINT', n)]);
      this.aid = names.map(n => id('mjOBJ_ACTUATOR', n));
      this.bodyId = LEGS.map(l => [id('mjOBJ_BODY', `${l}_hip`), id('mjOBJ_BODY', `${l}_shank`)]);
      this.footSite = LEGS.map(l => id('mjOBJ_SITE', `${l}_foot`));
      this.originSite = id('mjOBJ_SITE', 'body_origin');
      this.padLeg = new Map(LEGS.map((l, i) => [id('mjOBJ_GEOM', `${l}_foot_pad`), i]));
      const q = this.data.qpos;   // set_pose: base at (0, 0, start_z), identity attitude, then the standing joint angles
      q[0] = 0; q[1] = 0; q[2] = this.startZ; q[3] = 1; q[4] = q[5] = q[6] = 0;
      initial.flat().forEach((v, i) => { q[this.qadr[i]] = v; });
      mj.mj_forward(this.model, this.data);
      this.servo = new Servo(Object.assign({}, c.servo, { kp: p.kp, kd: p.kd, voltage: p.voltage }));
      this.home = this.robot.stance(p.height);
      this.time = 0; this.cycles = 0; this.blendProgress = 0;
      this.smooth = Object.fromEntries(c.smoothed.map(k => [k, Number(p[k])]));
      this.previousTargets = this.home.map(f => f.slice());
      this.commandedContact = [true, true, true, true];
      this.torqueFraction = 0; this.saturated = false;
      this.history = [];
    }

    step(n = 1) {
      const dt = this.timestep, p = this.params, c = this.c, mj = this.mj, d = this.data;
      for (let s = 0; s < n; s++) {
        const alpha = 1 - Math.exp(-dt / c.smoothing_time);
        for (const key of c.smoothed) this.smooth[key] += alpha * (p[key] - this.smooth[key]);
        const gait = this.gait();
        this.blendProgress = clamp(this.blendProgress + (p.walk ? dt : -dt) / c.blend_time, 0, 1);
        const gaitTime = this.cycles * gait.period;
        const t = Mojito.footTargets(gait, gaitTime, this.home, [this.smooth.vx, this.smooth.vy], this.smooth.yaw_rate);
        this.commandedContact = this.blendProgress > 0 ? t.contact : [true, true, true, true];
        const blend = smoothstep(this.blendProgress);
        // feet in the unposed body frame, then the body is moved over them: shifted by the sway and the height change, rotated by the posture angles
        let sway = [0, 0, 0], unposed = this.home;
        if (blend > 0) {
          sway = Mojito.bodySway(gait, gaitTime, this.home).map(v => blend * v);
          unposed = this.home.map((h, i) => h.map((v, k) => v + blend * (t.feet[i][k] - v)));
        }
        const offset = [sway[0], sway[1], sway[2] + this.smooth.body_height];
        let targets = Mojito.applyPosture(unposed, offset, this.smooth.body_roll, this.smooth.body_pitch, this.smooth.body_yaw);
        const limit = c.max_target_speed * dt;
        targets = targets.map((f, i) => f.map((v, k) => this.previousTargets[i][k] + clamp(v - this.previousTargets[i][k], -limit, limit)));
        this.previousTargets = targets;
        if (p.walk) this.cycles += dt / gait.period;
        const jointTargets = this.robot.solve(targets).flat();
        const qpos = d.qpos, qvel = d.qvel;
        const torque = this.servo.step(jointTargets, this.qadr.map(i => qpos[i]), this.dadr.map(i => qvel[i]), dt);
        const ctrl = d.ctrl; this.aid.forEach((a, i) => { ctrl[a] = torque[i]; });
        mj.mj_step(this.model, d);
        this.time += dt;
        this.torqueFraction = Math.max(...torque.map(Math.abs)) / this.servo.stallTorque;
        this.saturated = this.servo.lastSaturated;
        this.record();
      }
    }

    origin() { const x = this.data.site_xpos, i = this.originSite * 3; return [x[i], x[i + 1], x[i + 2]]; }
    rotation() { const m = this.data.site_xmat, i = this.originSite * 9; return [[m[i], m[i + 1], m[i + 2]], [m[i + 3], m[i + 4], m[i + 5]], [m[i + 6], m[i + 7], m[i + 8]]]; }

    record() {
      const o = this.origin(), R = this.data.site_xmat, i = this.originSite * 9;
      this.history.push([this.time, o[0], o[1], Math.atan2(R[i + 3], R[i])]);
      while (this.history.length && this.history[0][0] < this.time - this.c.history) this.history.shift();
    }

    // Body velocity (forward, left) in m/s over the last `history` s, in the body frame at the start of the window.
    measuredVelocity() {
      if (this.history.length < 2) return [0, 0];
      const [t0, x0, y0, yaw0] = this.history[0], [t1, x1, y1] = this.history[this.history.length - 1];
      if (t1 - t0 < 0.2) return [0, 0];
      const dx = x1 - x0, dy = y1 - y0, cs = Math.cos(yaw0), sn = Math.sin(yaw0);
      return [(cs * dx + sn * dy) / (t1 - t0), (-sn * dx + cs * dy) / (t1 - t0)];
    }

    // Normal force on each foot pad (N), summed over its contacts.
    footLoads() {
      const mj = this.mj, loads = [0, 0, 0, 0], contacts = this.data.contact, buf = new mj.DoubleBuffer(6);
      try {
        for (let i = 0; i < contacts.size(); i++) {
          const k = contacts.get(i), geoms = [k.geom1, k.geom2]; k.delete();
          const leg = geoms.map(g => this.padLeg.get(g)).filter(v => v !== undefined);
          if (!leg.length) continue;
          mj.mj_contactForce(this.model, this.data, i, buf);
          const view = buf.GetView ? buf.GetView() : buf.getView();   // the method is spelled GetView in 3.16.0
          leg.forEach(l => { loads[l] += view[0]; });
        }
      } finally { buf.delete(); contacts.delete(); }
      return loads;
    }

    // JSON-ready state for drawing and readouts (the same keys as live_sim.LiveSim.snapshot).
    snapshot() {
      const d = this.data, p = this.params, o = this.origin(), R = this.rotation(), deg = r => r * 180 / Math.PI;
      const tilt = deg(Math.acos(clamp(R[2][2], -1, 1))), loads = this.footLoads(), down = loads.map(v => v > this.c.contact_force);
      const at = (arr, i) => [arr[i * 3], arr[i * 3 + 1], arr[i * 3 + 2]];
      const legs = LEGS.map((_, i) => [at(d.xpos, this.bodyId[i][0]), at(d.xpos, this.bodyId[i][1]), at(d.site_xpos, this.footSite[i])]);
      const rounded = v => Math.round(v * 1000) / 1000;
      return {
        time: this.time, legs, origin: o, rotation: R, loads, down, commanded_down: this.commandedContact,
        torque_fraction: rounded(this.torqueFraction), saturated: this.saturated, tilt_deg: rounded(tilt),
        yaw_deg: deg(Math.atan2(R[1][0], R[0][0])), pitch_deg: deg(Math.asin(clamp(-R[2][0], -1, 1))), roll_deg: deg(Math.atan2(R[2][1], R[2][2])),
        measured_velocity: this.measuredVelocity(), feet_down: down.filter(Boolean).length,
        fell: tilt > 60 || o[2] < 0.5 * p.height, total_mass_kg: this.model.body_subtreemass[1],
        blend: smoothstep(this.blendProgress), cycle_phase: (this.cycles % 1 + 1) % 1, body_size: [p.body_length, p.body_width],
        stall_torque_nm: this.servo.stallTorque, stance_reachable: this.robot.reachable(this.home).every(Boolean),
        params: JSON.parse(JSON.stringify(p)),
      };
    }
  }

  /* ---------- scripts/gait/live_gait.Runner and its HTTP API, in process ---------- */
  // Advances the simulation at `speed` times real time and answers the same requests as the local server (meta, state, params, preset, reset).
  class BrowserRunner {
    constructor(sim, live, clock = { now: () => performance.now() / 1000 }) {
      this.sim = sim; this.live = live; this.clock = clock; this.paused = false; this.owed = 0; this.last = clock.now();
    }
    start() { if (!this.timer) { this.last = this.clock.now(); this.timer = setInterval(() => this.tick(), 16); } }
    stop() { clearInterval(this.timer); this.timer = null; }
    tick() {
      const now = this.clock.now(), elapsed = Math.min(now - this.last, 0.1); this.last = now;   // never catch up more than 0.1 s
      if (this.paused) return;
      this.owed += elapsed * this.sim.params.speed;
      const steps = Math.floor(this.owed / this.sim.timestep);
      if (steps) { this.sim.step(steps); this.owed -= steps * this.sim.timestep; }
    }
    get(path) {
      if (path === 'meta') return this.live.meta;
      if (path === 'state') { const s = this.sim.snapshot(); s.paused = this.paused; return s; }
      throw new Error('no such path ' + path);
    }
    post(path, body = {}) {
      if (path === 'params') {
        const { paused, ...changes } = body;
        if (this.sim.setParams(changes)) this.sim.reset();   // a new mass or size needs a new model
        if (paused !== undefined) this.paused = !!paused;
      } else if (path === 'preset') {
        const preset = this.live.meta.presets[body.name];
        if (preset && this.sim.setParams(preset)) this.sim.reset();
      } else if (path === 'reset') this.sim.reset();
      else throw new Error('no such path ' + path);
      return { ok: true };
    }
  }

  // Version of the WebAssembly build, and where it is fetched from (11 MB, cached by the browser). Override the base for a copy served locally.
  const MUJOCO_VERSION = '3.16.0';
  const mujocoBase = () => (root.MOJITO_MUJOCO_BASE || `https://cdn.jsdelivr.net/npm/@mujoco/mujoco@${MUJOCO_VERSION}/`);
  async function loadMujoco() { const mod = await import(mujocoBase() + 'mujoco.js'); return mod.default(); }
  async function startBrowserRunner(live) {
    const sim = new BrowserSim(await loadMujoco(), live, live.meta.presets.trot);
    const runner = new BrowserRunner(sim, live); runner.start(); return runner;
  }

  const api = { mjcfQuadruped, Servo, BrowserSim, BrowserRunner, loadMujoco, startBrowserRunner, MUJOCO_VERSION, rint, smoothstep };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.BrowserSim = api;
})(typeof self !== 'undefined' ? self : this);
