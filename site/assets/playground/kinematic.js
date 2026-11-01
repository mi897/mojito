/* Playground, kinematic lab: the project's own maths (assets/mojito.js) live, geometry and statics only. Needs common.js first. */
(function () {
  'use strict';
  if (!window.Mojito) return;   // data/constants.js missing, so mojito.js did not load: common.js says so on the page
  const M = window.Mojito, P = window.Playground, { D2R, $, fmt, slider, heading, buttons, kpi } = P;
  const matMul = (A, B) => A.map(r => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));

  /* =============================================================
   * Kinematic lab
   * ============================================================= */
  // gait sliders of a gait preset (mojito.js, from gaits.py) commanded at `vx` m/s
  const gaitSet = (gait, vx, yaw_rate = 0) => ({ period: gait.period, duty: gait.duty, lift: gait.lift, sway: gait.sway,
    o0: gait.offsets[0], o1: gait.offsets[1], o2: gait.offsets[2], o3: gait.offsets[3], vx, vy: 0, yaw_rate });
  // the Stage 2 design placeholders, trot, servo supply and total mass: all from data/constants.js, through mojito.js
  const DESIGN = M.DESIGN, DEFAULTS = Object.assign({
    abduction_offset: DESIGN.abduction_offset, upper_length: DESIGN.upper_length, lower_length: DESIGN.lower_length, body_length: DESIGN.body_length, body_width: DESIGN.body_width,
    stance_height: DESIGN.stance_height, front_left_scale: 1,
    px: 0, py: 0, pz: 0, roll: 0, pitch: 0, yaw: 0,
  }, gaitSet(M.TROT, M.GAIT_SPEEDS.trot), { mass: DESIGN.total_mass, volts: M.SERVO.voltage });
  const kin = { st: Object.assign({}, DEFAULTS), mode: 'gait', gaitName: 'trot', t: 0, playing: true, speed: 1, ctl: {},
    eff: { vx: DEFAULTS.vx, vy: DEFAULTS.vy, yaw_rate: DEFAULTS.yaw_rate }, pose: { x: 0, y: 0, yaw: 0 }, drive: null };
  // time constant of the easing of the commanded velocity (s): the live simulation's, so a steered robot does not jump (livemeta.js)
  const commandTau = () => (window.MOJITO_LIVE ? window.MOJITO_LIVE.constants.smoothing_time : 0);

  function robotOf(st) {
    const r = new M.Robot({ lengths: [st.abduction_offset, st.upper_length, st.lower_length], bodyLength: st.body_length, bodyWidth: st.body_width });
    r.lengths[0] = r.lengths[0].map((v, i) => (i ? v * st.front_left_scale : v));
    return r;
  }
  function gaitOf(st) {
    return { name: kin.gaitName, period: st.period, duty: st.duty, lift: st.lift, sway: st.sway, offsets: [st.o0, st.o1, st.o2, st.o3] };
  }
  // v: the velocity to evaluate at, default the sliders'; the frame being drawn uses kin.eff, the eased one
  function evalAt(t, v = kin.st) {
    const st = kin.st, robot = robotOf(st), standing = kin.mode === 'stand';
    return M.evaluate({
      robot, gait: standing ? null : gaitOf(st), t, velocity: [v.vx, v.vy], yawRate: v.yaw_rate, height: st.stance_height,
      posture: { x: st.px, y: st.py, z: st.pz, roll: st.roll * D2R, pitch: st.pitch * D2R, yaw: st.yaw * D2R }, mass: st.mass, volts: st.volts,
    });
  }
  function kinScene(res, t) {
    const st = kin.st, p = kin.pose;   // where the body is in the world: the dead-reckoned pose, which a changed command cannot make jump
    const c = Math.cos(p.yaw), s = Math.sin(p.yaw), Rz = [[c, -s, 0], [s, c, 0], [0, 0, 1]];
    const off = res.offset;
    return {
      origin: [p.x + c * off[0] - s * off[1], p.y + s * off[0] + c * off[1], st.stance_height + off[2]],
      R: matMul(Rz, res.R), bodySize: [st.body_length, st.body_width], points: res.points,
      down: res.loadsN.map(l => l > 1e-6), bad: res.reachable.map(r => !r),
    };
  }

  function kinSpecs() {
    return [
      ['Robot design (placeholders)', [['abduction_offset', 'abduction offset', 0.02, 0.09, 0.0025, ' mm', 1000], ['upper_length', 'upper leg', 0.08, 0.24, 0.005, ' mm', 1000],
        ['lower_length', 'lower leg', 0.08, 0.24, 0.005, ' mm', 1000], ['body_length', 'body length', 0.15, 0.5, 0.01, ' mm', 1000], ['body_width', 'body width', 0.08, 0.3, 0.01, ' mm', 1000],
        ['stance_height', 'stand height', 0.1, 0.32, 0.0025, ' mm', 1000], ['front_left_scale', 'FL leg scale', 0.8, 1.25, 0.01, '×', 1]]],
      ['Body posture (feet stay put)', [['px', 'shift forward', -0.06, 0.06, 0.0025, ' mm', 1000], ['py', 'shift left', -0.06, 0.06, 0.0025, ' mm', 1000],
        ['pz', 'raise', -0.06, 0.06, 0.0025, ' mm', 1000], ['roll', 'roll', -30, 30, 0.5, '°', 1], ['pitch', 'pitch', -30, 30, 0.5, '°', 1], ['yaw', 'yaw', -30, 30, 0.5, '°', 1]]],
      ['Gait timing', [['period', 'period', 0.2, 3, 0.05, ' s', 1], ['duty', 'duty (on ground)', 0.2, 0.95, 0.01, '', 1], ['lift', 'swing height', 0, 0.1, 0.005, ' mm', 1000],
        ['sway', 'crawl lean', 0, 0.06, 0.005, ' mm', 1000], ['o0', 'start FL', 0, 1, 0.01, '', 1], ['o1', 'start FR', 0, 1, 0.01, '', 1],
        ['o2', 'start RL', 0, 1, 0.01, '', 1], ['o3', 'start RR', 0, 1, 0.01, '', 1]]],
      ['Command', [['vx', 'forward', -0.4, 0.4, 0.01, ' m/s', 1], ['vy', 'left', -0.3, 0.3, 0.01, ' m/s', 1], ['yaw_rate', 'turn rate', -2.5, 2.5, 0.1, ' rad/s', 1]]],
      ['Load', [['mass', 'total mass', 0.5, 6, 0.01, ' kg', 1], ['volts', 'servo supply', 5, 6, 0.1, ' V', 1]]],
    ];
  }
  const PRESETS = {
    trot: { mode: 'gait', gait: 'trot', set: gaitSet(M.TROT, M.GAIT_SPEEDS.trot) },
    crawl: { mode: 'gait', gait: 'crawl', set: gaitSet(M.CRAWL, M.GAIT_SPEEDS.crawl) },
    pace: { mode: 'gait', gait: 'pace', set: gaitSet(M.PACE, M.GAIT_SPEEDS.pace) },
    spin: { mode: 'gait', gait: 'trot', set: gaitSet(M.TROT, 0, 0.6) },
    stand: { mode: 'stand', gait: 'trot', set: {} },
    twist: { mode: 'stand', gait: 'trot', set: { roll: 12, pitch: -8, yaw: 16, pz: 0.02 } },
  };

  function applyPreset(name) {
    const p = PRESETS[name], base = ['px', 'py', 'pz', 'roll', 'pitch', 'yaw'];
    kin.mode = p.mode; kin.gaitName = p.gait;
    base.forEach(k => kin.ctl[k].set(0));
    Object.entries(p.set).forEach(([k, v]) => kin.ctl[k].set(v));
    kin.engaged = false; kin.t = 0; kin.pose = { x: 0, y: 0, yaw: 0 }; kin.eff = { vx: kin.st.vx, vy: kin.st.vy, yaw_rate: kin.st.yaw_rate }; kinDirty('preset');
    if (kinView) kinView.recentre();
  }

  let kinView, kinPending = false, kinSamples = null;
  function kinDirty(why) { kinSamples = null; if (!kinPending) { kinPending = true; requestAnimationFrame(() => { kinPending = false; kinCharts(); kinFrame(); }); } }

  function sampleCycle() {
    if (kinSamples) return kinSamples;
    const standing = kin.mode === 'stand', n = standing ? 1 : 120, T = kin.st.period, out = [];
    for (let k = 0; k < n; k++) {
      const t = standing ? 0 : (k / n) * T, r = evalAt(t);
      out.push({ t, r, torquePerLeg: r.torque.map(tt => Math.max(...tt) / r.stall) });
    }
    return (kinSamples = out);
  }

  function kinCharts() {
    const st = kin.st, standing = kin.mode === 'stand', S = sampleCycle(), T = standing ? 1 : st.period, deg = 180 / Math.PI;
    const canon = r => (M.IS_RIGHT[0] ? r.q[0] : r.q[0]);   // FL is a left leg, so its angles are already canonical
    const jn = ['abduction', 'hip pitch', 'knee'], col = ['var(--blue)', 'var(--orange)', 'var(--aqua)'];
    const lim = M.LIMITS;
    const jointSeries = jn.map((name, j) => ({ name, color: col[j], points: S.map(s => [s.t, canon(s.r)[j] * deg]).concat(standing ? [[T, canon(S[0].r)[j] * deg]] : [[T, canon(S[0].r)[j] * deg]]), width: 2.2 }));
    const hl = lim.flatMap((l, j) => [{ y: l[0] * deg, color: col[j] }, { y: l[1] * deg, color: col[j] }]);
    $('ch-joints').innerHTML = '';
    const cj = Charts.line($('ch-joints'), { series: jointSeries, x: { label: standing ? '' : 'time in the cycle, s', domain: [0, T], fmt: v => +v.toFixed(2) },
      y: { label: 'degrees', fmt: v => Math.round(v) }, hlines: hl.map(h => Object.assign(h, { label: '' })), aspect: 1.7 });
    kin.cj = cj;
    const foot = S.map(s => [1000 * s.r.targets[0][0], 1000 * s.r.targets[0][2]]);
    $('ch-foot').innerHTML = '';
    kin.cf = Charts.line($('ch-foot'), { series: [{ name: 'foot target', color: 'var(--FL)', points: foot.concat([foot[0]]), width: 2.4 }, { name: 'now', color: 'var(--ink)', points: [foot[0]], markers: true, width: 0, markerSize: 5 }],
      x: { label: 'forward, mm', zero: false }, y: { label: 'height, mm', zero: false }, aspect: 1.7, legend: false });
    const legs = M.LEG_NAMES.map((name, i) => ({ name, color: `var(--${name})`, points: S.map(s => [s.t, s.torquePerLeg[i]]).concat([[T, S[0].torquePerLeg[i]]]), width: 2.2 }));
    $('ch-torque').innerHTML = '';
    kin.ct = Charts.line($('ch-torque'), { series: legs, x: { label: standing ? '' : 'time in the cycle, s', domain: [0, T], fmt: v => +v.toFixed(2) }, y: { label: 'fraction of stall', fmt: v => +v.toFixed(2), domain: [0, Math.max(1.05, ...legs.flatMap(l => l.points.map(p => p[1] * 1.05)))] },
      hlines: [{ y: 1, label: 'stall', color: 'var(--orange)' }, { y: 0.5, label: 'half of stall', color: 'var(--muted)' }], aspect: 1.7 });
    $('ch-timing').innerHTML = timingSvg(S, T);
    kin.tcursor = $('ch-timing').querySelector('.cursor');
    const cycle = kinCycleSummary(S);
    $('kin-cycle').innerHTML = cycle.html; $('kin-msgs').innerHTML = cycle.msgs;
  }

  function timingSvg(S, T) {
    const W = kin.tW = Math.max(300, $('ch-timing').clientWidth || 680), H = 150, l = 44, r = 12, rowH = (H - 28) / 4, pw = W - l - r;
    let g = '';
    M.LEG_NAMES.forEach((name, i) => {
      const y = 6 + i * rowH;
      g += `<text x="${l - 8}" y="${y + rowH * 0.62}" text-anchor="end" style="fill:var(--${name});font-weight:600">${name}</text><line x1="${l}" x2="${l + pw}" y1="${y + rowH * 0.9}" y2="${y + rowH * 0.9}" class="gridline"/>`;
      let start = null;
      S.forEach((s, k) => {
        const on = s.r.contact[i];
        if (on && start === null) start = k;
        if ((!on || k === S.length - 1) && start !== null) {
          const end = on ? S.length : k, x0 = l + pw * start / S.length, x1 = l + pw * end / S.length;
          g += `<rect x="${x0}" y="${y + rowH * 0.18}" width="${Math.max(1, x1 - x0)}" height="${rowH * 0.6}" rx="3" style="fill:var(--${name})"/>`; start = null;
        }
      });
    });
    for (let k = 0; k <= 4; k++) { const x = l + pw * k / 4; g += `<text x="${x}" y="${H - 6}" text-anchor="middle">${(T * k / 4).toFixed(2)} s</text>`; }
    return `<svg viewBox="0 0 ${W} ${H}"><g>${g}</g><line class="cursor" x1="${l}" x2="${l}" y1="2" y2="${H - 22}" style="stroke:var(--ink);stroke-width:1.6"/></svg>`;
  }

  function kinCycleSummary(S) {
    const st = kin.st, standing = kin.mode === 'stand';
    let worst = 0, worstK = 0, minFeet = 4, minMargin = Infinity, minStab = Infinity, bad = 0, track = 0;
    S.forEach((s, k) => {
      const p = s.r.peakFraction; if (p > worst) { worst = p; worstK = k; }
      minFeet = Math.min(minFeet, s.r.contact.filter(Boolean).length);
      minMargin = Math.min(minMargin, s.r.marginDeg);
      minStab = Math.min(minStab, s.r.stability);
      if (s.r.reachable.some(x => !x)) bad++;
      track = Math.max(track, s.r.trackingError);
    });
    const maxMass = worst > 0 ? st.mass * 0.5 / worst : Infinity, total = S.length;
    const msgs = [];
    if (bad) msgs.push(`<div class="w">${bad === total ? 'The' : fmt(100 * bad / total, 0) + '% of the cycle, the'} requested foot positions are outside the leg's reach or joint limits. The leg is drawn at the nearest pose the closed-form solution gives. Shorten the stride, lower the swing, or change the body height or lengths.</div>`);
    if (worst >= 1) msgs.push(`<div class="w">The weight needs more than the stall torque at ${fmt(worstK / total * st.period, 2)} s into the cycle: the servo cannot hold this pose at ${st.mass} kg. Lower the mass, raise the supply voltage, shorten the legs, or change the pose.</div>`);
    else if (worst >= 0.5) msgs.push(`<div class="w">Holding the weight needs ${fmt(worst * 100, 0)}% of the stall torque at the worst moment: more than the half-of-stall level assumed for continuous use.</div>`);
    if (!standing && minFeet <= 2 && !bad) msgs.push('<div class="muted small">Only two feet are down at the thinnest moment. Static torque is shown, but balance on two feet depends on motion that this lab does not simulate; see the Physics lab.</div>');
    if (!msgs.length) msgs.push('<div class="ok">Every foot target is reachable and the static torque stays under half of stall.</div>');
    const html = kpi('worst servo torque / stall', fmt(worst, 2), worst >= 1 ? 'bad' : worst >= 0.5 ? 'bad' : 'good', 'largest |Jᵀ F| over any joint, any moment of the cycle')
      + kpi('heaviest robot at ½ stall', Number.isFinite(maxMass) ? fmt(maxMass, 2) + ' kg' : '–', '', 'current mass scaled until the worst torque is half the stall torque; from weight on the feet alone')
      + kpi('closest joint to its limit', fmt(minMargin, 1) + '°', minMargin < 0.5 ? 'bad' : '')
      + kpi('fewest feet down', String(minFeet), minFeet <= 2 && !standing ? 'bad' : '')
      + kpi('body-centre margin', standing || minFeet >= 3 ? fmt(1000 * minStab, 0) + ' mm' : 'n/a (&lt;3 feet)', minStab < 0 && minFeet >= 3 ? 'bad' : '', 'distance from the body centre to the edge of the support polygon; negative = outside')
      + kpi('foot error from IK', fmt(1000 * track, 2) + ' mm', track > 1e-6 ? 'bad' : '', 'zero unless a target is outside the joint limits');
    return { html, msgs: msgs.join('') };
  }

  function kinFrame() {
    const st = kin.st, standing = kin.mode === 'stand', T = standing ? 1 : st.period;
    const res = evalAt(kin.t, kin.eff);
    kinView.setScene(kinScene(res, kin.t));
    const nDown = res.loadsN.filter(l => l > 1e-6).length;
    $('kin-kpis').innerHTML = kpi('feet carrying weight', nDown + ' of 4', nDown <= 2 ? 'bad' : '')
      + kpi('servo torque now / stall', fmt(res.peakFraction, 2), res.peakFraction >= 1 ? 'bad' : '', 'N·m needed: ' + fmt(res.peakTorque, 2))
      + kpi('stall torque', fmt(res.stall, 2) + ' N·m', '', `listed, unverified: straight line between ${fmt(M.SERVO.stall_5v, 2)} N·m at 5 V and ${fmt(M.SERVO.stall_68v, 2)} N·m at 6.8 V`)
      + kpi('closest joint to limit', fmt(res.marginDeg, 1) + '°', res.marginDeg < 0.5 ? 'bad' : '')
      + kpi('body-centre margin', nDown >= 3 ? fmt(1000 * res.stability, 0) + ' mm' : 'n/a', nDown >= 3 && res.stability < 0 ? 'bad' : '');
    const phase = standing ? 0 : (kin.t % T) / T;
    $('kin-scrub').value = phase;
    if (kin.cj && kin.cj.px) {
      const set = (F, line, x) => { if (!F) return; if (!F._cursor) F._cursor = Charts.svgEl('line', { style: 'stroke:var(--ink);stroke-width:1.4;stroke-dasharray:3 3' }, F.svg); F._cursor.setAttribute('x1', x); F._cursor.setAttribute('x2', x); F._cursor.setAttribute('y1', F.m.t); F._cursor.setAttribute('y2', F.m.t + F.ph); };
      const tc = standing ? 0 : kin.t % T;
      set(kin.cj, null, kin.cj.px(tc)); set(kin.ct, null, kin.ct.px(tc));
      if (kin.cf && kin.cf.svg) {
        const dot = kin.cf.svg.querySelectorAll('circle'); const last = dot[dot.length - 1];
        if (last && kin.cf.px) { last.setAttribute('cx', kin.cf.px(1000 * res.targets[0][0])); last.setAttribute('cy', kin.cf.py(1000 * res.targets[0][2])); }
      }
      if (kin.tcursor) { const x = 44 + ((kin.tW || 680) - 56) * phase; kin.tcursor.setAttribute('x1', x); kin.tcursor.setAttribute('x2', x); }
    }
  }

  /* Driving the lab: keyboard, gamepad and on-screen sticks (drive.js, touchpad.js), as on the live tab. They set this lab's own sliders, so the
   * sliders always show the command; the full-stick values are the live simulation's (livemeta.js). Walking is the gait animation, standing
   * (what pose mode uses) is the standing pose with the sliders' posture. Nothing is applied until there is an input, so the lab's own
   * presets and autoplay are not interrupted by an idle controller. */
  const KEYS_WALK = 'Drive: W S forward/back · A D sidestep · Q E turn · X Z stand height · Space keep walking · 1 2 trot/crawl · R reset · M pose mode. Gamepad: sticks, A, B, Y (mode), LB, RB.';
  const KEYS_POSE = 'Pose mode (feet stay put, body tilts): W S pitch nose down/up · A D roll left/right · Q E yaw left/right · M walk mode. Gamepad: left stick pitch and roll, right stick yaw, Y walk mode.';
  function initKinDrive() {
    const live = window.MOJITO_LIVE;
    if (!live || !window.MojitoDrive || !window.MojitoTouchpad) { $('kin-touch').hidden = true; $('kin-mode').hidden = true; return; }
    function kinPreset(name) {   // a preset button of the on-screen pad, or the 1, 2 and R keys
      applyPreset(name);
      if (live.meta.drive[name]) kin.drive.preset = name;   // trot and crawl have full-stick values; stand does not
      kin.drive.forget();
    }
    const toggleMode = () => kin.drive.setMode(kin.drive.mode === 'walk' ? 'pose' : 'walk');
    const touch = MojitoTouchpad.create($('kin-view'));
    const setTouch = on => { touch.setVisible(on); $('kin-touch').classList.toggle('on', on); $('kin-touch').setAttribute('aria-pressed', on); };
    setTouch(matchMedia('(pointer: coarse)').matches);   // on by default on a phone or tablet
    $('kin-touch').onclick = () => setTouch(!touch.visible);
    $('kin-mode').onclick = toggleMode;
    let chartTimer = null;   // the charts are redrawn a few times a second while the command changes, not on every input
    const chartsSoon = () => { kinSamples = null; if (!chartTimer) chartTimer = setTimeout(() => { chartTimer = null; kinCharts(); }, 300); };
    const set = (k, v) => kin.ctl[k].set(v);
    function apply(path, out) {
      if (path !== 'params') return;
      const pose = kin.drive.mode === 'pose', asked = pose || ['vx', 'vy', 'yaw_rate', 'body_roll', 'body_pitch', 'body_yaw', 'body_height'].some(k => out[k] !== 0);
      if (!asked && !kin.engaged) return;
      kin.engaged = true;
      if (pose) {
        kin.mode = 'stand'; set('vx', 0); set('vy', 0); set('yaw_rate', 0);
        set('roll', out.body_roll / D2R); set('pitch', out.body_pitch / D2R); set('yaw', out.body_yaw / D2R); set('pz', out.body_height);
      } else {
        kin.mode = out.walk ? 'gait' : 'stand';
        set('vx', out.vx); set('vy', out.vy); set('yaw_rate', out.yaw_rate);
        if (out.height !== undefined) set('stance_height', out.height);   // a held raise or lower input: the lab's own stand-height slider
        if (out.walk && !kin.playing) { kin.playing = true; $('kin-play').textContent = 'Pause'; }
      }
      kin.redraw = true; chartsSoon();
    }
    function onMode(mode) {
      $('kin-mode').textContent = 'Mode: ' + mode; $('kin-mode').classList.toggle('on', mode === 'pose'); $('kin-mode').setAttribute('aria-pressed', mode === 'pose');
      $('kin-keys').textContent = mode === 'pose' ? KEYS_POSE : KEYS_WALK;
      if (mode === 'walk') ['roll', 'pitch', 'yaw', 'pz'].forEach(k => set(k, 0));   // back to level, as the live simulation does
      else { kin.engaged = true; }
    }
    kin.drive = MojitoDrive.create({ touch, meta: { drive: live.meta.drive, pose: live.meta.pose, limits: { height: live.meta.limits.height }, presets: {} }, send: apply, getHeight: () => kin.st.stance_height,
      canvasActive: () => !$('kin').hidden, reset: () => kinPreset('stand'), setPreset: kinPreset, toggleChase: () => {}, onMode });
    $('kin-keys').textContent = KEYS_WALK;
  }

  function initKin() {
    const box = $('kin-controls');
    $('kin-stall-5v').textContent = fmt(M.SERVO.stall_5v, 2); $('kin-stall-68v').textContent = fmt(M.SERVO.stall_68v, 2);
    buttons(box, [['Trot', () => applyPreset('trot')], ['Crawl', () => applyPreset('crawl')], ['Pace', () => applyPreset('pace'), 'left legs together, then right: goes backward in physics'],
      ['Spin', () => applyPreset('spin')], ['Stand', () => applyPreset('stand')], ['Twist', () => applyPreset('twist'), 'posture with the feet planted']]);
    kinSpecs().forEach(([title, rows]) => {
      heading(box, title);
      rows.forEach(spec => { kin.ctl[spec[0]] = slider(box, spec, kin.st, k => { if (['period', 'duty', 'lift', 'sway', 'o0', 'o1', 'o2', 'o3'].includes(k)) kin.gaitName = 'custom'; kinDirty(k); }); });
    });
    buttons(box, [['Reset everything', () => { Object.entries(DEFAULTS).forEach(([k, v]) => kin.ctl[k].set(v)); applyPreset('trot'); }]]);
    kinView = new RobotView($('kin-view'), { az: -38, el: 24 });
    $('kin-play').onclick = () => { kin.playing = !kin.playing; $('kin-play').textContent = kin.playing ? 'Pause' : 'Play'; };
    $('kin-scrub').oninput = e => { kin.playing = false; $('kin-play').textContent = 'Play'; kin.t = parseFloat(e.target.value) * (kin.mode === 'stand' ? 1 : kin.st.period); kinFrame(); };
    $('kin-speed').onchange = e => { kin.speed = parseFloat(e.target.value); };
    $('kin-follow').onclick = e => { kinView.follow = !kinView.follow; e.target.setAttribute('aria-pressed', kinView.follow); e.target.classList.toggle('on', kinView.follow); kinView.recentre(); };
    $('kin-follow').classList.add('on');
    $('kin-zin').onclick = () => kinView.zoomBy(1.25); $('kin-zout').onclick = () => kinView.zoomBy(0.8);
    let last = performance.now();
    (function loop(now) {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      if (!$('kin').hidden) {
        const tau = commandTau(), st = kin.st, e = kin.eff;
        e.vx = M.easeTo(e.vx, st.vx, dt, tau); e.vy = M.easeTo(e.vy, st.vy, dt, tau); e.yaw_rate = M.easeTo(e.yaw_rate, st.yaw_rate, dt, tau);
        const moving = kin.playing && kin.mode === 'gait';
        if (moving) { kin.t += dt * kin.speed; kin.pose = M.advancePose(kin.pose, [e.vx, e.vy], e.yaw_rate, dt * kin.speed); }
        // a stopped lab is redrawn only while the eased command is still settling or the drive has just changed something
        if (moving || kin.redraw) { kin.redraw = false; kinFrame(); }
      }
      requestAnimationFrame(loop);
    })(last);
    applyPreset('trot');
    initKinDrive();
    window.addEventListener('themechange', kinCharts);
  }

  P.kin = { init: initKin, charts: kinCharts, redraw() { kinCharts(); kinFrame(); }, ready: () => Boolean(kinView) };
})();
