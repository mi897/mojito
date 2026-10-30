/* Playground: a live kinematic lab (assets/mojito.js) and a replay lab for the stored MuJoCo runs (data/sweep.js). */
(function () {
  'use strict';
  const M = window.Mojito, D2R = Math.PI / 180;
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–');

  /* ---------- small helpers ---------- */
  function slider(box, spec, state, onChange) {
    const [key, label, lo, hi, step, unit, scale] = spec;   // display = value * scale
    const row = document.createElement('div'); row.className = 'row';
    const dec = Math.max(0, Math.ceil(-Math.log10(step * (scale || 1)) - 1e-9));
    row.innerHTML = `<label for="${key}">${label}</label><input id="${key}" type="range" min="${lo}" max="${hi}" step="${step}"><output></output>`;
    const input = row.querySelector('input'), out = row.querySelector('output');
    const show = () => { out.textContent = (state[key] * (scale || 1)).toFixed(dec) + (unit || ''); };
    input.value = state[key]; show();
    input.addEventListener('input', () => { state[key] = parseFloat(input.value); show(); onChange(key); });
    box.append(row);
    return { set(v) { state[key] = v; input.value = v; show(); } };
  }
  function heading(box, text) { const h = document.createElement('h4'); h.textContent = text; box.append(h); }
  function buttons(box, items) {
    const d = document.createElement('div'); d.className = 'btnrow';
    items.forEach(([label, fn, title]) => { const b = document.createElement('button'); b.className = 'btn'; b.textContent = label; if (title) b.title = title; b.onclick = fn; d.append(b); });
    box.append(d);
  }
  function kpi(label, value, cls, hint) { return `<div class="kpi ${cls || ''}" ${hint ? `title="${hint}"` : ''}><b>${value}</b><span>${label}</span></div>`; }
  const matMul = (A, B) => A.map(r => [0, 1, 2].map(j => r[0] * B[0][j] + r[1] * B[1][j] + r[2] * B[2][j]));

  /* =============================================================
   * Kinematic lab
   * ============================================================= */
  const DEFAULTS = {
    la: 0.05, lu: 0.15, ll: 0.15, bl: 0.30, bw: 0.15, h: 0.2125, sFL: 1,
    px: 0, py: 0, pz: 0, roll: 0, pitch: 0, yaw: 0,
    period: 1.7, duty: 0.5, lift: 0.022, sway: 0, o0: 0, o1: 0.5, o2: 0.5, o3: 0,
    vx: 0.15, vy: 0, wz: 0, mass: 1.5, volts: 5,
  };
  const kin = { st: Object.assign({}, DEFAULTS), mode: 'gait', gaitName: 'trot', t: 0, playing: true, speed: 1, ctl: {} };

  function robotOf(st) {
    const r = new M.Robot({ lengths: [st.la, st.lu, st.ll], bodyLength: st.bl, bodyWidth: st.bw });
    r.lengths[0] = r.lengths[0].map((v, i) => (i ? v * st.sFL : v));
    return r;
  }
  function gaitOf(st) {
    return { name: kin.gaitName, period: st.period, duty: st.duty, lift: st.lift, sway: st.sway, offsets: [st.o0, st.o1, st.o2, st.o3] };
  }
  function evalAt(t) {
    const st = kin.st, robot = robotOf(st), standing = kin.mode === 'stand';
    return M.evaluate({
      robot, gait: standing ? null : gaitOf(st), t, velocity: [st.vx, st.vy], yawRate: st.wz, height: st.h,
      posture: { x: st.px, y: st.py, z: st.pz, roll: st.roll * D2R, pitch: st.pitch * D2R, yaw: st.yaw * D2R }, mass: st.mass, volts: st.volts,
    });
  }
  function kinScene(res, t) {
    const st = kin.st, moving = kin.mode === 'gait';
    const [angle, shift] = moving ? M.planarFlow(t, [st.vx, st.vy], st.wz) : [0, [0, 0]];
    const c = Math.cos(angle), s = Math.sin(angle), Rz = [[c, -s, 0], [s, c, 0], [0, 0, 1]];
    const off = res.offset;
    return {
      origin: [shift[0] + c * off[0] - s * off[1], shift[1] + s * off[0] + c * off[1], st.h + off[2]],
      R: matMul(Rz, res.R), bodySize: [st.bl, st.bw], points: res.points,
      down: res.loadsN.map(l => l > 1e-6), bad: res.reachable.map(r => !r),
    };
  }

  function kinSpecs() {
    return [
      ['Robot design (placeholders)', [['la', 'abduction offset', 0.02, 0.09, 0.0025, ' mm', 1000], ['lu', 'upper leg', 0.08, 0.24, 0.005, ' mm', 1000],
        ['ll', 'lower leg', 0.08, 0.24, 0.005, ' mm', 1000], ['bl', 'body length', 0.15, 0.5, 0.01, ' mm', 1000], ['bw', 'body width', 0.08, 0.3, 0.01, ' mm', 1000],
        ['h', 'stand height', 0.1, 0.32, 0.0025, ' mm', 1000], ['sFL', 'FL leg scale', 0.8, 1.25, 0.01, '×', 1]]],
      ['Body posture (feet stay put)', [['px', 'shift forward', -0.06, 0.06, 0.0025, ' mm', 1000], ['py', 'shift left', -0.06, 0.06, 0.0025, ' mm', 1000],
        ['pz', 'raise', -0.06, 0.06, 0.0025, ' mm', 1000], ['roll', 'roll', -30, 30, 0.5, '°', 1], ['pitch', 'pitch', -30, 30, 0.5, '°', 1], ['yaw', 'yaw', -30, 30, 0.5, '°', 1]]],
      ['Gait timing', [['period', 'period', 0.2, 3, 0.05, ' s', 1], ['duty', 'duty (on ground)', 0.2, 0.95, 0.01, '', 1], ['lift', 'swing height', 0, 0.1, 0.005, ' mm', 1000],
        ['sway', 'crawl lean', 0, 0.06, 0.005, ' mm', 1000], ['o0', 'start FL', 0, 1, 0.01, '', 1], ['o1', 'start FR', 0, 1, 0.01, '', 1],
        ['o2', 'start RL', 0, 1, 0.01, '', 1], ['o3', 'start RR', 0, 1, 0.01, '', 1]]],
      ['Command', [['vx', 'forward', -0.4, 0.4, 0.01, ' m/s', 1], ['vy', 'left', -0.3, 0.3, 0.01, ' m/s', 1], ['wz', 'turn rate', -2.5, 2.5, 0.1, ' rad/s', 1]]],
      ['Load', [['mass', 'total mass', 0.5, 6, 0.1, ' kg', 1], ['volts', 'servo supply', 5, 6, 0.1, ' V', 1]]],
    ];
  }
  const PRESETS = {
    trot: { mode: 'gait', gait: 'trot', set: { period: 1.7, duty: 0.5, lift: 0.022, sway: 0, o0: 0, o1: 0.5, o2: 0.5, o3: 0, vx: 0.15, vy: 0, wz: 0 } },
    crawl: { mode: 'gait', gait: 'crawl', set: { period: 2, duty: 0.85, lift: 0.03, sway: 0.025, o0: 0.75, o1: 0.25, o2: 0.5, o3: 0, vx: 0.04, vy: 0, wz: 0 } },
    pace: { mode: 'gait', gait: 'pace', set: { period: 1.7, duty: 0.5, lift: 0.03, sway: 0, o0: 0, o1: 0.5, o2: 0, o3: 0.5, vx: 0.1, vy: 0, wz: 0 } },
    spin: { mode: 'gait', gait: 'trot', set: { period: 1.7, duty: 0.5, lift: 0.022, sway: 0, o0: 0, o1: 0.5, o2: 0.5, o3: 0, vx: 0, vy: 0, wz: 0.6 } },
    stand: { mode: 'stand', gait: 'trot', set: {} },
    twist: { mode: 'stand', gait: 'trot', set: { roll: 12, pitch: -8, yaw: 16, pz: 0.02 } },
  };

  function applyPreset(name) {
    const p = PRESETS[name], base = ['px', 'py', 'pz', 'roll', 'pitch', 'yaw'];
    kin.mode = p.mode; kin.gaitName = p.gait;
    base.forEach(k => kin.ctl[k].set(0));
    Object.entries(p.set).forEach(([k, v]) => kin.ctl[k].set(v));
    kin.t = 0; kinDirty('preset');
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
    const res = evalAt(kin.t);
    kinView.setScene(kinScene(res, kin.t));
    const nDown = res.loadsN.filter(l => l > 1e-6).length;
    $('kin-kpis').innerHTML = kpi('feet carrying weight', nDown + ' of 4', nDown <= 2 ? 'bad' : '')
      + kpi('servo torque now / stall', fmt(res.peakFraction, 2), res.peakFraction >= 1 ? 'bad' : '', 'N·m needed: ' + fmt(res.peakTorque, 2))
      + kpi('stall torque', fmt(res.stall, 2) + ' N·m', '', 'listed, unverified: straight line between 19 kg·cm at 5 V and 21.5 kg·cm at 6.8 V')
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

  function initKin() {
    const box = $('kin-controls');
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
      if (kin.playing && kin.mode === 'gait' && !$('kin').hidden) { kin.t += dt * kin.speed; kinFrame(); }
      requestAnimationFrame(loop);
    })(last);
    applyPreset('trot');
    window.addEventListener('themechange', kinCharts);
  }

  /* =============================================================
   * Physics lab
   * ============================================================= */
  const phys = { sel: { gait: 'trot', speed: 0.15, kp: 10, mass: 1, lift: 0.03 }, t: 0, playing: true, speed: 0.5, run: null, built: false };
  let physView, physRobot;

  function findRun(sel) {
    const S = window.MOJITO_SWEEP;
    return S.runs.find(r => r.gait === sel.gait && Math.abs(r.speed_m_s - sel.speed) < 1e-9 && r.kp === sel.kp && r.mass_scale === sel.mass && Math.abs(r.lift_m - sel.lift) < 1e-9);
  }
  function frameAt(run, t) {
    const dt = window.MOJITO_SWEEP.frame_dt_s, x = Math.max(0, Math.min(run.frames.length - 1.0001, t / dt)), i = Math.floor(x), f = x - i, a = run.frames[i], b = run.frames[i + 1];
    return a.map((v, k) => (k === a.length - 1 ? a[k] : v + (b[k] - v) * f));
  }
  function physScene(run, t) {
    const fr = frameAt(run, t), S = window.MOJITO_SWEEP;
    const origin = [fr[0] / 1000, fr[1] / 1000, fr[2] / 1000], R = M.rotation(fr[3] / 100 * D2R, fr[4] / 100 * D2R, fr[5] / 100 * D2R);
    const feet = [0, 1, 2, 3].map(i => [fr[6 + 3 * i] / 1000, fr[7 + 3 * i] / 1000, fr[8 + 3 * i] / 1000]);
    const q = physRobot.solve(feet), loads = fr.slice(18, 22).map(v => v / 10), bits = fr[22];
    return { origin, R, bodySize: S.body_m, points: physRobot.jointPoints(q), down: loads.map(l => l > 0.5), bad: physRobot.reachable(feet).map(r => !r), fr, loads, bits };
  }

  function physSelectors() {
    const S = window.MOJITO_SWEEP, box = $('phys-controls');
    box.innerHTML = '';
    const sel = phys.sel, add = (id, label, opts, get, set) => {
      const row = document.createElement('div'); row.className = 'row';
      row.innerHTML = `<label for="${id}">${label}</label><select id="${id}">${opts.map(([v, t]) => `<option value="${v}">${t}</option>`).join('')}</select>`;
      const s = row.querySelector('select'); s.value = get(); s.onchange = () => { set(s.value); physChanged(); }; box.append(row);
    };
    heading(box, 'Gait');
    add('ps-gait', 'gait', Object.keys(S.axes.gait).map(g => [g, g]), () => sel.gait, v => { sel.gait = v; if (!S.axes.gait[v].includes(sel.speed)) sel.speed = S.axes.gait[v][Math.min(1, S.axes.gait[v].length - 1)]; });
    add('ps-speed', 'commanded speed', S.axes.gait[sel.gait].map(v => [v, v.toFixed(2) + ' m/s']), () => sel.speed, v => { sel.speed = parseFloat(v); });
    add('ps-lift', 'swing height', S.axes.lift_m.map(v => [v, Math.round(v * 1000) + ' mm']), () => sel.lift, v => { sel.lift = parseFloat(v); });
    heading(box, 'Servos and body');
    add('ps-kp', 'servo stiffness kp', S.axes.kp.map(v => [v, `${v} N·m/rad (kd ${S.axes.kd[v]})`]), () => sel.kp, v => { sel.kp = parseInt(v, 10); });
    add('ps-mass', 'total mass', S.axes.mass_scale.map(v => [v, (1.72 * v).toFixed(2) + ' kg' + (v === 1 ? ' (placeholder)' : '')]), () => sel.mass, v => { sel.mass = parseFloat(v); });
    heading(box, 'Try one of these');
    buttons(box, [
      ['Soft → stiff servos', () => setSel({ gait: 'trot', speed: 0.15, kp: 10, mass: 1, lift: 0.03 }, 'Compare kp 10 with kp 100: the body gets much closer to the commanded speed.'), 'docs/STAGE2.md: "Why the body falls short"'],
      ['Pace goes backward', () => setSel({ gait: 'pace', speed: 0.1, kp: 10, mass: 1, lift: 0.03 }, 'Only two feet on one side are down, so the body rolls onto the feet it is lifting.')],
      ['Taller swing on the pace', () => setSel({ gait: 'pace', speed: 0.1, kp: 10, mass: 1, lift: 0.06 })],
      ['Crawl', () => setSel({ gait: 'crawl', speed: 0.04, kp: 10, mass: 1, lift: 0.03 }, 'The crawl reaches its commanded speed at kp 100 but not at kp 10.')],
      ['Heavy robot', () => setSel({ gait: 'trot', speed: 0.15, kp: 10, mass: 1.75, lift: 0.03 })]]);
    const p = document.createElement('p'); p.id = 'phys-hint'; p.className = 'small muted'; box.append(p);
  }
  function setSel(s, hint) { Object.assign(phys.sel, s); physSelectors(); $('phys-hint').textContent = hint || ''; physChanged(); }

  function physChanged() {
    const S = window.MOJITO_SWEEP, sel = phys.sel;
    // keep dependent selectors in step
    const sp = $('ps-speed'); if (sp && sp.options.length !== S.axes.gait[sel.gait].length) physSelectors();
    ['ps-gait', 'ps-speed', 'ps-lift', 'ps-kp', 'ps-mass'].forEach(id => { const el = $(id); if (el) el.value = { 'ps-gait': sel.gait, 'ps-speed': sel.speed, 'ps-lift': sel.lift, 'ps-kp': sel.kp, 'ps-mass': sel.mass }[id]; });
    const run = findRun(sel); phys.run = run; phys.t = 0;
    const dur = (run.frames.length - 1) * S.frame_dt_s;
    phys.dur = dur;
    // commanded stance reference
    const commanded = sel.speed;
    const wk = run.frames.map((f, i) => [i * S.frame_dt_s + S.record_from_s, f[0] / 1000]);
    const x0 = run.frames.findIndex((f, i) => i * S.frame_dt_s + S.record_from_s >= 1.5);
    const startX = (run.frames[Math.max(0, x0)][0]) / 1000, startT = Math.max(0, x0) * S.frame_dt_s + S.record_from_s;
    const posSeries = [{ name: 'body, measured', color: 'var(--blue)', points: run.frames.map((f, i) => [i * S.frame_dt_s + S.record_from_s, f[0] / 1000 * 1000 - startX * 1000]).filter(p => p[0] >= startT - 0.001), width: 2.4 },
      { name: 'commanded speed', color: 'var(--orange)', dashed: true, width: 2, points: [[startT, 0], [S.record_from_s + dur, commanded * (S.record_from_s + dur - startT) * 1000]] }];
    $('pc-pos').innerHTML = ''; phys.cpos = Charts.line($('pc-pos'), { series: posSeries, x: { label: 'time in the run, s (walking starts at 1.5 s)', zero: false, fmt: v => +v.toFixed(1) }, y: { label: 'mm forward', zero: false }, aspect: 1.7 });
    const tilt = i => run.frames.map((f, k) => [k * S.frame_dt_s + S.record_from_s, f[i] / 100]);
    $('pc-tilt').innerHTML = ''; phys.ctilt = Charts.line($('pc-tilt'), { series: [{ name: 'roll', color: 'var(--blue)', points: tilt(3), width: 2.2 }, { name: 'pitch', color: 'var(--orange)', points: tilt(4), width: 2.2 }],
      x: { label: 'time in the run, s', zero: false, fmt: v => +v.toFixed(1) }, y: { label: 'degrees', zero: false, domain: undefined }, aspect: 1.7 });
    $('pc-timing').innerHTML = physTimingSvg(run);
    physKpis(run); physHeat(); physFrame();
  }

  function physTimingSvg(run) {
    const S = window.MOJITO_SWEEP, W = phys.tW = Math.max(300, $('pc-timing').clientWidth || 680), H = 170, l = 44, r = 12, rowH = (H - 28) / 4, pw = W - l - r, n = run.frames.length, t0 = S.record_from_s, T = n * S.frame_dt_s;
    let g = '';
    M.LEG_NAMES.forEach((name, i) => {
      const y = 6 + i * rowH;
      g += `<text x="${l - 8}" y="${y + rowH * 0.62}" text-anchor="end" style="fill:var(--${name});font-weight:600">${name}</text>`;
      run.frames.forEach((f, k) => {
        const x0 = l + pw * k / n, w = pw / n + 0.6;
        if (f[22] & (1 << i)) g += `<rect x="${x0}" y="${y + rowH * 0.08}" width="${w}" height="${rowH * 0.16}" style="fill:var(--ink)"/>`;
        if (f[18 + i] / 10 > 0.5) g += `<rect x="${x0}" y="${y + rowH * 0.32}" width="${w}" height="${rowH * 0.5}" style="fill:var(--${name})"/>`;
      });
    });
    for (let k = 0; k <= 5; k++) { const x = l + pw * k / 5; g += `<text x="${x}" y="${H - 6}" text-anchor="middle">${(t0 + T * k / 5).toFixed(1)} s</text>`; }
    return `<svg viewBox="0 0 ${W} ${H}"><g>${g}</g><line id="phys-cursor" x1="${l}" x2="${l}" y1="2" y2="${H - 22}" style="stroke:var(--ink);stroke-width:1.6"/></svg>`;
  }

  function physKpis(run) {
    const cmd = phys.sel.speed, ratio = cmd > 0 ? run.measured_speed_m_s / cmd : null;
    const msgs = [];
    if (!run.walked) msgs.push(`<div class="w">This run did not count as walked: the body tilted to ${fmt(run.max_tilt_deg, 0)}° or sank to ${fmt(run.min_body_height_mm, 0)} mm. Walking is judged upright within 30° and never below half the stance height.</div>`);
    if (cmd > 0 && run.measured_speed_m_s <= 0) msgs.push('<div class="w">The body moved backward although it was commanded forward.</div>');
    else if (cmd > 0 && ratio < 0.6) msgs.push(`<div class="muted">The body made ${fmt(100 * ratio, 0)}% of the commanded speed.</div>`);
    if (cmd === 0 && Math.abs(run.measured_speed_m_s) > 0.01) msgs.push(`<div class="muted">Marching in place, the body drifted ${run.measured_speed_m_s < 0 ? 'backward' : 'forward'} at ${fmt(Math.abs(run.measured_speed_m_s), 3)} m/s.</div>`);
    if (run.fraction_saturated > 0.05) msgs.push(`<div class="w">The servos were at their torque limit for ${fmt(100 * run.fraction_saturated, 0)}% of the walk.</div>`);
    if (run.walked && !msgs.length) msgs.push('<div class="ok">Upright throughout; speed close to the command.</div>');
    $('phys-msgs').innerHTML = msgs.join('');
    $('phys-kpis').innerHTML = kpi('measured speed', fmt(run.measured_speed_m_s, 3) + ' m/s', cmd > 0 && run.measured_speed_m_s <= 0 ? 'bad' : '', 'body displacement over the walking phase')
      + kpi('commanded speed', fmt(cmd, 2) + ' m/s') + kpi('upright and walked', run.walked ? 'yes' : 'no', run.walked ? 'good' : 'bad')
      + kpi('largest tilt', fmt(run.max_tilt_deg, 1) + '°', run.max_tilt_deg > 10 ? 'bad' : '') + kpi('lowest body height', fmt(run.min_body_height_mm, 0) + ' mm', '', 'the stance height is 212.5 mm')
      + kpi('peak servo torque / stall', fmt(run.peak_torque_fraction, 2), run.peak_torque_fraction >= 0.99 ? 'bad' : '', 'stall = 1.86 N·m at 5 V (unverified)')
      + kpi('servos saturated', fmt(100 * run.fraction_saturated, 0) + '% of walk', run.fraction_saturated > 0.05 ? 'bad' : '')
      + kpi('feet down, min / mean', run.feet_down_min + ' / ' + fmt(run.feet_down_mean, 1), '', 'feet carrying more than 0.5 N')
      + kpi('largest joint gap', fmt(run.max_joint_gap_deg, 0) + '°', '', 'commanded minus actual joint angle')
      + kpi('total mass', fmt(run.total_mass_kg, 2) + ' kg');
  }

  function physHeat() {
    const S = window.MOJITO_SWEEP, sel = phys.sel, kps = S.axes.kp, ms = S.axes.mass_scale;
    let html = '<table class="heat"><tr><th></th>' + ms.map(m => `<th>${(1.72 * m).toFixed(2)} kg</th>`).join('') + '</tr>';
    kps.forEach(kp => {
      html += `<tr><th>kp ${kp}</th>`;
      ms.forEach(m => {
        const r = findRun({ gait: sel.gait, speed: sel.speed, kp, mass: m, lift: sel.lift });
        const cls = (!r.walked ? 'fell ' : '') + (kp === sel.kp && m === sel.mass ? 'sel' : '');
        html += `<td class="${cls}" data-kp="${kp}" data-m="${m}" title="tilt ${r.max_tilt_deg}°, peak torque ${r.peak_torque_fraction}">${fmt(r.measured_speed_m_s, 3)}</td>`;
      });
      html += '</tr>';
    });
    $('phys-heat').innerHTML = html + '</table><p class="small muted">m/s measured, for commanded ' + sel.speed.toFixed(2) + ' m/s, ' + sel.gait + ', swing ' + Math.round(sel.lift * 1000) + ' mm.</p>';
    $('phys-heat').querySelectorAll('td').forEach(td => td.onclick = () => { sel.kp = parseInt(td.dataset.kp, 10); sel.mass = parseFloat(td.dataset.m); physChanged(); });
  }

  function physFrame() {
    const run = phys.run; if (!run) return;
    const S = window.MOJITO_SWEEP, sc = physScene(run, phys.t);
    // dashed line where the body would be at the commanded speed, on the floor (drawn as an overlay)
    const startIdx = Math.max(0, run.frames.findIndex((f, i) => i * S.frame_dt_s + S.record_from_s >= 1.5)), f0 = run.frames[startIdx];
    const tNow = phys.t + S.record_from_s, cmdX = f0[0] / 1000 + phys.sel.speed * Math.max(0, tNow - (startIdx * S.frame_dt_s + S.record_from_s));
    sc.overlay = (ctx, d, W, H) => {
      const sc2 = Math.min(W, H * 1.5) / 0.95 * physView.zoom, a = [cmdX, f0[1] / 1000, 0], b = [cmdX, f0[1] / 1000 + 0.16, 0], c0 = [cmdX, f0[1] / 1000 - 0.16, 0];
      const P = p => physView.project(p, W, H, sc2), pa = P(c0), pb = P(b);
      ctx.save(); ctx.setLineDash([6 * d, 5 * d]); ctx.strokeStyle = css('--orange'); ctx.lineWidth = 2 * d; ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke(); ctx.restore();
    };
    physView.setScene(sc);
    $('phys-scrub').value = phys.t / phys.dur;
    const cur = $('phys-cursor'); if (cur) { const x = 44 + ((phys.tW || 680) - 56) * (phys.t / (run.frames.length * S.frame_dt_s)); cur.setAttribute('x1', x); cur.setAttribute('x2', x); }
    [phys.cpos, phys.ctilt].forEach(F => {
      if (!F || !F.px) return;
      if (!F._cursor) F._cursor = Charts.svgEl('line', { style: 'stroke:var(--ink);stroke-width:1.4;stroke-dasharray:3 3' }, F.svg);
      const x = F.px(tNow); F._cursor.style.display = x < F.m.l ? 'none' : ''; F._cursor.setAttribute('x1', x); F._cursor.setAttribute('x2', x); F._cursor.setAttribute('y1', F.m.t); F._cursor.setAttribute('y2', F.m.t + F.ph);
    });
  }

  function initPhys() {
    const S = window.MOJITO_SWEEP;
    physRobot = new M.Robot({ lengths: S.lengths_m, bodyLength: S.body_m[0], bodyWidth: S.body_m[1] });
    physView = new RobotView($('phys-view'), { az: -38, el: 24, span: 1.2 });
    physSelectors(); physChanged();
    if (phys.built) return; phys.built = true;
    $('phys-play').onclick = () => { phys.playing = !phys.playing; $('phys-play').textContent = phys.playing ? 'Pause' : 'Play'; };
    $('phys-scrub').oninput = e => { phys.playing = false; $('phys-play').textContent = 'Play'; phys.t = parseFloat(e.target.value) * phys.dur; physFrame(); };
    $('phys-speed').onchange = e => { phys.speed = parseFloat(e.target.value); };
    $('phys-follow').classList.add('on');
    $('phys-follow').onclick = e => { physView.follow = !physView.follow; e.target.classList.toggle('on', physView.follow); physView.recentre(); };
    $('phys-zin').onclick = () => physView.zoomBy(1.25); $('phys-zout').onclick = () => physView.zoomBy(0.8);
    let last = performance.now();
    (function loop(now) {
      const dt = Math.min(0.1, (now - last) / 1000); last = now;
      if (phys.playing && phys.run && !$('phys').hidden) { phys.t += dt * phys.speed; if (phys.t > phys.dur) { phys.t = 0; physView.recentre(); } physFrame(); }
      requestAnimationFrame(loop);
    })(last);
    window.addEventListener('themechange', () => { if (phys.run) physChanged(); });
  }

  let rz; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (!$('kin').hidden) kinCharts(); else if (!$('phys').hidden && phys.run) physChanged(); }, 200); });

  /* ---------- tabs ---------- */
  function show(which) {
    $('kin').hidden = which !== 'kin'; $('phys').hidden = which !== 'phys'; $('live').hidden = which !== 'live';
    ['kin', 'phys', 'live'].forEach(k => $('tab-' + k).setAttribute('aria-selected', which === k));
    if (which === 'live') window.LivePhysics.start(); else window.LivePhysics.stop();
    history.replaceState(null, '', '#' + which);
    if (which === 'kin' && kinView) { kinCharts(); kinFrame(); }
    if (which === 'phys') {
      if ($('phys-controls').dataset.ready) { physChanged(); return; }
      $('phys-controls').innerHTML = '<p class="muted">Loading the 144 recorded runs…</p>';
      window.loadSweep().then(() => { $('phys-controls').dataset.ready = '1'; initPhys(); })
        .catch(() => { $('phys-controls').innerHTML = '<p class="w">The recorded runs (data/sweep.js) are missing. Run <code>python scripts/website/build_site.py</code>.</p>'; });
    }
  }
  document.addEventListener('DOMContentLoaded', () => {
    initKin();
    $('tab-kin').onclick = () => show('kin'); $('tab-phys').onclick = () => show('phys'); $('tab-live').onclick = () => show('live');
    if (location.hash === '#phys') show('phys'); else if (/^#live/.test(location.hash)) show('live');
  });
})();
