/* Playground, replay lab: the stored MuJoCo runs (data/sweep.js), replayed frame by frame, no interpolation between runs.
 * Needs common.js first. */
(function () {
  'use strict';
  const M = window.Mojito, P = window.Playground, { D2R, $, css, fmt, heading, buttons, kpi } = P;

  /* =============================================================
   * Physics lab
   * ============================================================= */
  const phys = { sel: { gait: 'trot', speed: 0.15, kp: 10, mass: 1, lift: 0.03 }, t: 0, playing: true, speed: 0.5, run: null, built: false };
  let physView, physRobot;

  // total mass (kg) of the runs at a mass scale: what the runs themselves recorded, else the placeholder total times the scale
  function sweepMass(scale) {
    const run = window.MOJITO_SWEEP.runs.find(r => r.mass_scale === scale);
    return run ? run.total_mass_kg : M.DESIGN.total_mass * scale;
  }
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
    add('ps-mass', 'total mass', S.axes.mass_scale.map(v => [v, sweepMass(v).toFixed(2) + ' kg' + (v === 1 ? ' (placeholder)' : '')]), () => sel.mass, v => { sel.mass = parseFloat(v); });
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
      + kpi('largest tilt', fmt(run.max_tilt_deg, 1) + '°', run.max_tilt_deg > 10 ? 'bad' : '') + kpi('lowest body height', fmt(run.min_body_height_mm, 0) + ' mm', '', `the stance height is ${fmt(1000 * window.MOJITO_SWEEP.stance_height_m, 1)} mm`)
      + kpi('peak servo torque / stall', fmt(run.peak_torque_fraction, 2), run.peak_torque_fraction >= 0.99 ? 'bad' : '', `stall = ${fmt(M.stallTorque(M.SERVO.voltage), 2)} N·m at ${M.SERVO.voltage} V (unverified)`)
      + kpi('servos saturated', fmt(100 * run.fraction_saturated, 0) + '% of walk', run.fraction_saturated > 0.05 ? 'bad' : '')
      + kpi('feet down, min / mean', run.feet_down_min + ' / ' + fmt(run.feet_down_mean, 1), '', 'feet carrying more than 0.5 N')
      + kpi('largest joint gap', fmt(run.max_joint_gap_deg, 0) + '°', '', 'commanded minus actual joint angle')
      + kpi('total mass', fmt(run.total_mass_kg, 2) + ' kg');
  }

  function physHeat() {
    const S = window.MOJITO_SWEEP, sel = phys.sel, kps = S.axes.kp, ms = S.axes.mass_scale;
    let html = '<table class="heat"><tr><th></th>' + ms.map(m => `<th>${sweepMass(m).toFixed(2)} kg</th>`).join('') + '</tr>';
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


  P.replay = { init: initPhys, changed: physChanged, hasRun: () => Boolean(phys.run) };
})();
