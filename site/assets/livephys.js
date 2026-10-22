/* Live physics tab: talks to scripts/website/serve_site.py on this machine (api/*). Absent when the site is hosted statically. */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const fmt = (v, d = 1) => (Number.isFinite(v) ? v.toFixed(d) : '–');
  const LEGS = ['FL', 'FR', 'RL', 'RR'];
  const SPEC = [
    ['Robot design (rebuilds the model on release)', true, [['la', 'abduction offset', 0.02, 0.09, 0.0025, ' mm', 1000], ['lu', 'upper leg', 0.08, 0.24, 0.005, ' mm', 1000], ['ll', 'lower leg', 0.08, 0.24, 0.005, ' mm', 1000],
      ['body_length', 'body length', 0.15, 0.5, 0.01, ' mm', 1000], ['body_width', 'body width', 0.08, 0.3, 0.01, ' mm', 1000], ['height', 'stand height', 0.1, 0.32, 0.0025, ' mm', 1000], ['mass', 'total mass', 0.5, 8, 0.1, ' kg', 1]]],
    ['Gait timing', false, [['period', 'period', 0.2, 4, 0.05, ' s', 1], ['duty', 'duty', 0.1, 0.95, 0.01, '', 1], ['lift', 'swing height', 0, 0.1, 0.005, ' mm', 1000], ['sway', 'crawl lean', 0, 0.06, 0.005, ' mm', 1000],
      ['o0', 'start FL', 0, 1, 0.01, '', 1], ['o1', 'start FR', 0, 1, 0.01, '', 1], ['o2', 'start RL', 0, 1, 0.01, '', 1], ['o3', 'start RR', 0, 1, 0.01, '', 1]]],
    ['Command', false, [['vx', 'forward', -0.5, 0.5, 0.01, ' m/s', 1], ['vy', 'left', -0.3, 0.3, 0.01, ' m/s', 1], ['yaw_rate', 'turn rate', -3, 3, 0.1, ' rad/s', 1]]],
    ['Servos and clock', false, [['kp', 'kp', 1, 300, 1, ' N·m/rad', 1], ['kd', 'kd', 0, 5, 0.05, '', 1], ['voltage', 'supply', 5, 6, 0.1, ' V', 1], ['speed', 'sim speed', 0.1, 2, 0.1, '×', 1]]],
  ];
  const L = { chase: true, drive: null, lastYaw: null, built: false, state: null, history: [], dragging: null, pending: {}, timer: null, ctl: {}, polling: false, view: null };

  const send = (path, body) => fetch('api/' + path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) }).catch(() => {});
  const queue = change => { Object.assign(L.pending, change); clearTimeout(L.timer); L.timer = setTimeout(() => { send('params', L.pending); L.pending = {}; }, 60); };
  const dec = (step, scale) => Math.max(0, Math.ceil(-Math.log10(step * scale) - 1e-9));

  const resetRun = () => { L.history = []; L.view.recentre(); if (L.drive) L.drive.forget(); send('reset'); };
  const choosePreset = name => { if (L.drive) L.drive.forget(); send('preset', { name }); };
  function setChase(on) {
    L.chase = on; $('live-chase').classList.toggle('on', on); $('live-chase').setAttribute('aria-pressed', on);
    if (on) { L.lastYaw = null; L.view.follow = true; $('live-follow').classList.add('on'); L.view.recentre(); }
  }
  function setDrive(on) { L.drive.setOn(on); $('live-drive').classList.toggle('on', on); $('live-drive').setAttribute('aria-pressed', on); $('live-drive').textContent = 'Drive: ' + (on ? 'on' : 'off'); }
  function build(meta) {
    const box = $('live-controls'); box.innerHTML = '';
    const btn = (label, fn, id) => { const b = document.createElement('button'); b.className = 'btn'; b.textContent = label; if (id) b.id = id; b.onclick = fn; return b; };
    const row = document.createElement('div'); row.className = 'btnrow';
    row.append(btn('Walk', () => send('params', { walk: !(L.state && L.state.params.walk) }), 'live-walk'), btn('Pause', () => send('params', { paused: !(L.state && L.state.paused) }), 'live-pause'),
      btn('Reset', () => resetRun()));
    box.append(row);
    const pr = document.createElement('div'); pr.className = 'btnrow';
    Object.keys(meta.presets).forEach(n => pr.append(btn(n[0].toUpperCase() + n.slice(1), () => choosePreset(n))));
    pr.append(btn('Default design', () => send('params', { la: meta.defaults.la, lu: meta.defaults.lu, ll: meta.defaults.ll, body_length: meta.defaults.body_length, body_width: meta.defaults.body_width, height: meta.defaults.height, mass: meta.defaults.mass })));
    box.append(pr);
    SPEC.forEach(([title, rebuild, rows]) => {
      const h = document.createElement('h4'); h.textContent = title; box.append(h);
      rows.forEach(([key, label, lo, hi, step, unit, scale]) => {
        const r = document.createElement('div'); r.className = 'row';
        r.innerHTML = `<label for="lp-${key}">${label}</label><input id="lp-${key}" type="range" min="${lo}" max="${hi}" step="${step}"><output></output>`;
        const input = r.querySelector('input'), out = r.querySelector('output'), show = v => { out.textContent = (v * scale).toFixed(dec(step, scale)) + unit; };
        input.addEventListener('pointerdown', () => { L.dragging = key; }); input.addEventListener('pointerup', () => { L.dragging = null; });
        input.addEventListener('input', () => {
          const v = parseFloat(input.value); show(v);
          if (rebuild) return;                       // sent on release: each change rebuilds the model
          if (key[0] === 'o' && key.length === 2) queue({ offsets: [0, 1, 2, 3].map(i => parseFloat($('lp-o' + i).value)) }); else queue({ [key]: v });
        });
        if (rebuild) input.addEventListener('change', () => send('params', { [key]: parseFloat(input.value) }));
        L.ctl[key] = { input, show }; box.append(r);
      });
    });
  }
  function sync() {
    const p = L.state.params;
    const set = (k, v) => { const c = L.ctl[k]; if (!c || L.dragging === k || L.pending[k] !== undefined) return; c.input.value = v; c.show(v); };
    Object.keys(L.ctl).forEach(k => { if (!/^o\d$/.test(k) && p[k] !== undefined) set(k, p[k]); });
    p.offsets.forEach((v, i) => { if (!(L.dragging && /^o\d$/.test(L.dragging))) set('o' + i, v); });
    $('live-walk').textContent = p.walk ? 'Stand' : 'Walk'; $('live-walk').classList.toggle('on', p.walk);
    $('live-pause').classList.toggle('on', L.state.paused);
  }
  function scene(s) {
    const R = s.rotation, o = s.origin, yaw = Math.atan2(R[1][0], R[0][0]), c = Math.cos(yaw), sn = Math.sin(yaw), reach = s.body_size[0] * 0.6, [cf, cl] = L.drive && L.drive.on ? L.drive.cmd : [0, 0];
    const ahead = (dx, dy, len) => { const n = Math.hypot(dx, dy) || 1, u = [(c * dx - sn * dy) / n, (sn * dx + c * dy) / n]; return { from: [o[0] + u[0] * reach, o[1] + u[1] * reach], to: [o[0] + u[0] * (reach + len), o[1] + u[1] * (reach + len)] }; };
    const arrows = [];   // the view already draws the heading arrow on the body
    if (Math.hypot(cf, cl) > 1e-3) arrows.push(Object.assign(ahead(cf, cl, 0.1 + 0.5 * Math.hypot(cf, cl)), { colour: '--blue' }));
    const overlay = (ctx, d) => {
      const v = s.measured_velocity, lim = L.drive ? L.drive.limits() : null;
      ctx.fillStyle = css('--ink'); ctx.font = `600 ${22 * d}px system-ui`; ctx.fillText(fmt(Math.hypot(v[0], v[1]), 2) + ' m/s', 14 * d, 30 * d);
      ctx.font = `${12 * d}px system-ui`; ctx.fillStyle = css('--muted');
      if (lim) { ctx.fillText(`${L.drive.preset}, full stick ${lim.vx} m/s${L.drive.hold ? ', walking on the spot' : ''}`, 14 * d, 48 * d); ctx.fillText(L.drive.pad ? 'gamepad: ' + L.drive.pad.slice(0, 38) : 'keyboard (connect a gamepad and press a button)', 14 * d, 64 * d); }
    };
    const inv = p => { const d = [p[0] - o[0], p[1] - o[1], p[2] - o[2]]; return [0, 1, 2].map(j => R[0][j] * d[0] + R[1][j] * d[1] + R[2][j] * d[2]); };
    return { origin: o, R, bodySize: s.body_size, points: s.legs.map(l => [inv(l[0]), inv(l[0]), inv(l[1]), inv(l[2])]), down: s.down, bad: [false, false, false, false], arrows, overlay };
  }
  function kpi(label, value, cls) { return `<div class="kpi ${cls || ''}"><b>${value}</b><span>${label}</span></div>`; }
  function readouts(s) {
    const p = s.params;
    $('live-kpis').innerHTML = kpi('forward speed, m/s', `${fmt(s.measured_velocity[0], 3)} <small class="muted">(cmd ${fmt(p.vx, 2)})</small>`) + kpi('sideways, m/s', `${fmt(s.measured_velocity[1], 3)} <small class="muted">(cmd ${fmt(p.vy, 2)})</small>`)
      + kpi('tilt', fmt(s.tilt_deg, 1) + '°', s.tilt_deg > 20 ? 'bad' : '') + kpi('feet down', String(s.feet_down), s.feet_down <= 2 && p.walk ? 'bad' : '')
      + kpi('servo torque / stall', fmt(s.torque_fraction, 2) + (s.saturated ? ' sat' : ''), s.saturated ? 'bad' : '') + kpi('total mass', fmt(s.total_mass_kg, 2) + ' kg') + kpi('time', fmt(s.time, 1) + ' s');
    const m = [];
    if (s.fell) m.push('<div class="w">The robot has fallen. Press Reset.</div>');
    if (!s.stance_reachable) m.push('<div class="w">This design cannot reach its stand height with these leg lengths: the legs are too short for the body height. Lengthen the legs or lower the stand height, then Reset.</div>');
    if (!m.length && !p.walk) m.push('<div class="muted">Standing. Press Walk.</div>');
    $('live-msgs').innerHTML = m.join('');
  }
  function drawTiming() {
    const c = $('live-timing'), r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1; if (!r.width || !L.state) return;
    c.width = Math.round(r.width * d); c.height = Math.round(r.height * d); const x = c.getContext('2d'), W = c.width, H = c.height, T = 4, t1 = L.state.time, left = 34 * d, rowH = H / 4;
    x.clearRect(0, 0, W, H); x.font = `${11 * d}px system-ui`;
    LEGS.forEach((n, i) => {
      const y = i * rowH; x.fillStyle = css('--muted'); x.fillText(n, 4 * d, y + rowH * 0.62);
      for (let k = 0; k < L.history.length - 1; k++) {
        const a = L.history[k], b = L.history[k + 1]; if (b.t < t1 - T) continue;
        const xa = Math.max(left, left + (a.t - (t1 - T)) / T * (W - left)), xb = left + (b.t - (t1 - T)) / T * (W - left);
        if (a.down[i]) { x.fillStyle = css('--' + n); x.fillRect(xa, y + rowH * 0.38, xb - xa + 1, rowH * 0.46); }
        if (a.cmd[i]) { x.fillStyle = css('--ink'); x.fillRect(xa, y + rowH * 0.18, xb - xa + 1, rowH * 0.12); }
      }
    });
  }
  async function poll() {
    if (!L.running) return;
    try {
      const s = await (await fetch('api/state', { cache: 'no-store' })).json();
      if (L.state && s.time < L.state.time) L.history = [];
      L.state = s; L.history.push({ t: s.time, down: s.down, cmd: s.commanded_down });
      while (L.history.length && L.history[0].t < s.time - 4.5) L.history.shift();
      if (L.chase) { // the camera turns with the robot; a drag changes the angle and the change is kept
        const yaw = s.yaw_deg; if (L.lastYaw !== null) L.view.az += ((yaw - L.lastYaw + 540) % 360) - 180; L.lastYaw = yaw;
      }
      L.drive.sync(s.params); L.view.setScene(scene(s)); sync(); readouts(s); drawTiming();
    } catch (e) { /* server stopped */ }
    setTimeout(poll, 33);
  }
  async function start() {
    if (L.running) return; const note = $('live-note');
    let meta;
    try { const r = await fetch('api/meta', { cache: 'no-store' }); if (!r.ok) throw 0; meta = await r.json(); } catch (e) {
      $('live-ui').hidden = true; note.hidden = false; return;
    }
    note.hidden = true; $('live-ui').hidden = false;
    if (!L.built) {
      L.built = true; L.view = new RobotView($('live-view'), { az: 180 + 20, el: 24, zoom: 1.4, span: 1.2 }); build(meta);
      L.drive = MojitoDrive.create({ meta, send, canvasActive: () => !!L.running, reset: resetRun, setPreset: choosePreset, toggleChase: () => setChase(!L.chase) });
      $('live-chase').classList.add('on'); $('live-chase').onclick = () => setChase(!L.chase);
      $('live-drive').classList.add('on'); $('live-drive').onclick = () => setDrive(!L.drive.on);
      $('live-keys').textContent = 'Drive: W S forward/back · A D sidestep · Q E turn · Space keep walking · 1 2 trot/crawl · R reset · C chase cam. Gamepad: sticks, A, B, X, LB, RB.';
      $('live-follow').classList.add('on'); $('live-follow').onclick = e => { L.view.follow = !L.view.follow; if (!L.view.follow) setChase(false); e.target.classList.toggle('on', L.view.follow); L.view.recentre(); };
      $('live-zin').onclick = () => L.view.zoomBy(1.25); $('live-zout').onclick = () => L.view.zoomBy(0.8);
      window.addEventListener('themechange', () => { L.view.draw(); drawTiming(); });
    }
    L.running = true; poll();
  }
  window.LivePhysics = { start, stop() { if (L.running && L.drive) { L.drive.keys.clear(); L.drive.forget(); send('params', { vx: 0, vy: 0, yaw_rate: 0, walk: false }); } L.running = false; } };
})();
