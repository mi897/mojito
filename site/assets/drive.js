/* Drive mode for the live physics pages (scripts/gait/live_gait.html and the playground's Live physics tab):
 * keyboard or gamepad -> {vx, vy, yaw_rate, walk} (walk mode) or {body_roll, body_pitch, body_yaw, body_height} (pose mode, the body tilts on
 * planted feet) sent to the local server. The full-stick values come from the server (live_sim.DRIVE and POSE, served in /meta).
 * Nothing here knows about drawing; the page passes callbacks. */
(function () {
  'use strict';
  const DEAD = 0.15;            // stick dead zone
  const STOP_DELAY = 500;       // ms of stepping after the last input, so the robot stops cleanly
  const stick = v => { const a = Math.abs(v); return a < DEAD ? 0 : Math.sign(v) * (a - DEAD) / (1 - DEAD); };
  const KEYMAP = { KeyW: 'fwd', ArrowUp: 'fwd', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', KeyD: 'right', KeyQ: 'turnl', ArrowLeft: 'turnl', KeyE: 'turnr', ArrowRight: 'turnr', KeyX: 'up', KeyZ: 'downh' };

  /* Pure part. keys: Set of 'fwd','back','left','right','turnl','turnr'. padAxes: [lx, ly, rx, ry] or null (standard layout,
   * stick up = negative y). A moved stick wins over keys. Returns [forward, left, turn] as fractions of full stick (-1..1);
   * forward and left together are limited to a unit circle so a diagonal is no faster than straight. */
  function fractions(keys, padAxes, dpad = 0) {
    let c = [(keys.has('fwd') ? 1 : 0) - (keys.has('back') ? 1 : 0), (keys.has('left') ? 1 : 0) - (keys.has('right') ? 1 : 0), (keys.has('turnl') ? 1 : 0) - (keys.has('turnr') ? 1 : 0)];
    if (padAxes) {
      const p = [Math.max(-1, Math.min(1, -stick(padAxes[1]) + dpad)), -stick(padAxes[0]), -stick(padAxes[2])];
      if (p.some(v => v !== 0)) c = p;
    }
    const n = Math.hypot(c[0], c[1]);
    if (n > 1) { c[0] /= n; c[1] /= n; }
    return c;
  }
  /* Which axes are the sticks. The standard layout is [left x, left y, right x, right y]; browsers only promise it when
   * pad.mapping === 'standard'. Other controllers (many Android ones) order the axes differently and rest a trigger axis at -1, so:
   * a saved calibration (calibrate() below) wins; else a non-standard pad is read as left stick = axes 0 and 1 and right stick = the
   * first two axes from 2 up that rested near 0 when first seen (a trigger rests at +-1). Returns [lx, ly, rx, ry] in the standard
   * sense: x right = +, y down = +. */
  function stickAxes(axes, mapping, rest, calibration) {
    if (calibration) { const g = m => (m && axes[m.i] !== undefined ? axes[m.i] * m.sign : 0); return [g(calibration.lx), g(calibration.ly), g(calibration.rx), g(calibration.ry)]; }
    if (mapping === 'standard') return [axes[0], axes[1], axes[2], axes[3]];
    const free = []; for (let i = 2; i < axes.length; i++) if (Math.abs(rest[i]) < 0.5) free.push(i);
    return [axes[0], axes[1], free.length ? axes[free[0]] : 0, free.length > 1 ? axes[free[1]] : 0];
  }
  /* Pose mode: the same inputs tilt the body. Returns [roll, pitch, yaw, height] fractions (-1..1), spring-loaded to zero on release.
   * Flight-stick sense: forward (W, stick up) pitches the nose down, right (D, stick right) rolls the right side down,
   * Q or the right stick left yaws the body left (counter-clockwise from above), the right stick up (X) raises the body, down (Z) lowers it. Right-hand rules about x forward, y left, z up. */
  function poseFractions(keys, padAxes) {
    let c = [(keys.has('right') ? 1 : 0) - (keys.has('left') ? 1 : 0), (keys.has('fwd') ? 1 : 0) - (keys.has('back') ? 1 : 0), (keys.has('turnl') ? 1 : 0) - (keys.has('turnr') ? 1 : 0), (keys.has('up') ? 1 : 0) - (keys.has('downh') ? 1 : 0)];
    if (padAxes) {
      const p = [stick(padAxes[0]), -stick(padAxes[1]), -stick(padAxes[2]), -stick(padAxes[3])];
      if (p.some(v => v !== 0)) c = p;
    }
    return c;
  }
  const scalePose = (c, lim) => [c[0] * lim.body_roll, c[1] * lim.body_pitch, c[2] * lim.body_yaw, c[3] * lim.body_height];
  const scale = (c, lim) => [c[0] * lim.vx, c[1] * lim.vy, c[2] * lim.yaw_rate];
  /* The preset whose stance height and period match the running parameters, or null. */
  function detectPreset(params, presets) {
    return Object.keys(presets).find(n => Math.abs(presets[n].height - params.height) < 1e-6 && Math.abs(presets[n].period - params.period) < 1e-6) || null;
  }

  const MAPS_KEY = 'mojito-gamepad-maps';
  function loadMaps() { try { return JSON.parse(localStorage.getItem(MAPS_KEY)) || {}; } catch (e) { return {}; } }
  function saveMaps(m) { try { localStorage.setItem(MAPS_KEY, JSON.stringify(m)); } catch (e) { /* private window: the map lasts until reload */ } }
  const CHOICE_KEY = 'mojito-gamepad-choice';
  function loadChoice() { try { return localStorage.getItem(CHOICE_KEY) || 'auto'; } catch (e) { return 'auto'; } }
  const CALIB_STEPS = [['ly', 'push the LEFT stick UP and release'], ['lx', 'push the LEFT stick RIGHT and release'], ['rx', 'push the RIGHT stick RIGHT and release'], ['ry', 'push the RIGHT stick UP and release']];

  /* opts: send(path, body), meta (from /meta), reset(), setPreset(name), toggleChase(), canvasActive() -> bool (keys are ignored when false),
   touch (optional, touchpad.js): {axes(), active()}; while a finger is down its sticks replace the gamepad's. */
  function create(opts) {
    const D = { maps: loadMaps(), rest: {}, calib: null, padInfo: null, on: true, mode: 'walk', pose: [0, 0, 0, 0], hold: false, keys: new Set(), cmd: [0, 0, 0], pad: null, preset: 'trot', sent: null, idleSince: 0, lastButtons: [], sigs: {}, active: -1, list: [], choice: loadChoice() };
    const typing = e => e.target && e.target.tagName === 'INPUT' && e.target.type !== 'range' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT';
    const onKeyDown = e => {
      if (!opts.canvasActive() || typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (KEYMAP[e.code]) { D.keys.add(KEYMAP[e.code]); e.preventDefault(); return; }
      if (e.repeat) return;
      if (e.code === 'Space') { D.hold = !D.hold; e.preventDefault(); }
      else if (e.code === 'Digit1') opts.setPreset('trot'); else if (e.code === 'Digit2') opts.setPreset('crawl');
      else if (e.code === 'KeyM') D.setMode(D.mode === 'walk' ? 'pose' : 'walk');
      else if (e.code === 'KeyR') opts.reset(); else if (e.code === 'KeyC') opts.toggleChase();
    };
    const onKeyUp = e => { if (KEYMAP[e.code]) D.keys.delete(KEYMAP[e.code]); };
    const onBlur = () => D.keys.clear();
    addEventListener('keydown', onKeyDown); addEventListener('keyup', onKeyUp); addEventListener('blur', onBlur);

    function readPad() {
      const pads = (navigator.getGamepads ? navigator.getGamepads() : []) || [];
      const cands = [...pads].filter(g => g && g.connected && g.axes.length >= 2);
      /* Several devices can show up for one controller (input-remapper and similar grab the real one and forward a clone), and the
       * silent one may come first. A pad chosen in the selector (D.choice, a pad id) wins; with "auto", follow whichever has moved
       * or been pressed most recently, and until then the first. */
      for (const g of cands) {
        const sig = g.axes.map(v => Math.round(v * 20)).join() + '|' + g.buttons.map(b => (b.pressed ? 1 : 0)).join('');
        if (D.sigs[g.index] !== undefined && D.sigs[g.index] !== sig) D.active = g.index;
        D.sigs[g.index] = sig;
      }
      D.list = cands.map(g => ({ index: g.index, id: g.id, live: g.index === D.active }));
      const pad = (D.choice !== 'auto' && cands.find(g => g.id === D.choice)) || cands.find(g => g.index === D.active) || cands[0];
      D.pad = pad ? pad.id : null;
      if (!pad) { D.lastButtons = []; D.padInfo = null; return null; }
      if (!D.rest[pad.id]) D.rest[pad.id] = Array.from(pad.axes);   // first reading: where a trigger axis rests
      D.padInfo = { id: pad.id, mapping: pad.mapping || 'none', axes: Array.from(pad.axes, v => Math.round(v * 100) / 100), pressed: pad.buttons.map((b, i) => (b.pressed ? i : -1)).filter(i => i >= 0), mapped: !!D.maps[pad.id] };
      if (D.calib) { calibrate(pad); D.lastButtons = pad.buttons.map(b => b.pressed); return null; }
      const was = D.lastButtons, now = pad.buttons.map(b => b.pressed), edge = i => now[i] && !was[i];
      if (opts.canvasActive()) {   // buttons act only while the page is in use; sticks are always read so the HUD stays truthful
        if (edge(0)) D.hold = !D.hold;
        if (edge(1)) opts.reset();
        if (edge(2)) opts.toggleChase();
        if (edge(3)) D.setMode(D.mode === 'walk' ? 'pose' : 'walk');
        if (edge(4)) opts.setPreset('trot');
        if (edge(5)) opts.setPreset('crawl');
      }
      D.lastButtons = now;
      return { axes: stickAxes(pad.axes, pad.mapping, D.rest[pad.id], D.maps[pad.id]), dpad: (now[12] ? 1 : 0) - (now[13] ? 1 : 0) };
    }
    /* Learn the stick axes of a controller that is not laid out as the standard one: four moves, each picks the axis that moved
     * most from where it rested and its sign. Saved per controller id in localStorage. */
    function calibrate(pad) {
      const c = D.calib, a = pad.axes;
      if (!c.rest) c.rest = Array.from(a);
      if (c.waitRelease) { if (a.every((v, i) => Math.abs(v - c.rest[i]) < 0.3)) c.waitRelease = false; return; }
      let best = -1, move = 0;
      for (let i = 0; i < a.length; i++) { const d = a[i] - c.rest[i]; if (Math.abs(d) > Math.abs(move)) { move = d; best = i; } }
      if (Math.abs(move) < 0.6) return;
      const key = CALIB_STEPS[c.step][0];
      c.map[key] = { i: best, sign: key === 'ly' || key === 'ry' ? -Math.sign(move) : Math.sign(move) };   // standard sense: up = -1 on ly and ry, right = +1 on lx and rx
      c.step++; c.waitRelease = true;
      if (c.step === CALIB_STEPS.length) { D.maps[pad.id] = c.map; saveMaps(D.maps); D.calib = null; D.sent = null; }
    }
    function tick() {
      let pad = readPad();
      if (opts.touch && opts.touch.active()) pad = { axes: opts.touch.axes(), dpad: 0 };   // fingers on the on-screen sticks
      if (D.calib) return;   // mapping: send nothing
      if (!D.on || !opts.canvasActive()) { D.cmd = [0, 0, 0]; return; }
      const now = performance.now();
      let out;
      if (D.mode === 'pose') {   // stand and tilt: no walking command
        D.cmd = [0, 0, 0]; D.pose = scalePose(poseFractions(D.keys, pad && pad.axes), opts.meta.pose);
        out = { vx: 0, vy: 0, yaw_rate: 0, walk: false, body_roll: D.pose[0], body_pitch: D.pose[1], body_yaw: D.pose[2], body_height: D.pose[3] };
      } else {
        const f = fractions(D.keys, pad && pad.axes, pad ? pad.dpad : 0), lim = opts.meta.drive[D.preset] || opts.meta.drive.trot;
        D.cmd = scale(f, lim); D.pose = [0, 0, 0, 0];
        const moving = f.some(v => v !== 0);
        if (moving) D.idleSince = now;
        out = { vx: D.cmd[0], vy: D.cmd[1], yaw_rate: D.cmd[2], walk: moving || D.hold || now - D.idleSince < STOP_DELAY, body_roll: 0, body_pitch: 0, body_yaw: 0, body_height: 0 };
      }
      const key = JSON.stringify(out);
      if (key !== D.sent) { D.sent = key; opts.send('params', out); }
    }
    const timer = setInterval(tick, 33);
    D.setMode = mode => { D.mode = mode; D.sent = null; D.hold = false; if (opts.onMode) opts.onMode(mode); };
    D.calibrate = () => { D.calib = D.calib ? null : { step: 0, rest: null, map: {}, waitRelease: false }; D.sent = null; };
    D.clearMap = () => { if (D.pad) { delete D.maps[D.pad]; saveMaps(D.maps); } };
    D.status = () => D.calib ? 'Mapping: ' + CALIB_STEPS[D.calib.step][1] + ' (step ' + (D.calib.step + 1) + ' of ' + CALIB_STEPS.length + ')'
      : D.padInfo ? `${D.padInfo.mapping === 'standard' ? 'standard' : 'non-standard'} mapping${D.padInfo.mapped ? ', your map' : ''} · axes ${D.padInfo.axes.join(' ')} · buttons ${D.padInfo.pressed.join(' ') || 'none'}` : 'no gamepad seen (press a button on it)';
    /* Keep a <select> in step with the pads the browser reports: "Auto" plus one entry per pad, a dot marking the one that last moved. */
    D.fillSelect = el => {
      const opts = [['auto', 'Auto (follow the pad that moves)']].concat(D.list.map(g => [g.id, (g.live ? '\u25cf ' : '') + g.index + ': ' + g.id]));
      const shown = Array.from(el.options, o => o.value + '|' + o.textContent).join('\n'), want = opts.map(o => o[0] + '|' + o[1]).join('\n');
      if (shown !== want) { el.textContent = ''; for (const [v, t] of opts) { const o = document.createElement('option'); o.value = v; o.textContent = t; el.appendChild(o); } }
      el.value = opts.some(o => o[0] === D.choice) ? D.choice : 'auto';
    };
    D.selectPad = v => { D.choice = v; D.sent = null; try { localStorage.setItem(CHOICE_KEY, v); } catch (e) { /* lasts until reload */ } };
    D.setOn = on => { D.on = on; D.sent = null; };
    D.forget = () => { D.sent = null; };                              // call after anything else changed the command (reset, preset)
    D.sync = params => { const p = detectPreset(params, opts.meta.presets); if (p) D.preset = p; };
    D.limits = () => opts.meta.drive[D.preset] || opts.meta.drive.trot;
    D.destroy = () => { clearInterval(timer); removeEventListener('keydown', onKeyDown); removeEventListener('keyup', onKeyUp); removeEventListener('blur', onBlur); };
    return D;
  }
  const api = { create, stickAxes, fractions, poseFractions, scalePose, scale, detectPreset, stick, DEAD };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else window.MojitoDrive = api;
})();
