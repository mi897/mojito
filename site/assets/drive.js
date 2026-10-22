/* Drive mode for the live physics pages (scripts/gait/live_gait.html and the playground's Live physics tab):
 * keyboard or gamepad -> forward, left and yaw command -> {vx, vy, yaw_rate, walk} sent to the local server.
 * The full-stick values per preset come from the server (live_sim.DRIVE, served in /meta), not from this file.
 * Nothing here knows about drawing; the page passes callbacks. */
(function () {
  'use strict';
  const DEAD = 0.15;            // stick dead zone
  const STOP_DELAY = 500;       // ms of stepping after the last input, so the robot stops cleanly
  const stick = v => { const a = Math.abs(v); return a < DEAD ? 0 : Math.sign(v) * (a - DEAD) / (1 - DEAD); };
  const KEYMAP = { KeyW: 'fwd', ArrowUp: 'fwd', KeyS: 'back', ArrowDown: 'back', KeyA: 'left', KeyD: 'right', KeyQ: 'turnl', ArrowLeft: 'turnl', KeyE: 'turnr', ArrowRight: 'turnr' };

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
  const scale = (c, lim) => [c[0] * lim.vx, c[1] * lim.vy, c[2] * lim.yaw_rate];
  /* The preset whose stance height and period match the running parameters, or null. */
  function detectPreset(params, presets) {
    return Object.keys(presets).find(n => Math.abs(presets[n].height - params.height) < 1e-6 && Math.abs(presets[n].period - params.period) < 1e-6) || null;
  }

  /* opts: send(path, body), meta (from /meta), reset(), setPreset(name), toggleChase(), canvasActive() -> bool (keys are ignored when false). */
  function create(opts) {
    const D = { on: true, hold: false, keys: new Set(), cmd: [0, 0, 0], pad: null, preset: 'trot', sent: null, idleSince: 0, lastButtons: [] };
    const typing = e => e.target && e.target.tagName === 'INPUT' && e.target.type !== 'range' || e.target.tagName === 'TEXTAREA' || e.target.tagName === 'SELECT';
    const onKeyDown = e => {
      if (!opts.canvasActive() || typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
      if (KEYMAP[e.code]) { D.keys.add(KEYMAP[e.code]); e.preventDefault(); return; }
      if (e.repeat) return;
      if (e.code === 'Space') { D.hold = !D.hold; e.preventDefault(); }
      else if (e.code === 'Digit1') opts.setPreset('trot'); else if (e.code === 'Digit2') opts.setPreset('crawl');
      else if (e.code === 'KeyR') opts.reset(); else if (e.code === 'KeyC') opts.toggleChase();
    };
    const onKeyUp = e => { if (KEYMAP[e.code]) D.keys.delete(KEYMAP[e.code]); };
    const onBlur = () => D.keys.clear();
    addEventListener('keydown', onKeyDown); addEventListener('keyup', onKeyUp); addEventListener('blur', onBlur);

    function readPad() {
      const pads = (navigator.getGamepads ? navigator.getGamepads() : []) || [];
      const pad = [...pads].find(g => g && g.connected && g.axes.length >= 4);
      D.pad = pad ? pad.id : null;
      if (!pad) { D.lastButtons = []; return null; }
      const was = D.lastButtons, now = pad.buttons.map(b => b.pressed), edge = i => now[i] && !was[i];
      if (opts.canvasActive()) {   // buttons act only while the page is in use; sticks are always read so the HUD stays truthful
        if (edge(0)) D.hold = !D.hold;
        if (edge(1)) opts.reset();
        if (edge(2)) opts.toggleChase();
        if (edge(4)) opts.setPreset('trot');
        if (edge(5)) opts.setPreset('crawl');
      }
      D.lastButtons = now;
      return { axes: pad.axes, dpad: (now[12] ? 1 : 0) - (now[13] ? 1 : 0) };
    }
    function tick() {
      const pad = readPad();
      if (!D.on || !opts.canvasActive()) { D.cmd = [0, 0, 0]; return; }
      const f = fractions(D.keys, pad && pad.axes, pad ? pad.dpad : 0);
      const lim = opts.meta.drive[D.preset] || opts.meta.drive.trot, now = performance.now();
      D.cmd = scale(f, lim);
      const moving = f.some(v => v !== 0);
      if (moving) D.idleSince = now;
      const out = { vx: D.cmd[0], vy: D.cmd[1], yaw_rate: D.cmd[2], walk: moving || D.hold || now - D.idleSince < STOP_DELAY };
      const key = JSON.stringify(out);
      if (key !== D.sent) { D.sent = key; opts.send('params', out); }
    }
    const timer = setInterval(tick, 33);
    D.setOn = on => { D.on = on; D.sent = null; };
    D.forget = () => { D.sent = null; };                              // call after anything else changed the command (reset, preset)
    D.sync = params => { const p = detectPreset(params, opts.meta.presets); if (p) D.preset = p; };
    D.limits = () => opts.meta.drive[D.preset] || opts.meta.drive.trot;
    D.destroy = () => { clearInterval(timer); removeEventListener('keydown', onKeyDown); removeEventListener('keyup', onKeyUp); removeEventListener('blur', onBlur); };
    return D;
  }
  const api = { create, fractions, scale, detectPreset, stick, DEAD };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else window.MojitoDrive = api;
})();
