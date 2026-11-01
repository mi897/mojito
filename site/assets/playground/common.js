/* Playground, shared part: helpers both labs use, the tabs, and the start-up. The labs are kinematic.js (live, assets/mojito.js)
 * and replay.js (the stored MuJoCo runs, data/sweep.js); the live-physics tab is livephys.js. Loaded first: the labs register
 * themselves on window.Playground, and the start-up below runs on DOMContentLoaded, after every script has been read. */
(function () {
  'use strict';
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

  // a visible one-line note, for a copy of the site built without data/ (mojito.js then refuses to load and the labs cannot start)
  function missingData(where) {
    const note = document.createElement('p'); note.className = 'w'; note.textContent = 'site data missing: run scripts/website/build_site.py';
    where.before(note);
  }
  const P = window.Playground = { D2R: Math.PI / 180, $, css, fmt, slider, heading, buttons, kpi, missingData };
  const ready = lab => Boolean(window.Mojito && P[lab]);

  let rz; window.addEventListener('resize', () => { clearTimeout(rz); rz = setTimeout(() => { if (!$('kin').hidden && ready('kin') && P.kin.ready()) P.kin.charts(); else if (!$('phys').hidden && ready('replay') && P.replay.hasRun()) P.replay.changed(); }, 200); });

  /* ---------- tabs ---------- */
  function show(which) {
    $('kin').hidden = which !== 'kin'; $('phys').hidden = which !== 'phys'; $('live').hidden = which !== 'live';
    ['kin', 'phys', 'live'].forEach(k => $('tab-' + k).setAttribute('aria-selected', which === k));
    if (window.LivePhysics) { if (which === 'live') window.LivePhysics.start(); else window.LivePhysics.stop(); }
    history.replaceState(null, '', '#' + which);
    if (which === 'kin' && ready('kin') && P.kin.ready()) P.kin.redraw();
    if (which === 'phys' && ready('replay')) {
      if ($('phys-controls').dataset.ready) { P.replay.changed(); return; }
      $('phys-controls').innerHTML = '<p class="muted">Loading the ' + (((window.MOJITO_RESULTS || {}).physics_sweep_summary || {}).n_runs || '') + ' recorded runs…</p>';
      window.loadSweep().then(() => { $('phys-controls').dataset.ready = '1'; P.replay.init(); })
        .catch(() => { $('phys-controls').innerHTML = '<p class="w">The recorded runs (data/sweep.js) are missing. Run <code>uv run --extra sim python scripts/website/build_site.py</code>.</p>'; });
    }
  }
  document.addEventListener('DOMContentLoaded', () => {
    // the tabs first, so that a lab that fails to start does not take the others with it
    $('tab-kin').onclick = () => show('kin'); $('tab-phys').onclick = () => show('phys'); $('tab-live').onclick = () => show('live');
    if (!window.Mojito) missingData($('tab-kin').parentElement);
    else if (ready('kin')) {
      try { P.kin.init(); } catch (e) { console.error(e); }
    }
    if (location.hash === '#phys') show('phys'); else if (/^#live/.test(location.hash)) show('live');
  });
})();
