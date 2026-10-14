/* Small SVG chart library: line/scatter and grouped bars, one y-axis per chart, light and dark through CSS variables.
 * Colours are CSS values such as 'var(--blue)', so charts follow the theme without redrawing. */
(function () {
  'use strict';
  const NS = 'http://www.w3.org/2000/svg';
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

  function niceTicks(lo, hi, n = 5) {
    if (!(hi > lo)) return [lo];
    const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), r = raw / mag;
    const step = (r < 1.5 ? 1 : r < 3 ? 2 : r < 7 ? 5 : 10) * mag, out = [];
    for (let v = Math.ceil(lo / step - 1e-9) * step; v <= hi + step * 1e-9; v += step) out.push(+v.toPrecision(12));
    return out;
  }
  function logTicks(lo, hi) {
    const out = [];
    for (let e = Math.floor(Math.log10(lo)); e <= Math.ceil(Math.log10(hi)); e++) out.push(Math.pow(10, e));
    return out;
  }
  const fmtNum = v => (Math.abs(v) >= 1000 || (Math.abs(v) < 0.01 && v !== 0) ? v.toExponential(0).replace('e+', 'e') : String(+v.toPrecision(4)));

  function frame(el, opts, h) {
    const W = Math.max(300, Math.round(el.clientWidth || 680)), H = h || Math.round(W / (opts.aspect || 1.9));
    const m = Object.assign({ l: 58, r: 16, t: 14, b: 46 }, opts.margin);
    if (opts.horizontal) m.l = opts.margin?.l ?? 120;
    const legend = (opts.legend === false || opts.series.length < 2) ? '' :
      '<div class="legend">' + opts.series.map(s => `<span><i style="background:${s.color}"></i>${esc(s.name)}</span>`).join('') + '</div>';
    el.classList.add('chart');
    el.innerHTML = legend + `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${esc(opts.alt || opts.y?.label || 'chart')}"></svg>`;
    return { svg: el.querySelector('svg'), W, H, m, pw: W - m.l - m.r, ph: H - m.t - m.b };
  }
  const svgEl = (tag, attrs, parent, text) => {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs) e.setAttribute(k, attrs[k]);
    if (text !== undefined) e.textContent = text;
    if (parent) parent.append(e);
    return e;
  };
  function tipOn(node, html) {
    node.addEventListener('pointermove', ev => window.MojitoSite.tip(html, ev.clientX, ev.clientY));
    node.addEventListener('pointerleave', () => window.MojitoSite.tip(null));
  }

  function scale(axis, vals, pad) {
    const log = !!axis.log;
    let lo = axis.domain ? axis.domain[0] : Math.min(...vals), hi = axis.domain ? axis.domain[1] : Math.max(...vals);
    if (!axis.domain) {
      if (log) { lo = Math.pow(10, Math.floor(Math.log10(lo))); hi = Math.pow(10, Math.ceil(Math.log10(hi))); }
      else { if (axis.zero !== false && lo > 0 && (axis.zero || lo < hi * 0.6)) lo = 0; const span = (hi - lo) || 1; hi += span * (pad ?? 0.04); if (lo !== 0) lo -= span * (pad ?? 0.04); }
    }
    const ticks = axis.ticks || (log ? logTicks(lo, hi) : niceTicks(lo, hi));
    if (!axis.domain && !log) { lo = Math.min(lo, ticks[0]); hi = Math.max(hi, ticks[ticks.length - 1]); }
    const f = log ? v => (Math.log10(v) - Math.log10(lo)) / (Math.log10(hi) - Math.log10(lo)) : v => (v - lo) / ((hi - lo) || 1);
    return { lo, hi, ticks, f };
  }

  function axes(g, F, sx, sy, opts) {
    const { m, pw, ph } = F;
    const gy = svgEl('g', { class: 'ax' }, g);
    sy.ticks.forEach(t => {
      const y = m.t + ph * (1 - sy.f(t));
      if (y < m.t - 1 || y > m.t + ph + 1) return;
      svgEl('line', { class: 'gridline', x1: m.l, x2: m.l + pw, y1: y, y2: y }, gy);
      svgEl('text', { x: m.l - 8, y: y + 4, 'text-anchor': 'end' }, gy, (opts.y.fmt || fmtNum)(t));
    });
    if (sx) {
      sx.ticks.forEach(t => {
        const x = m.l + pw * sx.f(t);
        if (x < m.l - 1 || x > m.l + pw + 1) return;
        svgEl('line', { class: 'axisline', x1: x, x2: x, y1: m.t + ph, y2: m.t + ph + 4 }, gy);
        svgEl('text', { x, y: m.t + ph + 18, 'text-anchor': 'middle' }, gy, (opts.x.fmt || fmtNum)(t));
      });
    }
    svgEl('line', { class: 'axisline', x1: m.l, x2: m.l + pw, y1: m.t + ph, y2: m.t + ph }, gy);
    if (opts.y.label) {
      const t = svgEl('text', { x: 14, y: m.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 14 ${m.t + ph / 2})` }, gy, opts.y.label);
      t.style.fill = 'var(--ink2)';
    }
    if (opts.x && opts.x.label) svgEl('text', { x: m.l + pw / 2, y: F.H - 6, 'text-anchor': 'middle' }, gy, opts.x.label);
  }

  /* series: [{name, color, points:[[x,y]...], dashed, markers, width, labelLast}] */
  function line(el, opts) {
    const F = frame(el, opts), { m, pw, ph, svg } = F;
    const all = opts.series.flatMap(s => s.points);
    const sx = scale(opts.x, all.map(p => p[0]), 0.02), sy = scale(opts.y, all.map(p => p[1]).concat((opts.hlines || []).map(h => h.y)));
    axes(svg, F, sx, sy, opts);
    const px = x => m.l + pw * sx.f(x), py = y => m.t + ph * (1 - sy.f(y));
    (opts.hlines || []).forEach(h => {
      svgEl('line', { x1: m.l, x2: m.l + pw, y1: py(h.y), y2: py(h.y), style: `stroke:${h.color || 'var(--muted)'};stroke-width:1.2;stroke-dasharray:5 4` }, svg);
      svgEl('text', { x: m.l + pw - 4, y: py(h.y) - 5, 'text-anchor': 'end', style: `fill:${h.color || 'var(--muted)'}` }, svg, h.label);
    });
    (opts.vlines || []).forEach(v => {
      svgEl('line', { x1: px(v.x), x2: px(v.x), y1: m.t, y2: m.t + ph, style: `stroke:${v.color || 'var(--muted)'};stroke-width:1.2;stroke-dasharray:5 4` }, svg);
      if (v.label) svgEl('text', { x: px(v.x) + 4, y: m.t + 12, style: `fill:${v.color || 'var(--muted)'}` }, svg, v.label);
    });
    const pts = [];
    opts.series.forEach(s => {
      const d = s.points.map((p, i) => (i ? 'L' : 'M') + px(p[0]).toFixed(1) + ' ' + py(p[1]).toFixed(1)).join('');
      if (s.width !== 0) svgEl('path', { d, fill: 'none', style: `stroke:${s.color};stroke-width:${s.width || 2.2};stroke-linejoin:round;stroke-linecap:round${s.dashed ? ';stroke-dasharray:6 4' : ''}` }, svg);
      if (s.markers) s.points.forEach(p => svgEl('circle', { cx: px(p[0]), cy: py(p[1]), r: s.markerSize || 3.6, style: `fill:${p[2] === 'open' ? 'var(--panel)' : s.color};stroke:${s.color};stroke-width:1.8` }, svg));
      s.points.forEach(p => pts.push({ x: px(p[0]), y: py(p[1]), s, p }));
    });
    const dot = svgEl('circle', { r: 6, fill: 'none', style: 'stroke:var(--ink);stroke-width:1.6;display:none' }, svg);
    const hit = svgEl('rect', { class: 'hit', x: m.l, y: m.t, width: pw, height: ph }, svg);
    hit.addEventListener('pointermove', ev => {
      const r = svg.getBoundingClientRect(), k = F.W / r.width, mx = (ev.clientX - r.left) * k, my = (ev.clientY - r.top) * k;
      let best = null, bd = 1e9;
      pts.forEach(q => { const d = Math.hypot(q.x - mx, (q.y - my) * 0.6); if (d < bd) { bd = d; best = q; } });
      if (!best || bd > 60) { dot.style.display = 'none'; window.MojitoSite.tip(null); return; }
      dot.setAttribute('cx', best.x); dot.setAttribute('cy', best.y); dot.style.display = '';
      const text = opts.tip ? opts.tip(best.s, best.p) : `<b>${esc(best.s.name)}</b><br>${esc(opts.x.label || 'x')}: ${(opts.x.fmt || fmtNum)(best.p[0])}<br>${esc(opts.y.label || 'y')}: ${(opts.y.fmt || fmtNum)(best.p[1])}`;
      window.MojitoSite.tip(text, ev.clientX, ev.clientY);
    });
    hit.addEventListener('pointerleave', () => { dot.style.display = 'none'; window.MojitoSite.tip(null); });
    return Object.assign(F, { px, py, sx, sy });
  }

  /* categories: [labels]; series: [{name, color, values:[...], notes?:[...]}]; values may be null. */
  function bars(el, opts) {
    const F = frame(el, opts), { m, pw, ph, svg } = F;
    const vals = opts.series.flatMap(s => s.values).filter(v => v != null);
    const sy = scale(Object.assign({ zero: true }, opts.y, opts.horizontal ? {} : {}), vals.concat(0));
    const n = opts.categories.length, k = opts.series.length;
    if (opts.horizontal) {
      // value axis along x
      const sx = scale(Object.assign({ zero: true }, opts.y), vals.concat(0)), band = ph / n, bh = Math.min(26, band * 0.78 / k);
      const gx = svgEl('g', { class: 'ax' }, svg);
      sx.ticks.forEach(t => {
        const x = m.l + pw * sx.f(t);
        svgEl('line', { class: 'gridline', x1: x, x2: x, y1: m.t, y2: m.t + ph }, gx);
        svgEl('text', { x, y: m.t + ph + 18, 'text-anchor': 'middle' }, gx, (opts.y.fmt || fmtNum)(t));
      });
      if (opts.y.label) svgEl('text', { x: m.l + pw / 2, y: F.H - 6, 'text-anchor': 'middle' }, gx, opts.y.label);
      opts.categories.forEach((c, i) => {
        const yc = m.t + band * (i + 0.5);
        svgEl('text', { x: m.l - 8, y: yc + 4, 'text-anchor': 'end' }, gx, c);
        opts.series.forEach((s, j) => {
          const v = s.values[i]; if (v == null) return;
          const y = yc - (k * bh) / 2 + j * bh, w = Math.max(1, pw * (sx.f(v) - sx.f(0)));
          const r = svgEl('rect', { x: m.l + pw * sx.f(0), y, width: w, height: bh - 2, rx: 3, style: `fill:${s.color}` }, svg);
          tipOn(r, `<b>${esc(c)}</b><br>${esc(s.name)}: ${(opts.y.fmt || fmtNum)(v)}${s.notes && s.notes[i] ? '<br>' + esc(s.notes[i]) : ''}`);
          if (opts.labels !== false) svgEl('text', { x: m.l + pw * sx.f(0) + w + 5, y: y + bh / 2 + 3, style: 'fill:var(--ink2)' }, svg, (opts.y.fmt || fmtNum)(v));
        });
      });
      return F;
    }
    axes(svg, F, null, sy, opts);
    const band = pw / n, bw = Math.min(46, band * 0.78 / k);
    opts.categories.forEach((c, i) => {
      const xc = m.l + band * (i + 0.5);
      svgEl('text', { x: xc, y: m.t + ph + 18, 'text-anchor': 'middle' }, svg, c);
      opts.series.forEach((s, j) => {
        const v = s.values[i]; if (v == null) return;
        const x = xc - (k * bw) / 2 + j * bw, y0 = m.t + ph * (1 - sy.f(0)), y1 = m.t + ph * (1 - sy.f(v));
        const r = svgEl('rect', { x, y: Math.min(y0, y1), width: bw - 2, height: Math.max(1, Math.abs(y0 - y1)), rx: 3, style: `fill:${s.color}` }, svg);
        tipOn(r, `<b>${esc(c)}</b><br>${esc(s.name)}: ${(opts.y.fmt || fmtNum)(v)}${s.notes && s.notes[i] ? '<br>' + esc(s.notes[i]) : ''}`);
        if (opts.labels !== false) svgEl('text', { x: x + (bw - 2) / 2, y: Math.min(y0, y1) - 5, 'text-anchor': 'middle', style: 'fill:var(--ink2)' }, svg, (opts.y.fmt || fmtNum)(v));
      });
    });
    return F;
  }

  window.Charts = { line, bars, niceTicks, svgEl };
})();
