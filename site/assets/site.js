/* Shared page chrome: navigation, footer, theme, numbers bound to the results JSON, maths rendering. */
(function () {
  'use strict';
  const PAGES = [['index.html', 'Overview'], ['progress.html', 'Progress'], ['theory.html', 'How it works'], ['results.html', 'Results'],
    ['demos.html', 'Demos'], ['playground.html', 'Playground'], ['reports.html', 'Full reports']];
  const REPO = 'https://github.com/mi897/mojito';
  const here = (location.pathname.split('/').pop() || 'index.html');

  try { const t = localStorage.getItem('mojito-theme'); if (t) document.documentElement.dataset.theme = t; } catch (e) { /* storage blocked */ }

  const logo = '<svg viewBox="0 0 32 32" fill="none" stroke="currentColor" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">'
    + '<rect x="6" y="9" width="20" height="7" rx="2" stroke="var(--blue)"/><path d="M9 16l-3 8 2 4M23 16l3 8-2 4M13 16l-1 6 1 6M19 16l1 6-1 6" stroke="var(--orange)"/></svg>';

  function chrome() {
    const nav = document.createElement('nav');
    nav.className = 'nav';
    nav.innerHTML = '<div class="nav-in"><a class="brand" href="index.html">' + logo + 'Mojito</a>'
      + PAGES.map(([h, n]) => `<a class="l" href="${h}"${h === here ? ' aria-current="page"' : ''}>${n}</a>`).join('')
      + `<span class="grow"></span><a class="l" href="${REPO}">GitHub</a><button id="theme" title="Switch light and dark" aria-label="Switch light and dark">◐</button></div>`;
    document.body.prepend(nav);
    nav.querySelector('#theme').onclick = () => {
      const dark = getComputedStyle(document.documentElement).getPropertyValue('--bg').trim() === '#121212';
      const next = dark ? 'light' : 'dark';
      document.documentElement.dataset.theme = next;
      try { localStorage.setItem('mojito-theme', next); } catch (e) { /* ignore */ }
      window.dispatchEvent(new Event('themechange'));
    };
    const info = window.MOJITO_BUILD || {};
    const foot = document.createElement('footer');
    foot.innerHTML = '<div class="in"><p><b>Everything on this site is a simulation of a robot that does not exist yet.</b> '
      + 'All dimensions, joint limits, masses and servo figures are placeholders (servo figures are unverified retailer numbers). '
      + 'Stages 0 and 1 are pure geometry; the physics results are MuJoCo with placeholder everything, one run per case.</p>'
      + `<p>Numbers are read from the committed <code>results/*.json</code>${info.commit ? ' at commit <code>' + info.commit + '</code>' : ''}`
      + `${info.date ? ' (site built ' + info.date + ')' : ''}. Source: <a href="${REPO}">github.com/mi897/mojito</a>.</p></div>`;
    document.body.append(foot);
  }

  // data-bind="metrics.learned_varied_legs.median_mm" data-fmt="2" data-scale="1000" data-unit=" mm"
  function resolve(path) {
    return path.split(path.includes('/') ? '/' : '.').reduce((o, k) => (o == null ? o : o[k]), window.MOJITO_RESULTS);
  }
  function bind() {
    if (!window.MOJITO_RESULTS) return;
    document.querySelectorAll('[data-bind]').forEach(el => {
      let v = resolve(el.dataset.bind);
      if (typeof v !== 'number') { if (v == null) { el.textContent = '?'; el.title = 'missing in results: ' + el.dataset.bind; } else el.textContent = v; return; }
      v *= parseFloat(el.dataset.scale || 1);
      const f = el.dataset.fmt;
      el.textContent = (f === undefined ? String(+v.toPrecision(3)) : v.toFixed(parseInt(f, 10))) + (el.dataset.unit || '');
    });
  }

  function math() {
    if (!window.renderMathInElement) return;
    document.querySelectorAll('.math-root, main').forEach(root => {
      window.renderMathInElement(root, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '\\(', right: '\\)', display: false }],
        throwOnError: false, ignoredClasses: ['no-math'] });
    });
  }
  function loadScript(src) {
    return new Promise((ok, fail) => { const s = document.createElement('script'); s.src = src; s.onload = ok; s.onerror = fail; document.head.append(s); });
  }
  window.MojitoSite = {
    loadScript,
    // One floating tooltip shared by the charts.
    tip(html, x, y) {
      let t = document.getElementById('tip');
      if (!t) { t = document.createElement('div'); t.id = 'tip'; t.className = 'tip'; document.body.append(t); }
      if (html === null) { t.style.display = 'none'; return; }
      t.innerHTML = html; t.style.display = 'block';
      const w = t.offsetWidth, h = t.offsetHeight;
      t.style.left = Math.min(window.innerWidth - w - 8, x + 14) + 'px';
      t.style.top = Math.max(8, Math.min(window.innerHeight - h - 8, y + 14)) + 'px';
    },
    css: name => getComputedStyle(document.documentElement).getPropertyValue(name).trim(),
    math, bind,
  };

  document.addEventListener('DOMContentLoaded', async () => {
    chrome(); bind();
    if (document.querySelector('.has-math')) {
      try {
        const base = 'vendor/katex/';
        const css = document.createElement('link'); css.rel = 'stylesheet'; css.href = base + 'katex.min.css'; document.head.append(css);
        await loadScript(base + 'katex.min.js'); await loadScript(base + 'auto-render.min.js'); math();
      } catch (e) { console.warn('maths rendering unavailable', e); }
    }
    document.dispatchEvent(new Event('mojito-ready'));
  });
})();
