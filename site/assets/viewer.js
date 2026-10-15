/* Draggable isometric-style 3D view of the quadruped on a floor grid (canvas 2D, no libraries).
 * scene = { origin:[x,y,z], R:3x3 (body to world), bodySize:[length,width], points: 4 legs x [shoulder,hip,knee,foot] x xyz in the body frame,
 *           down:[4 bool] (foot carrying weight: filled), bad:[4 bool] (target not reachable: red ring), labels:true } */
(function () {
  'use strict';
  const LEGS = ['FL', 'FR', 'RL', 'RR'];
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const mv = (R, v) => R.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);

  class RobotView {
    constructor(canvas, opts = {}) {
      this.c = canvas; this.ctx = canvas.getContext('2d');
      this.az = opts.az ?? -38; this.el = opts.el ?? 24; this.zoom = opts.zoom ?? 1; this.follow = opts.follow ?? true;
      this.home = { az: this.az, el: this.el, zoom: this.zoom };
      this.scene = null; this.anchor = [0, 0, 0]; this.span = opts.span ?? 0.7; this.grid = opts.grid ?? 0.1;
      this.autorotate = opts.autorotate ?? 0; this.wheel = opts.wheel ?? false;
      this.pointers = new Map(); this._pinch = null;
      canvas.addEventListener('pointerdown', e => { canvas.setPointerCapture(e.pointerId); this.pointers.set(e.pointerId, [e.clientX, e.clientY]); this._pinch = null; this.autorotate = 0; });
      canvas.addEventListener('pointermove', e => {
        const prev = this.pointers.get(e.pointerId); if (!prev) return;
        this.pointers.set(e.pointerId, [e.clientX, e.clientY]);
        if (this.pointers.size === 2) {
          const [a, b] = [...this.pointers.values()], d = Math.hypot(a[0] - b[0], a[1] - b[1]);
          if (this._pinch) this.zoom = Math.max(0.4, Math.min(6, this.zoom * d / this._pinch));
          this._pinch = d;
        } else if (this.pointers.size === 1) {
          this.az -= (e.clientX - prev[0]) * 0.4; this.el = Math.max(3, Math.min(88, this.el + (e.clientY - prev[1]) * 0.3));
        }
        this.draw();
      });
      const up = e => { this.pointers.delete(e.pointerId); this._pinch = null; };
      canvas.addEventListener('pointerup', up); canvas.addEventListener('pointercancel', up);
      canvas.addEventListener('wheel', e => {
        if (!this.wheel && !(e.ctrlKey || e.metaKey)) return;
        e.preventDefault(); this.zoom = Math.max(0.4, Math.min(6, this.zoom * Math.exp(-e.deltaY * 0.0015))); this.draw();
      }, { passive: false });
      canvas.addEventListener('dblclick', () => this.reset());
      window.addEventListener('themechange', () => this.draw());
      new ResizeObserver(() => this.draw()).observe(canvas);
    }
    reset() { Object.assign(this, this.home); this.draw(); }
    zoomBy(f) { this.zoom = Math.max(0.4, Math.min(6, this.zoom * f)); this.draw(); }
    setScene(scene) {
      this.scene = scene;
      if (this.follow || !this._anchored) { this.anchor = scene.origin.slice(); this._anchored = true; }
      this.draw();
    }
    recentre() { if (this.scene) this.anchor = this.scene.origin.slice(); }
    project(p, W, H, scale) {
      const az = this.az * Math.PI / 180, el = this.el * Math.PI / 180, a = this.anchor;
      const x = p[0] - a[0], y = p[1] - a[1], z = p[2];
      const right = -Math.sin(az) * x + Math.cos(az) * y, depth = Math.cos(az) * x + Math.sin(az) * y;
      return [W / 2 + right * scale, H * 0.62 - (z * Math.cos(el) - depth * Math.sin(el)) * scale, depth];
    }
    draw() {
      const c = this.c, ctx = this.ctx, r = c.getBoundingClientRect(), d = window.devicePixelRatio || 1;
      if (!r.width) return;
      const W = Math.round(r.width * d), H = Math.round(r.height * d);
      if (c.width !== W || c.height !== H) { c.width = W; c.height = H; }
      ctx.clearRect(0, 0, W, H);
      const s = this.scene; if (!s) return;
      const scale = Math.min(W, H * 1.5) / 0.95 * this.zoom, P = p => this.project(p, W, H, scale);
      // floor grid, fixed in the world, drawn around the anchor
      const g = this.grid, sp = this.span, ax = this.anchor, x0 = Math.floor((ax[0] - sp) / g) * g, y0 = Math.floor((ax[1] - sp) / g) * g;
      ctx.lineWidth = d; ctx.strokeStyle = css('--line'); ctx.beginPath();
      for (let x = x0; x <= ax[0] + sp; x += g) { const a = P([x, ax[1] - sp, 0]), b = P([x, ax[1] + sp, 0]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
      for (let y = y0; y <= ax[1] + sp; y += g) { const a = P([ax[0] - sp, y, 0]), b = P([ax[0] + sp, y, 0]); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); }
      ctx.stroke();
      const toW = p => { const q = mv(s.R, p); return [q[0] + s.origin[0], q[1] + s.origin[1], q[2] + s.origin[2]]; };
      const [L, Wd] = s.bodySize, body = [[1, 1], [1, -1], [-1, -1], [-1, 1]].map(([a, b]) => P(toW([a * L / 2, b * Wd / 2, 0])));
      const items = [];
      items.push({ depth: P(s.origin)[2] + 0.001, draw: () => {
        ctx.beginPath(); body.forEach((q, i) => (i ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]))); ctx.closePath();
        ctx.fillStyle = css('--body'); ctx.globalAlpha = 0.16; ctx.fill(); ctx.globalAlpha = 1;
        ctx.strokeStyle = css('--body'); ctx.lineWidth = 3.5 * d; ctx.lineJoin = 'round'; ctx.stroke();
        // forward arrow
        const a = P(toW([L * 0.12, 0, 0])), b = P(toW([L * 0.42, 0, 0])), h1 = P(toW([L * 0.34, Wd * 0.14, 0])), h2 = P(toW([L * 0.34, -Wd * 0.14, 0]));
        ctx.beginPath(); ctx.moveTo(a[0], a[1]); ctx.lineTo(b[0], b[1]); ctx.moveTo(h1[0], h1[1]); ctx.lineTo(b[0], b[1]); ctx.lineTo(h2[0], h2[1]);
        ctx.lineWidth = 2 * d; ctx.globalAlpha = 0.55; ctx.stroke(); ctx.globalAlpha = 1;
      } });
      LEGS.forEach((name, i) => {
        const pts = s.points[i].map(toW), proj = pts.map(P), foot = pts[3], sh = P([foot[0], foot[1], 0]), colour = css('--' + name);
        const depth = (proj[0][2] + proj[3][2]) / 2;
        items.push({ depth, draw: () => {
          ctx.fillStyle = css('--muted'); ctx.globalAlpha = 0.35; ctx.beginPath(); ctx.ellipse(sh[0], sh[1], 7 * d, 3 * d, 0, 0, 7); ctx.fill(); ctx.globalAlpha = 1;
          ctx.strokeStyle = colour; ctx.lineWidth = 4.5 * d; ctx.lineCap = 'round'; ctx.lineJoin = 'round';
          ctx.beginPath(); proj.forEach((q, k) => (k ? ctx.lineTo(q[0], q[1]) : ctx.moveTo(q[0], q[1]))); ctx.stroke();
          [proj[1], proj[2]].forEach(q => { ctx.beginPath(); ctx.arc(q[0], q[1], 3 * d, 0, 7); ctx.fillStyle = css('--panel'); ctx.fill(); ctx.lineWidth = 2 * d; ctx.stroke(); });
          ctx.beginPath(); ctx.arc(proj[3][0], proj[3][1], 7 * d, 0, 7);
          ctx.fillStyle = s.down && s.down[i] ? colour : css('--panel'); ctx.fill(); ctx.lineWidth = 2.5 * d; ctx.stroke();
          if (s.bad && s.bad[i]) { ctx.beginPath(); ctx.arc(proj[3][0], proj[3][1], 12 * d, 0, 7); ctx.strokeStyle = css('--orange'); ctx.lineWidth = 2.5 * d; ctx.setLineDash([4 * d, 3 * d]); ctx.stroke(); ctx.setLineDash([]); }
          if (s.labels !== false) { ctx.fillStyle = colour; ctx.font = `600 ${11 * d}px system-ui`; ctx.fillText(name, proj[0][0] + 7 * d, proj[0][1] - 7 * d); }
        } });
      });
      // depth grows toward the camera (project() puts nearer ground lower on screen), so paint the smallest first
      items.sort((a, b) => a.depth - b.depth).forEach(it => it.draw());
      if (s.overlay) s.overlay(ctx, d, W, H);
    }
  }
  window.RobotView = RobotView;
})();
