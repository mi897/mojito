/* On-screen controls for a phone or tablet: two floating thumb sticks over the bottom of the 3D view (the buttons stay in the page's own rows).
 * It only produces stick values in the same sense as a standard gamepad ([left x, left y, right x, right y], down and right are +),
 * which drive.js reads like any other controller (create({touch})). Nothing here talks to the server. */
(function () {
  'use strict';
  const RADIUS = 56;   // px of finger travel for a full stick

  /* Pure part: a finger at (dx, dy) pixels from where the stick was first touched -> stick value, limited to the unit circle. */
  function stickVector(dx, dy, radius = RADIUS) {
    let x = dx / radius, y = dy / radius;
    const n = Math.hypot(x, y);
    if (n > 1) { x /= n; y /= n; }
    return [x, y];
  }

  /* canvas: the 3D view; the pad is laid over its lower half. */
  function create(canvas) {
    const wrap = document.createElement('div');
    wrap.style.cssText = 'position:relative';
    canvas.parentNode.insertBefore(wrap, canvas); wrap.appendChild(canvas);
    const state = { axes: [0, 0, 0, 0], touches: 0, visible: false };
    const style = document.createElement('style');
    style.textContent = `.mtp-zone{position:absolute;bottom:0;width:50%;height:60%;touch-action:none;user-select:none;-webkit-user-select:none}
.mtp-ring,.mtp-knob{position:absolute;border-radius:50%;pointer-events:none;display:none}
.mtp-ring{width:${RADIUS * 2}px;height:${RADIUS * 2}px;margin:-${RADIUS}px 0 0 -${RADIUS}px;border:2px solid var(--muted,#888);opacity:.55}
.mtp-knob{width:44px;height:44px;margin:-22px 0 0 -22px;background:var(--blue,#2a78d6);opacity:.7}
.mtp-hint{position:absolute;bottom:8px;width:50%;text-align:center;font-size:12px;color:var(--muted,#888);pointer-events:none}
`;
    document.head.appendChild(style);
    const root = document.createElement('div'); root.hidden = true; wrap.appendChild(root);

    function zone(side) {   // 0 = left stick, 1 = right stick
      const el = document.createElement('div'); el.className = 'mtp-zone'; el.style[side ? 'right' : 'left'] = '0';
      const ring = document.createElement('div'), knob = document.createElement('div'); ring.className = 'mtp-ring'; knob.className = 'mtp-knob';
      const hint = document.createElement('div'); hint.className = 'mtp-hint'; hint.style[side ? 'right' : 'left'] = '0';
      hint.textContent = side ? 'turn' : 'move';
      el.append(ring, knob); root.append(el, hint);
      let id = null, origin = null;
      const place = (e, node) => { const r = el.getBoundingClientRect(); node.style.left = (e.clientX - r.left) + 'px'; node.style.top = (e.clientY - r.top) + 'px'; };
      el.addEventListener('pointerdown', e => {
        if (id !== null) return;
        id = e.pointerId; origin = [e.clientX, e.clientY]; el.setPointerCapture(id); state.touches++;
        place(e, ring); place(e, knob); ring.style.display = knob.style.display = 'block'; hint.style.display = 'none'; e.preventDefault();
      });
      el.addEventListener('pointermove', e => {
        if (e.pointerId !== id) return;
        const v = stickVector(e.clientX - origin[0], e.clientY - origin[1]);
        state.axes[side * 2] = v[0]; state.axes[side * 2 + 1] = v[1];
        knob.style.left = (parseFloat(ring.style.left) + v[0] * RADIUS) + 'px'; knob.style.top = (parseFloat(ring.style.top) + v[1] * RADIUS) + 'px';
      });
      const up = e => {
        if (e.pointerId !== id) return;
        id = null; state.touches--; state.axes[side * 2] = state.axes[side * 2 + 1] = 0; ring.style.display = knob.style.display = 'none'; hint.style.display = '';
      };
      el.addEventListener('pointerup', up); el.addEventListener('pointercancel', up); el.addEventListener('lostpointercapture', up);
    }
    zone(0); zone(1);
    return {
      axes: () => state.axes.slice(), active: () => state.visible && state.touches > 0,
      setVisible(on) { state.visible = on; root.hidden = !on; if (!on) { state.axes.fill(0); } },
      get visible() { return state.visible; },
    };
  }
  const api = { create, stickVector, RADIUS };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else window.MojitoTouchpad = api;
})();
