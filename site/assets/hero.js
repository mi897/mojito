/* The robot on the overview page: the trot from mojito.js, running live. */
(function () {
  'use strict';
  const M = window.Mojito;
  document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('hero-view');
    if (!canvas) return;
    const view = new RobotView(canvas, { az: -40, el: 22, zoom: 1.15 }), robot = new M.Robot(), reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let t = 0, last = performance.now(), mode = 'trot';
    const gaits = { trot: [M.TROT, [0.15, 0]], crawl: [M.CRAWL, [0.04, 0]] };
    function frame(now) {
      t += Math.min(0.1, (now - last) / 1000); last = now;
      const [gait, v] = gaits[mode], r = M.evaluate({ robot, gait, t, velocity: v, yawRate: 0, height: 0.2125, mass: 1.5, volts: 5 });
      const [, shift] = M.planarFlow(t, v, 0), off = r.offset;
      view.setScene({ origin: [shift[0] + off[0], shift[1] + off[1], 0.2125 + off[2]], R: r.R, bodySize: [robot.bodyLength, robot.bodyWidth], points: r.points, down: r.loadsN.map(l => l > 1e-6), bad: [0, 0, 0, 0] });
      if (!reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    document.querySelectorAll('[data-hero-gait]').forEach(b => b.onclick = () => {
      mode = b.dataset.heroGait; t = 0; document.querySelectorAll('[data-hero-gait]').forEach(x => x.classList.toggle('on', x === b));
      if (reduce) requestAnimationFrame(frame);
    });
  });
})();
