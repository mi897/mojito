/* The robot on the overview page: the trot from mojito.js, running live. Dimensions, speeds and mass are the
 * Python placeholders, from data/constants.js (loaded before mojito.js). */
(function () {
  'use strict';
  const M = window.Mojito;
  document.addEventListener('DOMContentLoaded', () => {
    const canvas = document.getElementById('hero-view');
    if (!canvas) return;
    if (!M) {   // data/constants.js missing, so mojito.js did not load
      const note = document.createElement('p'); note.className = 'w small'; note.textContent = 'site data missing: run scripts/website/build_site.py';
      canvas.replaceWith(note); return;
    }
    const view = new RobotView(canvas, { az: -40, el: 22, zoom: 1.15 }), robot = new M.Robot(), reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    let t = 0, last = performance.now(), mode = 'trot';
    const gaits = { trot: [M.TROT, [M.GAIT_SPEEDS.trot, 0]], crawl: [M.CRAWL, [M.GAIT_SPEEDS.crawl, 0]] }, height = M.DESIGN.stance_height;
    function frame(now) {
      t += Math.min(0.1, (now - last) / 1000); last = now;
      const [gait, v] = gaits[mode], r = M.evaluate({ robot, gait, t, velocity: v, yawRate: 0, height, mass: M.DESIGN.total_mass, volts: M.SERVO.voltage });
      const [, shift] = M.planarFlow(t, v, 0), off = r.offset;
      view.setScene({ origin: [shift[0] + off[0], shift[1] + off[1], height + off[2]], R: r.R, bodySize: [robot.bodyLength, robot.bodyWidth], points: r.points, down: r.loadsN.map(l => l > 1e-6), bad: [0, 0, 0, 0] });
      if (!reduce) requestAnimationFrame(frame);
    }
    requestAnimationFrame(frame);
    document.querySelectorAll('[data-hero-gait]').forEach(b => b.onclick = () => {
      mode = b.dataset.heroGait; t = 0; document.querySelectorAll('[data-hero-gait]').forEach(x => x.classList.toggle('on', x === b));
      if (reduce) requestAnimationFrame(frame);
    });
  });
})();
