// Runs the website's JavaScript maths on the cases in a JSON file (stdin) and prints the results as JSON.
// Used by tests/test_site_js.py, which compares them with the Python the port is meant to match.
const M = require('../site/assets/mojito.js');
const input = JSON.parse(require('fs').readFileSync(0, 'utf8'));
const robotFrom = c => { const r = new M.Robot({ bodyLength: c.body[0], bodyWidth: c.body[1] }); r.lengths = c.lengths; return r; };
const gaitOf = (name, g) => Object.assign({}, M[name.toUpperCase()], g);
const out = {};
out.leg = input.leg.map(c => ({
  forward: M.leg.forward(c.q, c.L), jacobian: M.leg.jacobian(c.q, c.L), points: M.leg.jointPositions(c.q, c.L),
  ik: M.leg.analyticIK(c.p, c.L),
}));
out.robot = input.robot.map(c => {
  const r = robotFrom(c), q = r.solve(c.feet), pts = r.jointPoints(c.q);
  return { q, points: pts, reach: r.reachable(c.feet), margin: r.limitMargin(c.q), stance: r.stance(c.height) };
});
out.posture = input.posture.map(c => M.applyPosture(c.feet, c.offset, c.roll, c.pitch, c.yaw));
out.rotation = input.posture.map(c => M.rotation(c.roll, c.pitch, c.yaw));
out.gait = input.gait.map(c => {
  const g = gaitOf(c.name, c.gait), ft = M.footTargets(g, c.t, c.home, c.v, c.w);
  return { feet: ft.feet, contact: ft.contact, swing: ft.swing, sway: M.bodySway(g, c.t, c.home) };
});
out.margin = input.margin.map(c => M.stabilityMargin(c.feet, c.contact, c.point));
out.loads = input.loads.map(c => M.footLoads(c.feet, c.contact, c.com, c.weight));
out.torque = input.torque.map(c => M.jointTorques(c.q, c.L, [0, 0, c.load]));
out.stall = input.stall.map(v => M.stallTorque(v));
process.stdout.write(JSON.stringify(out));
