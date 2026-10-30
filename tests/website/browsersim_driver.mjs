// Driver for tests/website/test_browsersim.py: reads one JSON request on stdin, answers one JSON line on stdout.
//   {op:'xml', design}                  -> {xml}
//   {op:'servo', params, steps}         -> {torques:[[..]], saturated:[..]}
//   {op:'run', params, schedule}        -> {snapshots:[..]}   schedule: [{changes?, steps}], one snapshot after each entry
// MOJITO_MUJOCO_DIR is the folder of the installed @mujoco/mujoco package.
import { createRequire } from 'module';
import { pathToFileURL } from 'url';
import fs from 'fs';
const require = createRequire(import.meta.url);
const here = new URL('.', import.meta.url).pathname;
const B = require(here + '../../site/assets/browsersim.js');
const live = JSON.parse(fs.readFileSync(here + '../../site/data/livemeta.js', 'utf8').replace(/^window\.MOJITO_LIVE=/, '').replace(/;\s*$/, ''));
const request = JSON.parse(fs.readFileSync(0, 'utf8'));
if (request.op === 'xml') {
  console.log(JSON.stringify({ xml: B.mjcfQuadruped(live.constants, request.design) }));
} else if (request.op === 'servo') {
  const servo = new B.Servo(Object.assign({}, live.constants.servo, request.params));
  const torques = [], saturated = [];
  for (const s of request.steps) { torques.push(servo.step(s.targets, s.angles, s.velocities, s.dt)); saturated.push(servo.lastSaturated); }
  console.log(JSON.stringify({ torques, saturated }));
} else {
  const mujoco = await (await import(pathToFileURL(process.env.MOJITO_MUJOCO_DIR + '/mujoco.js').href)).default();
  const sim = new B.BrowserSim(mujoco, live, request.params);
  const snapshots = [];
  for (const entry of request.schedule) {
    if (entry.changes && sim.setParams(entry.changes)) sim.reset();
    sim.step(entry.steps);
    snapshots.push(sim.snapshot());
  }
  console.log(JSON.stringify({ snapshots }));
}
