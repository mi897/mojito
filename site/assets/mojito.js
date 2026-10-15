/* Mojito kinematics in JavaScript: a port of mojito/robot/leg.py, body.py, gaits.py and statics.py.
 *
 * It uses the closed-form inverse kinematics (the reference the trained network is judged against),
 * not the trained network. tests/website/test_site_js.py checks every function here against the Python.
 * Units: metres and radians. Leg order FL, FR, RL, RR. Leg frame: x forward, y lateral, z up.
 * Body frame: x forward, y left, z up. R = Rz(yaw) Ry(pitch) Rx(roll), body to world.
 */
(function (root) {
  'use strict';

  const NOMINAL = [0.05, 0.15, 0.15];                    // abduction offset, upper, lower: Stage 2 design placeholders
  const STAGE01 = [0.04, 0.12, 0.12];                    // Stage 0-1 placeholders
  const LIMITS = [[-0.6, 0.6], [0.0, 1.5], [-2.3, -0.3]]; // abduction, hip pitch, knee
  const LEG_NAMES = ['FL', 'FR', 'RL', 'RR'];
  const IS_RIGHT = [false, true, false, true];
  const G = 9.81;
  const KGCM = 0.0980665;

  const clip = (x, lo, hi) => Math.min(hi, Math.max(lo, x));
  const add = (a, b) => a.map((v, i) => v + b[i]);
  const sub = (a, b) => a.map((v, i) => v - b[i]);
  const norm = a => Math.hypot(...a);

  /* ---------- one leg ---------- */
  const leg = {
    // Foot position for joint angles q = (abduction a, hip pitch b, knee k), lengths L = (l_a, l_u, l_l).
    forward(q, L) {
      const [a, b, k] = q, [la, lu, ll] = L, bk = b + k;
      const x = -lu * Math.sin(b) - ll * Math.sin(bk);
      const e = lu * Math.cos(b) + ll * Math.cos(bk);          // extension down the leg plane
      return [x, la * Math.cos(a) + e * Math.sin(a), la * Math.sin(a) - e * Math.cos(a)];
    },
    // d(foot)/dq, 3x3, row = x y z, column = joint.
    jacobian(q, L) {
      const [a, b, k] = q, [la, lu, ll] = L, bk = b + k;
      const p = leg.forward(q, L), e = lu * Math.cos(b) + ll * Math.cos(bk);
      const sa = Math.sin(a), ca = Math.cos(a), sl = Math.sin(bk), cl = Math.cos(bk);
      return [[0, -e, -ll * cl],
              [-p[2], p[0] * sa, -ll * sl * sa],
              [p[1], -p[0] * ca, ll * sl * ca]];
    },
    // Shoulder, hip, knee and foot, for drawing.
    jointPositions(q, L) {
      const [a, b] = q, [la, lu] = L, ue = lu * Math.cos(b);
      return [[0, 0, 0], [0, la * Math.cos(a), la * Math.sin(a)],
              [-lu * Math.sin(b), la * Math.cos(a) + ue * Math.sin(a), la * Math.sin(a) - ue * Math.cos(a)],
              leg.forward(q, L)];
    },
    // Closed-form IK. Unreachable targets are clamped to the nearest reachable distance, like leg.analytic_ik.
    analyticIK(p, L) {
      const [x, y, z] = p, [la, lu, ll] = L;
      const e = Math.sqrt(Math.max(y * y + z * z - la * la, 0));
      const a = Math.atan2(z, y) + Math.atan2(e, la);
      const ck = (x * x + e * e - lu * lu - ll * ll) / (2 * lu * ll);
      const k = -Math.acos(clip(ck, -1, 1));
      const b = Math.atan2(-x, e) - Math.atan2(ll * Math.sin(k), lu + ll * Math.cos(k));
      return [a, b, k];
    },
    mirrorTarget: p => [p[0], -p[1], p[2]],
    mirrorAngles: q => [-q[0], q[1], q[2]],
  };

  /* ---------- rotations and posture ---------- */
  function rotation(roll, pitch, yaw) {
    const cr = Math.cos(roll), sr = Math.sin(roll), cp = Math.cos(pitch), sp = Math.sin(pitch), cy = Math.cos(yaw), sy = Math.sin(yaw);
    return [[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr]];
  }
  const matVec = (R, v) => R.map(r => r[0] * v[0] + r[1] * v[1] + r[2] * v[2]);
  const matTVec = (R, v) => [0, 1, 2].map(j => R[0][j] * v[0] + R[1][j] * v[1] + R[2][j] * v[2]);

  // Move the body while the feet stay put: returns the feet in the posed body's frame, R^T (foot - offset).
  function applyPosture(feet, offset, roll, pitch, yaw) {
    const R = rotation(roll, pitch, yaw);
    return feet.map(f => matTVec(R, sub(f, offset)));
  }
  // Body-frame points to the world for a body at `position` with the given attitude.
  function toWorld(points, position, R) {
    return points.map(p => add(matVec(R, p), position));
  }

  /* ---------- the robot: four legs on a body ---------- */
  class Robot {
    constructor(opts = {}) {
      this.lengths = [0, 1, 2, 3].map(() => (opts.lengths || NOMINAL).slice());   // 4 x 3, may be edited at any time
      this.bodyLength = opts.bodyLength ?? 0.30;
      this.bodyWidth = opts.bodyWidth ?? 0.15;
      this.limits = opts.limits || LIMITS;
    }
    get shoulders() {
      const hx = this.bodyLength / 2, hy = this.bodyWidth / 2;
      return [[hx, hy, 0], [hx, -hy, 0], [-hx, hy, 0], [-hx, -hy, 0]];
    }
    stance(height = 0.2125) {
      const s = this.shoulders;
      return s.map((p, i) => [p[0], p[1] + (IS_RIGHT[i] ? -1 : 1) * this.lengths[i][0], -height]);
    }
    toCanonical(feet) {
      const s = this.shoulders;
      return feet.map((f, i) => { const p = sub(f, s[i]); return IS_RIGHT[i] ? leg.mirrorTarget(p) : p; });
    }
    // Joint angles (4 x 3) in each leg's own convention (right legs have the abduction sign flipped).
    solve(feet) {
      return this.toCanonical(feet).map((p, i) => {
        const q = leg.analyticIK(p, this.lengths[i]);
        return IS_RIGHT[i] ? leg.mirrorAngles(q) : q;
      });
    }
    canonicalAngles(q) { return q.map((qi, i) => (IS_RIGHT[i] ? leg.mirrorAngles(qi) : qi)); }
    // Shoulder, hip, knee, foot of every leg in the body frame: 4 x 4 x 3.
    jointPoints(q) {
      const s = this.shoulders, c = this.canonicalAngles(q);
      return c.map((qi, i) => leg.jointPositions(qi, this.lengths[i]).map(p => add(IS_RIGHT[i] ? leg.mirrorTarget(p) : p, s[i])));
    }
    feet(q) { return this.jointPoints(q).map(p => p[3]); }
    // True where a target is inside the joint limits (judged by the exact solution).
    reachable(feet) {
      const t = this.toCanonical(feet);
      return t.map((p, i) => {
        const q = leg.analyticIK(p, this.lengths[i]);
        const hit = norm(sub(leg.forward(q, this.lengths[i]), p)) < 1e-9;
        return hit && q.every((v, j) => v >= this.limits[j][0] - 1e-12 && v <= this.limits[j][1] + 1e-12);
      });
    }
    // Smallest distance in radians from any joint to its limit, per leg.
    limitMargin(q) {
      return this.canonicalAngles(q).map(qi => Math.min(...qi.map((v, j) => Math.min(v - this.limits[j][0], this.limits[j][1] - v))));
    }
  }

  /* ---------- gaits ---------- */
  const TROT = { name: 'trot', period: 0.5, duty: 0.5, offsets: [0, 0.5, 0.5, 0], lift: 0.03, sway: 0 };
  const CRAWL = { name: 'crawl', period: 2.0, duty: 0.85, offsets: [0.75, 0.25, 0.5, 0], lift: 0.03, sway: 0.025 };
  const PACE = { name: 'pace', period: 0.5, duty: 0.5, offsets: [0, 0.5, 0, 0.5], lift: 0.03, sway: 0 };

  const mod1 = x => ((x % 1) + 1) % 1;

  // Heading angle and displacement after `T` seconds of constant planar motion (vx, vy, yaw rate w).
  function planarFlow(T, v, w) {
    const angle = w * T;
    if (Math.abs(w) < 1e-9) return [angle, [T * v[0], T * v[1]]];
    const ic = Math.sin(angle) / w, is = (1 - Math.cos(angle)) / w;
    return [angle, [ic * v[0] - is * v[1], is * v[0] + ic * v[1]]];
  }
  // Body-frame position of a grounded foot that sits at `home` when T = 0.
  function stancePoint(home, T, v, w) {
    const [angle, shift] = planarFlow(T, v, w), c = Math.cos(angle), s = Math.sin(angle);
    return [c * home[0] - s * home[1] + shift[0], s * home[0] + c * home[1] + shift[1], home[2]];
  }
  // Foot targets in the body frame at time t. Returns {feet (4x3), contact (4 bool), swing (4 progress 0..1)}.
  function footTargets(gait, t, home, v = [0, 0], w = 0) {
    const stanceDur = gait.duty * gait.period, feet = [], contact = [], swing = [];
    for (let i = 0; i < 4; i++) {
      const phase = mod1(t / gait.period - gait.offsets[i]);
      if (phase < gait.duty) {
        const progress = phase / gait.duty;
        feet.push(stancePoint(home[i], (0.5 - progress) * stanceDur, v, w));
        contact.push(true); swing.push(0);
      } else {
        const u = (phase - gait.duty) / (1 - gait.duty);
        const a = stancePoint(home[i], -0.5 * stanceDur, v, w), b = stancePoint(home[i], 0.5 * stanceDur, v, w);
        const blend = u - Math.sin(2 * Math.PI * u) / (2 * Math.PI);   // zero horizontal speed at both ends
        const p = a.map((x, k) => x + (b[k] - x) * blend);
        p[2] += gait.lift * Math.sin(Math.PI * u) ** 2;
        feet.push(p); contact.push(false); swing.push(u);
      }
    }
    return { feet, contact, swing };
  }
  // Horizontal body shift (x, y, 0) that leans away from the leg in the air (gaits that lift one foot at a time).
  function bodySway(gait, t, home) {
    const out = [0, 0, 0], swingLen = 1 - gait.duty;
    const starts = gait.offsets.map(o => mod1(o + gait.duty));
    const order = [0, 1, 2, 3].sort((i, j) => starts[i] - starts[j]);
    const gaps = order.map((leg_i, k) => {
      const next = starts[order[(k + 1) % 4]] + (k === 3 ? 1 : 0);
      return next - starts[leg_i] - swingLen;
    });
    if (gait.sway === 0 || gaps.some(g => g < -1e-9)) return out;
    const lean = home.map(h => [-Math.sign(h[0]) * gait.sway, -Math.sign(h[1]) * gait.sway]);
    const phase = mod1(t / gait.period);
    for (let k = 0; k < 4; k++) {
      const li = order[k], nx = order[(k + 1) % 4], since = mod1(phase - starts[li]);
      if (since < swingLen) { out[0] = lean[li][0]; out[1] = lean[li][1]; }
      else if (since < swingLen + gaps[k]) {
        const pr = (since - swingLen) / gaps[k], w = pr * pr * (3 - 2 * pr);   // smoothstep
        out[0] = lean[li][0] * (1 - w) + lean[nx][0] * w; out[1] = lean[li][1] * (1 - w) + lean[nx][1] * w;
      }
    }
    return out;
  }
  // Distance from `point` to the edge of the support polygon of the grounded feet (positive inside).
  function stabilityMargin(feetXY, contact, point) {
    const walk = [0, 1, 3, 2].filter(k => contact[k]).map(k => feetXY[k]);
    if (walk.length < 3) {
      if (!walk.length) return -Infinity;
      const s = walk[0], e = walk[walk.length - 1], d = sub(e, s);
      const f = clip(((point[0] - s[0]) * d[0] + (point[1] - s[1]) * d[1]) / Math.max(d[0] * d[0] + d[1] * d[1], 1e-12), 0, 1);
      return -Math.hypot(point[0] - (s[0] + f * d[0]), point[1] - (s[1] + f * d[1]));
    }
    let m = Infinity;
    for (let j = 0; j < walk.length; j++) {
      const s = walk[j], e = walk[(j + 1) % walk.length], d = sub(e, s);
      const cross = d[0] * (point[1] - s[1]) - d[1] * (point[0] - s[0]);
      m = Math.min(m, -cross / Math.hypot(d[0], d[1]));
    }
    return m;
  }

  /* ---------- statics: foot loads and joint torques ---------- */
  function solve3(A, b) {                       // Gaussian elimination, 3x3
    const M = A.map((r, i) => [...r, b[i]]);
    for (let c = 0; c < 3; c++) {
      let p = c; for (let r = c + 1; r < 3; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = 0; r < 3; r++) if (r !== c) { const f = M[r][c] / M[c][c]; for (let k = c; k < 4; k++) M[r][k] -= f * M[c][k]; }
    }
    return [M[0][3] / M[0][0], M[1][3] / M[1][1], M[2][3] / M[2][2]];
  }
  // Vertical loads on the grounded feet that balance `weight` at com (x, y).
  // 3 feet: exact (statics.foot_loads). 4 feet: the minimum-norm loads that balance, with any foot that would
  // pull on the ground dropped and the rest re-solved. 1-2 feet: the centre of mass projected onto the support line.
  // This is an assumption for the redundant cases; the MuJoCo runs show the real split is not even.
  function footLoads(feetXY, contact, com, weight = 1) {
    const idx = contact.map((c, i) => (c ? i : -1)).filter(i => i >= 0), loads = [0, 0, 0, 0];
    if (!idx.length) return loads;
    if (idx.length <= 2) {
      const a = feetXY[idx[0]], b = feetXY[idx[idx.length - 1]], d = sub(b, a), dd = d[0] * d[0] + d[1] * d[1];
      const f = dd < 1e-12 ? 0 : clip(((com[0] - a[0]) * d[0] + (com[1] - a[1]) * d[1]) / dd, 0, 1);
      loads[idx[0]] = weight * (1 - f); loads[idx[idx.length - 1]] += weight * f;
      return loads;
    }
    let active = idx.slice();
    for (let guard = 0; guard < 4; guard++) {
      const rows = [active.map(() => 1), active.map(i => feetXY[i][0]), active.map(i => feetXY[i][1])];
      const rhs = [weight, weight * com[0], weight * com[1]];
      // f = A^T (A A^T)^-1 rhs  (minimum norm; for 3 feet this is the exact solution)
      const AAT = rows.map(r => rows.map(s => r.reduce((acc, v, k) => acc + v * s[k], 0)));
      const y = solve3(AAT, rhs), f = active.map((_, k) => rows[0][k] * y[0] + rows[1][k] * y[1] + rows[2][k] * y[2]);
      const worst = Math.min(...f);
      if (worst >= -1e-9 || active.length <= 3) { active.forEach((i, k) => { loads[i] = f[k]; }); return loads; }
      active.splice(f.indexOf(worst), 1);
    }
    return loads;
  }
  // |joint torque| (abduction, hip pitch, knee) for a force F on the foot of a canonical (left) leg: |J^T F|.
  function jointTorques(q, L, F) {
    const J = leg.jacobian(q, L);
    return [0, 1, 2].map(j => Math.abs(J[0][j] * F[0] + J[1][j] * F[1] + J[2][j] * F[2]));
  }

  /* ---------- servo ---------- */
  const lerp = (x, x0, y0, x1, y1) => y0 + (y1 - y0) * (x - x0) / (x1 - x0);
  // Listed (unverified) stall torque, N·m: 19 kg·cm at 5 V and 21.5 kg·cm at 6.8 V, straight line between.
  const stallTorque = volts => lerp(volts, 5, 19 * KGCM, 6.8, 21.5 * KGCM);

  /* ---------- one frame of the whole body, for the playground ---------- */
  // opts: robot, gait, t, velocity [vx, vy], yawRate, height, posture {x,y,z,roll,pitch,yaw}, mass (kg), volts
  function evaluate(o) {
    const robot = o.robot, home = robot.stance(o.height);
    let feet, contact, swing, sway = [0, 0, 0];
    if (o.gait) {
      const ft = footTargets(o.gait, o.t, home, o.velocity, o.yawRate);
      feet = ft.feet; contact = ft.contact; swing = ft.swing;
      sway = bodySway(o.gait, o.t, home);
    } else { feet = home; contact = [true, true, true, true]; swing = [0, 0, 0, 0]; }
    const p = o.posture || {};
    const offset = add(sway, [p.x || 0, p.y || 0, p.z || 0]);
    const targets = applyPosture(feet, offset, p.roll || 0, p.pitch || 0, p.yaw || 0);
    const q = robot.solve(targets);
    const points = robot.jointPoints(q);
    const R = rotation(p.roll || 0, p.pitch || 0, p.yaw || 0);
    // Statics: weight of the whole robot on the grounded feet (the body centre is taken as the centre of mass).
    const feetXY = feet.map(f => [f[0] - offset[0], f[1] - offset[1]]);   // feet relative to the body centre, horizontal
    const loadsN = footLoads(feetXY, contact, [0, 0], o.mass * G);
    const gravityInBody = matTVec(R, [0, 0, 1]);
    const canon = robot.canonicalAngles(q), stall = stallTorque(o.volts);
    const torque = canon.map((qi, i) => {
      const F = gravityInBody.map(v => v * loadsN[i]);
      return jointTorques(qi, robot.lengths[i], IS_RIGHT[i] ? leg.mirrorTarget(F) : F);
    });
    const peak = Math.max(...torque.flat());
    const reach = robot.reachable(targets), margin = robot.limitMargin(q);
    return {
      targets, q, points, contact, swing, loadsN, torque, peakTorque: peak, peakFraction: peak / stall, stall,
      reachable: reach, marginDeg: Math.min(...margin) * 180 / Math.PI, R, offset,
      stability: stabilityMargin(feetXY, contact, [0, 0]),
      trackingError: Math.max(...targets.map((t, i) => norm(sub(points[i][3], t)))),
    };
  }

  const api = { NOMINAL, STAGE01, LIMITS, LEG_NAMES, IS_RIGHT, G, leg, rotation, matVec, matTVec, applyPosture, toWorld, Robot,
    TROT, CRAWL, PACE, planarFlow, stancePoint, footTargets, bodySway, stabilityMargin, footLoads, jointTorques,
    stallTorque, evaluate };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.Mojito = api;
})(typeof self !== 'undefined' ? self : this);
