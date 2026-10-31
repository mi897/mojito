"""The website's JavaScript maths (site/assets/mojito.js) against the Python it is a port of.

Every case is drawn at random in Python, evaluated by the Python modules and, through Node, by the JavaScript,
and the two must agree. Skipped when Node is not installed.
"""
import dataclasses
import json
import os
import shutil
import subprocess
import unittest

import numpy as np

from mojito.robot import body, gaits, leg, statics
from mojito.sim import actuators

NODE = shutil.which("node")
HERE = os.path.dirname(__file__)
GAIT_BY_NAME = {"trot": gaits.TROT, "crawl": gaits.CRAWL}


def _lengths(rng, n):
    return leg.NOMINAL_LENGTHS * (1 + rng.uniform(-0.2, 0.2, size=(n, 3)))


@unittest.skipIf(NODE is None, "node is not installed")
class TestSiteJavaScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(7)
        cases = {"leg": [], "robot": [], "posture": [], "gait": [], "margin": [], "loads": [], "torque": [], "stall": []}
        py = {"leg": [], "robot": [], "posture": [], "rotation": [], "gait": [], "margin": [], "loads": [], "torque": [], "stall": []}
        limits = leg.JOINT_LIMITS
        for _ in range(40):
            L = _lengths(rng, 1)[0]
            q = rng.uniform(limits[:, 0], limits[:, 1])
            p = rng.uniform([-0.25, -0.3, -0.35], [0.25, 0.3, 0.05])  # some inside the workspace, some not
            cases["leg"].append({"q": q.tolist(), "L": L.tolist(), "p": p.tolist()})
            py["leg"].append({"forward": leg.forward(q, L), "jacobian": leg.jacobian(q, L),
                              "points": leg.joint_positions(q, L), "ik": leg.analytic_ik(p, L)})
        for _ in range(20):
            lengths = _lengths(rng, 4)
            body_size = rng.uniform([0.2, 0.1], [0.4, 0.2])
            robot = body.Robot(body.AnalyticSolver(), lengths=lengths, body_length=body_size[0], body_width=body_size[1])
            height = float(rng.uniform(0.15, 0.25))
            feet = robot.stance(height) + rng.uniform(-0.03, 0.03, size=(4, 3))
            q = np.stack([rng.uniform(limits[:, 0], limits[:, 1]) for _ in range(4)])
            cases["robot"].append({"lengths": lengths.tolist(), "body": body_size.tolist(), "feet": feet.tolist(),
                                   "q": q.tolist(), "height": height})
            py["robot"].append({"q": robot.solve(feet), "points": robot.joint_points(q), "reach": robot.reachable(feet),
                                "margin": robot.limit_margin(q), "stance": robot.stance(height)})
        for _ in range(10):
            feet = rng.uniform(-0.2, 0.2, size=(4, 3))
            offset, (roll, pitch, yaw) = rng.uniform(-0.03, 0.03, 3), rng.uniform(-0.4, 0.4, 3)
            cases["posture"].append({"feet": feet.tolist(), "offset": offset.tolist(), "roll": roll, "pitch": pitch, "yaw": yaw})
            py["posture"].append(body.apply_posture(feet, offset, roll, pitch, yaw))
            py["rotation"].append(body.rotation(roll, pitch, yaw))
        robot = body.Robot(body.AnalyticSolver())
        home = robot.stance(0.2125)
        for name in ("trot", "crawl", "trot", "crawl", "trot", "crawl"):
            for _ in range(8):
                gait = dataclasses.replace(GAIT_BY_NAME[name], period=float(rng.uniform(0.3, 2.5)), duty=float(rng.uniform(0.3, 0.9)),
                                           lift=float(rng.uniform(0, 0.08)))
                if name == "crawl" and rng.random() < 0.5:
                    gait = dataclasses.replace(gait, duty=0.85, period=2.0)  # the one-at-a-time timing, where the sway acts
                t, v, w = float(rng.uniform(0, 5)), rng.uniform(-0.3, 0.3, 2), float(rng.choice([0.0, rng.uniform(-2, 2)]))
                feet, contact, swing = gaits.foot_targets(gait, t, home, v, w)
                py["gait"].append({"feet": feet[0], "contact": contact[0], "swing": swing[0],
                                   "sway": gaits.body_sway(gait, t, home)[0]})
                cases["gait"].append({"name": name, "t": t, "v": v.tolist(), "w": w, "home": home.tolist(),
                                      "gait": {"period": gait.period, "duty": gait.duty, "lift": gait.lift, "sway": gait.sway,
                                               "offsets": list(gait.offsets)}})
        for _ in range(30):
            feet = rng.uniform(-0.2, 0.2, size=(1, 4, 2))
            contact = rng.random(4) < 0.8
            contact[:2] = True
            point = rng.uniform(-0.05, 0.05, size=(1, 2))
            cases["margin"].append({"feet": feet[0].tolist(), "contact": contact.tolist(), "point": point[0].tolist()})
            py["margin"].append(gaits.stability_margin(feet, contact[None], point)[0])
        # feet on the same spot (a zero-length polygon edge): two of three, and two of four
        for contact in ([True, True, True, False], [True, True, True, True]):
            feet = rng.uniform(-0.2, 0.2, size=(1, 4, 2))
            feet[0, 1] = feet[0, 0]
            point = rng.uniform(-0.05, 0.05, size=(1, 2))
            cases["margin"].append({"feet": feet[0].tolist(), "contact": contact, "point": point[0].tolist()})
            py["margin"].append(gaits.stability_margin(feet, np.array([contact]), point)[0])
        for _ in range(20):
            feet = rng.uniform(-0.2, 0.2, size=(3, 2))
            com, weight = rng.uniform(-0.05, 0.05, 2), float(rng.uniform(5, 30))
            full = np.zeros((4, 2))
            full[[0, 1, 3]] = feet
            cases["loads"].append({"feet": full.tolist(), "contact": [True, True, False, True], "com": com.tolist(), "weight": weight})
            py["loads"].append(statics.foot_loads(feet, com, weight))
        for _ in range(20):
            L = _lengths(rng, 1)[0]
            q = rng.uniform(limits[:, 0], limits[:, 1])
            load = float(rng.uniform(0, 20))
            cases["torque"].append({"q": q.tolist(), "L": L.tolist(), "load": load})
            py["torque"].append(statics.joint_torques(q, L, load))
        for v in (5.0, 5.4, 6.0):
            cases["stall"].append(v)
            py["stall"].append(actuators.ServoParams(voltage=v).stall_torque)
        run = subprocess.run([NODE, os.path.join(HERE, "site_js_check.js")], input=json.dumps(cases), capture_output=True,
                             text=True, check=True)
        cls.js, cls.py, cls.cases = json.loads(run.stdout), py, cases

    def close(self, a, b, atol=1e-9):
        np.testing.assert_allclose(np.asarray(a, float), np.asarray(b, float), atol=atol, rtol=1e-9)

    def test_leg(self):
        for js, py in zip(self.js["leg"], self.py["leg"]):
            for key, js_key in (("forward", "forward"), ("jacobian", "jacobian"), ("points", "points"), ("ik", "ik")):
                self.close(js[js_key], py[key])

    def test_robot(self):
        for js, py in zip(self.js["robot"], self.py["robot"]):
            self.close(js["q"], py["q"])
            self.close(js["points"], py["points"])
            self.close(js["margin"], py["margin"])
            self.close(js["stance"], py["stance"])
            self.assertEqual(js["reach"], py["reach"].tolist())

    def test_posture_and_rotation(self):
        for js, py in zip(self.js["posture"], self.py["posture"]):
            self.close(js, py)
        for js, py in zip(self.js["rotation"], self.py["rotation"]):
            self.close(js, py)

    def test_gait_and_sway(self):
        swayed = 0
        for js, py in zip(self.js["gait"], self.py["gait"]):
            self.close(js["feet"], py["feet"])
            self.assertEqual(js["contact"], py["contact"].tolist())
            self.close(js["swing"], py["swing"])
            self.close(js["sway"], py["sway"])
            swayed += bool(np.any(np.asarray(py["sway"]) != 0))
        self.assertGreater(swayed, 3, "the cases must include the crawl lean, or its port is untested")

    def test_stability_margin(self):
        self.close(self.js["margin"], self.py["margin"])

    def test_three_foot_loads_equal_the_python_solution(self):
        for js, py, case in zip(self.js["loads"], self.py["loads"], self.cases["loads"]):
            self.close(np.asarray(js)[[0, 1, 3]], py, atol=1e-8)
            self.assertEqual(js[2], 0.0)

    def test_joint_torques_and_stall(self):
        self.close(self.js["torque"], self.py["torque"])
        self.close(self.js["stall"], self.py["stall"])


@unittest.skipIf(NODE is None, "node is not installed")
class TestSiteStatics(unittest.TestCase):
    """The four-foot load split is an assumption, so check what must hold of any answer: equilibrium and no pulling."""

    def test_four_foot_loads_balance_the_weight(self):
        script = """const M=require('../../site/assets/mojito.js');
        const feet=[[.15,.075],[.15,-.075],[-.15,.075],[-.15,-.075]];
        const out=[[0,0],[0.03,0.01],[0.2,0],[0,0.02]].map(c=>M.footLoads(feet,[true,true,true,true],c,10));
        console.log(JSON.stringify(out));"""
        run = subprocess.run([NODE, "-e", script], cwd=HERE, capture_output=True, text=True, check=True)
        feet = np.array([[.15, .075], [.15, -.075], [-.15, .075], [-.15, -.075]])
        for com, loads in zip([[0, 0], [0.03, 0.01], [0.2, 0], [0, 0.02]], json.loads(run.stdout)):
            loads = np.asarray(loads)
            self.assertAlmostEqual(loads.sum(), 10.0, places=9)
            if abs(com[0]) < 0.15:  # inside the support: no foot pulls, and all three equilibrium equations hold
                self.assertTrue(np.all(loads >= -1e-9))
                np.testing.assert_allclose(loads @ feet / 10.0, com, atol=1e-9)
            else:  # outside it the robot tips, and the loads say so with a negative entry
                self.assertTrue(np.any(loads < 0))



@unittest.skipIf(NODE is None, "node is not installed")
class TestSteering(unittest.TestCase):
    """The pose integrator and the command easing that the kinematic lab uses when it is steered live, against their closed forms."""

    def js(self, expression):
        script = "const M=require('../../site/assets/mojito.js');console.log(JSON.stringify(%s));" % expression
        return json.loads(subprocess.run([NODE, "-e", script], cwd=HERE, capture_output=True, text=True, check=True).stdout)

    def test_advance_pose_follows_the_closed_form_arc(self):
        # a constant command from rest: stepping the pose must land where planarFlow (the exact solution) says
        for velocity, rate in (([0.2, 0.0], 0.0), ([0.25, 0.1], 0.7), ([-0.1, 0.12], -1.3), ([0.0, 0.0], 0.9)):
            out = self.js("""(()=>{let p={x:0,y:0,yaw:0};const v=%s,w=%s,T=6,dt=1/120;for(let i=0;i<T/dt;i++)p=M.advancePose(p,v,w,dt);
                const [angle,shift]=M.planarFlow(T,v,w);return {p,angle,shift};})()""" % (json.dumps(velocity), rate))
            self.assertAlmostEqual(out["p"]["yaw"], out["angle"], places=9)
            self.assertAlmostEqual(out["p"]["x"], out["shift"][0], delta=2e-3)
            self.assertAlmostEqual(out["p"]["y"], out["shift"][1], delta=2e-3)

    def test_changing_the_command_does_not_move_the_body(self):
        # the old form, planarFlow(t, v) with an unbounded t, put the body at t * v: a new v at t = 100 s moved it by metres at once
        out = self.js("""(()=>{let p={x:0,y:0,yaw:0};for(let i=0;i<60000;i++)p=M.advancePose(p,[0.25,0],0,1/600);   // 100 s at 0.25 m/s
            const stopped=M.advancePose(p,[0,0],0,1/60);return {x:p.x,dx:Math.abs(stopped.x-p.x)};})()""")
        self.assertAlmostEqual(out["x"], 25.0, delta=0.01)
        self.assertEqual(out["dx"], 0)

    def test_ease_to_is_the_exponential_approach(self):
        out = self.js("""(()=>{let v=0.2;const target=-0.1,tau=0.3,dt=1/60;for(let i=0;i<60;i++)v=M.easeTo(v,target,dt,tau);   // 1 s
            return {v,jump:M.easeTo(5,2,0.1,0)};})()""")
        self.assertAlmostEqual(out["v"], -0.1 + 0.3 * np.exp(-1.0 / 0.3), places=9)   # target + (start - target) exp(-t / tau)
        self.assertEqual(out["jump"], 2)   # tau = 0 jumps


if __name__ == "__main__":
    unittest.main()
