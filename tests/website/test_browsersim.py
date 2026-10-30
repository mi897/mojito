"""The in-browser simulation (site/assets/browsersim.js, MuJoCo WebAssembly) against the Python it is a port of.

Needs node, `python scripts/website/build_site.py` (it writes site/data/livemeta.js) and the @mujoco/mujoco package: skipped if any is
missing. Install the package with `npm install --no-save @mujoco/mujoco@3.16.0` in the repository root (the version of the MuJoCo that the
Python tests run), or point MOJITO_MUJOCO_DIR at an installed copy. Run with:  python -m unittest discover tests
"""
import json
import os
import shutil
import subprocess
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
HERE = os.path.dirname(os.path.abspath(__file__))
NODE = shutil.which("node")
MUJOCO_DIR = os.environ.get("MOJITO_MUJOCO_DIR") or os.path.join(ROOT, "node_modules", "@mujoco", "mujoco")
LIVE_META = os.path.join(ROOT, "site", "data", "livemeta.js")

try:
    import mujoco
except ImportError:
    mujoco = None

READY = NODE is not None and mujoco is not None and os.path.isfile(os.path.join(MUJOCO_DIR, "mujoco.js")) and os.path.isfile(LIVE_META)


def node(request):
    run = subprocess.run([NODE, os.path.join(HERE, "browsersim_driver.mjs")], input=json.dumps(request), capture_output=True, text=True,
                         env=dict(os.environ, MOJITO_MUJOCO_DIR=MUJOCO_DIR), check=True)
    return json.loads(run.stdout)


@unittest.skipUnless(READY, "needs node, mujoco, site/data/livemeta.js (run build_site.py) and the @mujoco/mujoco npm package")
class TestBrowserSim(unittest.TestCase):
    def test_constants_come_from_python(self):
        from mojito.sim import live_sim

        with open(LIVE_META) as f:
            live = json.loads(f.read().replace("window.MOJITO_LIVE=", "", 1).rstrip().rstrip(";"))
        self.assertEqual(live, json.loads(json.dumps(live_sim.browser_constants())))   # the generated file is current

    def test_model_matches_the_python_model(self):
        # a design that is not the default one, so a copied constant would show
        from mojito.sim import live_sim, stand

        lengths, height, length, width, mass = (0.06, 0.13, 0.17), 0.2, 0.35, 0.18, 2.5
        masses = live_sim.scaled_masses(mass)
        design = dict(la=lengths[0], lu=lengths[1], ll=lengths[2], body_length=length, body_width=width, start_z=height + 0.012,
                      masses=masses, timestep=0.002)
        theirs = mujoco.MjModel.from_xml_string(node({"op": "xml", "design": design})["xml"])
        _, ours, *_ = stand._open(np.array(lengths), height, length, width, masses, 0.002)
        for name in ("nq", "nv", "nu", "nbody", "ngeom", "njnt", "nsite", "nsensor"):
            self.assertEqual(getattr(theirs, name), getattr(ours, name), name)
        for name in ("body_pos", "body_mass", "body_inertia", "geom_size", "geom_pos", "geom_friction", "geom_contype", "jnt_range",
                     "jnt_axis", "jnt_pos", "dof_damping", "dof_armature", "actuator_gear", "site_pos", "body_subtreemass"):
            np.testing.assert_allclose(getattr(theirs, name), getattr(ours, name), rtol=1e-7, atol=1e-12, err_msg=name)
        np.testing.assert_allclose(theirs.opt.gravity, ours.opt.gravity)
        self.assertEqual(theirs.opt.timestep, ours.opt.timestep)
        names = lambda m, t, n: [mujoco.mj_id2name(m, t, i) for i in range(n)]
        for kind, count in ((mujoco.mjtObj.mjOBJ_JOINT, ours.njnt), (mujoco.mjtObj.mjOBJ_BODY, ours.nbody), (mujoco.mjtObj.mjOBJ_ACTUATOR, ours.nu)):
            self.assertEqual(names(theirs, kind, count), names(ours, kind, count))

    def test_servo_matches_the_python_servo(self):
        from mojito.sim import actuators

        rng = np.random.default_rng(3)
        params = dict(kp=50.0, kd=0.9, voltage=5.6)
        # 120 steps pass the 20 ms latency; the first 8 are slow so that the servo is not saturated, the rest are wild so that it is
        steps = [dict(targets=rng.uniform(-1.5, 1.5, 12).tolist(), angles=rng.uniform(-1.5, 1.5, 12).tolist(),
                      velocities=rng.uniform(-8 if k >= 8 else -0.1, 8 if k >= 8 else 0.1, 12).tolist(), dt=0.002) for k in range(120)]
        theirs = node({"op": "servo", "params": params, "steps": steps})
        servo = actuators.Servo(actuators.ServoParams(**params))
        servo.reset()
        for k, s in enumerate(steps):
            torque = servo.step(s["targets"], s["angles"], s["velocities"], s["dt"])
            np.testing.assert_allclose(theirs["torques"][k], torque, rtol=1e-9, atol=1e-12, err_msg=f"step {k}")
            self.assertEqual(theirs["saturated"][k], bool(servo.last_saturated.any()), f"step {k}")
        self.assertTrue(any(theirs["saturated"]) and not all(theirs["saturated"]))   # the test did cover both cases

    def run_both(self, preset, params, seconds):
        from mojito.sim import live_sim

        start = dict(live_sim.PRESETS[preset], **params)
        theirs = node({"op": "run", "params": start, "schedule": [{"steps": int(seconds / 0.002)}]})["snapshots"][0]
        sim = live_sim.LiveSim(start)
        sim.step(int(seconds / 0.002))
        return theirs, sim.snapshot()

    def assert_close(self, theirs, ours, position=2e-3, angle=0.3):
        np.testing.assert_allclose(theirs["origin"], ours["origin"], atol=position)
        np.testing.assert_allclose(np.array(theirs["legs"]), np.array(ours["legs"]), atol=position)
        for key in ("roll_deg", "pitch_deg", "yaw_deg", "tilt_deg"):
            self.assertAlmostEqual(theirs[key], ours[key], delta=angle, msg=key)
        np.testing.assert_allclose(theirs["measured_velocity"], ours["measured_velocity"], atol=0.01)
        self.assertEqual(theirs["feet_down"], ours["feet_down"])
        self.assertEqual(theirs["fell"], ours["fell"])

    def test_standing_matches(self):
        theirs, ours = self.run_both("trot", dict(walk=False), 2.0)
        self.assert_close(theirs, ours, position=1e-4, angle=0.05)
        self.assertEqual(set(theirs), set(ours))   # the same keys, so the page cannot tell them apart

    def test_trot_matches(self):
        theirs, ours = self.run_both("trot", dict(walk=True, vx=0.25, vy=0.05, yaw_rate=0.3), 3.0)
        self.assert_close(theirs, ours)

    def test_crawl_matches(self):
        theirs, ours = self.run_both("crawl", dict(walk=True, vx=0.05), 3.0)
        self.assert_close(theirs, ours)

    def test_pose_matches(self):
        theirs, ours = self.run_both("trot", dict(walk=False, body_roll=0.15, body_pitch=-0.15, body_yaw=0.2, body_height=0.02), 2.5)
        self.assert_close(theirs, ours, position=1e-3, angle=0.1)

    def test_changing_design_and_params_while_running_matches(self):
        from mojito.sim import live_sim

        schedule = [dict(steps=500), dict(changes=dict(walk=True, vx=0.2), steps=750), dict(changes=dict(mass=2.4, lu=0.13), steps=500),
                    dict(changes=dict(kp=30.0, lift=0.03), steps=500)]
        theirs = node({"op": "run", "params": live_sim.PRESETS["trot"], "schedule": schedule})["snapshots"][-1]
        sim = live_sim.LiveSim(live_sim.PRESETS["trot"])
        for entry in schedule:
            if entry.get("changes") and sim.set_params(**entry["changes"]):
                sim.reset()
            sim.step(entry["steps"])
        self.assert_close(theirs, sim.snapshot())
        self.assertAlmostEqual(theirs["total_mass_kg"], 2.4, places=3)


if __name__ == "__main__":
    unittest.main()
