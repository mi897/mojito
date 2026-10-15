"""The local site server: static files, the physics API under /api, and its local-only guards."""
import http.client
import json
import os
import sys
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "website"))
import serve_site  # noqa: E402
from mojito.sim import live_sim  # noqa: E402

try:
    import mujoco  # noqa: F401
except ImportError:  # pragma: no cover
    mujoco = None


class FakeRunner:
    """Just enough of live_gait.Runner for the server tests that do not need physics."""

    def __init__(self):
        self.seen = []

    def state(self):
        return {"time": 0.0}

    def update(self, changes):
        self.seen.append(changes)

    def preset(self, name):
        self.seen.append(name)

    def reset(self):
        self.seen.append("reset")


class TestGuards(unittest.TestCase):
    def test_host_and_origin_checks(self):
        for host in ("127.0.0.1:8000", "localhost:8000", "localhost", "[::1]:8000", "LOCALHOST:1"):
            self.assertTrue(serve_site.host_is_local(host), host)
        for host in ("evil.example", "evil.example:8000", "127.0.0.1.evil.example", "", None, "10.0.0.5:8000"):
            self.assertFalse(serve_site.host_is_local(host), host)
        self.assertTrue(serve_site.origin_is_local(None))
        self.assertTrue(serve_site.origin_is_local("http://localhost:8000"))
        self.assertFalse(serve_site.origin_is_local("https://evil.example"))


class TestServer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.dir = tempfile.TemporaryDirectory()
        os.makedirs(os.path.join(cls.dir.name, "data"))
        with open(os.path.join(cls.dir.name, "index.html"), "w") as f:
            f.write("<title>hi</title>")
        with open(os.path.join(cls.dir.name, "data", "x.js"), "w") as f:
            f.write("window.X=1;")
        with open(os.path.join(os.path.dirname(cls.dir.name), "secret.txt"), "w") as f:
            f.write("outside the site")
        cls.runner = FakeRunner()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), serve_site.make_site_handler(cls.runner, cls.dir.name))
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        os.remove(os.path.join(os.path.dirname(cls.dir.name), "secret.txt"))
        cls.dir.cleanup()

    def get(self, path, headers=None, method="GET", body=None):
        conn = http.client.HTTPConnection("127.0.0.1", self.port)
        conn.request(method, path, body=body, headers=headers or {})
        response = conn.getresponse()
        return response.status, response.read(), response.getheader("Content-Type")

    def test_serves_static_files_and_the_index(self):
        status, body, kind = self.get("/")
        self.assertEqual((status, body), (200, b"<title>hi</title>"))
        self.assertTrue(kind.startswith("text/html"))
        status, body, kind = self.get("/data/x.js")
        self.assertEqual((status, body), (200, b"window.X=1;"))
        self.assertIn("javascript", kind)

    def test_cannot_read_outside_the_site_directory(self):
        for path in ("/../secret.txt", "/%2e%2e/secret.txt", "/data/../../secret.txt", "//etc/passwd"):
            status, body, _ = self.get(path)
            self.assertNotEqual(status, 200, path)
            self.assertNotIn(b"outside the site", body, path)

    def test_api_is_under_slash_api(self):
        status, body, _ = self.get("/api/state")
        self.assertEqual((status, json.loads(body)), (200, {"time": 0.0}))
        status, _, _ = self.get("/state")  # the bare path is not the API on this server
        self.assertEqual(status, 404)

    def test_post_reaches_the_runner(self):
        body = json.dumps({"kp": 50})
        status, _, _ = self.get("/api/params", method="POST", body=body, headers={"Content-Length": str(len(body))})
        self.assertEqual(status, 200)
        self.assertEqual(self.runner.seen[-1], {"kp": 50})

    def test_refuses_foreign_host_and_foreign_origin(self):
        self.assertEqual(self.get("/", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.get("/api/state", headers={"Host": "evil.example:80"})[0], 403)
        before = len(self.runner.seen)
        status, _, _ = self.get("/api/params", method="POST", body="{}", headers={"Origin": "https://evil.example", "Content-Length": "2"})
        self.assertEqual(status, 403)
        self.assertEqual(len(self.runner.seen), before)


@unittest.skipIf(mujoco is None, "mujoco not installed")
class TestRobotDesignIsLive(unittest.TestCase):
    def test_longer_legs_lower_the_stand_and_ask_for_a_rebuild(self):
        sim = live_sim.LiveSim()
        self.assertTrue(sim.set_params(lu=0.2, ll=0.2, height=0.3))
        sim.reset()
        self.assertTrue(sim.snapshot()["stance_reachable"])
        np.testing.assert_allclose(sim.robot.lengths[0], [0.05, 0.2, 0.2])
        self.assertAlmostEqual(sim.snapshot()["body_size"][0], 0.30)
        sim.set_params(body_length=0.4, body_width=0.2)
        sim.reset()
        self.assertEqual(sim.snapshot()["body_size"], [0.4, 0.2])

    def test_a_stance_the_legs_cannot_reach_is_reported(self):
        sim = live_sim.LiveSim({"height": 0.32, "lu": 0.08, "ll": 0.08})
        self.assertFalse(sim.snapshot()["stance_reachable"])

    def test_supply_voltage_changes_the_stall_torque_live(self):
        sim = live_sim.LiveSim()
        low = sim.snapshot()["stall_torque_nm"]
        self.assertFalse(sim.set_params(voltage=6.0))  # no rebuild needed
        self.assertGreater(sim.servo.params.stall_torque, low)

    def test_the_defaults_still_stand_on_the_default_design(self):
        sim = live_sim.LiveSim()
        sim.step(500)
        snap = sim.snapshot()
        self.assertFalse(snap["fell"])
        self.assertTrue(snap["stance_reachable"])


if __name__ == "__main__":
    unittest.main()
