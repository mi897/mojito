"""Live gait explorer: change gait and servo parameters while the physics runs, and watch the robot in a browser.

The physics (MuJoCo, servo model, IK) runs in this Python process; the browser only draws the state and sends
slider changes. Drag with the mouse to rotate the view, wheel to zoom. Needs mujoco. Nothing leaves your machine
(the server listens on 127.0.0.1 only).

    pip install numpy scipy mujoco
    python scripts/sim/live_gait.py             # then open http://127.0.0.1:8765
    python scripts/sim/live_gait.py --port 9000 --preset crawl

Placeholder dimensions, masses, servo figures and friction (docs/STAGE2.md); the model is the one the replays use.
"""
import argparse
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.sim import live_sim  # noqa: E402

PAGE = os.path.join(os.path.dirname(__file__), "live_gait.html")
FRAME_SECONDS = 1.0 / 60.0


class Runner:
    """Owns the simulation and a thread that advances it at `speed` times real time."""

    def __init__(self, sim):
        self.sim = sim
        self.lock = threading.Lock()
        self.paused = False
        self.stop = False
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        last = time.perf_counter()
        owed = 0.0  # simulated seconds still to run
        while not self.stop:
            now = time.perf_counter()
            elapsed, last = min(now - last, 0.1), now  # never try to catch up more than 0.1 s
            with self.lock:
                if not self.paused:
                    owed += elapsed * self.sim.params["speed"]
                    steps = int(owed / self.sim.timestep)
                    if steps:
                        self.sim.step(steps)
                        owed -= steps * self.sim.timestep
            time.sleep(max(0.0, FRAME_SECONDS - (time.perf_counter() - now)))

    def state(self):
        with self.lock:
            snap = self.sim.snapshot()
            snap["paused"] = self.paused
            return snap

    def update(self, changes):
        with self.lock:
            if self.sim.set_params(**{k: v for k, v in changes.items() if k != "paused"}):
                self.sim.reset()  # a new mass or size needs a new model
            if "paused" in changes:
                self.paused = bool(changes["paused"])

    def preset(self, name):
        with self.lock:
            self.sim.set_params(**live_sim.PRESETS[name])

    def reset(self):
        with self.lock:
            self.sim.reset()


def make_handler(runner):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _send(self, body, kind="application/json"):
            data = body if isinstance(body, bytes) else json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/state":
                self._send(runner.state())
            elif self.path in ("/", "/index.html"):
                with open(PAGE, "rb") as f:
                    self._send(f.read(), "text/html; charset=utf-8")
            elif self.path == "/favicon.ico":
                self.send_response(204)
                self.end_headers()
            elif self.path == "/meta":
                self._send({"limits": live_sim.LIMITS, "presets": live_sim.PRESETS, "defaults": live_sim.DEFAULTS})
            else:
                self.send_error(404)

        def do_POST(self):
            length = int(self.headers.get("Content-Length", 0))
            try:
                payload = json.loads(self.rfile.read(length) or b"{}")
            except ValueError:
                return self.send_error(400)
            if self.path == "/params":
                runner.update(payload)
            elif self.path == "/preset" and payload.get("name") in live_sim.PRESETS:
                runner.preset(payload["name"])
            elif self.path == "/reset":
                runner.reset()
            else:
                return self.send_error(404)
            self._send({"ok": True})

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--preset", choices=sorted(live_sim.PRESETS), default="trot")
    parser.add_argument("--walk", action="store_true", help="start walking at once instead of standing")
    args = parser.parse_args()
    sim = live_sim.LiveSim(dict(live_sim.PRESETS[args.preset], walk=args.walk))
    runner = Runner(sim)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(runner))
    print(f"open http://127.0.0.1:{args.port}   (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop = True


if __name__ == "__main__":
    main()
