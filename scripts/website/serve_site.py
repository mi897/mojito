"""Serve the website on this machine with live physics behind its playground (needs mujoco). Local only.

The static pages are served exactly as published; in addition the playground's "Live physics" tab talks to the
MuJoCo simulation running in this process (mojito/sim/live_sim.py, the same model the replays use), so any robot
dimension, gait or servo setting can be tried in real physics, not only the 144 stored runs.

    uv sync --extra sim                                            # or ./scripts/setup.sh
    uv run --extra sim python scripts/website/build_site.py        # once, and after results change
    uv run --extra sim python scripts/website/serve_site.py        # then open http://127.0.0.1:8000/playground.html#live
    uv run --extra sim python scripts/website/serve_site.py --port 9000 --preset crawl

Local only: the server listens on 127.0.0.1, refuses requests whose Host is not a loopback name (a defence against
DNS rebinding), and refuses POSTs from other origins. Nothing leaves your machine. Placeholder dimensions, masses,
servo figures and friction (docs/STAGE2.md).
"""
import argparse
import mimetypes
import os
import sys
from http.server import ThreadingHTTPServer
from urllib.parse import urlparse

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gait"))  # live_gait
from live_gait import Runner, host_is_local, make_handler, origin_is_local  # noqa: E402,F401  (the guards are shared)
from mojito.results import repo_path  # noqa: E402
from mojito.sim import live_sim  # noqa: E402

SITE = repo_path("site")


def make_site_handler(runner, site_dir=SITE):
    api = make_handler(runner)
    root = os.path.realpath(site_dir)

    class Handler(api):  # inherits live_gait's `_local_only` Host and Origin check
        def do_GET(self):
            if not self._local_only():
                return
            path = urlparse(self.path).path
            if path.startswith("/api/"):
                self.path = path[len("/api"):]  # /api/state -> /state, handled by live_gait's handler
                return api.do_GET(self)
            if path in ("", "/"):
                path = "/index.html"
            full = os.path.realpath(os.path.join(root, path.lstrip("/")))
            if os.path.commonpath([root, full]) != root or not os.path.isfile(full):
                return self.send_error(404)
            with open(full, "rb") as f:
                kind = mimetypes.guess_type(full)[0] or "application/octet-stream"
                self._send(f.read(), kind + ("; charset=utf-8" if kind.startswith("text/") or kind.endswith("javascript") else ""))

        def do_POST(self):
            if not self._local_only():
                return
            path = urlparse(self.path).path
            if not path.startswith("/api/"):
                return self.send_error(404)
            self.path = path[len("/api"):]
            return api.do_POST(self)

    return Handler


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--preset", choices=sorted(live_sim.PRESETS), default="trot")
    args = parser.parse_args()
    if not os.path.exists(os.path.join(SITE, "data", "results.js")):
        print("site/data is missing: run `uv run --extra sim python scripts/website/build_site.py` first", file=sys.stderr)
        sys.exit(1)
    runner = Runner(live_sim.LiveSim(dict(live_sim.PRESETS[args.preset], walk=False)))
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_site_handler(runner))
    print(f"open http://127.0.0.1:{args.port}/playground.html#live   (Ctrl+C to stop; this server is local only)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        runner.stop = True


if __name__ == "__main__":
    main()
