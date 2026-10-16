"""Render the social preview image (site/assets/og-image.png, 1200x630) from the live hero.

The image is the website's own RobotView drawing the trot from mojito.js, frozen mid-stride, next to the
project's one-line description. Needs a Chromium-family browser (google-chrome, chromium) on the PATH.
Run it again after changing mojito.js or the viewer.
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
SITE = os.path.abspath(os.path.join(ROOT, "site"))
OUT = os.path.join(SITE, "assets", "og-image.png")
STRIDE_TIME_S = 1.05  # seconds into the 1.7 s trot (phase 0.62, as 0.31 s was in the 0.5 s trot): two diagonal feet down, two swinging
MASK = "linear-gradient(to right,transparent 0,#000 16%),linear-gradient(to bottom,transparent 0,#000 14%,#000 82%,transparent 100%)"

PAGE = """<!doctype html><html data-theme="light"><head><meta charset="utf-8">
<link rel="stylesheet" href="file://{site}/assets/site.css">
<style>
html,body{{margin:0;width:1200px;height:630px;overflow:hidden;background:var(--bg)}}
.og{{position:relative;height:630px}}
.text{{position:absolute;left:64px;top:0;width:470px;height:630px;display:flex;flex-direction:column;justify-content:center;z-index:2}}
h1{{font-size:60px;line-height:1.08;margin:0 0 22px}}
.sub{{font-size:25px;color:var(--ink2);line-height:1.4;margin:0}}
.tag{{margin-top:34px;font:600 22px var(--sans);color:var(--blue)}}
canvas{{position:absolute;left:400px;top:40px;width:820px;height:560px;border:0;background:transparent;
  -webkit-mask-image:{mask};mask-image:{mask};-webkit-mask-composite:source-in;mask-composite:intersect}}
</style></head><body><div class="og"><div class="text">
<h1>Mojito</h1><p class="sub">Robots that learn to move by themselves: from simulation first, then from real interaction.</p>
<div class="tag">A 12-motor quadruped</div></div>
<canvas id="v"></canvas></div>
<script src="file://{site}/assets/mojito.js"></script><script src="file://{site}/assets/viewer.js"></script>
<script>
const cv = document.getElementById('v'), cx = cv.getContext('2d'), set = Object.getOwnPropertyDescriptor(CanvasRenderingContext2D.prototype, 'lineWidth').set;
Object.defineProperty(cx, 'lineWidth', {{ get() {{ return 1; }}, set(v) {{ set.call(cx, v * 1.7); }} }});  // thicker strokes: this is read at thumbnail size
const M = window.Mojito, view = new RobotView(cv, {{ az: -40, el: 20, zoom: 1.12, span: 1.1 }}), robot = new M.Robot();
const t = {t}, v = [0.15, 0], r = M.evaluate({{ robot, gait: M.TROT, t, velocity: v, yawRate: 0, height: 0.2125, mass: 1.5, volts: 5 }});
const [, shift] = M.planarFlow(t, v, 0), off = r.offset;
view.setScene({{ origin: [shift[0] + off[0], shift[1] + off[1], 0.2125 + off[2]], R: r.R, bodySize: [robot.bodyLength, robot.bodyWidth],
  points: r.points, down: r.loadsN.map(l => l > 1e-6), bad: [0, 0, 0, 0] }});
</script></body></html>
"""


def find_browser():
    for name in ("google-chrome", "chromium", "chromium-browser", "chrome"):
        path = shutil.which(name)
        if path:
            return path
    sys.exit("No Chromium-family browser found on PATH (tried google-chrome, chromium, chromium-browser).")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=OUT)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory() as tmp:
        page = os.path.join(tmp, "og.html")
        with open(page, "w") as f:
            f.write(PAGE.format(site=SITE, mask=MASK, t=STRIDE_TIME_S))
        subprocess.run([find_browser(), "--headless=new", "--no-sandbox", "--allow-file-access-from-files",
                        "--window-size=1200,630", "--hide-scrollbars", "--virtual-time-budget=2000",
                        f"--screenshot={args.out}", f"file://{page}"], check=True, capture_output=True)
    print(f"wrote {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
