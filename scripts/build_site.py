"""Assemble the website data from the committed results (no MuJoCo needed).

The pages in site/ never hold a measured number of their own: they read `data/results.js`, which this script writes
from `results/*.json`, so a regenerated result shows up on the site after a rebuild. It also copies the plots and
animations from results/, the interactive leg demo from docs/, and the Markdown reports.

    python scripts/build_site.py                     # writes site/data/ and site/media/
    python -m http.server -d site 8000               # then open http://127.0.0.1:8000

Generated files (site/data, site/media) are not committed; the `site` workflow runs this before publishing.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

# results/<name>.json -> key in window.MOJITO_RESULTS
RESULT_FILES = ["metrics", "calibration", "body_metrics", "torque_envelope", "stand_test", "sit_to_stand", "gait_replay",
                "gait_speed_diagnosis", "pace_diagnosis", "training_log"]
DOCS = {"README": "README.md", "PLAN": "docs/PLAN.md", "STAGE2": "docs/STAGE2.md", "NAMING": "docs/NAMING.md", "CLAUDE": "CLAUDE.md"}
MEDIA_SUFFIXES = (".gif", ".png")


def write_js(path, name, payload):
    with open(path, "w") as f:
        f.write(f"window.{name}=")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")


def git(*args):
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def build(out):
    results_dir = os.path.join(ROOT, "results")
    data_dir, media_dir = os.path.join(out, "data"), os.path.join(out, "media")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(media_dir, exist_ok=True)

    results = {}
    for name in RESULT_FILES:
        path = os.path.join(results_dir, name + ".json")
        if os.path.exists(path):
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"missing results/{name}.json: sections that use it will show '?'", file=sys.stderr)
    write_js(os.path.join(data_dir, "results.js"), "MOJITO_RESULTS", results)

    sweep = os.path.join(results_dir, "physics_sweep.json")
    if os.path.exists(sweep):
        with open(sweep) as f:
            write_js(os.path.join(data_dir, "sweep.js"), "MOJITO_SWEEP", json.load(f))
    else:
        print("missing results/physics_sweep.json: run scripts/site_sweep.py (needs mujoco) for the physics playground", file=sys.stderr)

    docs = {}
    for key, rel in DOCS.items():
        with open(os.path.join(ROOT, rel)) as f:
            docs[key] = f.read()
    write_js(os.path.join(data_dir, "docs.js"), "MOJITO_DOCS", docs)

    copied = 0
    for name in sorted(os.listdir(results_dir)):
        if name.endswith(MEDIA_SUFFIXES):
            shutil.copy2(os.path.join(results_dir, name), os.path.join(media_dir, name))
            copied += 1
    shutil.copy2(os.path.join(ROOT, "docs", "demo.html"), os.path.join(media_dir, "leg_demo.html"))

    commit = git("rev-parse", "--short", "HEAD")
    write_js(os.path.join(data_dir, "build.js"), "MOJITO_BUILD",
             {"commit": commit, "date": datetime.date.today().isoformat(), "branch": git("rev-parse", "--abbrev-ref", "HEAD")})
    print(f"site data in {data_dir}: {len(results)} result files, {len(docs)} documents, {copied} images/animations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "site"), help="site directory to write data/ and media/ into")
    build(parser.parse_args().out)
