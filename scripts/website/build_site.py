"""Assemble the website data from the committed results (no MuJoCo needed).

The pages in site/ never hold a measured number of their own: they read `data/results.js`, which this script writes
from the JSON files under `results/`, so a regenerated result shows up on the site after a rebuild. It also copies the plots and
animations from results/, the interactive leg demo from docs/, and the Markdown reports.

    python scripts/website/build_site.py                     # writes site/data/ and site/media/
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

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")

# results/<name>.json -> key in window.MOJITO_RESULTS
RESULT_FILES = ["metrics", "calibration", "body_metrics", "torque_envelope", "stand_test", "sit_to_stand", "gait_replay",
                "gait_speed_diagnosis", "pace_diagnosis", "trot_period_sweep", "trot_period_robustness", "trot_1p7_parameter_sweep", "trot_1p7_one_at_a_time", "trot_1p7_kp_kd_grid_summary", "training_log"]
DOCS = {"README": "README.md", "USAGE": "docs/USAGE.md", "RESULTS": "docs/RESULTS.md", "PHYSICS": "docs/PHYSICS.md", "PLAN": "docs/PLAN.md", "STAGE2": "docs/STAGE2.md", "NAMING": "docs/NAMING.md", "CLAUDE": "CLAUDE.md"}
MEDIA_SUFFIXES = (".gif", ".png")


def write_js(path, name, payload):
    with open(path, "w") as f:
        f.write(f"window.{name}=")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")


def find_result(results_dir, filename):
    """Path of results/<group>/<filename>, or None. File names are unique across the group folders."""
    for folder, dirs, files in os.walk(results_dir):
        dirs[:] = sorted(d for d in dirs if d != "try")
        if filename in files:
            return os.path.join(folder, filename)
    return None


def media_files(results_dir):
    """Every plot and animation under results/ (not the uncommitted results/gait/try/), sorted by file name."""
    found = []
    for folder, dirs, files in os.walk(results_dir):
        dirs[:] = [d for d in dirs if d != "try"]
        found += [os.path.join(folder, n) for n in files if n.endswith(MEDIA_SUFFIXES)]
    return sorted(found, key=os.path.basename)


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
        path = find_result(results_dir, name + ".json")
        if path:
            with open(path) as f:
                results[name] = json.load(f)
        else:
            print(f"missing {name}.json under results/: sections that use it will show '?'", file=sys.stderr)
    write_js(os.path.join(data_dir, "results.js"), "MOJITO_RESULTS", results)

    sweep = find_result(results_dir, "physics_sweep.json")
    if sweep:
        with open(sweep) as f:
            write_js(os.path.join(data_dir, "sweep.js"), "MOJITO_SWEEP", json.load(f))
    else:
        print("missing physics_sweep.json under results/: run scripts/website/site_sweep.py (needs mujoco) for the physics playground", file=sys.stderr)

    docs = {}
    for key, rel in DOCS.items():
        with open(os.path.join(ROOT, rel)) as f:
            docs[key] = f.read()
    write_js(os.path.join(data_dir, "docs.js"), "MOJITO_DOCS", docs)

    copied = 0
    for path in media_files(results_dir):
        shutil.copy2(path, os.path.join(media_dir, os.path.basename(path)))
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
