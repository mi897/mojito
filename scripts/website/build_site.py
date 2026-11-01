"""Assemble the website data from the committed results (no MuJoCo needed).

The pages in site/ never hold a measured number of their own: they read `data/results.js`, which this script writes
from the JSON files under `results/`, so a regenerated result shows up on the site after a rebuild. It also copies the plots and
animations from results/, the interactive leg demo from docs/, and the Markdown reports.

    uv run --extra sim python scripts/website/build_site.py                     # writes site/data/ and site/media/
    python -m http.server -d site 8000               # then open http://127.0.0.1:8000

Generated files (site/data, site/media) are not committed; the `site` workflow runs this before publishing.
Each result's `provenance` block (mojito/results.py) is moved out of `data/results.js` into `data/build.js`
(`MOJITO_BUILD.results_provenance`), so the pages see the same result shapes as before. `data/constants.js` holds the Python
constants that site/assets/mojito.js needs (`site_constants`), and `data/livemeta.js` those of the browser simulation.
"""
import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys

from mojito.geometry.placeholders import JOINT_LIMITS, SIM_QUADRUPED, STAGE1_QUADRUPED
from mojito.results import ROOT, without_provenance
from mojito.robot import gaits, statics
from mojito.sim import actuators, live_sim  # NumPy only (MuJoCo is imported lazily): the constants the browser simulation needs

# results/<name>.json -> key in window.MOJITO_RESULTS
RESULT_FILES = ["metrics", "calibration", "body_metrics", "torque_envelope", "stand_test", "sit_to_stand", "gait_replay",
                "gait_speed_diagnosis", "pace_diagnosis", "trot_period_sweep", "trot_period_robustness", "trot_1p7_parameter_sweep", "trot_1p7_one_at_a_time", "trot_1p7_kp_kd_grid_summary", "trot_1p7_combined_summary", "training_log"]
DOCS = {"README": "README.md", "USAGE": "docs/USAGE.md", "RESULTS": "docs/RESULTS.md", "PHYSICS": "docs/PHYSICS.md", "PLAN": "docs/PLAN.md", "STAGE2": "docs/STAGE2.md", "NAMING": "docs/NAMING.md", "CLAUDE": "CLAUDE.md"}
MEDIA_SUFFIXES = (".gif", ".png")
# physics_sweep.json keys copied into window.MOJITO_RESULTS.physics_sweep_summary
SWEEP_SUMMARY_KEYS = ("n_runs", "walk_from_s", "record_from_s", "frame_dt_s", "stance_height_m")


def _design(design):
    """A QuadrupedDesign's dimensions (m), and its total mass (kg) when it has masses."""
    out = {key: getattr(design, key) for key in ("abduction_offset", "upper_length", "lower_length", "body_length", "body_width",
                                                  "stance_height")}
    if design.masses is not None:
        out["total_mass"] = round(design.total_mass, 6)  # 1.72, not the float sum 1.7200000000000002
    return out


def site_constants() -> dict:
    """The Python constants that site/assets/mojito.js and the pages using it need (window.MOJITO_CONSTANTS, data/constants.js):
    the placeholder designs, joint limits, gait presets and the speed each starts at, gravity and the listed servo figures.
    Generated so no number is copied by hand into the JavaScript."""
    servo = actuators.ServoParams()
    return {
        "joint_limits": [[float(low), float(high)] for low, high in JOINT_LIMITS],
        "sim_quadruped": _design(SIM_QUADRUPED),
        "stage1_quadruped": _design(STAGE1_QUADRUPED),
        "gaits": {gait.name: {"period": gait.period, "duty": gait.duty, "offsets": list(gait.offsets), "lift": gait.lift,
                              "sway": gait.sway} for gait in (gaits.TROT, gaits.CRAWL, gaits.PACE)},
        "gait_speeds": {name: preset["vx"] for name, preset in live_sim.PRESETS.items()},  # m/s each preset starts at
        "gravity": statics.G,
        # N·m stall torque and rad/s no-load speed at 5 V and at 6.8 V (listed, unverified), and the default supply voltage
        "servo": {"voltage": servo.voltage, "stall_5v": servo.stall_5v, "stall_68v": servo.stall_68v,
                  "noload_5v": servo.noload_5v, "noload_68v": servo.noload_68v},
    }


def constants_js() -> str:
    """data/constants.js. It sets globalThis, not window, so that Node (the tests) can load it as well as the pages."""
    return "globalThis.MOJITO_CONSTANTS=" + json.dumps(site_constants(), separators=(",", ":")) + ";\n"


def write_js(path, name, payload):
    with open(path, "w") as f:
        f.write(f"window.{name}=")
        json.dump(payload, f, separators=(",", ":"))
        f.write(";\n")


def load_result(path):
    """(the numbers of a results JSON file, its provenance block or None)."""
    with open(path) as f:
        return without_provenance(json.load(f))


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


def sweep_summary(sweep):
    """The few numbers of physics_sweep.json that pages outside the physics lab bind (the full file is 2 MB, loaded on demand)."""
    return {key: sweep[key] for key in SWEEP_SUMMARY_KEYS if key in sweep}


def collect_results(results_dir):
    """(results by name for data/results.js, provenance by name, physics_sweep.json's numbers or None). The results include
    `physics_sweep_summary` (sweep_summary) when the sweep is there."""
    results, provenance = {}, {}
    for name in RESULT_FILES:
        path = find_result(results_dir, name + ".json")
        if path:
            results[name], record = load_result(path)
            if record is not None:
                provenance[name] = record
        else:
            print(f"missing {name}.json under results/: sections that use it will show '?'", file=sys.stderr)
    sweep = None
    path = find_result(results_dir, "physics_sweep.json")
    if path:
        sweep, record = load_result(path)
        if record is not None:
            provenance["physics_sweep"] = record
        results["physics_sweep_summary"] = sweep_summary(sweep)
    return results, provenance, sweep


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

    results, provenance, sweep = collect_results(results_dir)
    write_js(os.path.join(data_dir, "results.js"), "MOJITO_RESULTS", results)
    if sweep is not None:
        write_js(os.path.join(data_dir, "sweep.js"), "MOJITO_SWEEP", sweep)
    else:
        print("missing physics_sweep.json under results/: run scripts/website/site_sweep.py (needs mujoco) for the physics playground", file=sys.stderr)

    write_js(os.path.join(data_dir, "livemeta.js"), "MOJITO_LIVE", live_sim.browser_constants())
    with open(os.path.join(data_dir, "constants.js"), "w") as f:
        f.write(constants_js())

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
             {"commit": commit, "date": datetime.date.today().isoformat(), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
              "results_provenance": provenance})
    print(f"site data in {data_dir}: {len(results)} result files, {len(docs)} documents, {copied} images/animations")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", default=os.path.join(ROOT, "site"), help="site directory to write data/ and media/ into")
    build(parser.parse_args().out)
