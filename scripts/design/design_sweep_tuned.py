"""The design sweep with the gait re-tuned per design. Writes results/design/design_sweep_tuned.json.

Takes the best GaitSettings per design, mass and gait from results/design/design_tuning.json (scripts/design/design_tune.py,
chosen at 0.10 m/s on the nominal masses and latency) and runs them at 0.05, 0.10, 0.15 m/s for 10 draws each, the same draws as
scripts/design/design_sweep_draws.py (masses times a uniform 0.95 to 1.05, latency 18 to 22 ms), 10 s per run. The untuned
settings are in design_sweep_draws.json for comparison. A tuned setting is only as good as its search: it was picked on one speed and
the nominal run, so the other two speeds and the draws are a check on it, not part of the fit.

    uv run --extra sim python scripts/design/design_sweep_tuned.py        # needs mujoco; 66 entries x 3 speeds x 10 draws = 1980 runs, about 20 min on 4 cores
"""
import json
import multiprocessing

from mojito.sim import design_sweep
from mojito.results import results_path, write_result

OUTPUT = results_path("design", "design_sweep_tuned.json")
DRAWS = 10


def one(cell):
    design, gait, settings, scale, speed, draw = cell
    walk = design_sweep.evaluate_walk(design, gait, settings, scale, speed, draw=draw)
    return {**walk, "mass_scale": scale, "draw": draw, "design": design.__dict__}


def main():
    with open(results_path("design", "design_tuning.json")) as f:
        tuning = json.load(f)
    cells = []
    for e in tuning["entries"]:
        design = design_sweep.Design(**e["design"])
        settings = design_sweep.GaitSettings(**e["best"])
        for speed in design_sweep.SPEEDS:
            for draw in range(DRAWS):
                cells.append((design, e["gait"], settings, e["mass_scale"], speed, draw))
    runs = []
    with multiprocessing.Pool() as pool:
        for n, r in enumerate(pool.imap(one, cells, chunksize=4), 1):
            runs.append(r)
            if n % 100 == 0:
                print(f"{n}/{len(cells)}", flush=True)
    out = {"what": "Walks with the gait settings re-tuned per design, mass and gait (design_tuning.json), at three speeds and 10 draws "
                   "each, 10 s per run. All dimensions, masses, servo figures, friction and joint limits are placeholders.",
           "draws": DRAWS, "speeds_m_s": list(design_sweep.SPEEDS), "settings": {
               f"{e['gait']} {e['mass_scale']} {e['design']}": e["best"] for e in tuning["entries"]}, "runs": runs}
    write_result(OUTPUT, out, indent=None)
    print(f"{sum(design_sweep.is_valid(r) for r in runs)} of {len(runs)} valid")


if __name__ == "__main__":
    main()
