"""Re-tune the gait for each design. Writes results/design/design_tuning.json.

For each of the 11 designs, three masses and both gaits, runs every GaitSettings in design_sweep.settings_grid()
(5 knee bends x 3 swing heights x 3 (kp, kd) pairs = 45) at the commanded speed design_sweep.TUNE_SPEED, on the nominal masses
and latency (no draw), 10 s per run. The best by design_sweep.tuning_score (valid first, then speed ratio closest to 1, then lowest
peak torque) is kept per design, mass and gait, with all candidates. Period, duty, offsets and stride are not tuned.

    uv run --extra sim python scripts/design/design_tune.py        # needs mujoco; 11 x 3 x 2 x 45 = 2970 runs, about 35 min on 4 cores
"""
import dataclasses
import multiprocessing

from mojito.sim import design_sweep
from mojito.results import results_path, write_result

OUTPUT = results_path("design", "design_tuning.json")


def one(cell):
    design, gait, settings, scale = cell
    return design_sweep.evaluate_walk(design, gait, settings, scale, design_sweep.TUNE_SPEED)


def main():
    designs, settings = design_sweep.design_grid(), design_sweep.settings_grid()
    cells = [(d, g, st, s) for d in designs for s in design_sweep.MASS_SCALES for g in design_sweep.GAITS for st in settings]
    results = []
    with multiprocessing.Pool() as pool:
        for n, r in enumerate(pool.imap(one, cells, chunksize=2), 1):
            results.append(r)
            if n % 100 == 0:
                print(f"{n}/{len(cells)}", flush=True)
    entries, i = [], 0
    for d in designs:
        for s in design_sweep.MASS_SCALES:
            for g in design_sweep.GAITS:
                candidates = results[i:i + len(settings)]
                i += len(settings)
                best = max(range(len(settings)), key=lambda k: design_sweep.tuning_score(candidates[k]))
                entries.append({"design": dataclasses.asdict(d), "mass_scale": s, "gait": g,
                                "default": dataclasses.asdict(design_sweep.default_settings(g)),
                                "best": dataclasses.asdict(settings[best]), "best_walk": candidates[best],
                                "n_valid": sum(design_sweep.is_valid(c) for c in candidates),
                                "candidates": [{"settings": dataclasses.asdict(st), **c} for st, c in zip(settings, candidates)]})
    out = {"what": "Grid search of knee bend, swing height and (kp, kd) per design, mass and gait at 0.10 m/s on the nominal masses and "
                   "latency, 10 s per run; best = valid, then speed ratio closest to 1, then lowest peak torque. All dimensions, masses, "
                   "servo figures, friction and joint limits are placeholders.",
           "tune_speed_m_s": design_sweep.TUNE_SPEED, "knee_grid_rad": list(design_sweep.KNEE_GRID),
           "lift_grid_m": list(design_sweep.LIFT_GRID), "gain_grid": [list(g) for g in design_sweep.GAIN_GRID],
           "run_time_s": design_sweep.RUN_TIME, "entries": entries}
    write_result(OUTPUT, out, indent=None)
    print(f"{sum(design_sweep.is_valid(e['best_walk']) for e in entries)} of {len(entries)} have a valid best")


if __name__ == "__main__":
    main()
