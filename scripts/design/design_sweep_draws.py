"""The design sweep repeated with random draws. Writes results/design/design_sweep_draws.json.

The same 11 designs, three masses, stand test, trot and crawl at 0.05, 0.10, 0.15 m/s and 10 s per run as
scripts/design/design_sweep.py, but each cell is run for 10 draws. Draw k seeds a generator that scales every body
mass by a uniform 0.95 to 1.05 and draws the servo latency uniformly from 18 to 22 ms; the same 10 draws in every cell
(as in scripts/gait/sweeps/trot_1p7_combined.py). The spread is from those draws only, not model error.

    python scripts/design/design_sweep_draws.py        # needs mujoco; 330 design-mass-draw cells of 7 runs, about 25 min on 4 cores
"""
import multiprocessing

from mojito.sim import design_sweep
from mojito.results import results_path, write_result

OUTPUT = results_path("design", "design_sweep_draws.json")
DRAWS = 10


def one(cell):
    design, mass_scale, draw = cell
    return design_sweep.evaluate_design(design, mass_scale, draw=draw)


def main():
    cells = [(d, s, k) for d in design_sweep.design_grid() for s in design_sweep.MASS_SCALES for k in range(DRAWS)]
    out_cells = []
    with multiprocessing.Pool() as pool:
        for n, result in enumerate(pool.imap(one, cells, chunksize=1), 1):
            out_cells.append(result)
            if n % 30 == 0:
                print(f"{n}/{len(cells)}", flush=True)
    out = {"what": "The design sweep (11 designs, three masses, stand, trot and crawl at three speeds, 10 s per run) with 10 random "
                   "draws per cell: every body mass times a uniform 0.95 to 1.05 and servo latency uniform 18 to 22 ms, the same 10 draws "
                   "in every cell. Gait timings and servo gains not re-tuned. All dimensions, masses, servo figures, friction and "
                   "joint limits are placeholders.",
           "draws": DRAWS, "mass_spread": design_sweep.MASS_SPREAD, "latency_s_range": list(design_sweep.LATENCY),
           "speeds_m_s": list(design_sweep.SPEEDS), "mass_scales": list(design_sweep.MASS_SCALES),
           "run_time_s": design_sweep.RUN_TIME, "cells": out_cells}
    write_result(OUTPUT, out, indent=None)
    print(f"{sum(c['stand']['stood'] for c in out_cells)} of {len(out_cells)} stood")


if __name__ == "__main__":
    main()
