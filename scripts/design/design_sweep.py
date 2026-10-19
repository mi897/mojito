"""Design sweep: which dimensions stand and walk in physics? Writes results/design/design_sweep.json.

Nine leg designs (upper and lower length 100, 150, 200 mm) at the placeholder body (300 x 150 mm), plus two body
sizes (240 x 120 and 360 x 180 mm) at the placeholder legs; each at three mass scales (1.04, 1.72, 3.01 kg, the
placeholder masses times 0.6, 1.0, 1.75); each stood once and walked with the trot and the crawl at 0.05, 0.10, 0.15 m/s,
each run 10 s of simulated time (the walks stand 1 s, blend 0.5 s, walk 8.5 s). One run per cell, no random draws. Gait timings and servo gains are not re-tuned; the standing height scales
so each leg keeps the placeholder knee bend (docs/STAGE2.md, work package 5).

    python scripts/design/design_sweep.py        # needs mujoco; 33 design-and-mass cells of 7 runs, parallel over the CPU cores
"""
import json
import multiprocessing
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.sim import design_sweep  # noqa: E402

OUTPUT = "results/design/design_sweep.json"


def one(cell):
    design, mass_scale = cell
    return design_sweep.evaluate_design(design, mass_scale)


def main():
    cells = [(d, s) for d in design_sweep.design_grid() for s in design_sweep.MASS_SCALES]
    with multiprocessing.Pool() as pool:
        cells_out = pool.map(one, cells, chunksize=1)
    out = {"what": "Stand once and walk the trot and the crawl at three commanded speeds for 11 designs and three masses, one run per "
                   "cell. Closed-form IK and the servo model into the actuated MuJoCo model, body free, 5 V. Gait timings and servo "
                   "gains are those tuned on the 150 mm legs (not re-tuned); standing height = the height at which the leg has the placeholder knee bend. "
                   "All dimensions, masses, servo figures and friction are placeholders.",
           "speeds_m_s": list(design_sweep.SPEEDS), "mass_scales": list(design_sweep.MASS_SCALES),
           "run_time_s": design_sweep.RUN_TIME, "abduction_offset_m": design_sweep.ABDUCTION_OFFSET, "cells": cells_out}
    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(out, f, indent=1)
    stood = sum(c["stand"]["stood"] for c in cells_out)
    walked = sum(w["walked"] for c in cells_out for w in c["walk"])
    print(f"{stood} of {len(cells_out)} stood; {walked} of {sum(len(c['walk']) for c in cells_out)} walks")


if __name__ == "__main__":
    main()
