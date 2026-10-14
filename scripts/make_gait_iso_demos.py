"""Isometric animations of three gait replays, with the gait timing chart underneath.

    results/trot_iso.gif         trot at 0.15 m/s
    results/trot_turn_iso.gif    trot turning on the spot at 1.2 rad/s
    results/crawl_iso.gif        crawl at 0.04 m/s

Same model as scripts/gait_replay.py (placeholder dimensions, masses, servo figures, friction).
For other timings use scripts/try_gait.py.

    python scripts/make_gait_iso_demos.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from mojito import gait_plots, gait_replay, gaits  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "results")
CASES = [
    ("trot_iso", gaits.TROT, (0.15, 0.0), 0.0, 3, "Trot at 0.15 m/s commanded"),
    ("trot_turn_iso", gaits.TROT, (0.0, 0.0), 1.2, 3, "Trot turning at 1.2 rad/s commanded"),
    ("crawl_iso", gaits.CRAWL, (0.04, 0.0), 0.0, 2, "Crawl at 0.04 m/s commanded"),
]


def main():
    for name, gait, velocity, yaw_rate, cycles, title in CASES:
        r = gait_replay.run_gait(gait, velocity, yaw_rate, cycles=cycles, record_every=20)
        path = os.path.join(OUT, name + ".gif")
        gait_plots.save_isometric(r, path, title)
        print(path)


if __name__ == "__main__":
    main()
