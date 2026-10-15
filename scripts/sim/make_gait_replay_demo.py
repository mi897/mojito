"""Side-view animations of the gait replays (docs/STAGE2.md, work package 4).

Drawn from the MuJoCo state: shoulder, knee and foot of the two left legs and the body bar. A filled
foot carries more than 0.5 N. The dashed orange line is where the body would be if it moved at the commanded
velocity from the end of the blend; the grey line is the body's actual position along x. The lower
panels show the largest joint torque as a fraction of the stall torque. Placeholder dimensions, masses,
servo figures and friction. The turning run is drawn from the side, so only its drift is visible.

    results/gait/trot_replay.gif    trot at 0, 0.05, 0.15, 0.30 m/s and turning at 1.2 rad/s
    results/gait/crawl_replay.gif   crawl at 0, 0.02, 0.04, 0.08 m/s

    python scripts/sim/make_gait_replay_demo.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.robot import gaits
from mojito.sim import gait_replay  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK = "#333333"
OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "gait")
EVERY = 20  # simulation steps per frame (40 ms), played at 25 frames per second
STAND, RAMP = 1.0, 0.5


def case(gait, vx, yaw_rate, cycles):
    r = gait_replay.run_gait(gait, (vx, 0.0), yaw_rate, cycles=cycles, stand_time=STAND, ramp_time=RAMP,
                             record_every=EVERY)
    title = f"{gait.name} {vx:g} m/s" if yaw_rate == 0 else f"{gait.name}, turn {yaw_rate:g} rad/s"
    idx = np.arange(0, len(r["time_s"]), EVERY)
    trace = r["torque_nm"].max(1)[idx] / r["servo_limit_nm"]
    return dict(frames=r["frames"], t=r["time_s"][idx], trace=trace, vx=vx, title=title,
                x0=r["origin_m"][int(round((STAND + RAMP) / 0.002)), 0])


def animate(cases, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = min(len(c["frames"]) for c in cases)
    fig, axes = plt.subplots(2, len(cases), figsize=(3.6 * len(cases), 4.2), gridspec_kw={"height_ratios": [3, 1.3]})

    def draw(i):
        for c, cs in enumerate(cases):
            a, b = axes[0, c], axes[1, c]
            a.clear()
            b.clear()
            fr, t = cs["frames"][i], cs["t"][i]
            a.axhline(0, color=INK, lw=2)
            a.set_xlim(-0.35, 0.4)
            a.set_ylim(-0.03, 0.30)
            a.set_aspect("equal")
            origin = fr["origin"]
            half = fr["R"] @ np.array([0.15, 0, 0])
            walking = max(0.0, t - STAND - RAMP)
            commanded_x = cs["vx"] * walking
            a.axvline(commanded_x, color=ORANGE, ls="--", lw=1)
            a.axvline(origin[0] - cs["x0"], color="#999999", lw=1)
            x = lambda p: p[0] - cs["x0"]  # noqa: E731
            a.plot([x(origin) - half[0], x(origin) + half[0]], [origin[2] - half[2], origin[2] + half[2]], color=INK,
                   lw=6, solid_capstyle="round")
            for leg, (name, colour) in enumerate((("FL", BLUE), ("RL", ORANGE))):
                pts = fr["legs"][name]
                a.plot(pts[:, 0] - cs["x0"], pts[:, 2], "-", color=colour, lw=3)
                down = fr["loads"][0 if name == "FL" else 2] > 0.5
                a.plot(pts[2, 0] - cs["x0"], pts[2, 2], "o", ms=6, color=colour, mfc=colour if down else "white")
            slip = origin[0] - cs["x0"] - commanded_x
            a.set_title(f"{cs['title']}\nt = {t:.2f} s   actual minus commanded {slip * 1000:+.0f} mm", fontsize=8, color=INK)
            a.set_xticks([])
            a.set_yticks([])
            b.plot(cs["t"], cs["trace"], color=BLUE, lw=1.2)
            b.axhline(1, color=ORANGE, ls="--", lw=1)
            b.axvline(t, color=INK, lw=0.8)
            b.set_ylim(0, 1.1)
            b.set_xlim(0, cs["t"][-1])
            b.set_ylabel("torque / stall", fontsize=7)
            b.set_xlabel("s", fontsize=7)
            b.tick_params(labelsize=7)

    FuncAnimation(fig, draw, frames=range(n)).save(path, writer=PillowWriter(fps=25))
    print(path)


def main():
    animate([case(gaits.TROT, v, w, 4) for v, w in ((0.0, 0.0), (0.05, 0.0), (0.15, 0.0), (0.30, 0.0), (0.0, 1.2))],
            os.path.join(OUT, "trot_replay.gif"))
    animate([case(gaits.CRAWL, v, 0.0, 2) for v in (0.0, 0.02, 0.04, 0.08)], os.path.join(OUT, "crawl_replay.gif"))


if __name__ == "__main__":
    main()
