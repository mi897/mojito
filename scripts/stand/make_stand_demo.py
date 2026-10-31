"""Side-view animations of the stand test and the sit-to-stand test.

Drawn from the MuJoCo state (shoulder, knee and foot of the two left legs, and the body bar). The
lower panels show the largest joint torque as a fraction of the stall torque. Placeholder
dimensions, masses and servo figures. The body is drawn only as a bar: after a collapse it passes through the
floor because only the foot pads collide in the model.

    results/stand/stand_demo.gif        4 kg stands, 6 kg collapses (kp 100)
    results/stand/sit_to_stand_demo.gif 3 kg rises from a crouch in 2 s, 5 kg cannot (kp 10)

    uv run --extra sim python scripts/stand/make_stand_demo.py
"""
import os

import numpy as np

from mojito.sim import actuators, stand
from mojito.palette import BLUE, INK, ORANGE
from mojito.results import results_path

OUT = results_path("stand")
EVERY = 10  # simulation steps per frame (20 ms)


def light(total):
    return {"body": total - 4 * 4e-4, "hip": 1e-4, "thigh": 1e-4, "shank": 1e-4, "foot": 1e-4}


def stand_case(mass, servo):
    r = stand.run_stand(masses=light(mass), servo=servo, duration=3.0, record_every=EVERY)
    return r["frames"], np.arange(len(r["frames"])) * EVERY * 0.002, r["torque_trace"]


def sit_case(mass, servo):
    r = stand.run_sit_to_stand(masses=light(mass), servo=servo, rise_time=2.0, record_every=EVERY)
    return r["frames"], r["time_s"][::EVERY], r["peak_torque_nm"][::EVERY] / r["servo_limit_nm"]


def animate(cases, titles, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = min(len(c[0]) for c in cases)
    fig, axes = plt.subplots(2, len(cases), figsize=(4.6 * len(cases), 6), gridspec_kw={"height_ratios": [3, 1.3]})

    def draw(i):
        for c, (frames, t, trace) in enumerate(cases):
            a = axes[0, c]
            a.clear()
            fr = frames[i]
            a.axhline(0, color=INK, lw=2)
            a.set_xlim(-0.35, 0.35)
            a.set_ylim(-0.3, 0.35)
            a.set_aspect("equal")
            half = fr["R"] @ np.array([0.15, 0, 0])
            o = fr["origin"]
            a.plot([o[0] - half[0], o[0] + half[0]], [o[2] - half[2], o[2] + half[2]], color=INK, lw=6,
                   solid_capstyle="round")
            for p, col in (("FL", BLUE), ("RL", ORANGE)):
                pts = fr["legs"][p]
                a.plot(pts[:, 0], pts[:, 2], "-o", color=col, lw=3, ms=4)
            a.set_title(f"{titles[c]}\nt = {t[i]:.2f} s", fontsize=9, color=INK)
            a.set_xticks([])
            a.set_yticks([])
            b = axes[1, c]
            b.clear()
            b.plot(t[:len(trace)], trace, color=BLUE, lw=1.5)
            b.axhline(1, color=ORANGE, ls="--", lw=1)
            b.axvline(t[i], color=INK, lw=0.8)
            b.set_ylim(0, 1.1)
            b.set_xlim(0, t[-1])
            b.set_ylabel("largest torque / stall", fontsize=8)
            b.set_xlabel("s", fontsize=8)

    FuncAnimation(fig, draw, frames=range(0, n, 2)).save(path, writer=PillowWriter(fps=20))
    print(path)


def main():
    stiff = actuators.ServoParams(kp=100.0, kd=1.0)
    animate([stand_case(4.0, stiff), stand_case(6.0, stiff)],
            ["4 kg, kp 100: stands", "6 kg, kp 100: knees saturate, collapses"],
            os.path.join(OUT, "stand_demo.gif"))
    soft = actuators.ServoParams()
    animate([sit_case(3.0, soft), sit_case(5.0, soft)],
            ["3 kg: crouch to stand in 2 s (starts at 1 s)", "5 kg: knees saturate on the way up, collapses"],
            os.path.join(OUT, "sit_to_stand_demo.gif"))


if __name__ == "__main__":
    main()
