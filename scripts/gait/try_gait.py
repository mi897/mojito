"""Try a gait variation in physics: timing table, timing chart, isometric GIF, measured numbers.

Start from the built-in trot or crawl and override any timing. Outputs go to results/gait/try/<name>.* (not committed).
Needs mujoco and matplotlib.

    python scripts/gait/try_gait.py --base trot --name slow_trot --period 0.8 --velocity 0.1
    python scripts/gait/try_gait.py --base trot --name pace --offsets 0,0.5,0,0.5        # left legs together, then right
    python scripts/gait/try_gait.py --base trot --name short_flight --duty 0.4            # all feet in the air for 0.1 of the cycle
    python scripts/gait/try_gait.py --base crawl --name crawl_diag --order FL,RR,FR,RL    # lift order; offsets are worked out
    python scripts/gait/try_gait.py --base crawl --name fast_crawl --period 1.2 --duty 0.8 --velocity 0.06 --kp 100 --kd 1

Timing, all as fractions of the cycle (see gaits.Gait): `period` seconds per cycle; `duty` the fraction of it a foot is
on the ground; `offsets` the cycle phase at which each leg (FL, FR, RL, RR) starts its stance. The leg lifts at
offset + duty (mod 1). `--order` replaces `--offsets` for one-leg-at-a-time gaits and spaces the lifts evenly.
`--sway` is the crawl's body lean in metres; it only acts when no two swings overlap (duty >= 0.75 with four legs).
"""
import argparse
import dataclasses
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from mojito.robot import body, gaits
from mojito.sim import actuators, gait_plots, gait_replay  # noqa: E402

OUT = os.path.join(os.path.dirname(__file__), "..", "..", "results", "gait", "try")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", choices=("trot", "crawl"), default="trot", help="gait to start from")
    p.add_argument("--name", default="variant", help="output file stem in results/gait/try/")
    p.add_argument("--period", type=float, help="seconds per cycle")
    p.add_argument("--duty", type=float, help="fraction of the cycle on the ground, 0 to 1 exclusive")
    p.add_argument("--offsets", help="four numbers: stance-start phase of FL,FR,RL,RR, e.g. 0,0.5,0.5,0")
    p.add_argument("--order", help="lift order for one-at-a-time gaits, e.g. FR,RL,FL,RR (replaces --offsets)")
    p.add_argument("--lift", type=float, help="swing height, m")
    p.add_argument("--sway", type=float, help="body lean away from the lifted leg, m")
    p.add_argument("--velocity", type=float, nargs="+", default=[0.1], help="vx [vy] in m/s")
    p.add_argument("--yaw-rate", type=float, default=0.0, help="rad/s")
    p.add_argument("--cycles", type=int, default=3)
    p.add_argument("--kp", type=float, help="servo position gain, N·m/rad (default 10)")
    p.add_argument("--kd", type=float, help="servo damping (default 0.2)")
    p.add_argument("--mass", type=float, help="total mass, kg, all in the body (default: placeholder masses, 1.72 kg)")
    p.add_argument("--no-gif", action="store_true")
    return p.parse_args()


def build_gait(a):
    gait = gaits.TROT if a.base == "trot" else gaits.CRAWL
    changes = {k: v for k, v in dict(period=a.period, duty=a.duty, lift=a.lift, sway=a.sway).items() if v is not None}
    duty = changes.get("duty", gait.duty)
    if a.order:
        legs = [body.LEG_NAMES.index(n) for n in a.order.split(",")]
        if sorted(legs) != [0, 1, 2, 3]:
            raise SystemExit("--order needs each of FL,FR,RL,RR once")
        changes["offsets"] = gaits.sequence_offsets(legs, duty)
    elif a.offsets:
        changes["offsets"] = tuple(float(x) for x in a.offsets.split(","))
    return dataclasses.replace(gait, name=a.name, **changes)


def describe(gait, velocity, yaw_rate):
    print(f"gait {gait.name}: period {gait.period:g} s, duty {gait.duty:g}  ->  stance {gait.stance_time:.3f} s, swing {gait.swing_time:.3f} s")
    starts = gait.swing_starts()
    for name, offset, lift in zip(body.LEG_NAMES, gait.offsets, starts):
        print(f"  {name}: stance starts at phase {offset:.2f}, lifts at phase {lift:.2f} ({lift * gait.period:.3f} s into the cycle)")
    t = np.linspace(0, gait.period, 2000, endpoint=False)
    home = body.Robot(body.AnalyticSolver()).stance()
    _, contact, _ = gaits.foot_targets(gait, t, home, velocity, yaw_rate)
    down = contact.sum(1)
    print(f"  feet commanded down: min {down.min()}, mean {down.mean():.2f}, max {down.max()}")
    speed = np.hypot(*velocity)
    if speed > 0:
        print(f"  stride (foot on the ground) {speed * gait.stance_time * 1000:.0f} mm, swing foot ground speed about {speed * gait.stance_time / gait.swing_time:.2f} m/s")
    if gait.sway and np.any(np.diff(np.append(np.sort(starts), np.sort(starts)[0] + 1.0)) - (1 - gait.duty) < -1e-9):
        print("  note: swings overlap, so the crawl lean (--sway) is switched off")


def main():
    a = parse_args()
    gait = build_gait(a)
    velocity = (a.velocity[0], a.velocity[1] if len(a.velocity) > 1 else 0.0)
    describe(gait, velocity, a.yaw_rate)
    servo = actuators.ServoParams(**{k: v for k, v in dict(kp=a.kp, kd=a.kd).items() if v is not None})
    masses = None
    if a.mass:
        masses = {"body": a.mass - 4 * 4e-4, "hip": 1e-4, "thigh": 1e-4, "shank": 1e-4, "foot": 1e-4}
    r = gait_replay.run_gait(gait, velocity, a.yaw_rate, cycles=a.cycles, servo=servo, masses=masses,
                             record_every=0 if a.no_gif else 20)
    print(f"result: walked={r['walked']}  tilt max {r['max_tilt_deg']:.1f}°  measured velocity "
          f"{r['measured_velocity_m_s'][0]:+.3f}, {r['measured_velocity_m_s'][1]:+.3f} m/s (commanded {velocity[0]:+.3f}, {velocity[1]:+.3f})  "
          f"yaw {r['measured_yaw_rate_rad_s']:+.2f} rad/s")
    print(f"  peak torque {r['peak_torque_fraction_of_limit']:.2f} of stall, saturated {100 * r['fraction_of_walking_saturated']:.1f}% of walking, "
          f"feet loaded min {r['feet_down_min']}, max slip step {r['slip_max_mm_per_step']:.2f} mm")
    os.makedirs(OUT, exist_ok=True)
    stem = os.path.join(OUT, a.name)
    title = f"{gait.name}: period {gait.period:g} s, duty {gait.duty:g}, command {velocity[0]:g} m/s"
    gait_plots.save_timing_chart(r, stem + "_timing.png", title)
    if not a.no_gif:
        gait_plots.save_isometric(r, stem + "_iso.gif", title)
    keep = ("gait", "walked", "max_tilt_deg", "measured_velocity_m_s", "measured_yaw_rate_rad_s", "peak_torque_fraction_of_limit",
            "fraction_of_walking_saturated", "feet_down_min", "slip_max_mm_per_step", "commanded_velocity_m_s")
    with open(stem + ".json", "w") as f:
        json.dump({"gait": dataclasses.asdict(gait), "velocity": velocity, "yaw_rate": a.yaw_rate,
                   "result": {k: r[k] for k in keep}}, f, indent=2)
    print("wrote", stem + "_timing.png" + ("" if a.no_gif else ", " + stem + "_iso.gif"), "and .json")


if __name__ == "__main__":
    main()
