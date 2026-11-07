"""Mass budget and printed-link strength of the first robot (work package 6, part 1).

Listed servo and battery figures plus assumed placeholders for everything else (`mojito/robot/budget.py`): what the robot
weighs against the 1.5 kg working target, and whether printed PLA links carry the servo's torque. Reads the static mass
bound from results/body/torque_envelope.json and the walking peaks from results/gait/gait_replay.json (run those first).

    results/body/mass_budget.json

    uv run python scripts/body/mass_budget.py [--safety-factor 1.5] [--material PLA|PETG]
"""
import argparse
import dataclasses
import json

from mojito.geometry.placeholders import SIM_QUADRUPED
from mojito.results import results_path, write_result
from mojito.robot import budget
from mojito.sim import actuators  # NumPy only

TARGET_KG = 1.5  # working target, docs/STAGE2.md ("How heavy can it be?"): a recommendation, not a result
MATERIALS = {"PLA": budget.PLA, "PETG": budget.PETG}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--safety-factor", type=float, default=1.5, help="on the servo stall torque, the most a link can ever be asked for")
    ap.add_argument("--material", choices=sorted(MATERIALS), default="PLA")
    ap.add_argument("--stress-concentration", type=float, default=3.0, help="assumed, for screw holes and corners")
    args = ap.parse_args()
    material, design, tube = MATERIALS[args.material], SIM_QUADRUPED, budget.DEFAULT_TUBE

    parts = budget.default_parts(design, material, tube)
    tot = budget.totals(parts)
    masses = dict(design.masses)
    sim_current = masses["body"] + 4 * sum(masses[r] for r in ("hip", "thigh", "shank", "foot"))

    envelope = json.load(open(results_path("body", "torque_envelope.json")))
    trot_bound = envelope["max_total_mass_kg"]["trot_2_feet"]["5.0 V derate 0.5"]
    replay = json.load(open(results_path("gait", "gait_replay.json")))
    stall = actuators.ServoParams(voltage=5.0).stall_torque
    peak_fraction = max(r["peak_torque_fraction_of_limit"] for r in replay["runs"])

    upper, lower = float(design.upper_length), float(design.lower_length)
    # a link's root moment equals the torque of the joint that drives it; the servo cannot give more than its stall torque
    strength = {}
    for name, length in (("thigh", upper), ("shank", lower)):
        load = stall * args.safety_factor
        stress = budget.bending_stress(load, tube, args.stress_concentration)
        strength[name] = {
            "design_torque_nm": round(load, 3),
            "peak_stress_mpa": round(stress / 1e6, 1),
            "allowable_mpa": round(material.strength / 1e6, 1),
            "margin": round(material.strength / stress, 1),
            "tip_sag_at_stall_mm": round(budget.tip_deflection(stall, length, tube, material) * 1e3, 2),
        }

    result = {
        "what": "mass budget and printed-link strength of the first robot. Servo and battery masses are retailer listings "
                "(unverified); every other mass, the link section, the material figures and the stress concentration are my assumptions",
        "material": dataclasses.asdict(material), "link_section_m": dataclasses.asdict(tube),
        "target_kg": TARGET_KG, "total_kg": round(tot["total"], 3),
        "over_target_kg": round(tot["total"] - TARGET_KG, 3),
        "by_group_kg": {k: round(v, 3) for k, v in tot["by_group"].items()},
        "by_source_kg": {k: round(v, 3) for k, v in tot["by_source"].items()},
        "by_role_kg": {k: round(v, 3) for k, v in tot["by_role"].items()},
        "role_masses_for_simulator_kg": {k: round(v, 4) for k, v in budget.role_masses(parts).items()},
        "simulator_masses_now_kg": {"total": round(sim_current, 3), **masses},
        "static_trot_bound_kg_5V_half_stall": trot_bound,
        "total_over_trot_bound": round(tot["total"] / trot_bound, 2),
        "leg_scale_that_fits_trot_bound": round(budget.leg_scale_to_fit(tot["total"], trot_bound), 2),
        "parts": [{"name": p.name, "count": p.count, "mass_each_kg": round(p.mass_each, 4), "total_kg": round(p.mass, 4),
                   "where": p.where, "group": p.group, "source": p.source} for p in parts],
        "strength": {
            "safety_factor_on_stall": args.safety_factor, "servo_stall_nm_5V": round(stall, 3),
            "stress_concentration": args.stress_concentration,
            "walking_peak_fraction_of_servo_limit_max_over_runs": peak_fraction, "links": strength,
        },
    }
    write_result(results_path("body", "mass_budget.json"), result)
    print(json.dumps({k: v for k, v in result.items() if k != "parts"}, indent=1))
    for p in parts:
        print(f"{p.name:42s} {p.count:2d} x {p.mass_each*1e3:6.1f} g = {p.mass*1e3:7.1f} g  {p.where:6s} {p.source}")


if __name__ == "__main__":
    main()
