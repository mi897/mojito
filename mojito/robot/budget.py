"""Mass budget and link strength of the first robot: what it weighs, and whether printed links carry the servo torque.

Pure NumPy, no MuJoCo. Every part carries a `source`: "listed" (a retailer figure, unverified, see docs/STAGE2.md) or
"assumed" (my placeholder until the owner names the part). Nothing here is a measurement of a built robot.

Layout assumed (the owner's decision A): the abduction servos sit in the body, the hip-pitch servo on the hip link, the
knee servo on the thigh at the knee. `role_masses` gives the mass at each role the MuJoCo model builds
(`mojito.sim.mjcf`: body, hip, thigh, shank, foot), so a budget can be fed to the simulator.

Units: metres, kilograms, newtons, N·m, pascals.
"""
from __future__ import annotations

from dataclasses import dataclass

ROLES = ("body", "hip", "thigh", "shank", "foot")
LEGS = 4


@dataclass(frozen=True)
class Part:
    name: str
    count: int
    mass_each: float  # kg
    where: str        # a role in ROLES: the link of one leg for hip/thigh/shank/foot (count is the total over all legs), else the body
    group: str        # "servo", "battery", "electronics", "hardware" or "frame"
    source: str       # "listed" or "assumed"
    size: tuple | None = None  # box, m (length, width, height), for the volume check later

    @property
    def mass(self) -> float:
        return self.count * self.mass_each


@dataclass(frozen=True)
class Material:
    """Printed-plastic figures. Assumed typical values for FDM PLA, not measured on the owner's printer."""
    name: str
    density: float            # kg/m^3, solid
    youngs_modulus: float     # Pa, along the print layers
    strength: float           # Pa, tensile along the layers (across layers it is lower)


PLA = Material("PLA", density=1240.0, youngs_modulus=2.0e9, strength=40.0e6)
PETG = Material("PETG", density=1270.0, youngs_modulus=1.5e9, strength=35.0e6)


@dataclass(frozen=True)
class Tube:
    """Rectangular hollow section: depth is in the bending plane (the leg swings in x-z), width across it."""
    depth: float = 0.020
    width: float = 0.014
    wall: float = 0.002

    @property
    def area(self) -> float:
        return self.depth * self.width - (self.depth - 2 * self.wall) * (self.width - 2 * self.wall)

    @property
    def second_moment(self) -> float:
        """I = (b h^3 - b_i h_i^3) / 12 about the axis across the bending plane."""
        inner_depth, inner_width = self.depth - 2 * self.wall, self.width - 2 * self.wall
        return (self.width * self.depth**3 - inner_width * inner_depth**3) / 12


DEFAULT_TUBE = Tube()


def tube_mass(length: float, tube: Tube, material: Material) -> float:
    return material.density * tube.area * length


def plate_mass(length: float, width: float, thickness: float, material: Material, fill: float) -> float:
    """A printed plate; `fill` is the effective solid fraction after infill (assumed)."""
    return material.density * length * width * thickness * fill


def bending_stress(moment: float, tube: Tube, concentration: float = 1.0) -> float:
    """Peak stress sigma = K M (h/2) / I at the root of a link carrying `moment`; K allows for holes and corners."""
    return concentration * moment * (tube.depth / 2) / tube.second_moment


def tip_deflection(torque: float, length: float, tube: Tube, material: Material) -> float:
    """Tip sag of a cantilever of `length` whose root moment is `torque` with the load at the tip: F = T/L, delta = F L^3/(3EI)."""
    return torque * length**2 / (3 * material.youngs_modulus * tube.second_moment)


def default_parts(design, material: Material = PLA, tube: Tube = DEFAULT_TUBE, plate_thickness: float = 0.003,
                  plate_fill: float = 0.25, plate_margin: float = 0.06) -> list:
    """The parts list for `design` (a `placeholders.QuadrupedDesign`): listed servo and battery, assumed everything else."""
    servo = dict(mass_each=0.060, group="servo", source="listed", size=(0.040, 0.020, 0.0405))
    parts = [
        Part("servo, abduction", LEGS, where="body", **servo),
        Part("servo, hip pitch", LEGS, where="hip", **servo),
        Part("servo, knee", LEGS, where="thigh", **servo),
        Part("battery, 2S 1500 mAh", 1, 0.076, "body", "battery", "listed", (0.072, 0.034, 0.026)),
        Part("ESP32 board", 1, 0.010, "body", "electronics", "assumed", (0.052, 0.028, 0.010)),
        Part("PCA9685 servo driver", 1, 0.010, "body", "electronics", "assumed", (0.062, 0.025, 0.010)),
        Part("servo-rail regulator", 2, 0.025, "body", "electronics", "assumed", (0.050, 0.025, 0.015)),
        Part("logic regulator", 1, 0.005, "body", "electronics", "assumed", (0.025, 0.015, 0.010)),
        Part("IMU board", 1, 0.003, "body", "electronics", "assumed", (0.020, 0.015, 0.005)),
        Part("wiring, connectors", 1, 0.060, "body", "hardware", "assumed"),
        Part("fuse, switch", 1, 0.020, "body", "hardware", "assumed", (0.040, 0.020, 0.020)),
    ]
    for joint, role in (("abduction", "body"), ("hip pitch", "hip"), ("knee", "thigh")):
        parts.append(Part(f"Hall sensor board, {joint}", LEGS, 0.003, role, "electronics", "assumed", (0.015, 0.015, 0.005)))
        parts.append(Part(f"horn, screws, bracket, {joint}", LEGS, 0.015, role, "hardware", "assumed"))
    abduction_offset, upper, lower = (float(x) for x in design.lengths)
    parts += [
        Part("hip link", LEGS, tube_mass(abduction_offset, tube, material), "hip", "frame", "assumed"),
        Part("thigh tube", LEGS, tube_mass(upper, tube, material), "thigh", "frame", "assumed"),
        Part("shank tube", LEGS, tube_mass(lower, tube, material), "shank", "frame", "assumed"),
        Part("foot pad", LEGS, 0.010, "foot", "frame", "assumed"),
        Part("body plates, top and bottom", 2, plate_mass(design.body_length + plate_margin, design.body_width + plate_margin,
                                                        plate_thickness, material, plate_fill), "body", "frame", "assumed"),
    ]
    return parts


def totals(parts) -> dict:
    """Total mass, and the mass by group, by source and by role (kg)."""
    out = {"total": sum(p.mass for p in parts), "by_group": {}, "by_source": {}, "by_role": {role: 0.0 for role in ROLES}}
    for p in parts:
        out["by_group"][p.group] = out["by_group"].get(p.group, 0.0) + p.mass
        out["by_source"][p.source] = out["by_source"].get(p.source, 0.0) + p.mass
        out["by_role"][p.where] += p.mass
    return out


def role_masses(parts) -> dict:
    """Mass at each simulator role: the body once, and the hip, thigh, shank and foot of one leg."""
    by_role = totals(parts)["by_role"]
    return {role: (by_role[role] if role == "body" else by_role[role] / LEGS) for role in ROLES}


def leg_scale_to_fit(total_mass: float, bound_mass_at_design_lengths: float) -> float:
    """Leg-length scale at which a static mass bound reaches `total_mass`, given the bound at the design's lengths.

    The bound is inversely proportional to leg length (docs/STAGE2.md), so scale = bound / total. Links also get
    lighter with length, which this ignores, so it is a little conservative."""
    return bound_mass_at_design_lengths / total_mass
