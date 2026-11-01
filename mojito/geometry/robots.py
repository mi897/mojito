"""Generators for the example robots in robots/.

The checked-in .urdf files are produced by `python -m mojito.geometry.robots`, and a
test keeps them in sync, so the examples cannot drift from these numbers.
Everything here is built from the `urdf` module; it is also how a `LegConfig`
is turned into a URDF for the compatibility path.
"""
from __future__ import annotations

import json
import os

from . import placeholders
from .urdf import Joint, URDFModel, write


def _leg_joints(prefix: str, parent: str, mount: tuple, right: bool, lengths, limits) -> tuple:
    """Abduction (x), hip pitch (y), knee (y), fixed foot. Right legs mirror the lateral offset."""
    abduction_offset, upper_length, lower_length = (float(v) for v in lengths)
    side = -1.0 if right else 1.0
    abduction_limits, hip_pitch_limits, knee_limits = limits

    def revolute(name, parent_link, child_link, xyz, axis, joint_limits):
        return Joint(name, "revolute", parent_link, child_link, xyz=xyz, axis=axis,
                     lower=joint_limits[0], upper=joint_limits[1])

    joints = [
        revolute(f"{prefix}_abduction", parent, f"{prefix}_hip", mount, (1.0, 0.0, 0.0),
                 _mirror_limits(abduction_limits, right)),
        revolute(f"{prefix}_hip_pitch", f"{prefix}_hip", f"{prefix}_thigh", (0.0, side * abduction_offset, 0.0),
                 (0.0, 1.0, 0.0), hip_pitch_limits),
        revolute(f"{prefix}_knee", f"{prefix}_thigh", f"{prefix}_shank", (0.0, 0.0, -upper_length), (0.0, 1.0, 0.0),
                 knee_limits),
        Joint(f"{prefix}_foot", "fixed", f"{prefix}_shank", f"{prefix}_foot", xyz=(0.0, 0.0, -lower_length)),
    ]
    return [f"{prefix}_hip", f"{prefix}_thigh", f"{prefix}_shank", f"{prefix}_foot"], joints


def _mirror_limits(joint_limits, right):
    """(lower, upper) of a joint whose angle changes sign on a right leg: (-upper, -lower)."""
    return (-joint_limits[1], -joint_limits[0]) if right else tuple(joint_limits)


def leg_urdf(lengths=None, limits=None, name: str = "leg") -> URDFModel:
    """A single left leg hanging from a body link: the Stage 0 placeholder."""
    lengths = placeholders.NOMINAL_LENGTHS if lengths is None else lengths
    limits = placeholders.JOINT_LIMITS if limits is None else limits
    links, joints = _leg_joints("L", "body", (0.0, 0.0, 0.0), False, lengths, limits)
    return URDFModel(name, ["body"] + links, {j.name: j for j in joints})


def legged_urdf(name: str, mounts: dict, lengths=None, limits=None, body_size=None) -> URDFModel:
    """A body with one leg per entry in `mounts` (name -> (x, y, z)); y < 0 is a right leg."""
    lengths = placeholders.NOMINAL_LENGTHS if lengths is None else lengths
    limits = placeholders.JOINT_LIMITS if limits is None else limits
    links, joints = ["body"], {}
    for leg_name, xyz in mounts.items():
        leg_links, leg_joints = _leg_joints(leg_name, "body", xyz, xyz[1] < 0, lengths, limits)
        links += leg_links
        joints.update({joint.name: joint for joint in leg_joints})
    return URDFModel(name, links, joints)


def quadruped_urdf(body_length=placeholders.BODY_LENGTH, body_width=placeholders.BODY_WIDTH, lengths=None, limits=None) -> URDFModel:
    """The Stage 1 placeholder quadruped. Leg order FL, FR, RL, RR matches mojito.robot.body."""
    x, y = body_length / 2, body_width / 2
    return legged_urdf("mojito_quadruped", {"FL": (x, y, 0.0), "FR": (x, -y, 0.0),
                                            "RL": (-x, y, 0.0), "RR": (-x, -y, 0.0)}, lengths, limits)


def hexapod_urdf(body_length=0.36, body_width=0.12, lengths=None, limits=None) -> URDFModel:
    """Three legs a side, to show the pipeline is not tied to four legs."""
    x_positions, y = (body_length / 2, 0.0, -body_length / 2), body_width / 2
    mounts = {}
    for tag, x in zip(("F", "M", "R"), x_positions):
        mounts[f"{tag}L"] = (x, y, 0.0)
        mounts[f"{tag}R"] = (x, -y, 0.0)
    return legged_urdf("mojito_hexapod", mounts, lengths, limits)


def planar_urdf(upper=0.15, lower=0.15) -> URDFModel:
    """A two-joint planar leg (hip and knee about y): fewer DoF than the quadruped."""
    joints = [
        Joint("hip", "revolute", "body", "thigh", axis=(0.0, 1.0, 0.0), lower=0.0, upper=1.5),
        Joint("knee", "revolute", "thigh", "shank", xyz=(0.0, 0.0, -upper), axis=(0.0, 1.0, 0.0), lower=-2.3, upper=-0.3),
        Joint("foot", "fixed", "shank", "foot", xyz=(0.0, 0.0, -lower)),
    ]
    return URDFModel("mojito_planar", ["body", "thigh", "shank", "foot"], {joint.name: joint for joint in joints})


QUADRUPED_MANIFEST = {
    "length_tolerance": 0.10,
    "limbs": [
        {"name": "FL", "tip": "FL_foot", "group": "leg"},
        {"name": "FR", "tip": "FR_foot", "group": "leg", "mirror_of": "FL"},
        {"name": "RL", "tip": "RL_foot", "group": "leg"},
        {"name": "RR", "tip": "RR_foot", "group": "leg", "mirror_of": "RL"},
    ],
    # Positions are divided by this to make them dimensionless: upper + lower leg length.
    "groups": {"leg": {"reach": placeholders.LEG_REACH}},
}

HEXAPOD_MANIFEST = {
    "length_tolerance": 0.10,
    "limbs": [{"name": name, "tip": f"{name}_foot", "group": "leg",
               **({"mirror_of": name[0] + "L"} if name[1] == "R" else {})}  # a right leg mirrors the left one beside it
              for name in ("FL", "FR", "ML", "MR", "RL", "RR")],
    "groups": {"leg": {"reach": placeholders.LEG_REACH}},
}


def write_examples(directory: str = "robots") -> list:
    os.makedirs(directory, exist_ok=True)
    written = []
    for stem, model, manifest in (
        ("quadruped", quadruped_urdf(), QUADRUPED_MANIFEST),
        ("hexapod", hexapod_urdf(), HEXAPOD_MANIFEST),
        ("planar", planar_urdf(), {"length_tolerance": 0.10}),
    ):
        os.makedirs(os.path.join(directory, stem), exist_ok=True)
        write(model, os.path.join(directory, stem, f"{stem}.urdf"))
        with open(os.path.join(directory, stem, f"{stem}.json"), "w") as manifest_file:
            json.dump(manifest, manifest_file, indent=2)
            manifest_file.write("\n")
        written += [f"{stem}/{stem}.urdf", f"{stem}/{stem}.json"]
    return written


if __name__ == "__main__":
    print("wrote", ", ".join(write_examples()))
