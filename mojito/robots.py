"""Generators for the example robots in robots/.

The checked-in .urdf files are produced by `python -m mojito.robots`, and a
test keeps them in sync, so the examples cannot drift from these numbers.
Everything here is built from the `urdf` module; it is also how a `LegConfig`
is turned into a URDF for the compatibility path.
"""
from __future__ import annotations

import json
import os

from . import leg
from .urdf import Joint, URDFModel, write


def _leg_joints(prefix: str, parent: str, mount: tuple, right: bool, lengths, limits) -> tuple:
    """Abduction (x), hip pitch (y), knee (y), fixed foot. Right legs mirror the lateral offset."""
    la, lu, ll = (float(v) for v in lengths)
    side = -1.0 if right else 1.0
    a, h, k = limits
    mk = lambda name, parent_, child, xyz, axis, lim: Joint(
        name, "revolute", parent_, child, xyz=xyz, axis=axis, lower=lim[0], upper=lim[1])
    p = prefix
    joints = [
        mk(f"{p}_abduction", parent, f"{p}_hip", mount, (1.0, 0.0, 0.0), _flip(a, right)),
        mk(f"{p}_hip_pitch", f"{p}_hip", f"{p}_thigh", (0.0, side * la, 0.0), (0.0, 1.0, 0.0), h),
        mk(f"{p}_knee", f"{p}_thigh", f"{p}_shank", (0.0, 0.0, -lu), (0.0, 1.0, 0.0), k),
        Joint(f"{p}_foot", "fixed", f"{p}_shank", f"{p}_foot", xyz=(0.0, 0.0, -ll)),
    ]
    return [f"{p}_hip", f"{p}_thigh", f"{p}_shank", f"{p}_foot"], joints


def _flip(lim, right):
    return (-lim[1], -lim[0]) if right else tuple(lim)


def leg_urdf(lengths=None, limits=None, name: str = "leg") -> URDFModel:
    """A single left leg hanging from a body link: the Stage 0 placeholder."""
    lengths = leg.NOMINAL_LENGTHS if lengths is None else lengths
    limits = leg.JOINT_LIMITS if limits is None else limits
    links, joints = _leg_joints("L", "body", (0.0, 0.0, 0.0), False, lengths, limits)
    return URDFModel(name, ["body"] + links, {j.name: j for j in joints})


def legged_urdf(name: str, mounts: dict, lengths=None, limits=None, body_size=None) -> URDFModel:
    """A body with one leg per entry in `mounts` (name -> (x, y, z)); y < 0 is a right leg."""
    lengths = leg.NOMINAL_LENGTHS if lengths is None else lengths
    limits = leg.JOINT_LIMITS if limits is None else limits
    links, joints = ["body"], {}
    for leg_name, xyz in mounts.items():
        l, js = _leg_joints(leg_name, "body", xyz, xyz[1] < 0, lengths, limits)
        links += l
        joints.update({j.name: j for j in js})
    return URDFModel(name, links, joints)


def quadruped_urdf(body_length=0.24, body_width=0.10, lengths=None, limits=None) -> URDFModel:
    """The Stage 1 placeholder quadruped. Leg order FL, FR, RL, RR matches mojito.body."""
    x, y = body_length / 2, body_width / 2
    return legged_urdf("mojito_quadruped", {"FL": (x, y, 0.0), "FR": (x, -y, 0.0),
                                            "RL": (-x, y, 0.0), "RR": (-x, -y, 0.0)}, lengths, limits)


def hexapod_urdf(body_length=0.36, body_width=0.12, lengths=None, limits=None) -> URDFModel:
    """Three legs a side, to show the pipeline is not tied to four legs."""
    xs, y = (body_length / 2, 0.0, -body_length / 2), body_width / 2
    mounts = {}
    for tag, x in zip(("F", "M", "R"), xs):
        mounts[f"{tag}L"] = (x, y, 0.0)
        mounts[f"{tag}R"] = (x, -y, 0.0)
    return legged_urdf("mojito_hexapod", mounts, lengths, limits)


def planar_urdf(upper=0.15, lower=0.15) -> URDFModel:
    """A two-joint planar leg (hip and knee about y): fewer DoF than the quadruped."""
    j = [
        Joint("hip", "revolute", "body", "thigh", axis=(0.0, 1.0, 0.0), lower=0.0, upper=1.5),
        Joint("knee", "revolute", "thigh", "shank", xyz=(0.0, 0.0, -upper), axis=(0.0, 1.0, 0.0), lower=-2.3, upper=-0.3),
        Joint("foot", "fixed", "shank", "foot", xyz=(0.0, 0.0, -lower)),
    ]
    return URDFModel("mojito_planar", ["body", "thigh", "shank", "foot"], {x.name: x for x in j})


QUADRUPED_MANIFEST = {
    "length_tolerance": 0.10,
    "limbs": [
        {"name": "FL", "tip": "FL_foot", "group": "leg"},
        {"name": "FR", "tip": "FR_foot", "group": "leg", "mirror_of": "FL"},
        {"name": "RL", "tip": "RL_foot", "group": "leg"},
        {"name": "RR", "tip": "RR_foot", "group": "leg", "mirror_of": "RL"},
    ],
    # Positions are divided by this to make them dimensionless. 0.24 m = upper + lower leg.
    "groups": {"leg": {"reach": 0.24}},
}

HEXAPOD_MANIFEST = {
    "length_tolerance": 0.10,
    "limbs": [{"name": n, "tip": f"{n}_foot", "group": "leg", **({"mirror_of": n[0] + "L"} if n[1] == "R" else {})}
              for n in ("FL", "FR", "ML", "MR", "RL", "RR")],
    "groups": {"leg": {"reach": 0.24}},
}


def write_examples(directory: str = "robots") -> list:
    os.makedirs(directory, exist_ok=True)
    out = []
    for stem, model, manifest in (
        ("quadruped", quadruped_urdf(), QUADRUPED_MANIFEST),
        ("hexapod", hexapod_urdf(), HEXAPOD_MANIFEST),
        ("planar", planar_urdf(), {"length_tolerance": 0.10}),
    ):
        write(model, os.path.join(directory, f"{stem}.urdf"))
        with open(os.path.join(directory, f"{stem}.json"), "w") as f:
            json.dump(manifest, f, indent=2)
            f.write("\n")
        out += [f"{stem}.urdf", f"{stem}.json"]
    return out


if __name__ == "__main__":
    print("wrote", ", ".join(write_examples()))
