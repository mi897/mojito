"""The alignment layer between a URDF and the learned model.

    URDF + manifest  ->  RobotSpec  ->  LimbSpec (one per limb)  ->  network

A `LimbSpec` is everything the network needs to know about one limb: how many
joints it has, their limits, the kinematic chain, and which link lengths are
free parameters. It is derived from the URDF, never typed in twice, and it is
saved inside the weight file next to a signature. Loading weights against a URDF
therefore fails loudly, with a readable diff, if the two no longer describe the
same limb (see `describe_mismatch`).

The manifest is a small JSON file kept beside the URDF (so the URDF stays a
plain, portable URDF):

    {
      "base_link": "body",                      # default: the URDF root
      "length_tolerance": 0.10,                 # default fractional spread of link lengths
      "limbs": [                                # default: one limb per leaf link
        {"name": "FL", "tip": "FL_foot", "group": "leg"},
        {"name": "FR", "tip": "FR_foot", "mirror_of": "FL"}
      ],
      "groups": {"leg": {"reach": 0.24}}        # optional per-group overrides
    }

Limbs whose chains are structurally identical share one network (a "group").
A limb that is the mirror image of another, e.g. a right leg and a left leg,
joins the same group; the sign flips that relate them are worked out and
checked numerically rather than assumed.

Parameters. Every link whose joint origin has a non-zero translation gets one
free length: the scale of that translation, with its direction fixed by the
URDF. That is the "link lengths" input of the network. Joint origins' rotations,
axes and limits are structure: they are not learned inputs and must match.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field, replace

import numpy as np

from . import kinematics, urdf as urdf_mod
from .kinematics import Chain
from .limb_model import Mirror

_ROUND = 9
_EPS = 1e-12


class SpecError(ValueError):
    """The URDF and manifest cannot be turned into a consistent model description."""


class SpecMismatch(SpecError):
    """Saved weights and a URDF-derived spec describe different limbs."""


# ----------------------------------------------------------------- small helpers
def rpy_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis rotation: R = Rz(yaw) Ry(pitch) Rx(roll), as a (3, 3) NumPy array."""
    roll, pitch, yaw = (float(angle) for angle in rpy)
    return kinematics.rpy_rotation(roll, pitch, yaw)


def _rounded_list(values):
    """`values` as nested lists of floats rounded to 1e-9, so signatures ignore float noise."""
    return np.round(np.asarray(values, dtype=float), _ROUND).tolist()


def _exact_list(values):
    """`values` as nested lists of floats, unrounded: what a weight file stores."""
    return np.asarray(values, dtype=float).tolist()


def _diff(first, second, path="") -> list:
    """Human-readable differences between two JSON-like structures."""
    differences = []
    if isinstance(first, dict) and isinstance(second, dict):
        for key in sorted(set(first) | set(second)):
            if key not in first:
                differences.append(f"{path}{key}: missing in first, {second[key]!r} in second")
            elif key not in second:
                differences.append(f"{path}{key}: {first[key]!r} in first, missing in second")
            else:
                differences += _diff(first[key], second[key], f"{path}{key}.")
    elif isinstance(first, list) and isinstance(second, list):
        if len(first) != len(second):
            differences.append(f"{path.rstrip('.')}: length {len(first)} vs {len(second)}")
        else:
            for index, (first_item, second_item) in enumerate(zip(first, second)):
                differences += _diff(first_item, second_item, f"{path.rstrip('.')}[{index}].")
    elif first != second:
        differences.append(f"{path.rstrip('.')}: {first!r} vs {second!r}")
    return differences


# ----------------------------------------------------------------- one limb
@dataclass(frozen=True, eq=False)
class LimbSpec:
    """One serial limb, derived from a URDF. Also usable wherever a `leg.LegConfig` is."""

    name: str
    chain: Chain
    limits: np.ndarray  # (dof, 2)
    joint_types: tuple  # "revolute" / "prismatic" per joint
    joint_names: tuple  # URDF joint names, in joint_angles order
    nominal: np.ndarray  # (n_params,) link lengths [m], as read from the URDF
    param_joints: tuple  # URDF joint whose origin translation each parameter scales
    length_tolerance: float = 0.10
    reach: float = 0.0  # length scale that makes positions dimensionless
    mount_R: np.ndarray = field(default_factory=lambda: np.eye(3))  # limb frame in the base frame
    mount_t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Satisfies limb_model.LimbModel, so networks and samplers take a LimbSpec or a leg.LegConfig.
    backend_agnostic = True  # forward/jacobian accept NumPy arrays and torch tensors

    @property
    def dof(self) -> int:
        return self.chain.dof

    @property
    def n_params(self) -> int:
        return self.chain.n_params

    mirror = None  # a URDF limb's mirror belongs to its group Member, not to the limb itself

    def forward(self, joint_angles, link_lengths):
        return kinematics.fk(self.chain, joint_angles, link_lengths)

    def jacobian(self, joint_angles, link_lengths):
        return kinematics.jacobian(self.chain, joint_angles, link_lengths)

    def points(self, joint_angles, link_lengths):
        return kinematics.points(self.chain, joint_angles, link_lengths)

    def numeric_ik(self, target_position, link_lengths, **solver_options):
        return kinematics.numeric_ik(self.chain, self.limits, target_position, link_lengths, **solver_options)

    # ---- identity ----
    def structure(self, exact: bool = False) -> dict:
        """Everything that must match between a URDF and saved weights. No lengths.

        Rounded to 1e-9 unless `exact`, so signatures ignore float noise.
        """
        chain, as_list = self.chain, _exact_list if exact else _rounded_list
        return dict(
            dof=self.dof,
            n_params=self.n_params,
            joint_types=list(self.joint_types),
            limits=as_list(self.limits),
            elements=[
                dict(dir=as_list(chain.dirs[element]), param=int(chain.param_index[element]), rot=as_list(chain.rot[element]),
                     joint=int(chain.joint_index[element]), axis=as_list(chain.axes[element]))
                for element in range(len(chain.param_index))
            ],
        )

    @property
    def signature(self) -> str:
        return hashlib.sha256(json.dumps(self.structure(), sort_keys=True).encode()).hexdigest()[:16]

    def describe_mismatch(self, other: "LimbSpec") -> list:
        return _diff(self.structure(), other.structure())

    # ---- serialisation (stored inside weight files) ----
    def to_dict(self) -> dict:
        description = self.structure(exact=True)
        description.update(name=self.name, joint_names=list(self.joint_names), nominal=self.nominal.tolist(),
                           param_joints=list(self.param_joints), length_tolerance=self.length_tolerance,
                           reach=self.reach, mount_R=self.mount_R.tolist(), mount_t=self.mount_t.tolist(),
                           signature=self.signature)
        return description

    @classmethod
    def from_dict(cls, description: dict) -> "LimbSpec":
        elements = description["elements"]
        joint_types = description["joint_types"]
        rotations = np.array([element["rot"] for element in elements], dtype=float).reshape(len(elements), 3, 3)
        chain = Chain(
            dirs=np.array([element["dir"] for element in elements], dtype=float),
            param_index=tuple(int(element["param"]) for element in elements),
            rot=rotations,
            has_rot=tuple(not np.allclose(rotation, np.eye(3)) for rotation in rotations),
            joint_index=tuple(int(element["joint"]) for element in elements),
            axes=np.array([element["axis"] for element in elements], dtype=float),
            prismatic=tuple(element["joint"] >= 0 and joint_types[element["joint"]] == "prismatic" for element in elements),
        )
        spec = cls(
            name=description.get("name", "limb"), chain=chain, limits=np.array(description["limits"], dtype=float),
            joint_types=tuple(joint_types), joint_names=tuple(description.get("joint_names", ())),
            nominal=np.array(description["nominal"], dtype=float), param_joints=tuple(description.get("param_joints", ())),
            length_tolerance=float(description["length_tolerance"]), reach=float(description["reach"]),
            mount_R=np.array(description.get("mount_R", np.eye(3)), dtype=float),
            mount_t=np.array(description.get("mount_t", np.zeros(3)), dtype=float),
        )
        if "signature" in description and description["signature"] != spec.signature:
            raise SpecError("stored limb description is corrupt: its signature does not match its contents")
        return spec

    def with_normalisation(self, nominal, length_tolerance, reach) -> "LimbSpec":
        """Same limb, different input scaling. Used when a network trained at other nominal lengths is loaded."""
        return replace(self, nominal=np.asarray(nominal, dtype=float), length_tolerance=float(length_tolerance),
                       reach=float(reach))

    def check_params(self, params, tolerance: float | None = None) -> list:
        """Warnings for any length outside the range the network was trained on."""
        tolerance = self.length_tolerance if tolerance is None else tolerance
        params = np.asarray(params, dtype=float)
        deviation = params / self.nominal - 1.0  # fractional distance from nominal
        return [f"parameter {i} ({self.param_joints[i] if i < len(self.param_joints) else '?'}): "
                f"{params[i]*1000:.1f} mm is {deviation[i]:+.1%} from nominal, outside the trained ±{tolerance:.0%}"
                for i in range(self.n_params) if abs(deviation[i]) > tolerance + 1e-9]


def build_limb(model: urdf_mod.URDFModel, base_link: str, tip: str, name: str | None = None,
               length_tolerance: float = 0.10, reach: float | None = None) -> LimbSpec:
    """Turn the joints between `base_link` and `tip` into a LimbSpec."""
    name = name or tip
    joints = model.chain(base_link, tip)
    if not joints:
        raise SpecError(f"limb {name!r}: tip {tip!r} is the base link itself")
    for joint in joints:
        if joint.mimic:
            raise SpecError(f"limb {name!r}: joint {joint.name!r} uses <mimic>, which is not supported")
        if joint.type in ("floating", "planar"):
            raise SpecError(f"limb {name!r}: joint {joint.name!r} is {joint.type}; only revolute, continuous, "
                            "prismatic and fixed joints are supported")
    for joint in joints[:-1]:  # serial: no branching below the first joint
        if len(model.children(joint.child)) != 1:
            raise SpecError(f"limb {name!r}: link {joint.child!r} branches; only serial chains are supported")

    # Everything up to and including the first movable joint's origin is the mount.
    mount_R, mount_t, first_movable = np.eye(3), np.zeros(3), 0
    while first_movable < len(joints) and not joints[first_movable].movable:
        mount_t = mount_t + mount_R @ np.array(joints[first_movable].xyz)
        mount_R = mount_R @ rpy_matrix(joints[first_movable].rpy)
        first_movable += 1
    if first_movable == len(joints):
        raise SpecError(f"limb {name!r} has no movable joint between {base_link!r} and {tip!r}")
    first = joints[first_movable]
    mount_t = mount_t + mount_R @ np.array(first.xyz)
    mount_R = mount_R @ rpy_matrix(first.rpy)

    # One entry per chain element (the fields of kinematics.Chain) ...
    dirs, param_index, rotations, joint_index, axes, prismatic = [], [], [], [], [], []
    # ... and one per free length or per movable joint.
    nominal, param_joints, joint_types, joint_names, limits = [], [], [], [], []

    def add_joint(joint):
        axis = np.array(joint.axis, dtype=float)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < _EPS:
            raise SpecError(f"limb {name!r}: joint {joint.name!r} has a zero axis")
        joint_index.append(len(joint_names))
        axes.append(axis / axis_norm)
        prismatic.append(joint.type == "prismatic")
        joint_types.append("prismatic" if joint.type == "prismatic" else "revolute")
        joint_names.append(joint.name)
        lower, upper = joint.limits
        if not upper > lower:
            raise SpecError(f"limb {name!r}: joint {joint.name!r} has an empty range [{lower}, {upper}]")
        limits.append((lower, upper))

    dirs.append(np.zeros(3))
    param_index.append(-1)
    rotations.append(np.eye(3))
    add_joint(first)
    for joint in joints[first_movable + 1:]:
        xyz = np.array(joint.xyz, dtype=float)
        length = np.linalg.norm(xyz)
        if length > _EPS:
            dirs.append(xyz / length)
            param_index.append(len(nominal))
            nominal.append(length)
            param_joints.append(joint.name)
        else:
            dirs.append(np.zeros(3))
            param_index.append(-1)
        rotations.append(rpy_matrix(joint.rpy))
        if joint.movable:
            add_joint(joint)
        else:
            joint_index.append(-1)
            axes.append(np.zeros(3))
            prismatic.append(False)
    if not nominal:
        raise SpecError(f"limb {name!r} has no link with a non-zero length, so there is nothing to parameterise")

    rotations = np.array(rotations)
    chain = Chain(
        dirs=np.array(dirs), param_index=tuple(param_index), rot=rotations,
        has_rot=tuple(not np.allclose(rotation, np.eye(3), atol=1e-12) for rotation in rotations),
        joint_index=tuple(joint_index), axes=np.array(axes), prismatic=tuple(prismatic),
    )
    nominal = np.array(nominal)
    return LimbSpec(
        name=name, chain=chain, limits=np.array(limits, dtype=float), joint_types=tuple(joint_types),
        joint_names=tuple(joint_names), nominal=nominal, param_joints=tuple(param_joints),
        length_tolerance=length_tolerance, reach=float(nominal.sum() if reach is None else reach),
        mount_R=mount_R, mount_t=mount_t,
    )


def derive_mirror(canonical: LimbSpec, other: LimbSpec, axis: str = "y", seed: int = 0) -> Mirror:
    """Find and verify the sign flips that make `other` the mirror image of `canonical`.

    Tries every joint_sign in {+1, -1}^dof and keeps the first for which, at 16 random joint values and lengths,
        other.forward(joint_sign * joint_values) == position_flip * canonical.forward(joint_values)
    and the canonical limits times joint_sign are the other limb's limits.
    Raises SpecError if no assignment of joint signs works.
    """
    if (canonical.dof, canonical.n_params) != (other.dof, other.n_params):
        raise SpecError(f"limbs {canonical.name!r} and {other.name!r} have different joint or parameter counts")
    position_flip = np.ones(3)
    position_flip["xyz".index(axis)] = -1.0
    rng = np.random.default_rng(seed)
    lower, upper = canonical.limits[:, 0], canonical.limits[:, 1]
    joint_values = rng.uniform(lower, upper, size=(16, canonical.dof))
    params = canonical.nominal * (1 + rng.uniform(-0.1, 0.1, size=(16, canonical.n_params)))
    expected = canonical.forward(joint_values, params) * position_flip
    for joint_sign in np.array(np.meshgrid(*[[1.0, -1.0]] * canonical.dof)).reshape(canonical.dof, -1).T:
        mirrored = other.forward(joint_values * joint_sign, params)
        if not np.allclose(mirrored, expected, atol=1e-9):
            continue
        mapped_limits = np.sort(canonical.limits * joint_sign[:, None], axis=1)
        if not np.allclose(mapped_limits, other.limits, atol=1e-9):
            continue
        return Mirror(tuple(position_flip), tuple(joint_sign))
    raise SpecError(f"limb {other.name!r} is not the {axis}-mirror of {canonical.name!r}: no joint sign "
                    "assignment reproduces its kinematics and joint limits")


# ----------------------------------------------------------------- whole robot
@dataclass
class Member:
    """One physical limb inside a group."""

    name: str
    limb: LimbSpec  # this limb's own spec (own lengths, own mount)
    mirror: Mirror | None  # None for the canonical limb

    @property
    def mirrored(self) -> bool:
        return self.mirror is not None


@dataclass
class Group:
    """Limbs that share one network."""

    name: str
    canonical: LimbSpec
    members: list = field(default_factory=list)


@dataclass
class RobotSpec:
    name: str
    base_link: str
    urdf: urdf_mod.URDFModel = field(repr=False)
    limbs: dict = field(default_factory=dict)  # name -> LimbSpec, in order
    groups: dict = field(default_factory=dict)  # name -> Group

    def group_of(self, limb: str) -> Group:
        for group in self.groups.values():
            if any(member.name == limb for member in group.members):
                return group
        raise KeyError(limb)

    def member(self, limb: str) -> Member:
        return next(member for member in self.group_of(limb).members if member.name == limb)

    def params(self) -> dict:
        """Current link lengths of every limb, as read from the URDF."""
        return {name: limb.nominal.copy() for name, limb in self.limbs.items()}

    def write_back(self, params: dict) -> urdf_mod.URDFModel:
        """A copy of the URDF with the given link lengths applied (for example calibrated ones)."""
        return apply_params(self.urdf, self.limbs, params)

    def summary(self) -> str:
        lines = [f"robot {self.name!r}: base {self.base_link!r}, {len(self.limbs)} limbs, {len(self.groups)} model group(s)"]
        for group in self.groups.values():
            canonical = group.canonical
            lines.append(f"  group {group.name!r}: {canonical.dof} DoF, {canonical.n_params} lengths, "
                         f"signature {canonical.signature}, reach {canonical.reach*1000:.0f} mm, "
                         f"tolerance ±{canonical.length_tolerance:.0%}")
            lines.append("    joints: " + ", ".join(
                f"{joint_name} [{lower:+.2f}, {upper:+.2f}]"
                for joint_name, (lower, upper) in zip(canonical.joint_names, canonical.limits)))
            lines.append("    lengths [mm]: " + ", ".join(
                f"{joint_name}={length*1000:.1f}" for joint_name, length in zip(canonical.param_joints, canonical.nominal)))
            for member in group.members:
                mirror = member.mirror
                tag = (f"mirror (position_flip={[float(v) for v in mirror.position_flip]}, "
                       f"joint_sign={[float(v) for v in mirror.joint_sign]})"
                       if mirror else "canonical")
                lines.append(f"    - {member.name}: {tag}, mount at {np.round(member.limb.mount_t, 4).tolist()}")
        return "\n".join(lines)

    def validate(self) -> list:
        """Problems that are not outright errors. Empty list = clean."""
        issues = []
        for group in self.groups.values():
            for member in group.members:
                if member.limb.signature != group.canonical.signature and member.mirror is None:
                    issues.append(f"limb {member.name!r} no longer matches group {group.name!r}")
                if member.mirror is not None:
                    mirror_axis = "xyz"[int(np.argmin(member.mirror.position_flip))]  # the axis whose sign flips
                    try:
                        derive_mirror(group.canonical, member.limb, axis=mirror_axis)
                    except SpecError as error:
                        issues.append(str(error))
                issues += [f"limb {member.name!r}: {warning}" for warning in group.canonical.check_params(member.limb.nominal)]
        for name, limb in self.limbs.items():
            mid_range = limb.limits.mean(axis=1)
            if not np.all(np.isfinite(limb.forward(mid_range, limb.nominal))):
                issues.append(f"limb {name!r}: forward kinematics is not finite at mid-range")
        return issues


def apply_params(model: urdf_mod.URDFModel, limbs: dict, params: dict) -> urdf_mod.URDFModel:
    """Copy of `model` with each limb's link lengths replaced. params: limb name -> (n_params,)."""
    updated = copy.deepcopy(model)
    for name, link_lengths in params.items():
        limb = limbs[name]
        link_lengths = np.asarray(link_lengths, dtype=float)
        if link_lengths.shape != (limb.n_params,):
            raise SpecError(f"limb {name!r} takes {limb.n_params} lengths, got shape {link_lengths.shape}")
        if not np.all(link_lengths > 0):
            raise SpecError(f"limb {name!r}: lengths must be positive, got {link_lengths.tolist()}")
        for element, length_index in enumerate(limb.chain.param_index):
            if length_index < 0:  # this element has no free length
                continue
            joint = updated.joints[limb.param_joints[length_index]]
            joint.xyz = tuple(float(v) for v in limb.chain.dirs[element] * link_lengths[length_index])
    return updated


def params_from_urdf(model: urdf_mod.URDFModel, limb: LimbSpec) -> np.ndarray:
    """Read a limb's link lengths out of a (possibly updated) URDF."""
    return np.array([np.linalg.norm(model.joints[joint_name].xyz) for joint_name in limb.param_joints])


# ----------------------------------------------------------------- building from URDF + manifest
def from_urdf(model: urdf_mod.URDFModel, manifest: dict | None = None) -> RobotSpec:
    manifest = manifest or {}
    base = manifest.get("base_link") or model.root()
    if base not in model.links:
        raise SpecError(f"base_link {base!r} is not a link in the URDF")
    length_tolerance = float(manifest.get("length_tolerance", 0.10))
    group_overrides = manifest.get("groups", {})

    declared = manifest.get("limbs")
    if declared is None:
        declared = [dict(name=tip, tip=tip) for tip in model.leaves(base)]
    if not declared:
        raise SpecError("no limbs: the URDF has no leaf links below the base")
    names = [entry.get("name", entry.get("tip")) for entry in declared]
    if len(set(names)) != len(names):
        raise SpecError(f"duplicate limb names in manifest: {names}")

    spec = RobotSpec(name=model.name, base_link=base, urdf=model)
    n_unnamed_groups = 0
    for entry in declared:
        name = entry.get("name", entry.get("tip"))
        if "tip" not in entry:
            raise SpecError(f"manifest limb {name!r} has no 'tip' link")
        group_name = entry.get("group")
        reach = group_overrides.get(group_name, {}).get("reach") if group_name else None
        limb = build_limb(model, base, entry["tip"], name, entry.get("length_tolerance", length_tolerance), reach)
        spec.limbs[name] = limb

        target, mirror = None, None
        if entry.get("mirror_of"):
            source = entry["mirror_of"]
            if source not in spec.limbs:
                raise SpecError(f"limb {name!r}: mirror_of {source!r} must be declared before it")
            target = spec.group_of(source)
            mirror = derive_mirror(target.canonical, limb, entry.get("mirror_axis", "y"))
        else:
            # A limb that names an existing group is checked against that group only (same signature, then mirror), so it
            # is not matched to another group that happens to share its signature or mirror and then rejected.
            candidates = [spec.groups[group_name]] if group_name in spec.groups else list(spec.groups.values())
            for group in candidates:
                if group.canonical.signature == limb.signature:
                    target = group
                    break
            if target is None:
                for group in candidates:
                    try:
                        mirror = derive_mirror(group.canonical, limb)
                        target = group
                        break
                    except SpecError:
                        continue
            if target is not None and group_name and group_name != target.name:
                target, mirror = None, None
            if target is None and group_name in spec.groups:
                raise SpecError(
                    f"limb {name!r} is declared in group {group_name!r} but differs from it:\n  "
                    + "\n  ".join(spec.groups[group_name].canonical.describe_mismatch(limb)))
        if target is None:
            new_group_name = group_name or f"group{n_unnamed_groups}"
            n_unnamed_groups += 0 if group_name else 1
            target = Group(new_group_name, limb)
            spec.groups[new_group_name] = target
        target.members.append(Member(name, limb, mirror))
    return spec


def load_manifest(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def load_robot(urdf_path: str, manifest_path: str | None = None) -> RobotSpec:
    """Read a URDF file (and optional manifest) and build the RobotSpec."""
    return from_urdf(urdf_mod.parse(urdf_path), load_manifest(manifest_path))


def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(description="Show how a URDF maps onto model groups, and check it.")
    parser.add_argument("urdf")
    parser.add_argument("--manifest")
    args = parser.parse_args(argv)
    spec = load_robot(args.urdf, args.manifest)
    print(spec.summary())
    issues = spec.validate()
    print("\nvalidation:", "ok" if not issues else "")
    for issue in issues:
        print("  -", issue)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
