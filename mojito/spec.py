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
import math
from dataclasses import dataclass, field, replace

import numpy as np

from . import kinematics, urdf as urdf_mod
from .kinematics import Chain

_ROUND = 9
_EPS = 1e-12


class SpecError(ValueError):
    """The URDF and manifest cannot be turned into a consistent model description."""


class SpecMismatch(SpecError):
    """Saved weights and a URDF-derived spec describe different limbs."""


# ----------------------------------------------------------------- small helpers
def rpy_matrix(rpy) -> np.ndarray:
    """URDF fixed-axis rotation: R = Rz(yaw) Ry(pitch) Rx(roll)."""
    r, p, y = (float(a) for a in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _r(a):
    return np.round(np.asarray(a, dtype=float), _ROUND).tolist()


def _diff(a, b, path="") -> list:
    """Human-readable differences between two JSON-like structures."""
    out = []
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a:
                out.append(f"{path}{k}: missing in first, {b[k]!r} in second")
            elif k not in b:
                out.append(f"{path}{k}: {a[k]!r} in first, missing in second")
            else:
                out += _diff(a[k], b[k], f"{path}{k}.")
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path.rstrip('.')}: length {len(a)} vs {len(b)}")
        else:
            for i, (x, y) in enumerate(zip(a, b)):
                out += _diff(x, y, f"{path.rstrip('.')}[{i}].")
    elif a != b:
        out.append(f"{path.rstrip('.')}: {a!r} vs {b!r}")
    return out


@dataclass(frozen=True)
class Mirror:
    """How a limb relates to the canonical limb of its group.

    FK_this(sign_q * q, params) == flip_p * FK_canonical(q, params)
    """

    flip_p: tuple = (1.0, -1.0, 1.0)
    sign_q: tuple = (-1.0, 1.0, 1.0)

    def target(self, p):
        """Map a target for this limb into the canonical limb's frame (and back)."""
        return np.asarray(p, dtype=float) * np.asarray(self.flip_p)

    def angles(self, q):
        """Map canonical joint values to this limb's convention (and back)."""
        return np.asarray(q, dtype=float) * np.asarray(self.sign_q)


# ----------------------------------------------------------------- one limb
@dataclass(frozen=True, eq=False)
class LimbSpec:
    """One serial limb, derived from a URDF. Also usable wherever a `leg.LegConfig` is."""

    name: str
    chain: Chain
    limits: np.ndarray  # (dof, 2)
    joint_types: tuple  # "revolute" / "prismatic" per joint
    joint_names: tuple  # URDF joint names, in q order
    nominal: np.ndarray  # (n_params,) link lengths [m], as read from the URDF
    param_joints: tuple  # URDF joint whose origin translation each parameter scales
    length_tolerance: float = 0.10
    reach: float = 0.0  # length scale that makes positions dimensionless
    mount_R: np.ndarray = field(default_factory=lambda: np.eye(3))  # limb frame in the base frame
    mount_t: np.ndarray = field(default_factory=lambda: np.zeros(3))

    backend_agnostic = True  # forward/jacobian accept NumPy arrays and torch tensors

    @property
    def dof(self) -> int:
        return self.chain.dof

    @property
    def n_params(self) -> int:
        return self.chain.n_params

    mirror = None  # LegConfig compatibility; a limb's mirror is a property of its group member

    def forward(self, q, params):
        return kinematics.fk(self.chain, q, params)

    def jacobian(self, q, params):
        return kinematics.jacobian(self.chain, q, params)

    def points(self, q, params):
        return kinematics.points(self.chain, q, params)

    def numeric_ik(self, p, params, **kw):
        return kinematics.numeric_ik(self.chain, self.limits, p, params, **kw)

    # ---- identity ----
    def structure(self, exact: bool = False) -> dict:
        """Everything that must match between a URDF and saved weights. No lengths.

        Rounded to 1e-9 unless `exact`, so signatures ignore float noise.
        """
        ch, r = self.chain, (lambda a: np.asarray(a, dtype=float).tolist()) if exact else _r
        return dict(
            dof=self.dof,
            n_params=self.n_params,
            joint_types=list(self.joint_types),
            limits=r(self.limits),
            elements=[
                dict(dir=r(ch.dirs[e]), param=int(ch.param_index[e]), rot=r(ch.rot[e]),
                     joint=int(ch.joint_index[e]), axis=r(ch.axes[e]))
                for e in range(len(ch.param_index))
            ],
        )

    @property
    def signature(self) -> str:
        return hashlib.sha256(json.dumps(self.structure(), sort_keys=True).encode()).hexdigest()[:16]

    def describe_mismatch(self, other: "LimbSpec") -> list:
        return _diff(self.structure(), other.structure())

    # ---- serialisation (stored inside weight files) ----
    def to_dict(self) -> dict:
        d = self.structure(exact=True)
        d.update(name=self.name, joint_names=list(self.joint_names), nominal=self.nominal.tolist(),
                 param_joints=list(self.param_joints), length_tolerance=self.length_tolerance,
                 reach=self.reach, mount_R=self.mount_R.tolist(), mount_t=self.mount_t.tolist(),
                 signature=self.signature)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "LimbSpec":
        els = d["elements"]
        rot = np.array([e["rot"] for e in els], dtype=float).reshape(len(els), 3, 3)
        chain = Chain(
            dirs=np.array([e["dir"] for e in els], dtype=float),
            param_index=tuple(int(e["param"]) for e in els),
            rot=rot,
            has_rot=tuple(not np.allclose(r, np.eye(3)) for r in rot),
            joint_index=tuple(int(e["joint"]) for e in els),
            axes=np.array([e["axis"] for e in els], dtype=float),
            prismatic=tuple(e["joint"] >= 0 and d["joint_types"][e["joint"]] == "prismatic" for e in els),
        )
        spec = cls(
            name=d.get("name", "limb"), chain=chain, limits=np.array(d["limits"], dtype=float),
            joint_types=tuple(d["joint_types"]), joint_names=tuple(d.get("joint_names", ())),
            nominal=np.array(d["nominal"], dtype=float), param_joints=tuple(d.get("param_joints", ())),
            length_tolerance=float(d["length_tolerance"]), reach=float(d["reach"]),
            mount_R=np.array(d.get("mount_R", np.eye(3)), dtype=float),
            mount_t=np.array(d.get("mount_t", np.zeros(3)), dtype=float),
        )
        if "signature" in d and d["signature"] != spec.signature:
            raise SpecError("stored limb description is corrupt: its signature does not match its contents")
        return spec

    def with_normalisation(self, nominal, length_tolerance, reach) -> "LimbSpec":
        """Same limb, different input scaling. Used when a network trained at other nominal lengths is loaded."""
        return replace(self, nominal=np.asarray(nominal, dtype=float), length_tolerance=float(length_tolerance),
                       reach=float(reach))

    def check_params(self, params, tolerance: float | None = None) -> list:
        """Warnings for any length outside the range the network was trained on."""
        tol = self.length_tolerance if tolerance is None else tolerance
        params = np.asarray(params, dtype=float)
        dev = params / self.nominal - 1.0
        return [f"parameter {i} ({self.param_joints[i] if i < len(self.param_joints) else '?'}): "
                f"{params[i]*1000:.1f} mm is {dev[i]:+.1%} from nominal, outside the trained ±{tol:.0%}"
                for i in range(self.n_params) if abs(dev[i]) > tol + 1e-9]


def build_limb(model: urdf_mod.URDFModel, base_link: str, tip: str, name: str | None = None,
               length_tolerance: float = 0.10, reach: float | None = None) -> LimbSpec:
    """Turn the joints between `base_link` and `tip` into a LimbSpec."""
    name = name or tip
    joints = model.chain(base_link, tip)
    if not joints:
        raise SpecError(f"limb {name!r}: tip {tip!r} is the base link itself")
    for j in joints:
        if j.mimic:
            raise SpecError(f"limb {name!r}: joint {j.name!r} uses <mimic>, which is not supported")
        if j.type in ("floating", "planar"):
            raise SpecError(f"limb {name!r}: joint {j.name!r} is {j.type}; only revolute, continuous, "
                            "prismatic and fixed joints are supported")
    for j in joints[:-1]:  # serial: no branching below the first joint
        if len(model.children(j.child)) != 1:
            raise SpecError(f"limb {name!r}: link {j.child!r} branches; only serial chains are supported")

    # Everything up to and including the first movable joint's origin is the mount.
    mount_R, mount_t, k = np.eye(3), np.zeros(3), 0
    while k < len(joints) and not joints[k].movable:
        mount_t = mount_t + mount_R @ np.array(joints[k].xyz)
        mount_R = mount_R @ rpy_matrix(joints[k].rpy)
        k += 1
    if k == len(joints):
        raise SpecError(f"limb {name!r} has no movable joint between {base_link!r} and {tip!r}")
    first = joints[k]
    mount_t = mount_t + mount_R @ np.array(first.xyz)
    mount_R = mount_R @ rpy_matrix(first.rpy)

    dirs, pidx, rots, jidx, axes, prismatic = [], [], [], [], [], []
    nominal, param_joints, jtypes, jnames, limits = [], [], [], [], []

    def add_joint(j):
        a = np.array(j.axis, dtype=float)
        n = np.linalg.norm(a)
        if n < _EPS:
            raise SpecError(f"limb {name!r}: joint {j.name!r} has a zero axis")
        jidx.append(len(jnames))
        axes.append(a / n)
        prismatic.append(j.type == "prismatic")
        jtypes.append("prismatic" if j.type == "prismatic" else "revolute")
        jnames.append(j.name)
        lo, hi = j.limits
        if not hi > lo:
            raise SpecError(f"limb {name!r}: joint {j.name!r} has an empty range [{lo}, {hi}]")
        limits.append((lo, hi))

    dirs.append(np.zeros(3)); pidx.append(-1); rots.append(np.eye(3))
    add_joint(first)
    for j in joints[k + 1:]:
        xyz = np.array(j.xyz, dtype=float)
        length = np.linalg.norm(xyz)
        if length > _EPS:
            dirs.append(xyz / length)
            pidx.append(len(nominal))
            nominal.append(length)
            param_joints.append(j.name)
        else:
            dirs.append(np.zeros(3))
            pidx.append(-1)
        rots.append(rpy_matrix(j.rpy))
        if j.movable:
            add_joint(j)
        else:
            jidx.append(-1)
            axes.append(np.zeros(3))
            prismatic.append(False)
    if not nominal:
        raise SpecError(f"limb {name!r} has no link with a non-zero length, so there is nothing to parameterise")

    rots = np.array(rots)
    chain = Chain(
        dirs=np.array(dirs), param_index=tuple(pidx), rot=rots,
        has_rot=tuple(not np.allclose(r, np.eye(3), atol=1e-12) for r in rots),
        joint_index=tuple(jidx), axes=np.array(axes), prismatic=tuple(prismatic),
    )
    nominal = np.array(nominal)
    return LimbSpec(
        name=name, chain=chain, limits=np.array(limits, dtype=float), joint_types=tuple(jtypes),
        joint_names=tuple(jnames), nominal=nominal, param_joints=tuple(param_joints),
        length_tolerance=length_tolerance, reach=float(nominal.sum() if reach is None else reach),
        mount_R=mount_R, mount_t=mount_t,
    )


def derive_mirror(canonical: LimbSpec, other: LimbSpec, axis: str = "y", seed: int = 0) -> Mirror:
    """Find and verify the sign flips that make `other` the mirror image of `canonical`.

    Raises SpecError if no assignment of joint signs works.
    """
    if (canonical.dof, canonical.n_params) != (other.dof, other.n_params):
        raise SpecError(f"limbs {canonical.name!r} and {other.name!r} have different joint or parameter counts")
    flip = np.ones(3)
    flip["xyz".index(axis)] = -1.0
    rng = np.random.default_rng(seed)
    lo, hi = canonical.limits[:, 0], canonical.limits[:, 1]
    q = rng.uniform(lo, hi, size=(16, canonical.dof))
    params = canonical.nominal * (1 + rng.uniform(-0.1, 0.1, size=(16, canonical.n_params)))
    want = canonical.forward(q, params) * flip
    for signs in np.array(np.meshgrid(*[[1.0, -1.0]] * canonical.dof)).reshape(canonical.dof, -1).T:
        got = other.forward(q * signs, params)
        if not np.allclose(got, want, atol=1e-9):
            continue
        mapped = np.sort(canonical.limits * signs[:, None], axis=1)
        if not np.allclose(mapped, other.limits, atol=1e-9):
            continue
        return Mirror(tuple(flip), tuple(signs))
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
        for g in self.groups.values():
            if any(m.name == limb for m in g.members):
                return g
        raise KeyError(limb)

    def member(self, limb: str) -> Member:
        return next(m for m in self.group_of(limb).members if m.name == limb)

    def params(self) -> dict:
        """Current link lengths of every limb, as read from the URDF."""
        return {n: l.nominal.copy() for n, l in self.limbs.items()}

    def write_back(self, params: dict) -> urdf_mod.URDFModel:
        """A copy of the URDF with the given link lengths applied (for example calibrated ones)."""
        return apply_params(self.urdf, self.limbs, params)

    def summary(self) -> str:
        lines = [f"robot {self.name!r}: base {self.base_link!r}, {len(self.limbs)} limbs, {len(self.groups)} model group(s)"]
        for g in self.groups.values():
            c = g.canonical
            lines.append(f"  group {g.name!r}: {c.dof} DoF, {c.n_params} lengths, signature {c.signature}, "
                         f"reach {c.reach*1000:.0f} mm, tolerance ±{c.length_tolerance:.0%}")
            lines.append(f"    joints: " + ", ".join(
                f"{n} [{lo:+.2f}, {hi:+.2f}]" for n, (lo, hi) in zip(c.joint_names, c.limits)))
            lines.append(f"    lengths [mm]: " + ", ".join(
                f"{j}={v*1000:.1f}" for j, v in zip(c.param_joints, c.nominal)))
            for m in g.members:
                tag = f"mirror (flip_p={list(m.mirror.flip_p)}, sign_q={list(m.mirror.sign_q)})" if m.mirror else "canonical"
                lines.append(f"    - {m.name}: {tag}, mount at {np.round(m.limb.mount_t, 4).tolist()}")
        return "\n".join(lines)

    def validate(self) -> list:
        """Problems that are not outright errors. Empty list = clean."""
        issues = []
        for g in self.groups.values():
            for m in g.members:
                if m.limb.signature != g.canonical.signature and m.mirror is None:
                    issues.append(f"limb {m.name!r} no longer matches group {g.name!r}")
                if m.mirror is not None:
                    try:
                        derive_mirror(g.canonical, m.limb, axis="xyz"[int(np.argmin(m.mirror.flip_p))])
                    except SpecError as e:
                        issues.append(str(e))
                issues += [f"limb {m.name!r}: {w}" for w in g.canonical.check_params(m.limb.nominal)]
        for n, l in self.limbs.items():
            q = l.limits.mean(axis=1)
            if not np.all(np.isfinite(l.forward(q, l.nominal))):
                issues.append(f"limb {n!r}: forward kinematics is not finite at mid-range")
        return issues


def apply_params(model: urdf_mod.URDFModel, limbs: dict, params: dict) -> urdf_mod.URDFModel:
    """Copy of `model` with each limb's link lengths replaced. params: limb name -> (n_params,)."""
    out = copy.deepcopy(model)
    for name, p in params.items():
        limb = limbs[name]
        p = np.asarray(p, dtype=float)
        if p.shape != (limb.n_params,):
            raise SpecError(f"limb {name!r} takes {limb.n_params} lengths, got shape {p.shape}")
        if not np.all(p > 0):
            raise SpecError(f"limb {name!r}: lengths must be positive, got {p.tolist()}")
        elems = [e for e in range(len(limb.chain.param_index)) if limb.chain.param_index[e] >= 0]
        for e in elems:
            i = limb.chain.param_index[e]
            j = out.joints[limb.param_joints[i]]
            j.xyz = tuple(float(v) for v in limb.chain.dirs[e] * p[i])
    return out


def params_from_urdf(model: urdf_mod.URDFModel, limb: LimbSpec) -> np.ndarray:
    """Read a limb's link lengths out of a (possibly updated) URDF."""
    return np.array([np.linalg.norm(model.joints[j].xyz) for j in limb.param_joints])


# ----------------------------------------------------------------- building from URDF + manifest
def from_urdf(model: urdf_mod.URDFModel, manifest: dict | None = None) -> RobotSpec:
    manifest = manifest or {}
    base = manifest.get("base_link") or model.root()
    if base not in model.links:
        raise SpecError(f"base_link {base!r} is not a link in the URDF")
    tol = float(manifest.get("length_tolerance", 0.10))
    overrides = manifest.get("groups", {})

    decl = manifest.get("limbs")
    if decl is None:
        decl = [dict(name=t, tip=t) for t in model.leaves(base)]
    if not decl:
        raise SpecError("no limbs: the URDF has no leaf links below the base")
    names = [d.get("name", d.get("tip")) for d in decl]
    if len(set(names)) != len(names):
        raise SpecError(f"duplicate limb names in manifest: {names}")

    spec = RobotSpec(name=model.name, base_link=base, urdf=model)
    auto = 0
    for d in decl:
        name = d.get("name", d.get("tip"))
        if "tip" not in d:
            raise SpecError(f"manifest limb {name!r} has no 'tip' link")
        gname = d.get("group")
        reach = overrides.get(gname, {}).get("reach") if gname else None
        limb = build_limb(model, base, d["tip"], name, d.get("length_tolerance", tol), reach)
        spec.limbs[name] = limb

        target, mirror = None, None
        if d.get("mirror_of"):
            src = d["mirror_of"]
            if src not in spec.limbs:
                raise SpecError(f"limb {name!r}: mirror_of {src!r} must be declared before it")
            target = spec.group_of(src)
            mirror = derive_mirror(target.canonical, limb, d.get("mirror_axis", "y"))
        else:
            for g in spec.groups.values():
                if g.canonical.signature == limb.signature:
                    target = g
                    break
            if target is None:
                for g in spec.groups.values():
                    try:
                        mirror = derive_mirror(g.canonical, limb)
                        target = g
                        break
                    except SpecError:
                        continue
            if target is not None and gname and gname != target.name:
                target, mirror = None, None
            if target is None and gname in spec.groups:
                raise SpecError(
                    f"limb {name!r} is declared in group {gname!r} but differs from it:\n  "
                    + "\n  ".join(spec.groups[gname].canonical.describe_mismatch(limb)))
        if target is None:
            gn = gname or f"group{auto}"
            auto += 0 if gname else 1
            target = Group(gn, limb)
            spec.groups[gn] = target
        target.members.append(Member(name, limb, mirror))
    return spec


def load_manifest(path: str | None) -> dict:
    if path is None:
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_robot(urdf_path: str, manifest_path: str | None = None) -> RobotSpec:
    """Read a URDF file (and optional manifest) and build the RobotSpec."""
    return from_urdf(urdf_mod.parse(urdf_path), load_manifest(manifest_path))


def main(argv=None):
    import argparse

    ap = argparse.ArgumentParser(description="Show how a URDF maps onto model groups, and check it.")
    ap.add_argument("urdf")
    ap.add_argument("--manifest")
    args = ap.parse_args(argv)
    spec = load_robot(args.urdf, args.manifest)
    print(spec.summary())
    issues = spec.validate()
    print("\nvalidation:", "ok" if not issues else "")
    for i in issues:
        print("  -", i)
    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
