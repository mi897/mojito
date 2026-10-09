"""Read and write URDF robot descriptions. Pure Python: no NumPy, no torch.

Only the kinematic skeleton is interpreted (links, joints, origins, axes,
limits). Meshes, inertias, materials and transmissions are not parsed, but
when a file is read from disk they are kept in the XML tree and written back
untouched, so editing a joint and saving does not strip a real robot's
visuals. `<xacro>` is not supported: expand it to plain URDF first.

This module knows nothing about learning. mojito/spec.py turns a URDFModel
into the structures the network is built from.
"""
from __future__ import annotations

import copy
import math
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

JOINT_TYPES = ("revolute", "continuous", "prismatic", "fixed", "floating", "planar")


class URDFError(ValueError):
    """The file is not a URDF this project can use. The message names the culprit."""


@dataclass
class Joint:
    name: str
    type: str
    parent: str
    child: str
    xyz: tuple = (0.0, 0.0, 0.0)  # origin translation in the parent link frame [m]
    rpy: tuple = (0.0, 0.0, 0.0)  # origin rotation, fixed-axis roll/pitch/yaw [rad]
    axis: tuple = (1.0, 0.0, 0.0)  # in the joint frame
    lower: float | None = None
    upper: float | None = None
    mimic: str | None = None

    @property
    def movable(self) -> bool:
        return self.type != "fixed"

    @property
    def limits(self) -> tuple:
        """(lower, upper). Continuous joints are treated as +/- pi."""
        if self.type == "continuous":
            return (-math.pi, math.pi)
        if self.lower is None or self.upper is None:
            raise URDFError(f"joint {self.name!r} ({self.type}) has no <limit lower upper>")
        return (self.lower, self.upper)


@dataclass
class URDFModel:
    name: str
    links: list  # link names, in file order
    joints: dict  # name -> Joint, in file order
    _tree: ET.Element | None = field(default=None, repr=False, compare=False)

    # ---- structure queries ----
    def root(self) -> str:
        children = {j.child for j in self.joints.values()}
        roots = [l for l in self.links if l not in children]
        return roots[0]

    def parent_joint(self, link: str) -> Joint | None:
        for j in self.joints.values():
            if j.child == link:
                return j
        return None

    def chain(self, base: str, tip: str) -> list:
        """Joints from `base` down to `tip`, in order. Raises if tip is not below base."""
        if tip not in self.links:
            raise URDFError(f"link {tip!r} does not exist")
        out, link = [], tip
        while link != base:
            j = self.parent_joint(link)
            if j is None:
                raise URDFError(f"link {tip!r} is not a descendant of {base!r}")
            out.append(j)
            link = j.parent
        return out[::-1]

    def children(self, link: str) -> list:
        return [j for j in self.joints.values() if j.parent == link]

    def leaves(self, below: str | None = None) -> list:
        start = below or self.root()
        out, stack = [], [start]
        while stack:
            l = stack.pop()
            kids = self.children(l)
            if not kids and l != start:
                out.append(l)
            stack.extend(j.child for j in reversed(kids))
        return out


# --------------------------------------------------------------------------- parse
def _floats(text: str | None, n: int, default: tuple, where: str) -> tuple:
    if text is None:
        return default
    parts = text.split()
    try:
        vals = tuple(float(x) for x in parts)
    except ValueError:
        raise URDFError(f"{where}: cannot read numbers from {text!r}") from None
    if len(vals) != n:
        raise URDFError(f"{where}: expected {n} numbers, got {len(vals)} in {text!r}")
    return vals


def _parse_joint(el: ET.Element) -> Joint:
    name = el.get("name")
    if not name:
        raise URDFError("a <joint> has no name")
    jtype = el.get("type")
    if jtype not in JOINT_TYPES:
        raise URDFError(f"joint {name!r}: unknown type {jtype!r}")
    parent, child = el.find("parent"), el.find("child")
    if parent is None or child is None or not parent.get("link") or not child.get("link"):
        raise URDFError(f"joint {name!r} needs <parent link=...> and <child link=...>")
    origin, axis, limit, mimic = el.find("origin"), el.find("axis"), el.find("limit"), el.find("mimic")
    where = f"joint {name!r}"
    kw = {}
    if limit is not None:
        for key in ("lower", "upper"):
            if limit.get(key) is not None:
                kw[key] = float(limit.get(key))
        if jtype == "revolute" and "lower" not in kw:
            kw["lower"] = 0.0  # URDF default when only <limit> is present
        if jtype == "revolute" and "upper" not in kw:
            kw["upper"] = 0.0
    return Joint(
        name=name, type=jtype, parent=parent.get("link"), child=child.get("link"),
        xyz=_floats(origin.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0), where + " origin xyz"),
        rpy=_floats(origin.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0), where + " origin rpy"),
        axis=_floats(axis.get("xyz") if axis is not None else None, 3, (1.0, 0.0, 0.0), where + " axis"),
        mimic=mimic.get("joint") if mimic is not None else None,
        **kw,
    )


def _check(model: URDFModel) -> None:
    link_set = set(model.links)
    if len(link_set) != len(model.links):
        raise URDFError("duplicate link names")
    seen_children = {}
    for j in model.joints.values():
        for end in (j.parent, j.child):
            if end not in link_set:
                raise URDFError(f"joint {j.name!r} refers to link {end!r}, which is not defined")
        if j.child in seen_children:
            raise URDFError(f"link {j.child!r} has two parent joints ({seen_children[j.child]!r}, {j.name!r}); "
                            "closed loops are not supported")
        seen_children[j.child] = j.name
    roots = [l for l in model.links if l not in seen_children]
    if len(roots) != 1:
        raise URDFError(f"expected exactly one root link, found {roots or 'none (cycle)'}")
    # every link must reach the root, which also rules out cycles
    for l in model.links:
        hops, cur = 0, l
        while cur in seen_children:
            cur = model.joints[seen_children[cur]].parent
            hops += 1
            if hops > len(model.links):
                raise URDFError(f"cycle in the joint tree near link {l!r}")


def parse(source: str) -> URDFModel:
    """Parse a URDF from a file path or from the XML text itself."""
    text = source
    if "<" not in source:
        if not os.path.exists(source):
            raise URDFError(f"no such file: {source}")
        with open(source, encoding="utf-8") as f:
            text = f.read()
    try:
        root = ET.fromstring(text)
    except ET.ParseError as e:
        raise URDFError(f"not valid XML: {e}") from None
    if root.tag != "robot":
        raise URDFError(f"root element is <{root.tag}>, expected <robot>")
    links = [l.get("name") for l in root.findall("link")]
    if not links or not all(links):
        raise URDFError("every <link> needs a name, and there must be at least one")
    joints = {}
    for el in root.findall("joint"):
        j = _parse_joint(el)
        if j.name in joints:
            raise URDFError(f"duplicate joint name {j.name!r}")
        joints[j.name] = j
    model = URDFModel(name=root.get("name", "robot"), links=links, joints=joints, _tree=root)
    _check(model)
    return model


# --------------------------------------------------------------------------- write
def _fmt(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def _sub(parent: ET.Element, tag: str) -> ET.Element:
    el = parent.find(tag)
    return el if el is not None else ET.SubElement(parent, tag)


def _sync_joint(el: ET.Element, j: Joint) -> None:
    el.set("type", j.type)
    _sub(el, "parent").set("link", j.parent)
    _sub(el, "child").set("link", j.child)
    origin = _sub(el, "origin")
    origin.set("xyz", _fmt(j.xyz))
    origin.set("rpy", _fmt(j.rpy))
    if j.movable:
        _sub(el, "axis").set("xyz", _fmt(j.axis))
    if j.lower is not None or j.upper is not None:
        limit = _sub(el, "limit")
        if j.lower is not None:
            limit.set("lower", repr(float(j.lower)))
        if j.upper is not None:
            limit.set("upper", repr(float(j.upper)))
        for key in ("effort", "velocity"):  # required by the URDF schema
            limit.attrib.setdefault(key, "0")


def to_xml(model: URDFModel) -> ET.ElementTree:
    root = copy.deepcopy(model._tree) if model._tree is not None else ET.Element("robot")
    root.set("name", model.name)
    have_links = {l.get("name") for l in root.findall("link")}
    for name in model.links:
        if name not in have_links:
            ET.SubElement(root, "link", name=name)
    elements = {e.get("name"): e for e in root.findall("joint")}
    for j in model.joints.values():
        el = elements.get(j.name)
        if el is None:
            el = ET.SubElement(root, "joint", name=j.name)
        _sync_joint(el, j)
    for name, el in elements.items():
        if name not in model.joints:
            root.remove(el)
    ET.indent(root)
    return ET.ElementTree(root)


def write(model: URDFModel, path: str | None = None) -> str:
    """Serialise to URDF text; also saved to `path` when given."""
    text = ET.tostring(to_xml(model).getroot(), encoding="unicode") + "\n"
    if path is not None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    return text
