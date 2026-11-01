"""Read and write URDF robot descriptions. Pure Python: no NumPy, no torch.

Only the kinematic skeleton is interpreted (links, joints, origins, axes,
limits). Meshes, inertias, materials and transmissions are not parsed, but
when a file is read from disk they are kept in the XML tree and written back
untouched, so editing a joint and saving does not strip a real robot's
visuals. `<xacro>` is not supported: expand it to plain URDF first.

This module knows nothing about learning. mojito/geometry/spec.py turns a URDFModel
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
        children = {joint.child for joint in self.joints.values()}
        roots = [link for link in self.links if link not in children]
        return roots[0]

    def parent_joint(self, link: str) -> Joint | None:
        for joint in self.joints.values():
            if joint.child == link:
                return joint
        return None

    def chain(self, base: str, tip: str) -> list:
        """Joints from `base` down to `tip`, in order. Raises if tip is not below base."""
        if tip not in self.links:
            raise URDFError(f"link {tip!r} does not exist")
        joints_up, link = [], tip  # collected from the tip upwards
        while link != base:
            joint = self.parent_joint(link)
            if joint is None:
                raise URDFError(f"link {tip!r} is not a descendant of {base!r}")
            joints_up.append(joint)
            link = joint.parent
        return joints_up[::-1]

    def children(self, link: str) -> list:
        return [joint for joint in self.joints.values() if joint.parent == link]

    def leaves(self, below: str | None = None) -> list:
        start = below or self.root()
        leaf_links, stack = [], [start]
        while stack:
            link = stack.pop()
            child_joints = self.children(link)
            if not child_joints and link != start:
                leaf_links.append(link)
            stack.extend(joint.child for joint in reversed(child_joints))
        return leaf_links


# --------------------------------------------------------------------------- parse
def _floats(text: str | None, n_numbers: int, default: tuple, where: str) -> tuple:
    if text is None:
        return default
    parts = text.split()
    try:
        values = tuple(float(part) for part in parts)
    except ValueError:
        raise URDFError(f"{where}: cannot read numbers from {text!r}") from None
    if len(values) != n_numbers:
        raise URDFError(f"{where}: expected {n_numbers} numbers, got {len(values)} in {text!r}")
    return values


def _parse_joint(element: ET.Element) -> Joint:
    name = element.get("name")
    if not name:
        raise URDFError("a <joint> has no name")
    joint_type = element.get("type")
    if joint_type not in JOINT_TYPES:
        raise URDFError(f"joint {name!r}: unknown type {joint_type!r}")
    parent, child = element.find("parent"), element.find("child")
    if parent is None or child is None or not parent.get("link") or not child.get("link"):
        raise URDFError(f"joint {name!r} needs <parent link=...> and <child link=...>")
    origin, axis, limit, mimic = element.find("origin"), element.find("axis"), element.find("limit"), element.find("mimic")
    where = f"joint {name!r}"
    limit_values = {}
    if limit is not None:
        for key in ("lower", "upper"):
            if limit.get(key) is not None:
                limit_values[key] = float(limit.get(key))
        if joint_type == "revolute" and "lower" not in limit_values:
            limit_values["lower"] = 0.0  # URDF default when only <limit> is present
        if joint_type == "revolute" and "upper" not in limit_values:
            limit_values["upper"] = 0.0
    return Joint(
        name=name, type=joint_type, parent=parent.get("link"), child=child.get("link"),
        xyz=_floats(origin.get("xyz") if origin is not None else None, 3, (0.0, 0.0, 0.0), where + " origin xyz"),
        rpy=_floats(origin.get("rpy") if origin is not None else None, 3, (0.0, 0.0, 0.0), where + " origin rpy"),
        axis=_floats(axis.get("xyz") if axis is not None else None, 3, (1.0, 0.0, 0.0), where + " axis"),
        mimic=mimic.get("joint") if mimic is not None else None,
        **limit_values,
    )


def _check(model: URDFModel) -> None:
    link_set = set(model.links)
    if len(link_set) != len(model.links):
        raise URDFError("duplicate link names")
    seen_children = {}
    for joint in model.joints.values():
        for end in (joint.parent, joint.child):
            if end not in link_set:
                raise URDFError(f"joint {joint.name!r} refers to link {end!r}, which is not defined")
        if joint.child in seen_children:
            raise URDFError(f"link {joint.child!r} has two parent joints ({seen_children[joint.child]!r}, {joint.name!r}); "
                            "closed loops are not supported")
        seen_children[joint.child] = joint.name
    roots = [link for link in model.links if link not in seen_children]
    if len(roots) != 1:
        raise URDFError(f"expected exactly one root link, found {roots or 'none (cycle)'}")
    # every link must reach the root, which also rules out cycles
    for link in model.links:
        hops, current = 0, link
        while current in seen_children:
            current = model.joints[seen_children[current]].parent
            hops += 1
            if hops > len(model.links):
                raise URDFError(f"cycle in the joint tree near link {link!r}")


def parse(source: str) -> URDFModel:
    """Parse a URDF from a file path or from the XML text itself."""
    text = source
    if "<" not in source:
        if not os.path.exists(source):
            raise URDFError(f"no such file: {source}")
        with open(source, encoding="utf-8") as urdf_file:
            text = urdf_file.read()
    try:
        root = ET.fromstring(text)
    except ET.ParseError as error:
        raise URDFError(f"not valid XML: {error}") from None
    if root.tag != "robot":
        raise URDFError(f"root element is <{root.tag}>, expected <robot>")
    links = [link.get("name") for link in root.findall("link")]
    if not links or not all(links):
        raise URDFError("every <link> needs a name, and there must be at least one")
    joints = {}
    for element in root.findall("joint"):
        joint = _parse_joint(element)
        if joint.name in joints:
            raise URDFError(f"duplicate joint name {joint.name!r}")
        joints[joint.name] = joint
    model = URDFModel(name=root.get("name", "robot"), links=links, joints=joints, _tree=root)
    _check(model)
    return model


# --------------------------------------------------------------------------- write
def _format_numbers(values) -> str:
    return " ".join(repr(float(v)) for v in values)


def _find_or_add(parent: ET.Element, tag: str) -> ET.Element:
    element = parent.find(tag)
    return element if element is not None else ET.SubElement(parent, tag)


def _sync_joint(element: ET.Element, joint: Joint) -> None:
    element.set("type", joint.type)
    _find_or_add(element, "parent").set("link", joint.parent)
    _find_or_add(element, "child").set("link", joint.child)
    origin = _find_or_add(element, "origin")
    origin.set("xyz", _format_numbers(joint.xyz))
    origin.set("rpy", _format_numbers(joint.rpy))
    if joint.movable:
        _find_or_add(element, "axis").set("xyz", _format_numbers(joint.axis))
    if joint.lower is not None or joint.upper is not None:
        limit = _find_or_add(element, "limit")
        if joint.lower is not None:
            limit.set("lower", repr(float(joint.lower)))
        if joint.upper is not None:
            limit.set("upper", repr(float(joint.upper)))
        for key in ("effort", "velocity"):  # required by the URDF schema
            limit.attrib.setdefault(key, "0")


def to_xml(model: URDFModel) -> ET.ElementTree:
    root = copy.deepcopy(model._tree) if model._tree is not None else ET.Element("robot")
    root.set("name", model.name)
    have_links = {link.get("name") for link in root.findall("link")}
    for name in model.links:
        if name not in have_links:
            ET.SubElement(root, "link", name=name)
    elements = {joint_element.get("name"): joint_element for joint_element in root.findall("joint")}
    for joint in model.joints.values():
        element = elements.get(joint.name)
        if element is None:
            element = ET.SubElement(root, "joint", name=joint.name)
        _sync_joint(element, joint)
    for name, element in elements.items():
        if name not in model.joints:
            root.remove(element)
    ET.indent(root)
    return ET.ElementTree(root)


def write(model: URDFModel, path: str | None = None) -> str:
    """Serialise to URDF text; also saved to `path` when given."""
    text = ET.tostring(to_xml(model).getroot(), encoding="unicode") + "\n"
    if path is not None:
        with open(path, "w", encoding="utf-8") as urdf_file:
            urdf_file.write(text)
    return text
