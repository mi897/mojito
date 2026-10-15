"""Mojito: robots that learn to move.

Stage 0: learned leg inverse kinematics.  Stage 1: whole body and scripted gaits.
"""
from .geometry import kinematics, robots, spec, urdf
from .robot import body, gaits, leg
from .learning.backend import get_backend, load_model, make_model, set_backend
from .learning.model import Adam, IKNet
from .geometry.spec import load_robot

__all__ = ["leg", "body", "gaits", "kinematics", "robots", "spec", "urdf", "load_robot", "IKNet", "Adam", "get_backend", "set_backend", "make_model", "load_model"]
