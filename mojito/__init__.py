"""Mojito: robots that learn to move.

Stage 0: learned leg inverse kinematics.  Stage 1: whole body and scripted gaits.
"""
from . import body, gaits, leg
from .backend import get_backend, load_model, make_model, set_backend
from .model import Adam, IKNet

__all__ = ["leg", "body", "gaits", "IKNet", "Adam", "get_backend", "set_backend", "make_model", "load_model"]
