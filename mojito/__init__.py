"""Mojito: robots that learn to move.

Stage 0: learned leg inverse kinematics.  Stage 1: whole body and scripted gaits.
"""
from . import body, gaits, leg
from .model import Adam, IKNet

__all__ = ["leg", "body", "gaits", "IKNet", "Adam"]
