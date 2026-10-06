"""Mojito: robots that learn to move. Stage 0 - learned leg inverse kinematics."""
from . import leg
from .model import Adam, IKNet

__all__ = ["leg", "IKNet", "Adam"]
