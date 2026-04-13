"""Public API for the ps_kinematics package.

Expose commonly used classes and helpers so callers can do:

    from ps_kinematics import HandLandmarkProcessor, apply_tuning_overrides

The heavy implementations live in submodules (core, kinematics, utils).
"""

from .core import HandLandmarkProcessor, MultiHandOfflineTracker
from .kinematics import KinematicAnalyzer
from .utils import PipelineConfig, apply_tuning_overrides

__all__ = [
    "HandLandmarkProcessor",
    "MultiHandOfflineTracker",
    "KinematicAnalyzer",
    "apply_tuning_overrides",
    "PipelineConfig",
]
