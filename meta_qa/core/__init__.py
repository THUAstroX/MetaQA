"""
MetaQA Core Module.

Core components for trajectory-based RL environment with MetaDrive.
- Trajectory tracking and vehicle state management
- Environment wrapper for MetaDrive
- Action space definitions
"""

from .trajectory_tracker import TrajectoryTracker, VehicleState, VehicleParams
from .env import TrajectoryEnv
from .action_space import TrajectoryActionSpace, TrajectoryActionSpaceNormalized

__all__ = [
    # Trajectory Tracking
    "TrajectoryTracker",
    "VehicleState",
    "VehicleParams",
    # Environment
    "TrajectoryEnv",
    # Action Space
    "TrajectoryActionSpace",
    "TrajectoryActionSpaceNormalized",
]