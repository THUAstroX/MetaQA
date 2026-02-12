"""
Data Collection Scripts.

Trajectory-based offline RL data collection.

All observations use historical trajectory, and all actions are future waypoint trajectories.

Parameters:
    --frequency    Data frequency: original (~12Hz) or keyframe (2Hz)
    --qa           Include QA annotations (default: true, use --qa false to disable)
    --map          Include map features (default: true, use --map false to disable)
    --history_sec  Historical observation window (default: 0.5s)
    --future_sec   Future trajectory horizon (default: 2.0s)

Usage:
    # Default: collect with QA + map
    python -m meta_qa.scripts.data_collect.collect_data
    
    # Disable QA or map
    python -m meta_qa.scripts.data_collect.collect_data --qa false
    python -m meta_qa.scripts.data_collect.collect_data --map false
    python -m meta_qa.scripts.data_collect.collect_data --qa false --map false
"""

from .collect_data import (
    FrequencyMode,
    CostType,
    CollectionConfig,
    TrajectoryDataCollector,
    MAP_TYPE_CODES,
    MAP_TYPE_NAMES,
)

__all__ = [
    "FrequencyMode",
    "CostType",
    "CollectionConfig",
    "TrajectoryDataCollector",
    "MAP_TYPE_CODES",
    "MAP_TYPE_NAMES",
]
