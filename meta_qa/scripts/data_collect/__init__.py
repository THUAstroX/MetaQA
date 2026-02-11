"""
Data Collection Scripts.

Unified data collection for offline RL datasets.

Supported modes:
    - original:     Original sensor frequency (~12Hz) without QA
    - original_qa:  Original frequency with QA at keyframes  
    - keyframe:     Keyframe data only (2Hz)
    - keyframe_qa:  Keyframe data with QA annotations
    - trajectory:   Trajectory-based observation/action

Usage:
    python -m meta_qa.scripts.data_collect.collect_data --mode original
    python -m meta_qa.scripts.data_collect.collect_data --mode keyframe_qa --cost_type ttc
    python -m meta_qa.scripts.data_collect.collect_data --mode trajectory --history_sec 0.5 --future_sec 2.0
"""

from .collect_data import (
    CollectionMode,
    ObservationType,
    CostType,
    CollectionConfig,
    UnifiedDataCollector,
)

__all__ = [
    "CollectionMode",
    "ObservationType",
    "CostType",
    "CollectionConfig",
    "UnifiedDataCollector",
]
