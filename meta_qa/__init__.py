"""
MetaQA - Trajectory-based RL Data Pipeline for Autonomous Driving with QA.

Modules:
    - core: Core components (env, action space, trajectory tracker, config)
    - tools: Data processing, QA loading, visualization, surrounding vehicle extraction
    - cost: Cost interfaces (collision, ttc, kinematic)

Quick Start:
    from meta_qa.core import TrajectoryEnv, TrajectoryTracker
    from meta_qa.tools import NuScenesQALoader, ReplayOriginalEnv
    from meta_qa.cost import CollisionCost, TTCCost, KinematicCost
"""

__version__ = "0.1.0"
