"""
MetaQA - Trajectory-based RL Data Pipeline for Autonomous Driving with QA.

Modules:
    - core: Core components (env, action space, trajectory tracker, config)
    - tools: QA loading, visualization, and GIF generation
    - cost: Cost interfaces (collision, ttc, kinematic)
    - data: Data structures and I/O utilities

Quick Start:
    from meta_qa.core import TrajectoryEnv, TrajectoryTracker
    from meta_qa.tools import NuScenesQALoader, GIFGenerator
    from meta_qa.data import save_dataset_hdf5
    from meta_qa.cost import CollisionCost, TTCCost, KinematicCost
"""

__version__ = "0.1.0"
