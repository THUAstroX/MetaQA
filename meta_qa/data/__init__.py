"""
Data Module for Offline RL Dataset Collection.

This module provides data structures and I/O utilities for offline 
reinforcement learning datasets.

Components:
    - structures: MDP/CMDP transition data classes
    - io: Dataset save/load utilities (HDF5, NPZ)

Note: QA loading moved to meta_qa.tools.qa_loader
"""

from .structures import (
    MDPTransition,
    CMDPTransition,
    SurroundingVehicleInfo,
)
from .io import (
    save_dataset_hdf5,
    load_dataset_hdf5,
    save_dataset_npz,
    load_dataset_npz,
    get_dataset_info,
)

__all__ = [
    # Structures
    "MDPTransition",
    "CMDPTransition",
    "SurroundingVehicleInfo",
    # I/O
    "save_dataset_hdf5",
    "load_dataset_hdf5",
    "save_dataset_npz",
    "load_dataset_npz",
    "get_dataset_info",
]
