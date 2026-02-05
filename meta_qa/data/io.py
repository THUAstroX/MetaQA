"""
Dataset I/O Utilities.

Functions for saving and loading offline RL datasets in various formats.
Supports HDF5 (recommended) and NPZ formats.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, List


def save_dataset_hdf5(
    data: Dict[str, np.ndarray],
    path: str,
    metadata: Optional[Dict[str, Any]] = None,
    compression: str = "gzip",
    compression_opts: int = 4,
) -> None:
    """
    Save dataset in HDF5 format.
    
    HDF5 is the recommended format for large datasets as it supports:
    - Compression (gzip, lzf)
    - Partial loading (memory efficient)
    - Hierarchical structure
    - Metadata attributes
    
    Args:
        data: Dictionary containing arrays (observations, actions, rewards, etc.)
        path: Output file path (should end with .h5 or .hdf5)
        metadata: Optional metadata dictionary to store as attributes
        compression: Compression type ("gzip", "lzf", or None)
        compression_opts: Compression level (1-9 for gzip)
        
    Example:
        save_dataset_hdf5(
            {
                "observations": obs_array,
                "actions": act_array,
                "rewards": rew_array,
            },
            "dataset.h5",
            metadata={"num_episodes": 100, "env_name": "nuplan"}
        )
    """
    import h5py
    
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    with h5py.File(path, "w") as f:
        # Create datasets with compression
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if compression:
                    f.create_dataset(
                        key,
                        data=value,
                        compression=compression,
                        compression_opts=compression_opts,
                    )
                else:
                    f.create_dataset(key, data=value)
        
        # Add metadata
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    f.attrs[key] = value
                elif isinstance(value, (list, tuple)):
                    f.attrs[key] = str(value)
    
    # Print summary
    total_samples = len(data.get("observations", data.get("states", [])))
    file_size = os.path.getsize(path) / (1024 * 1024)
    print(f"Dataset saved to {path}")
    print(f"  - Total samples: {total_samples}")
    print(f"  - File size: {file_size:.2f} MB")
    print(f"  - Keys: {list(data.keys())}")


def load_dataset_hdf5(
    path: str,
    keys: Optional[List[str]] = None,
    load_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Load dataset from HDF5 format.
    
    Args:
        path: Path to HDF5 file
        keys: Optional list of keys to load (loads all if None)
        load_metadata: Whether to load metadata attributes
        
    Returns:
        Dictionary containing loaded data and optionally metadata
        
    Example:
        data = load_dataset_hdf5("dataset.h5")
        obs = data["observations"]
        metadata = data["metadata"]
    """
    import h5py
    
    data = {}
    with h5py.File(path, "r") as f:
        # Load specified keys or all keys
        load_keys = keys if keys else list(f.keys())
        for key in load_keys:
            if key in f:
                data[key] = f[key][:]
        
        # Load metadata
        if load_metadata:
            data["metadata"] = dict(f.attrs)
    
    return data


def save_dataset_npz(
    data: Dict[str, np.ndarray],
    path: str,
    compressed: bool = True,
) -> None:
    """
    Save dataset in NPZ format.
    
    NPZ is a simple numpy format, good for smaller datasets or quick prototyping.
    
    Args:
        data: Dictionary containing arrays
        path: Output file path (should end with .npz)
        compressed: Whether to use compression
        
    Example:
        save_dataset_npz(
            {"observations": obs, "actions": acts},
            "dataset.npz"
        )
    """
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    
    if compressed:
        np.savez_compressed(path, **data)
    else:
        np.savez(path, **data)
    
    total_samples = len(data.get("observations", data.get("states", [])))
    print(f"Dataset saved to {path}")
    print(f"  - Total samples: {total_samples}")
    print(f"  - Keys: {list(data.keys())}")


def load_dataset_npz(path: str) -> Dict[str, np.ndarray]:
    """
    Load dataset from NPZ format.
    
    Args:
        path: Path to NPZ file
        
    Returns:
        Dictionary containing loaded arrays
    """
    data = np.load(path)
    return {k: data[k] for k in data.files}


def get_dataset_info(path: str) -> Dict[str, Any]:
    """
    Get information about a saved dataset without loading all data.
    
    Useful for inspecting large datasets without loading into memory.
    
    Args:
        path: Path to dataset file
        
    Returns:
        Dictionary with dataset information including shapes, dtypes, and metadata
        
    Example:
        info = get_dataset_info("large_dataset.h5")
        print(f"Samples: {info['shapes']['observations'][0]}")
        print(f"Obs dim: {info['shapes']['observations'][1]}")
    """
    import h5py
    
    if path.endswith(".h5") or path.endswith(".hdf5"):
        with h5py.File(path, "r") as f:
            info = {
                "format": "hdf5",
                "path": path,
                "keys": list(f.keys()),
                "metadata": dict(f.attrs),
                "shapes": {key: f[key].shape for key in f.keys()},
                "dtypes": {key: str(f[key].dtype) for key in f.keys()},
                "total_size_mb": sum(f[key].nbytes for key in f.keys()) / (1024 * 1024),
                "file_size_mb": os.path.getsize(path) / (1024 * 1024),
            }
            
            # Compute compression ratio
            if info["total_size_mb"] > 0:
                info["compression_ratio"] = info["total_size_mb"] / info["file_size_mb"]
            
    elif path.endswith(".npz"):
        data = np.load(path)
        info = {
            "format": "npz",
            "path": path,
            "keys": list(data.keys()),
            "shapes": {key: data[key].shape for key in data.keys()},
            "dtypes": {key: str(data[key].dtype) for key in data.keys()},
            "file_size_mb": os.path.getsize(path) / (1024 * 1024),
        }
    else:
        info = {"format": "unknown", "path": path}
    
    return info


def convert_to_d4rl_format(
    data: Dict[str, np.ndarray],
    include_timeouts: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert dataset to D4RL-compatible format.
    
    D4RL format:
        - observations: (N, obs_dim)
        - actions: (N, action_dim)
        - rewards: (N,)
        - terminals: (N,)
        - timeouts: (N,) [optional]
        - next_observations: (N, obs_dim) [optional]
    
    For Safe RL (CMDP):
        - costs: (N,) [additional]
    
    Args:
        data: Dataset dictionary
        include_timeouts: Whether to include timeout flags
        
    Returns:
        D4RL-compatible dictionary
    """
    result = {}
    
    # Required keys
    result["observations"] = data.get("observations", data.get("states"))
    result["actions"] = data["actions"]
    result["rewards"] = data["rewards"]
    result["terminals"] = data.get("terminals", data.get("dones"))
    
    # Optional keys
    if "next_observations" in data:
        result["next_observations"] = data["next_observations"]
    
    if include_timeouts and "timeouts" in data:
        result["timeouts"] = data["timeouts"]
    
    # Safe RL extension
    if "costs" in data:
        result["costs"] = data["costs"]
    
    return result


def merge_datasets(
    datasets: List[Dict[str, np.ndarray]],
    output_path: Optional[str] = None,
) -> Dict[str, np.ndarray]:
    """
    Merge multiple datasets into one.
    
    Args:
        datasets: List of dataset dictionaries to merge
        output_path: Optional path to save merged dataset
        
    Returns:
        Merged dataset dictionary
    """
    if not datasets:
        raise ValueError("No datasets to merge")
    
    # Get common keys
    common_keys = set(datasets[0].keys())
    for d in datasets[1:]:
        common_keys &= set(d.keys())
    
    # Remove metadata key if present
    common_keys.discard("metadata")
    
    # Merge arrays
    merged = {}
    for key in common_keys:
        arrays = [d[key] for d in datasets]
        merged[key] = np.concatenate(arrays, axis=0)
    
    # Merge metadata
    merged["metadata"] = {
        "num_source_datasets": len(datasets),
        "total_samples": len(merged.get("observations", merged.get("states", []))),
    }
    
    # Save if path provided
    if output_path:
        if output_path.endswith(".h5") or output_path.endswith(".hdf5"):
            save_dataset_hdf5(merged, output_path, merged.get("metadata"))
        else:
            save_dataset_npz(merged, output_path)
    
    return merged
