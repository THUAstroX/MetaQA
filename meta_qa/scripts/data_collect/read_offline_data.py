"""
Read and inspect offline RL datasets.

Usage:
    python -m meta_qa.scripts.data_collect.read_offline_data
    python -m meta_qa.scripts.data_collect.read_offline_data --file my_dataset.h5
"""

import os
import sys
import argparse
import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Default output directory (project root)
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "outputs", "offline_data")


def load_hdf5(filepath: str) -> dict:
    """Load HDF5 dataset."""
    import h5py
    
    data = {}
    with h5py.File(filepath, 'r') as f:
        for key in f.keys():
            data[key] = f[key][:]
        # Load attributes
        data['_attrs'] = dict(f.attrs)
    return data


def load_npz(filepath: str) -> dict:
    """Load NPZ dataset."""
    data = dict(np.load(filepath, allow_pickle=True))
    return data


def print_dataset_info(data: dict, filepath: str):
    """Print dataset information."""
    print("=" * 60)
    print(f"Dataset: {os.path.basename(filepath)}")
    print("=" * 60)
    
    # Print attributes if available
    if '_attrs' in data:
        attrs = data['_attrs']
        print("\nMetadata:")
        for key, value in attrs.items():
            print(f"  {key}: {value}")
    
    print("\nData Arrays:")
    print("-" * 60)
    
    for key, value in data.items():
        if key == '_attrs':
            continue
        if isinstance(value, np.ndarray):
            print(f"  {key}:")
            print(f"    Shape: {value.shape}")
            print(f"    Dtype: {value.dtype}")
            if value.size > 0:
                if np.issubdtype(value.dtype, np.floating):
                    print(f"    Range: [{value.min():.4f}, {value.max():.4f}]")
                    print(f"    Mean:  {value.mean():.4f}")
                elif np.issubdtype(value.dtype, np.integer):
                    print(f"    Range: [{value.min()}, {value.max()}]")
                elif value.dtype == bool:
                    print(f"    True count: {value.sum()}")
    
    # Dataset statistics
    print("\n" + "=" * 60)
    print("Statistics")
    print("=" * 60)
    
    if 'observations' in data:
        n_samples = len(data['observations'])
        print(f"Total samples: {n_samples}")
    
    if 'episode_starts' in data:
        n_episodes = len(data['episode_starts'])
        print(f"Total episodes: {n_episodes}")
        if n_samples > 0 and n_episodes > 0:
            print(f"Avg episode length: {n_samples / n_episodes:.1f}")
    
    if 'rewards' in data:
        print(f"Total reward: {data['rewards'].sum():.2f}")
        print(f"Avg reward per step: {data['rewards'].mean():.4f}")
    
    if 'costs' in data:
        costs = data['costs']
        print(f"Total cost: {costs.sum():.2f}")
        print(f"Avg cost per step: {costs.mean():.4f}")
        print(f"Steps with cost > 0: {(costs > 0).sum()} ({100*(costs > 0).mean():.1f}%)")
    
    if 'terminals' in data:
        print(f"Terminal states: {data['terminals'].sum()}")


def sample_data(data: dict, n_samples: int = 3):
    """Print sample data points."""
    print("\n" + "=" * 60)
    print(f"Sample Data (first {n_samples} steps)")
    print("=" * 60)
    
    n = min(n_samples, len(data.get('observations', [])))
    
    for i in range(n):
        print(f"\n--- Step {i} ---")
        
        if 'observations' in data:
            obs = data['observations'][i]
            print(f"Observation: shape={obs.shape}, first 5 values={obs[:5]}")
        
        if 'actions' in data:
            act = data['actions'][i]
            # Reshape to waypoints
            waypoints = act.reshape(-1, 2)
            print(f"Action (trajectory): {len(waypoints)} waypoints")
            print(f"  First waypoint: ({waypoints[0, 0]:.2f}, {waypoints[0, 1]:.2f})")
            print(f"  Last waypoint:  ({waypoints[-1, 0]:.2f}, {waypoints[-1, 1]:.2f})")
        
        if 'rewards' in data:
            print(f"Reward: {data['rewards'][i]:.4f}")
        
        if 'costs' in data:
            print(f"Cost: {data['costs'][i]:.4f}")
        
        if 'terminals' in data:
            print(f"Terminal: {data['terminals'][i]}")


def list_datasets(output_dir: str):
    """List available datasets."""
    print("=" * 60)
    print(f"Available datasets in: {output_dir}")
    print("=" * 60)
    
    if not os.path.exists(output_dir):
        print("  (directory does not exist)")
        return []
    
    files = []
    for f in os.listdir(output_dir):
        if f.endswith('.h5') or f.endswith('.npz'):
            filepath = os.path.join(output_dir, f)
            size = os.path.getsize(filepath) / 1024  # KB
            files.append((f, size))
    
    if not files:
        print("  (no datasets found)")
        return []
    
    for f, size in sorted(files):
        if size > 1024:
            print(f"  {f} ({size/1024:.1f} MB)")
        else:
            print(f"  {f} ({size:.1f} KB)")
    
    return [f[0] for f in files]


def main():
    parser = argparse.ArgumentParser(description="Read and inspect offline RL datasets")
    parser.add_argument(
        "--file", "-f",
        type=str,
        default=None,
        help="Dataset file to read (in output/datasets/ or full path)"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--samples", "-s",
        type=int,
        default=3,
        help="Number of sample data points to show"
    )
    args = parser.parse_args()
    
    # List datasets
    if args.list or args.file is None:
        files = list_datasets(DEFAULT_OUTPUT_DIR)
        if args.file is None and files:
            print(f"\nUse --file <filename> to inspect a dataset")
            # Auto-select first file if only listing
            if not args.list and len(files) > 0:
                args.file = files[0]
                print(f"\nAuto-selecting: {args.file}\n")
    
    if args.file is None:
        return
    
    # Resolve file path
    if os.path.isabs(args.file):
        filepath = args.file
    else:
        # Try as relative to project root first
        filepath = os.path.join(project_root, args.file)
        # If not found, try relative to DEFAULT_OUTPUT_DIR
        if not os.path.exists(filepath):
            filepath = os.path.join(DEFAULT_OUTPUT_DIR, args.file)
    
    if not os.path.exists(filepath):
        print(f"Error: File not found: {filepath}")
        return
    
    # Load dataset
    print(f"\nLoading: {filepath}\n")
    
    if filepath.endswith('.h5'):
        data = load_hdf5(filepath)
    elif filepath.endswith('.npz'):
        data = load_npz(filepath)
    else:
        print(f"Error: Unsupported format: {filepath}")
        return
    
    # Print info
    print_dataset_info(data, filepath)
    
    # Print samples
    if args.samples > 0:
        sample_data(data, args.samples)


if __name__ == "__main__":
    main()
