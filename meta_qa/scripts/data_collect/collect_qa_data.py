#!/usr/bin/env python3
"""
QA-Enhanced Offline Data Collection.

This script extends the standard offline data collection to include
NuScenes-QA annotations, allowing the creation of datasets that combine:
1. MDP/CMDP trajectory data from MetaDrive replay
2. QA annotations from NuScenes-QA
3. References to original camera images

The enhanced dataset can be used for:
- Multi-modal offline RL (vision + language + control)
- VQA training with driving context
- Language-conditioned decision making

Usage:
    python -m meta_qa.scripts.data_collect.collect_qa_data \\
        --scenario-dir ./dataset/Scenario_Data/exp_nuscenes_converted \\
        --qa-dir ./dataset/QA_Data \\
        --nuscenes-dir ./dataset/Scenario_Data/exp_nuscenes/v1.0-mini \\
        --output outputs/offline_data/qa_enhanced_dataset.h5
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
import pickle
import h5py

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from meta_qa.tools import (
    NuScenesQALoader,
    ScenarioQAMatcher,
    SampleQAData,
    QAItem,
)


@dataclass
class QAEnhancedTransition:
    """
    MDP transition enhanced with QA annotations.
    
    Extends standard (s, a, r, s', done) with:
    - QA question-answer pairs for the current frame
    - Image path references
    - Scene context
    """
    # Standard MDP
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool
    
    # QA enhancement
    qa_items: List[Dict[str, str]] = field(default_factory=list)
    image_paths: Dict[str, str] = field(default_factory=dict)
    sample_token: Optional[str] = None
    scene_name: Optional[str] = None
    frame_idx: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "state": self.state.tolist() if isinstance(self.state, np.ndarray) else self.state,
            "action": self.action.tolist() if isinstance(self.action, np.ndarray) else self.action,
            "reward": float(self.reward),
            "next_state": self.next_state.tolist() if isinstance(self.next_state, np.ndarray) else self.next_state,
            "done": bool(self.done),
            "qa_items": self.qa_items,
            "image_paths": self.image_paths,
            "sample_token": self.sample_token,
            "scene_name": self.scene_name,
            "frame_idx": self.frame_idx,
        }


class QAEnhancedDataCollector:
    """
    Collects offline RL data enhanced with QA annotations.
    
    This collector:
    1. Replays scenarios using MetaDrive
    2. Collects trajectory data at each step
    3. Associates QA annotations from NuScenes-QA
    4. Stores image path references for multi-modal learning
    """
    
    def __init__(
        self,
        qa_loader: NuScenesQALoader,
        matcher: ScenarioQAMatcher,
        output_format: str = "hdf5",
        include_image_paths: bool = False,
        num_waypoints: int = 20,
    ):
        """
        Initialize the collector.
        
        Args:
            qa_loader: Loaded QA data
            matcher: Scenario matcher
            output_format: Output format ('hdf5' or 'json')
            include_image_paths: Whether to include 6-camera image paths (default: False)
            num_waypoints: Number of future waypoints for trajectory action (default: 20)
        """
        self.qa_loader = qa_loader
        self.matcher = matcher
        self.output_format = output_format
        self.include_image_paths = include_image_paths
        self.num_waypoints = num_waypoints
        
        # Data storage
        self.transitions: List[QAEnhancedTransition] = []
        self.episode_starts: List[int] = []
        self.scene_info: Dict[str, Dict] = {}
        
        # Statistics
        self.total_steps = 0
        self.total_qa_items = 0
    
    def collect_from_scenario(
        self,
        scene_name: str,
        use_metadrive: bool = False,
    ) -> int:
        """
        Collect data from a single scenario.
        
        Args:
            scene_name: Scene to collect from
            use_metadrive: Whether to use MetaDrive for full simulation
            
        Returns:
            Number of transitions collected
        """
        # Load scenario
        scenario = self.matcher.load_scenario(scene_name)
        if scenario is None:
            print(f"Could not load scenario: {scene_name}")
            return 0
        
        # Get QA data
        scene_qa = self.qa_loader.get_qa_for_scene(scene_name)
        
        # Record episode start
        self.episode_starts.append(len(self.transitions))
        
        # Get scenario length and metadata
        length = scenario.get('length', 0)
        metadata = scenario.get('metadata', {})
        
        # Store scene info
        self.scene_info[scene_name] = {
            'length': length,
            'description': scene_qa.description if scene_qa else '',
            'qa_count': scene_qa.total_qa_count if scene_qa else 0,
            'keyframe_count': len(scene_qa.sample_order) if scene_qa else 0,
        }
        
        # Extract ego trajectory from scenario
        ego_track = scenario.get('tracks', {}).get('ego', {})
        ego_state = ego_track.get('state', {})
        positions = ego_state.get('position', np.zeros((length, 3)))
        velocities = ego_state.get('velocity', np.zeros((length, 2)))
        headings = ego_state.get('heading', np.zeros(length))
        
        if isinstance(positions, dict):
            positions = np.array(positions.get('data', np.zeros((length, 3))))
        if isinstance(velocities, dict):
            velocities = np.array(velocities.get('data', np.zeros((length, 2))))
        if isinstance(headings, dict):
            headings = np.array(headings.get('data', np.zeros(length)))
        
        num_collected = 0
        
        for t in range(length - 1):
            # Build state from scenario data
            if t < len(positions):
                pos = positions[t][:2] if len(positions[t]) >= 2 else np.zeros(2)
                vel = velocities[t] if t < len(velocities) else np.zeros(2)
                heading = headings[t] if t < len(headings) else 0.0
                speed = np.linalg.norm(vel)
                
                state = np.array([pos[0], pos[1], vel[0], vel[1], heading, speed])
            else:
                state = np.zeros(6)
            
            # Build action (trajectory waypoints to future positions)
            # Extract future waypoints relative to current position
            waypoints = []
            current_pos = state[:2]
            
            for i in range(1, self.num_waypoints + 1):
                future_idx = t + i
                if future_idx < len(positions):
                    future_pos = positions[future_idx][:2]
                    # Store relative displacement from current position
                    waypoint = future_pos - current_pos
                    waypoints.append(waypoint)
                else:
                    # Pad with last valid waypoint if we run out of data
                    if waypoints:
                        waypoints.append(waypoints[-1])
                    else:
                        waypoints.append(np.zeros(2))
            
            # Flatten waypoints to action vector: [x1, y1, x2, y2, ...]
            action = np.array(waypoints).flatten()
            
            # Build next state
            if t + 1 < len(positions):
                next_pos = positions[t + 1][:2]
                next_vel = velocities[t + 1] if t + 1 < len(velocities) else np.zeros(2)
                next_heading = headings[t + 1] if t + 1 < len(headings) else 0.0
                next_speed = np.linalg.norm(next_vel)
                next_state = np.array([next_pos[0], next_pos[1], next_vel[0], next_vel[1], 
                                       next_heading, next_speed])
            else:
                next_state = state.copy()
            
            # Get QA data for this timestep
            sample_qa = self.matcher.get_qa_for_timestep(scene_name, t)
            qa_items = []
            image_paths = {}
            sample_token = None
            
            if sample_qa:
                sample_token = sample_qa.sample_token
                qa_items = [
                    {"question": q.question, "answer": q.answer, "type": q.template_type}
                    for q in sample_qa.qa_items
                ]
                if self.include_image_paths:
                    image_paths = self.qa_loader.get_all_images_for_sample(
                        sample_qa.sample_token, 
                        absolute=False
                    )
                self.total_qa_items += len(sample_qa.qa_items)
            
            # Create transition
            transition = QAEnhancedTransition(
                state=state.astype(np.float32),
                action=action.astype(np.float32),
                reward=0.0,  # Will be computed if using MetaDrive
                next_state=next_state.astype(np.float32),
                done=(t == length - 2),
                qa_items=qa_items,
                image_paths=image_paths,
                sample_token=sample_token,
                scene_name=scene_name,
                frame_idx=t,
            )
            
            self.transitions.append(transition)
            num_collected += 1
        
        self.total_steps += num_collected
        return num_collected
    
    def collect_all_matching_scenes(self) -> int:
        """Collect data from all scenes that have QA data."""
        matching_scenes = self.matcher.get_matching_scenes()
        total = 0
        
        for i, scene_name in enumerate(matching_scenes):
            print(f"Collecting from {scene_name} ({i+1}/{len(matching_scenes)})...")
            num = self.collect_from_scenario(scene_name)
            total += num
            print(f"  Collected {num} transitions")
        
        return total
    
    def save_hdf5(self, output_path: str):
        """Save collected data to HDF5 format."""
        with h5py.File(output_path, 'w') as f:
            # Save arrays
            n = len(self.transitions)
            state_dim = self.transitions[0].state.shape[0] if n > 0 else 6
            action_dim = self.transitions[0].action.shape[0] if n > 0 else 2
            
            f.create_dataset('observations', (n, state_dim), dtype=np.float32)
            f.create_dataset('actions', (n, action_dim), dtype=np.float32)
            f.create_dataset('rewards', (n,), dtype=np.float32)
            f.create_dataset('next_observations', (n, state_dim), dtype=np.float32)
            f.create_dataset('terminals', (n,), dtype=bool)
            f.create_dataset('episode_starts', data=np.array(self.episode_starts))
            
            # Fill arrays
            for i, t in enumerate(self.transitions):
                f['observations'][i] = t.state
                f['actions'][i] = t.action
                f['rewards'][i] = t.reward
                f['next_observations'][i] = t.next_state
                f['terminals'][i] = t.done
            
            # Save QA data as JSON strings (HDF5 doesn't support complex nested structures well)
            qa_data = []
            for t in self.transitions:
                qa_data.append({
                    'qa_items': t.qa_items,
                    'image_paths': t.image_paths,
                    'sample_token': t.sample_token,
                    'scene_name': t.scene_name,
                    'frame_idx': t.frame_idx,
                })
            
            # Store as JSON string
            qa_json = json.dumps(qa_data)
            f.attrs['qa_data'] = qa_json
            f.attrs['scene_info'] = json.dumps(self.scene_info)
            f.attrs['total_qa_items'] = self.total_qa_items
            
        print(f"Saved {n} transitions to {output_path}")
    
    def save_json(self, output_path: str):
        """Save collected data to JSON format (for smaller datasets)."""
        data = {
            'transitions': [t.to_dict() for t in self.transitions],
            'episode_starts': self.episode_starts,
            'scene_info': self.scene_info,
            'stats': {
                'total_steps': self.total_steps,
                'total_qa_items': self.total_qa_items,
                'num_episodes': len(self.episode_starts),
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved {len(self.transitions)} transitions to {output_path}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        return {
            'total_transitions': len(self.transitions),
            'total_episodes': len(self.episode_starts),
            'total_qa_items': self.total_qa_items,
            'scenes_collected': list(self.scene_info.keys()),
            'scene_info': self.scene_info,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Collect QA-enhanced offline RL data"
    )
    
    parser.add_argument("--scenario-dir", type=str,
                       default=os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes_converted"),
                       help="Path to ScenarioNet converted directory")
    parser.add_argument("--qa-dir", type=str,
                       default=os.path.join(project_root, "dataset", "QA_Data"),
                       help="Path to QA data directory")
    parser.add_argument("--nuscenes-dir", type=str,
                       default=os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes", "v1.0-mini"),
                       help="Path to NuScenes data directory")
    parser.add_argument("--output", type=str, default=os.path.join(project_root, "outputs", "offline_data", "qa_enhanced_dataset.h5"),
                       help="Output file path")
    parser.add_argument("--format", type=str, default="hdf5",
                       choices=["hdf5", "json"],
                       help="Output format")
    parser.add_argument("--scenes", type=str, nargs="*", default=None,
                       help="Specific scenes to collect (default: all)")
    parser.add_argument("--include-image-paths", action="store_true",
                       help="Include 6-camera image paths in dataset (default: False)")
    parser.add_argument("--num-waypoints", type=int, default=20,
                       help="Number of future waypoints for trajectory action (default: 20)")
    
    args = parser.parse_args()
    
    # Load QA data
    print("Loading QA data...")
    qa_loader = NuScenesQALoader(
        qa_data_dir=args.qa_dir,
        nuscenes_dataroot=args.nuscenes_dir,
    ).load()
    
    # Create matcher
    print("Creating scenario matcher...")
    matcher = ScenarioQAMatcher(qa_loader, args.scenario_dir)
    
    # Create collector
    collector = QAEnhancedDataCollector(
        qa_loader=qa_loader,
        matcher=matcher,
        output_format=args.format,
        include_image_paths=args.include_image_paths,
        num_waypoints=args.num_waypoints,
    )
    
    # Collect data
    if args.scenes:
        for scene in args.scenes:
            print(f"Collecting from {scene}...")
            collector.collect_from_scenario(scene)
    else:
        print("Collecting from all matching scenes...")
        collector.collect_all_matching_scenes()
    
    # Print stats
    stats = collector.get_stats()
    print("\n=== Collection Statistics ===")
    print(f"Total transitions: {stats['total_transitions']}")
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total QA items: {stats['total_qa_items']}")
    print(f"Scenes: {stats['scenes_collected']}")
    
    # Save
    if args.format == "hdf5":
        collector.save_hdf5(args.output)
    else:
        collector.save_json(args.output)


if __name__ == "__main__":
    main()
