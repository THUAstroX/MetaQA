#!/usr/bin/env python3
"""
Unified Offline RL Data Collection Script.

This script provides a unified interface for collecting various types of
offline RL datasets from NuScenes/ScenarioNet data.

Supported Collection Modes:
    1. original:    Original sensor frequency (~12Hz) without QA
    2. original_qa: Original frequency with QA annotations at keyframes
    3. keyframe:    Keyframe data only (2Hz)
    4. keyframe_qa: Keyframe data with QA annotations
    5. trajectory:  Trajectory-based observation/action (history + future)

Observation Types:
    - state:      Ego state + lane info + surrounding vehicles (instant)
    - trajectory: Historical trajectory (ego + surrounding) over time window

Action Types:
    - waypoint:   Future trajectory waypoints
    - control:    Low-level control (steering, acceleration)

Cost Functions (for CMDP):
    - none:       MDP without cost
    - collision:  Binary collision cost
    - ttc:        Time-to-collision based cost

Output Format:
    HDF5 file containing:
    - observations: (N, obs_dim)
    - actions: (N, action_dim)
    - rewards: (N,)
    - terminals: (N,)
    - costs: (N,) [optional, for CMDP]
    - qa_data: QA annotations [optional]
    - metadata: Collection parameters

Usage Examples:
    # Original frequency data
    python -m meta_qa.scripts.data_collect.collect_data --mode original
    
    # Original frequency with QA
    python -m meta_qa.scripts.data_collect.collect_data --mode original_qa
    
    # Keyframe only
    python -m meta_qa.scripts.data_collect.collect_data --mode keyframe
    
    # Trajectory-based with history 0.5s and future 2.0s
    python -m meta_qa.scripts.data_collect.collect_data \\
        --mode trajectory --history_sec 0.5 --future_sec 2.0
    
    # CMDP with TTC cost
    python -m meta_qa.scripts.data_collect.collect_data \\
        --mode original --cost_type ttc
    
    # Specific scenes
    python -m meta_qa.scripts.data_collect.collect_data \\
        --mode original_qa --scenes scene-0061 scene-0103
"""

import os
import sys
import argparse
import json
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Default paths
DEFAULT_NUSCENES_ROOT = os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes", "v1.0-mini")
DEFAULT_SCENARIO_DIR = os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes_converted")
DEFAULT_QA_PATH = os.path.join(project_root, "dataset", "QA_Data", "NuScenes_QA_val.json")
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "outputs", "offline_data")


class CollectionMode(Enum):
    """Data collection modes."""
    ORIGINAL = "original"           # Original frequency (~12Hz) without QA
    ORIGINAL_QA = "original_qa"     # Original frequency with QA at keyframes
    KEYFRAME = "keyframe"           # Keyframe only (2Hz)
    KEYFRAME_QA = "keyframe_qa"     # Keyframe with QA
    TRAJECTORY = "trajectory"       # Trajectory-based obs/action


class ObservationType(Enum):
    """Observation types."""
    STATE = "state"           # Instant state (ego + lane + surrounding)
    TRAJECTORY = "trajectory" # Historical trajectory


class CostType(Enum):
    """Cost function types."""
    NONE = "none"
    COLLISION = "collision"
    TTC = "ttc"


@dataclass
class CollectionConfig:
    """Configuration for data collection."""
    # Mode
    mode: CollectionMode = CollectionMode.ORIGINAL
    
    # Paths
    nuscenes_root: str = ""
    scenario_dir: str = ""
    qa_path: str = ""
    output_path: str = ""
    
    # Scene selection
    scenes: List[str] = field(default_factory=list)
    max_scenes: Optional[int] = None
    
    # Frequency
    original_fps: float = 12.0
    keyframe_fps: float = 2.0
    
    # Trajectory parameters
    history_sec: float = 0.5
    future_sec: float = 2.0
    
    # Observation
    max_vehicles: int = 10
    obs_type: ObservationType = ObservationType.STATE
    
    # Cost (CMDP)
    cost_type: CostType = CostType.NONE
    ttc_threshold: float = 3.0
    
    # Options
    include_images: bool = False
    verbose: bool = True


class UnifiedDataCollector:
    """
    Unified data collector for offline RL datasets.
    
    Supports multiple collection modes, observation types, and cost functions.
    """
    
    # Observation dimensions for state-based observation
    EGO_STATE_DIM = 5       # x, y, vx, vy, heading (or vx, vy, heading, speed + lane)
    LANE_INFO_DIM = 3       # dist_to_left, dist_to_right, lane_width
    PER_VEHICLE_DIM = 6     # rel_x, rel_y, rel_vx, rel_vy, heading, distance
    
    def __init__(self, config: CollectionConfig):
        """
        Initialize the collector.
        
        Args:
            config: Collection configuration
        """
        self.config = config
        
        # Compute dimensions
        self._compute_dimensions()
        
        # Data storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.terminals = []
        self.costs = []
        self.episode_starts = []
        self.episode_infos = []
        
        # QA storage (for QA modes)
        self.qa_data = []
        
        # Frame metadata (timestamps, is_keyframe, etc.)
        self.frame_metadata = []
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.total_qa_items = 0
        
        # Lazy-loaded components
        self._replay_env = None
        self._qa_loader = None
        self._cost_function = None
        
    def _compute_dimensions(self):
        """Compute observation and action dimensions based on config."""
        cfg = self.config
        
        if cfg.mode == CollectionMode.TRAJECTORY or cfg.obs_type == ObservationType.TRAJECTORY:
            # Trajectory-based observation
            history_steps = max(1, int(cfg.history_sec * cfg.original_fps))
            future_steps = max(1, int(cfg.future_sec * cfg.original_fps))
            
            ego_dim = 5  # x, y, vx, vy, heading
            vehicle_dim = 5  # rel_x, rel_y, rel_vx, rel_vy, heading
            waypoint_dim = 5  # rel_x, rel_y, rel_vx, rel_vy, heading
            
            self.obs_dim = history_steps * (ego_dim + cfg.max_vehicles * vehicle_dim)
            self.action_dim = future_steps * waypoint_dim
            self.history_steps = history_steps
            self.future_steps = future_steps
        else:
            # State-based observation
            self.obs_dim = self.EGO_STATE_DIM + self.LANE_INFO_DIM + cfg.max_vehicles * self.PER_VEHICLE_DIM
            self.action_dim = 40  # 20 waypoints * 2 (x, y)
            self.history_steps = 1
            self.future_steps = 20
    
    def _get_replay_env(self):
        """Get or create replay environment."""
        if self._replay_env is None:
            from meta_qa.tools import ReplayOriginalEnv
            
            self._replay_env = ReplayOriginalEnv(
                nuscenes_dataroot=self.config.nuscenes_root,
                scenario_dir=self.config.scenario_dir,
                qa_loader=self._get_qa_loader(),
                original_fps=self.config.original_fps,
            )
            self._replay_env.load()
        return self._replay_env
    
    def _get_qa_loader(self):
        """Get or create QA loader (only for QA modes)."""
        if self._qa_loader is None:
            if self.config.mode in [CollectionMode.ORIGINAL_QA, CollectionMode.KEYFRAME_QA]:
                qa_path = self.config.qa_path
                # qa_path can be a file or directory
                if os.path.isfile(qa_path):
                    qa_data_dir = os.path.dirname(qa_path)
                else:
                    qa_data_dir = qa_path
                
                if os.path.exists(qa_data_dir):
                    try:
                        from meta_qa.tools import NuScenesQALoader
                        self._qa_loader = NuScenesQALoader(
                            qa_data_dir=qa_data_dir,
                            nuscenes_dataroot=self.config.nuscenes_root,
                        )
                        self._qa_loader.load()
                        if self.config.verbose:
                            print(f"Loaded QA data: {len(self._qa_loader.qa_items)} items")
                            print(f"  Samples with QA: {len(self._qa_loader.sample_to_qa)}")
                    except Exception as e:
                        print(f"Warning: Could not load QA data: {e}")
                        import traceback
                        traceback.print_exc()
        return self._qa_loader
    
    def _get_cost_function(self):
        """Get or create cost function."""
        if self._cost_function is None and self.config.cost_type != CostType.NONE:
            from meta_qa.cost import CollisionCost, TTCCost
            
            if self.config.cost_type == CostType.COLLISION:
                self._cost_function = CollisionCost()
            elif self.config.cost_type == CostType.TTC:
                self._cost_function = TTCCost(threshold=self.config.ttc_threshold)
        return self._cost_function
    
    def collect(self) -> Dict[str, Any]:
        """
        Collect data according to configuration.
        
        Returns:
            Dictionary containing collected dataset
        """
        cfg = self.config
        
        if cfg.verbose:
            print("=" * 60)
            print("Unified Data Collection")
            print("=" * 60)
            print(f"Mode: {cfg.mode.value}")
            print(f"Cost type: {cfg.cost_type.value}")
            print(f"Observation dim: {self.obs_dim}")
            print(f"Action dim: {self.action_dim}")
            if cfg.mode == CollectionMode.TRAJECTORY:
                print(f"History: {cfg.history_sec}s ({self.history_steps} steps)")
                print(f"Future: {cfg.future_sec}s ({self.future_steps} steps)")
            print("=" * 60)
        
        # Get scenes to collect
        replay_env = self._get_replay_env()
        
        if cfg.scenes:
            scenes = cfg.scenes
        else:
            scenes = replay_env.nuscenes_processor.get_available_scenes()
            if cfg.max_scenes:
                scenes = scenes[:cfg.max_scenes]
        
        if cfg.verbose:
            print(f"\nCollecting from {len(scenes)} scenes...")
        
        # Collect from each scene
        for scene_name in scenes:
            try:
                self._collect_scene(scene_name)
            except Exception as e:
                print(f"Error collecting {scene_name}: {e}")
                import traceback
                traceback.print_exc()
        
        # Build final dataset
        return self._build_dataset()
    
    def _collect_scene(self, scene_name: str):
        """Collect data from a single scene."""
        cfg = self.config
        replay_env = self._get_replay_env()
        
        if cfg.verbose:
            print(f"\nCollecting {scene_name}...")
        
        if not replay_env.load_scene(scene_name):
            print(f"  Failed to load scene")
            return
        
        scene_info = replay_env.get_scene_info()
        if cfg.verbose:
            print(f"  Frames: {scene_info['num_frames']}, Samples: {scene_info['num_samples']}")
        
        # Collect all frames first
        all_frames = list(replay_env.iterate_frames())
        
        if len(all_frames) < self.history_steps + self.future_steps:
            print(f"  Scene too short, skipping")
            return
        
        # Mark episode start
        episode_start_idx = len(self.observations)
        self.episode_starts.append(episode_start_idx)
        
        scene_steps = 0
        scene_qa_items = 0
        
        # Determine which frames to process based on mode
        if cfg.mode in [CollectionMode.KEYFRAME, CollectionMode.KEYFRAME_QA]:
            # Only process keyframes
            frame_indices = [i for i, f in enumerate(all_frames) if f.is_sample]
        else:
            # Process all frames (original frequency)
            frame_indices = list(range(len(all_frames)))
        
        # Process frames
        for current_idx in frame_indices:
            # Skip if not enough history or future
            if current_idx < self.history_steps - 1:
                continue
            if current_idx >= len(all_frames) - self.future_steps:
                continue
            
            frame = all_frames[current_idx]
            
            # Build observation
            if cfg.mode == CollectionMode.TRAJECTORY or cfg.obs_type == ObservationType.TRAJECTORY:
                obs = self._build_trajectory_observation(all_frames, current_idx)
            else:
                obs = self._build_state_observation(frame)
            
            # Build action
            if cfg.mode == CollectionMode.TRAJECTORY:
                action = self._build_trajectory_action(all_frames, current_idx)
            else:
                action = self._build_waypoint_action(all_frames, current_idx)
            
            # Compute cost
            cost = 0.0
            cost_fn = self._get_cost_function()
            if cost_fn and frame.ego_state:
                cost = self._compute_cost(frame)
            
            # Reward (simple forward progress reward)
            reward = 1.0
            
            # Terminal
            is_terminal = (current_idx == frame_indices[-1]) or \
                         (current_idx >= len(all_frames) - self.future_steps - 1)
            
            # Store data
            self.observations.append(obs)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(is_terminal)
            self.costs.append(cost)
            
            # Store frame metadata
            self.frame_metadata.append({
                'timestamp': frame.timestamp,
                'is_sample': frame.is_sample,
                'sample_index': frame.sample_index,
                'interpolation_ratio': frame.interpolation_ratio,
            })
            
            # Store QA data if applicable
            if cfg.mode in [CollectionMode.ORIGINAL_QA, CollectionMode.KEYFRAME_QA]:
                if frame.is_sample and self._qa_loader is not None:
                    # Get sample_token for this keyframe
                    sample_token = frame.sample_token or frame.token
                    sample_qa = self._qa_loader.get_qa_for_sample(sample_token)
                    if sample_qa and sample_qa.qa_items:
                        for qa in sample_qa.qa_items:
                            self.qa_data.append({
                                'step_index': len(self.observations) - 1,
                                'scene': scene_name,
                                'sample_index': frame.sample_index,
                                'sample_token': sample_token,
                                'question': qa.question,
                                'answer': qa.answer,
                                'template_type': qa.template_type,
                                'num_hop': qa.num_hop,
                            })
                            scene_qa_items += 1
            
            scene_steps += 1
        
        # Store episode info
        self.episode_infos.append({
            'scene_name': scene_name,
            'num_steps': scene_steps,
            'num_qa_items': scene_qa_items,
            'duration': scene_info['duration_seconds'],
        })
        
        self.total_steps += scene_steps
        self.total_episodes += 1
        self.total_qa_items += scene_qa_items
        
        if cfg.verbose:
            print(f"  Collected {scene_steps} transitions, {scene_qa_items} QA items")
    
    def _build_state_observation(self, frame) -> np.ndarray:
        """Build state-based observation from a single frame."""
        cfg = self.config
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        if frame.ego_state:
            ego = frame.ego_state
            # Ego state
            obs[0] = ego.velocity[0]
            obs[1] = ego.velocity[1]
            obs[2] = ego.heading
            obs[3] = np.linalg.norm(ego.velocity)
            obs[4] = 0.0  # Placeholder for additional ego info
        
        # Lane info (placeholder - would need additional data)
        obs[5] = 0.0  # dist_to_left
        obs[6] = 0.0  # dist_to_right
        obs[7] = 3.5  # lane_width (default)
        
        # Surrounding vehicles
        base_offset = self.EGO_STATE_DIM + self.LANE_INFO_DIM
        
        if frame.ego_state and frame.surrounding_states:
            ego_pos = np.array(frame.ego_state.position[:2])
            ego_heading = frame.ego_state.heading
            cos_h, sin_h = np.cos(-ego_heading), np.sin(-ego_heading)
            
            # Sort by distance
            sorted_surrounding = sorted(
                frame.surrounding_states,
                key=lambda x: np.linalg.norm(np.array(x[1].position[:2]) - ego_pos)
            )
            
            for i, (vid, state) in enumerate(sorted_surrounding[:cfg.max_vehicles]):
                base_idx = base_offset + i * self.PER_VEHICLE_DIM
                
                # Relative position
                rel_pos = np.array(state.position[:2]) - ego_pos
                local_x = rel_pos[0] * cos_h - rel_pos[1] * sin_h
                local_y = rel_pos[0] * sin_h + rel_pos[1] * cos_h
                
                # Relative velocity
                rel_vel = np.array(state.velocity[:2])
                local_vx = rel_vel[0] * cos_h - rel_vel[1] * sin_h
                local_vy = rel_vel[0] * sin_h + rel_vel[1] * cos_h
                
                obs[base_idx + 0] = local_x
                obs[base_idx + 1] = local_y
                obs[base_idx + 2] = local_vx
                obs[base_idx + 3] = local_vy
                obs[base_idx + 4] = state.heading - ego_heading
                obs[base_idx + 5] = np.linalg.norm(rel_pos)
        
        return obs
    
    def _build_trajectory_observation(self, frames: List, current_idx: int) -> np.ndarray:
        """Build trajectory-based observation from historical frames."""
        cfg = self.config
        
        ego_dim = 5
        vehicle_dim = 5
        total_dim = self.history_steps * (ego_dim + cfg.max_vehicles * vehicle_dim)
        obs = np.zeros(total_dim, dtype=np.float32)
        
        # Get current frame for reference
        current_frame = frames[current_idx]
        if not current_frame.ego_state:
            return obs
        
        current_pos = np.array(current_frame.ego_state.position[:2])
        current_heading = current_frame.ego_state.heading
        cos_h, sin_h = np.cos(-current_heading), np.sin(-current_heading)
        
        def transform_to_local(pos, vel):
            rel_pos = pos - current_pos
            local_pos = np.array([
                rel_pos[0] * cos_h - rel_pos[1] * sin_h,
                rel_pos[0] * sin_h + rel_pos[1] * cos_h
            ])
            local_vel = np.array([
                vel[0] * cos_h - vel[1] * sin_h,
                vel[0] * sin_h + vel[1] * cos_h
            ])
            return local_pos, local_vel
        
        # Collect ego history
        start_idx = max(0, current_idx - self.history_steps + 1)
        ego_history = []
        surrounding_history = {}  # vid -> list of states
        
        for idx in range(start_idx, current_idx + 1):
            frame = frames[idx]
            if frame.ego_state:
                pos = np.array(frame.ego_state.position[:2])
                vel = np.array(frame.ego_state.velocity[:2])
                heading = frame.ego_state.heading
                ego_history.append((pos, vel, heading))
            
            for vid, state in frame.surrounding_states:
                if vid not in surrounding_history:
                    surrounding_history[vid] = []
                surrounding_history[vid].append((
                    np.array(state.position[:2]),
                    np.array(state.velocity[:2]),
                    state.heading
                ))
        
        # Fill ego observation
        n_ego = len(ego_history)
        ego_offset = self.history_steps - n_ego
        
        for i, (pos, vel, heading) in enumerate(ego_history):
            idx = (ego_offset + i) * ego_dim
            local_pos, local_vel = transform_to_local(pos, vel)
            obs[idx + 0] = local_pos[0]
            obs[idx + 1] = local_pos[1]
            obs[idx + 2] = local_vel[0]
            obs[idx + 3] = local_vel[1]
            obs[idx + 4] = heading - current_heading
        
        # Fill surrounding vehicle observations
        ego_total = self.history_steps * ego_dim
        
        # Sort vehicles by distance at current frame
        vehicle_distances = {}
        for vid, hist in surrounding_history.items():
            if hist:
                dist = np.linalg.norm(hist[-1][0] - current_pos)
                vehicle_distances[vid] = dist
        
        sorted_vehicles = sorted(vehicle_distances.keys(), key=lambda v: vehicle_distances[v])
        
        for v_idx, vid in enumerate(sorted_vehicles[:cfg.max_vehicles]):
            hist = surrounding_history[vid]
            n_hist = len(hist)
            hist_offset = self.history_steps - n_hist
            
            for i, (pos, vel, heading) in enumerate(hist):
                base_idx = ego_total + v_idx * self.history_steps * vehicle_dim + (hist_offset + i) * vehicle_dim
                local_pos, local_vel = transform_to_local(pos, vel)
                obs[base_idx + 0] = local_pos[0]
                obs[base_idx + 1] = local_pos[1]
                obs[base_idx + 2] = local_vel[0]
                obs[base_idx + 3] = local_vel[1]
                obs[base_idx + 4] = heading - current_heading
        
        return obs
    
    def _build_waypoint_action(self, frames: List, current_idx: int) -> np.ndarray:
        """Build waypoint action (20 future positions)."""
        action = np.zeros(40, dtype=np.float32)  # 20 waypoints * 2 (x, y)
        
        current_frame = frames[current_idx]
        if not current_frame.ego_state:
            return action
        
        current_pos = np.array(current_frame.ego_state.position[:2])
        current_heading = current_frame.ego_state.heading
        cos_h, sin_h = np.cos(-current_heading), np.sin(-current_heading)
        
        waypoints = []
        for idx in range(current_idx + 1, min(current_idx + 21, len(frames))):
            frame = frames[idx]
            if frame.ego_state:
                pos = np.array(frame.ego_state.position[:2])
                rel_pos = pos - current_pos
                local_x = rel_pos[0] * cos_h - rel_pos[1] * sin_h
                local_y = rel_pos[0] * sin_h + rel_pos[1] * cos_h
                waypoints.extend([local_x, local_y])
        
        # Pad if necessary
        while len(waypoints) < 40:
            if len(waypoints) >= 2:
                waypoints.extend([waypoints[-2], waypoints[-1]])
            else:
                waypoints.extend([0.0, 0.0])
        
        action[:40] = waypoints[:40]
        return action
    
    def _build_trajectory_action(self, frames: List, current_idx: int) -> np.ndarray:
        """Build trajectory action (future waypoints with velocity and heading)."""
        waypoint_dim = 5
        action = np.zeros(self.future_steps * waypoint_dim, dtype=np.float32)
        
        current_frame = frames[current_idx]
        if not current_frame.ego_state:
            return action
        
        current_pos = np.array(current_frame.ego_state.position[:2])
        current_heading = current_frame.ego_state.heading
        cos_h, sin_h = np.cos(-current_heading), np.sin(-current_heading)
        
        n_waypoints = 0
        for idx in range(current_idx, min(current_idx + self.future_steps + 1, len(frames))):
            frame = frames[idx]
            if frame.ego_state:
                pos = np.array(frame.ego_state.position[:2])
                vel = np.array(frame.ego_state.velocity[:2])
                heading = frame.ego_state.heading
                
                rel_pos = pos - current_pos
                local_x = rel_pos[0] * cos_h - rel_pos[1] * sin_h
                local_y = rel_pos[0] * sin_h + rel_pos[1] * cos_h
                local_vx = vel[0] * cos_h - vel[1] * sin_h
                local_vy = vel[0] * sin_h + vel[1] * cos_h
                
                base = n_waypoints * waypoint_dim
                action[base + 0] = local_x
                action[base + 1] = local_y
                action[base + 2] = local_vx
                action[base + 3] = local_vy
                action[base + 4] = heading - current_heading
                
                n_waypoints += 1
                if n_waypoints >= self.future_steps:
                    break
        
        # Extrapolate if necessary
        if n_waypoints > 0 and n_waypoints < self.future_steps:
            last_base = (n_waypoints - 1) * waypoint_dim
            for i in range(n_waypoints, self.future_steps):
                base = i * waypoint_dim
                # Simple copy of last waypoint
                action[base:base + waypoint_dim] = action[last_base:last_base + waypoint_dim]
        
        return action
    
    def _compute_cost(self, frame) -> float:
        """Compute cost for a frame."""
        cost_fn = self._get_cost_function()
        if not cost_fn or not frame.ego_state:
            return 0.0
        
        try:
            from meta_qa.cost import CostState, EgoState, SurroundingVehicle
            
            ego = EgoState(
                position=np.array(frame.ego_state.position[:2]),
                velocity=np.array(frame.ego_state.velocity[:2]),
                heading=frame.ego_state.heading,
                speed=np.linalg.norm(frame.ego_state.velocity[:2]),
            )
            
            surrounding = []
            for vid, state in frame.surrounding_states:
                pos = np.array(state.position[:2])
                vel = np.array(state.velocity[:2])
                dist = np.linalg.norm(pos - ego.position)
                surrounding.append(SurroundingVehicle(
                    position=pos,
                    velocity=vel,
                    heading=state.heading,
                    distance=dist,
                ))
            
            cost_state = CostState(ego=ego, surrounding=surrounding, info={})
            result = cost_fn.compute(cost_state)
            return result.value
        except Exception as e:
            return 0.0
    
    def _build_dataset(self) -> Dict[str, Any]:
        """Build final dataset dictionary."""
        dataset = {
            'observations': np.array(self.observations, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'terminals': np.array(self.terminals, dtype=bool),
            'episode_starts': np.array(self.episode_starts, dtype=np.int64),
        }
        
        if self.config.cost_type != CostType.NONE:
            dataset['costs'] = np.array(self.costs, dtype=np.float32)
        
        return dataset
    
    def save(self, output_path: str):
        """Save collected data to HDF5 file."""
        import h5py
        
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        
        dataset = self._build_dataset()
        
        with h5py.File(output_path, 'w') as f:
            # Save main arrays
            for key, value in dataset.items():
                f.create_dataset(key, data=value, compression='gzip')
            
            # Save QA data if present
            if self.qa_data:
                qa_grp = f.create_group('qa_data')
                qa_grp.attrs['num_items'] = len(self.qa_data)
                
                # Save as JSON string (HDF5 doesn't handle complex nested structures well)
                qa_json = json.dumps(self.qa_data)
                qa_grp.create_dataset('items', data=qa_json)
            
            # Save frame metadata
            if self.frame_metadata:
                meta_grp = f.create_group('frame_metadata')
                timestamps = np.array([m['timestamp'] for m in self.frame_metadata])
                is_sample = np.array([m['is_sample'] for m in self.frame_metadata])
                sample_indices = np.array([m['sample_index'] for m in self.frame_metadata])
                interp_ratios = np.array([m['interpolation_ratio'] for m in self.frame_metadata])
                
                meta_grp.create_dataset('timestamps', data=timestamps)
                meta_grp.create_dataset('is_sample', data=is_sample)
                meta_grp.create_dataset('sample_indices', data=sample_indices)
                meta_grp.create_dataset('interpolation_ratios', data=interp_ratios)
            
            # Save configuration metadata
            f.attrs['mode'] = self.config.mode.value
            f.attrs['cost_type'] = self.config.cost_type.value
            f.attrs['obs_dim'] = self.obs_dim
            f.attrs['action_dim'] = self.action_dim
            f.attrs['history_sec'] = self.config.history_sec
            f.attrs['future_sec'] = self.config.future_sec
            f.attrs['history_steps'] = self.history_steps
            f.attrs['future_steps'] = self.future_steps
            f.attrs['original_fps'] = self.config.original_fps
            f.attrs['max_vehicles'] = self.config.max_vehicles
            f.attrs['total_steps'] = self.total_steps
            f.attrs['total_episodes'] = self.total_episodes
            f.attrs['total_qa_items'] = self.total_qa_items
        
        if self.config.verbose:
            print(f"\nSaved dataset to {output_path}")
            print(f"  Total episodes: {self.total_episodes}")
            print(f"  Total steps: {self.total_steps}")
            print(f"  Total QA items: {self.total_qa_items}")
            print(f"  Observation shape: ({self.total_steps}, {self.obs_dim})")
            print(f"  Action shape: ({self.total_steps}, {self.action_dim})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        rewards = np.array(self.rewards) if self.rewards else np.array([0])
        costs = np.array(self.costs) if self.costs else np.array([0])
        
        return {
            'mode': self.config.mode.value,
            'cost_type': self.config.cost_type.value,
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'total_qa_items': self.total_qa_items,
            'avg_episode_length': self.total_steps / max(1, self.total_episodes),
            'obs_dim': self.obs_dim,
            'action_dim': self.action_dim,
            'avg_reward': float(np.mean(rewards)),
            'avg_cost': float(np.mean(costs)),
            'total_cost': float(np.sum(costs)),
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified offline RL data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Collection Modes:
  original     - Original sensor frequency (~12Hz) without QA
  original_qa  - Original frequency with QA at keyframes
  keyframe     - Keyframe data only (2Hz)
  keyframe_qa  - Keyframe data with QA annotations
  trajectory   - Trajectory-based observation/action

Examples:
  # Original frequency data
  python -m meta_qa.scripts.data_collect.collect_data --mode original
  
  # Keyframe with QA and TTC cost (CMDP)
  python -m meta_qa.scripts.data_collect.collect_data \\
      --mode keyframe_qa --cost_type ttc
  
  # Trajectory-based with custom windows
  python -m meta_qa.scripts.data_collect.collect_data \\
      --mode trajectory --history_sec 1.0 --future_sec 3.0
"""
    )
    
    # Mode
    parser.add_argument("--mode", type=str, default="original",
                       choices=["original", "original_qa", "keyframe", "keyframe_qa", "trajectory"],
                       help="Collection mode")
    
    # Data paths
    parser.add_argument("--nuscenes_root", type=str, default=DEFAULT_NUSCENES_ROOT,
                       help="Path to NuScenes dataset root")
    parser.add_argument("--scenario_dir", type=str, default=DEFAULT_SCENARIO_DIR,
                       help="Path to ScenarioNet converted data")
    parser.add_argument("--qa_path", type=str, default=DEFAULT_QA_PATH,
                       help="Path to NuScenes QA JSON file")
    
    # Scene selection
    parser.add_argument("--scene", type=str, default=None,
                       help="Specific scene to collect")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                       help="List of scenes to collect")
    parser.add_argument("--all_scenes", action="store_true",
                       help="Collect from all available scenes")
    parser.add_argument("--max_scenes", type=int, default=None,
                       help="Maximum number of scenes")
    
    # Trajectory parameters
    parser.add_argument("--history_sec", type=float, default=0.5,
                       help="History window in seconds (default: 0.5)")
    parser.add_argument("--future_sec", type=float, default=2.0,
                       help="Future horizon in seconds (default: 2.0)")
    parser.add_argument("--original_fps", type=float, default=12.0,
                       help="Original frame rate (default: 12.0)")
    
    # Observation
    parser.add_argument("--max_vehicles", type=int, default=10,
                       help="Maximum surrounding vehicles (default: 10)")
    parser.add_argument("--obs_type", type=str, default="state",
                       choices=["state", "trajectory"],
                       help="Observation type (state or trajectory)")
    
    # Cost (CMDP)
    parser.add_argument("--cost_type", type=str, default="none",
                       choices=["none", "collision", "ttc"],
                       help="Cost function type (none for MDP)")
    parser.add_argument("--ttc_threshold", type=float, default=3.0,
                       help="TTC threshold in seconds")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output file path")
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory")
    
    # Options
    parser.add_argument("--verbose", action="store_true", default=True,
                       help="Verbose output")
    parser.add_argument("--quiet", action="store_true",
                       help="Quiet mode")
    
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Build configuration
    config = CollectionConfig(
        mode=CollectionMode(args.mode),
        nuscenes_root=args.nuscenes_root,
        scenario_dir=args.scenario_dir,
        qa_path=args.qa_path,
        history_sec=args.history_sec,
        future_sec=args.future_sec,
        original_fps=args.original_fps,
        max_vehicles=args.max_vehicles,
        obs_type=ObservationType(args.obs_type),
        cost_type=CostType(args.cost_type),
        ttc_threshold=args.ttc_threshold,
        verbose=not args.quiet,
    )
    
    # Set scenes
    if args.scene:
        config.scenes = [args.scene]
    elif args.scenes:
        config.scenes = args.scenes
    elif not args.all_scenes:
        # Default to all scenes
        config.scenes = []  # Will be populated by collector
    
    config.max_scenes = args.max_scenes
    
    # Create collector
    collector = UnifiedDataCollector(config)
    
    # Collect data
    dataset = collector.collect()
    
    # Determine output path
    output_path = args.output
    if not output_path:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Auto-generate filename
        cost_suffix = f"_{args.cost_type}" if args.cost_type != "none" else ""
        if args.mode == "trajectory":
            filename = f"{args.mode}_h{args.history_sec}s_f{args.future_sec}s{cost_suffix}.h5"
        else:
            filename = f"{args.mode}{cost_suffix}.h5"
        
        output_path = os.path.join(args.output_dir, filename)
    
    # Save
    if collector.total_steps > 0:
        collector.save(output_path)
        
        # Print statistics
        stats = collector.get_statistics()
        if config.verbose:
            print("\n" + "=" * 60)
            print("Collection Statistics")
            print("=" * 60)
            for key, value in stats.items():
                print(f"  {key}: {value}")
    else:
        print("\nNo data collected!")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
