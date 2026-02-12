#!/usr/bin/env python3
"""
Unified Offline RL Data Collection Script.

This script provides a unified interface for collecting trajectory-based offline
RL datasets from NuScenes/ScenarioNet data. All collection uses trajectory-based
observation (history) and action (future waypoints).

Parameters:
    --frequency: Data frequency
        - original: Original sensor frequency (~12Hz)
        - keyframe: Keyframe only (2Hz)
    
    --qa: Include QA annotations at keyframes
    
    --history_sec: Historical observation window in seconds (default: 0.5)
    --future_sec:  Future trajectory horizon in seconds (default: 2.0)

Observation Format:
    Trajectory-based observation containing:
    - Ego trajectory: history_steps × (x, y, vx, vy, heading)
    - Surrounding vehicles: history_steps × max_vehicles × (rel_x, rel_y, rel_vx, rel_vy, heading)

Action Format:
    Future trajectory waypoints:
    - future_steps × (rel_x, rel_y, rel_vx, rel_vy, heading)

Cost Functions (for CMDP):
    - none:      MDP without cost
    - collision: Binary collision cost
    - ttc:       Time-to-collision based cost

Output Format:
    HDF5 file containing:
    - observations: (N, obs_dim) — Trajectory-based observations
    - actions: (N, action_dim) — Future waypoint actions
    - rewards: (N,) — MetaDrive ScenarioEnv reward (driving progress, penalties, etc.)
    - terminals: (N,)
    - costs: (N,) [optional, for CMDP]
    - qa_data: QA annotations [when --qa is set]
    - map_data: Per-episode nearby map features [when --map is set]
    - metadata: Collection parameters

Reward:
    Rewards are collected directly from MetaDrive's ScenarioEnv using
    ReplayEgoCarPolicy. The reward includes:
    - Driving reward (longitudinal progress along reference trajectory)
    - Heading penalty (misalignment with reference direction)
    - Lateral penalty (distance from reference path center)
    - Collision penalty (crash with vehicles/objects)
    - Terminal rewards (success +5.0 / out-of-road -5.0)

Usage Examples:
    # Original frequency (~12Hz) trajectory data
    python -m meta_qa.scripts.data_collect.collect_data
    
    # Original frequency with QA annotations
    python -m meta_qa.scripts.data_collect.collect_data --qa
    
    # Keyframe only (2Hz)
    python -m meta_qa.scripts.data_collect.collect_data --frequency keyframe
    
    # Custom history/future windows
    python -m meta_qa.scripts.data_collect.collect_data \\
        --history_sec 1.0 --future_sec 3.0
    
    # CMDP with TTC cost
    python -m meta_qa.scripts.data_collect.collect_data --cost_type ttc
    
    # Include map features
    python -m meta_qa.scripts.data_collect.collect_data --map
    
    # Specific scenes
    python -m meta_qa.scripts.data_collect.collect_data \\
        --qa --scenes scene-0061 scene-0103
"""

import os
import sys
import argparse
import json
import pickle
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


class FrequencyMode(Enum):
    """Data frequency modes."""
    ORIGINAL = "original"   # Original frequency (~12Hz)
    KEYFRAME = "keyframe"   # Keyframe only (2Hz)


class CostType(Enum):
    """Cost function types."""
    NONE = "none"
    COLLISION = "collision"
    TTC = "ttc"


# Map feature type codes for HDF5 storage
MAP_TYPE_CODES = {
    'LANE_SURFACE_STREET': 0,
    'LANE_SURFACE_UNSTRUCTURE': 1,
    'ROAD_LINE_SOLID_SINGLE_WHITE': 2,
    'ROAD_LINE_BROKEN_SINGLE_WHITE': 3,
    'ROAD_LINE_SOLID_SINGLE_YELLOW': 4,
    'ROAD_EDGE_SIDEWALK': 5,
    'CROSSWALK': 6,
}

# Reverse mapping for decoding
MAP_TYPE_NAMES = {v: k for k, v in MAP_TYPE_CODES.items()}


@dataclass
class CollectionConfig:
    """Configuration for data collection."""
    # Frequency mode
    frequency: FrequencyMode = FrequencyMode.ORIGINAL
    
    # QA flag
    include_qa: bool = True
    
    # Paths
    nuscenes_root: str = ""
    scenario_dir: str = ""
    qa_path: str = ""
    output_path: str = ""
    
    # Scene selection
    scenes: List[str] = field(default_factory=list)
    max_scenes: Optional[int] = None
    
    # Frequency parameters
    original_fps: float = 12.0
    keyframe_fps: float = 2.0
    
    # Trajectory parameters (core parameters)
    history_sec: float = 0.5
    future_sec: float = 2.0
    
    # Observation parameters
    max_vehicles: int = 10
    
    # Cost (CMDP)
    cost_type: CostType = CostType.NONE
    ttc_threshold: float = 3.0
    
    # Map data
    include_map: bool = True
    map_radius: float = 50.0  # meters — radius around ego trajectory to collect map features
    
    # Options
    verbose: bool = True


class TrajectoryDataCollector:
    """
    Trajectory-based data collector for offline RL datasets.
    
    All observations are trajectory-based (historical trajectory).
    All actions are future waypoint trajectories.
    """
    
    # Dimension constants
    EGO_DIM = 5             # x, y, vx, vy, heading
    VEHICLE_DIM = 5         # rel_x, rel_y, rel_vx, rel_vy, heading
    WAYPOINT_DIM = 5        # rel_x, rel_y, rel_vx, rel_vy, heading
    
    def __init__(self, config: CollectionConfig):
        """
        Initialize the collector.
        
        Args:
            config: Collection configuration
        """
        self.config = config
        
        # Compute dimensions based on history/future
        self._compute_dimensions()
        
        # Data storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.terminals = []
        self.costs = []
        self.episode_starts = []
        self.episode_infos = []
        
        # QA storage
        self.qa_data = []
        
        # Frame metadata
        self.frame_metadata = []
        
        # Map data storage (per-episode)
        self.map_data_per_episode = []
        
        # MetaDrive ScenarioEnv for reward computation
        self._metadrive_env = None
        self._scene_to_seed = {}  # scene_name -> seed index
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
        self.total_qa_items = 0
        
        # Lazy-loaded components
        self._replay_env = None
        self._qa_loader = None
        self._cost_function = None
        
    def _compute_dimensions(self):
        """Compute observation and action dimensions."""
        cfg = self.config
        
        # Get the effective FPS based on frequency mode
        if cfg.frequency == FrequencyMode.KEYFRAME:
            fps = cfg.keyframe_fps
        else:
            fps = cfg.original_fps
        
        # Compute steps
        self.history_steps = max(1, int(cfg.history_sec * fps))
        self.future_steps = max(1, int(cfg.future_sec * fps))
        
        # Observation: ego history + surrounding vehicles history
        # Shape: history_steps × (ego_dim + max_vehicles × vehicle_dim)
        self.obs_dim = self.history_steps * (self.EGO_DIM + cfg.max_vehicles * self.VEHICLE_DIM)
        
        # Action: future waypoints
        # Shape: future_steps × waypoint_dim
        self.action_dim = self.future_steps * self.WAYPOINT_DIM
    
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
        """Get or create QA loader (only when include_qa is True)."""
        if self._qa_loader is None and self.config.include_qa:
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
            print("Trajectory Data Collection")
            print("=" * 60)
            print(f"Frequency: {cfg.frequency.value} (~{cfg.original_fps if cfg.frequency == FrequencyMode.ORIGINAL else cfg.keyframe_fps}Hz)")
            print(f"Include QA: {cfg.include_qa}")
            print(f"Include Map: {cfg.include_map}" + (f" (radius={cfg.map_radius}m)" if cfg.include_map else ""))
            print(f"History: {cfg.history_sec}s ({self.history_steps} steps)")
            print(f"Future: {cfg.future_sec}s ({self.future_steps} steps)")
            print(f"Observation dim: {self.obs_dim}")
            print(f"Action dim: {self.action_dim}")
            print(f"Cost type: {cfg.cost_type.value}")
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
        try:
            for scene_name in scenes:
                try:
                    self._collect_scene(scene_name)
                except Exception as e:
                    print(f"Error collecting {scene_name}: {e}")
                    import traceback
                    traceback.print_exc()
        finally:
            # Clean up MetaDrive environment
            self._close_metadrive_env()
        
        # Build final dataset
        return self._build_dataset()
    
    def _collect_scene(self, scene_name: str):
        """Collect trajectory data from a single scene."""
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
        
        # Collect MetaDrive rewards for this scene (10Hz)
        metadrive_rewards = self._collect_metadrive_rewards(scene_name)
        if metadrive_rewards is not None and cfg.verbose:
            print(f"  MetaDrive rewards: {len(metadrive_rewards)} steps, "
                  f"mean={metadrive_rewards.mean():.3f}, range=[{metadrive_rewards.min():.3f}, {metadrive_rewards.max():.3f}]")
        
        # Mark episode start
        episode_start_idx = len(self.observations)
        self.episode_starts.append(episode_start_idx)
        
        scene_steps = 0
        scene_qa_items = 0
        
        # Determine which frames to process based on frequency mode
        if cfg.frequency == FrequencyMode.KEYFRAME:
            # Only process keyframes (2Hz)
            frame_indices = [i for i, f in enumerate(all_frames) if f.is_sample]
        else:
            # Process all frames (original frequency ~12Hz)
            frame_indices = list(range(len(all_frames)))
        
        # Process frames
        for current_idx in frame_indices:
            # Skip if not enough history or future
            if current_idx < self.history_steps - 1:
                continue
            if current_idx >= len(all_frames) - self.future_steps:
                continue
            
            frame = all_frames[current_idx]
            
            # Build trajectory observation (always trajectory-based)
            obs = self._build_trajectory_observation(all_frames, current_idx)
            
            # Build trajectory action (always future waypoints)
            action = self._build_trajectory_action(all_frames, current_idx)
            
            # Compute cost
            cost = 0.0
            cost_fn = self._get_cost_function()
            if cost_fn and frame.ego_state:
                cost = self._compute_cost(frame)
            
            # Reward from MetaDrive ScenarioEnv
            reward = 0.0
            if metadrive_rewards is not None and frame.ego_state:
                # Map 12Hz frame to nearest 10Hz MetaDrive step
                # ScenarioNet runs at 10Hz; frame timestamp gives elapsed time
                frame_time = (frame.timestamp - all_frames[0].timestamp) / 1e6  # seconds
                md_step = int(round(frame_time * 10.0))  # 10Hz step index
                md_step = min(md_step, len(metadrive_rewards) - 1)
                md_step = max(md_step, 0)
                reward = float(metadrive_rewards[md_step])
            
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
                'ego_world_x': frame.ego_state.position[0] if frame.ego_state else 0.0,
                'ego_world_y': frame.ego_state.position[1] if frame.ego_state else 0.0,
                'ego_world_heading': frame.ego_state.heading if frame.ego_state else 0.0,
                'scene_name': scene_name,
                'image_paths': frame.image_paths if hasattr(frame, 'image_paths') else {},
            })
            
            # Store QA data if applicable
            if cfg.include_qa and frame.is_sample and self._qa_loader is not None:
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
        
        # Collect map features if enabled
        map_features_count = 0
        if cfg.include_map:
            map_features_count = self._collect_map_features(all_frames, scene_name)
        
        # Store episode info
        self.episode_infos.append({
            'scene_name': scene_name,
            'num_steps': scene_steps,
            'num_qa_items': scene_qa_items,
            'num_map_features': map_features_count,
            'duration': scene_info['duration_seconds'],
        })
        
        self.total_steps += scene_steps
        self.total_episodes += 1
        self.total_qa_items += scene_qa_items
        
        if cfg.verbose:
            qa_str = f", {scene_qa_items} QA items" if cfg.include_qa else ""
            map_str = f", {map_features_count} map features" if cfg.include_map else ""
            print(f"  Collected {scene_steps} transitions{qa_str}{map_str}")
    
    def _init_metadrive_env(self):
        """
        Initialize MetaDrive ScenarioEnv for reward computation.
        
        Uses ReplayEgoCarPolicy to replay the recorded trajectory and
        collects the built-in MetaDrive reward at each 10Hz step.
        """
        if self._metadrive_env is not None:
            return
        
        from metadrive.envs.scenario_env import ScenarioEnv
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy
        
        cfg = self.config
        
        self._metadrive_env = ScenarioEnv(dict(
            agent_policy=ReplayEgoCarPolicy,
            use_render=False,
            data_directory=cfg.scenario_dir,
            num_scenarios=10,
            start_scenario_index=0,
            no_traffic=False,
            reactive_traffic=False,
        ))
        
        # Initialize engine by doing a reset
        self._metadrive_env.reset(seed=0)
        
        # Build scene_name -> seed mapping from data manager
        dm = self._metadrive_env.engine.data_manager
        self._scene_to_seed = {}
        for idx, fname in enumerate(dm.mapping.keys()):
            # Extract scene name from filename like "sd_nuscenes_v1.0-mini_scene-0061.pkl"
            # The scenario_id in metadata is like "scene-0061"
            for part in fname.replace('.pkl', '').split('_'):
                if part.startswith('scene-'):
                    self._scene_to_seed[part] = idx
                    break
            else:
                # Fallback: try to extract scene-XXXX pattern
                import re
                m = re.search(r'(scene-\d+)', fname)
                if m:
                    self._scene_to_seed[m.group(1)] = idx
        
        if cfg.verbose:
            print(f"  MetaDrive env initialized: {len(self._scene_to_seed)} scenes mapped")
    
    def _close_metadrive_env(self):
        """Close MetaDrive environment."""
        if self._metadrive_env is not None:
            try:
                self._metadrive_env.close()
            except Exception:
                pass
            self._metadrive_env = None
    
    def _collect_metadrive_rewards(self, scene_name: str) -> Optional[np.ndarray]:
        """
        Collect rewards from MetaDrive ScenarioEnv for a specific scene.
        
        Runs the scene replay in MetaDrive's physics engine with ReplayEgoCarPolicy
        to get the actual MetaDrive reward at each 10Hz simulation step.
        
        Args:
            scene_name: Scene name (e.g. "scene-0061")
            
        Returns:
            Array of rewards at 10Hz, or None if scene not found
        """
        self._init_metadrive_env()
        
        seed = self._scene_to_seed.get(scene_name)
        if seed is None:
            if self.config.verbose:
                print(f"  Warning: no MetaDrive mapping for {scene_name}, using reward=0")
            return None
        
        env = self._metadrive_env
        obs, info = env.reset(seed=seed)
        
        # Verify we got the right scenario
        meta = env.engine.data_manager.current_scenario.get('metadata', {})
        loaded_id = meta.get('scenario_id', '')
        if loaded_id != scene_name:
            if self.config.verbose:
                print(f"  Warning: expected {scene_name}, got {loaded_id}")
        
        # Step through collecting rewards
        rewards = []
        for step in range(2000):  # Safety limit
            obs, reward, terminated, truncated, info = env.step([0, 0])
            rewards.append(reward)
            if info.get('replay_done', False):
                break
        
        return np.array(rewards, dtype=np.float32)

    def _collect_map_features(self, all_frames: List, scene_name: str) -> int:
        """
        Collect nearby map features from the ScenarioNet pkl file.
        
        Extracts map features (lanes, road lines, crosswalks, road edges) that
        are within map_radius of the ego trajectory, and stores them per-episode.
        
        Args:
            all_frames: All frames in the scene
            scene_name: Scene name for finding the scenario file
            
        Returns:
            Number of map features collected
        """
        replay_env = self._get_replay_env()
        
        # Get the scenario data via the trajectory matcher
        scenario_data = None
        if replay_env.trajectory_matcher and replay_env.trajectory_matcher.interpolator.scenario_data:
            scenario_data = replay_env.trajectory_matcher.interpolator.scenario_data
        
        if scenario_data is None or 'map_features' not in scenario_data:
            self.map_data_per_episode.append(None)
            return 0
        
        map_features = scenario_data['map_features']
        
        # Collect ego trajectory world positions for proximity filtering
        ego_positions = []
        for frame in all_frames:
            if frame.ego_state:
                ego_positions.append(frame.ego_state.position[:2])
        
        if not ego_positions:
            self.map_data_per_episode.append(None)
            return 0
        
        ego_positions = np.array(ego_positions)  # (T, 2)
        radius = self.config.map_radius
        
        # Filter map features by distance to ego trajectory
        collected_points = []
        collected_offsets = [0]
        collected_types = []
        
        for feat_id, feat in map_features.items():
            feat_type = feat.get('type', '')
            if feat_type not in MAP_TYPE_CODES:
                continue
            
            # Get the geometry points
            if 'polyline' in feat:
                pts = np.array(feat['polyline'])[:, :2]  # take x, y only (some have z)
            elif 'polygon' in feat:
                pts = np.array(feat['polygon'])[:, :2]
            else:
                continue
            
            if len(pts) < 2:
                continue
            
            # Check if any point is within radius of any ego position
            # Efficient: compute min distance from feature to ego trajectory
            # Use bounding box pre-filter for speed
            feat_min = pts.min(axis=0)
            feat_max = pts.max(axis=0)
            ego_min = ego_positions.min(axis=0) - radius
            ego_max = ego_positions.max(axis=0) + radius
            
            if (feat_min[0] > ego_max[0] or feat_max[0] < ego_min[0] or
                feat_min[1] > ego_max[1] or feat_max[1] < ego_min[1]):
                continue
            
            # More precise check: min distance from any feature point to any ego point
            # Use sampled ego positions for efficiency
            n_ego = len(ego_positions)
            sample_step = max(1, n_ego // 20)  # sample ~20 ego positions
            ego_sampled = ego_positions[::sample_step]
            
            min_dist = np.inf
            for ep in ego_sampled:
                dists = np.linalg.norm(pts - ep, axis=1)
                min_dist = min(min_dist, dists.min())
                if min_dist < radius:
                    break
            
            if min_dist >= radius:
                continue
            
            # Add this feature
            collected_points.append(pts)
            collected_offsets.append(collected_offsets[-1] + len(pts))
            collected_types.append(MAP_TYPE_CODES[feat_type])
        
        if not collected_types:
            self.map_data_per_episode.append(None)
            return 0
        
        # Store as arrays
        map_episode_data = {
            'points': np.vstack(collected_points).astype(np.float32),     # (total_pts, 2)
            'offsets': np.array(collected_offsets, dtype=np.int32),         # (num_features + 1,)
            'types': np.array(collected_types, dtype=np.int32),            # (num_features,)
        }
        self.map_data_per_episode.append(map_episode_data)
        
        return len(collected_types)
    
    def _build_trajectory_observation(self, frames: List, current_idx: int) -> np.ndarray:
        """
        Build trajectory-based observation from historical frames.
        
        Observation structure:
            [ego_history] + [vehicle_0_history] + ... + [vehicle_N_history]
            
            ego_history: history_steps × (x, y, vx, vy, heading) in local frame
            vehicle_i_history: history_steps × (rel_x, rel_y, rel_vx, rel_vy, heading)
        """
        cfg = self.config
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        
        # Get current frame for reference (local coordinate frame)
        current_frame = frames[current_idx]
        if not current_frame.ego_state:
            return obs
        
        current_pos = np.array(current_frame.ego_state.position[:2])
        current_heading = current_frame.ego_state.heading
        cos_h, sin_h = np.cos(-current_heading), np.sin(-current_heading)
        
        def transform_to_local(pos, vel):
            """Transform world coordinates to local (ego-centric) frame."""
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
        
        # Collect history
        start_idx = max(0, current_idx - self.history_steps + 1)
        ego_history = []
        surrounding_history = {}  # vid -> list of (pos, vel, heading)
        
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
            idx = (ego_offset + i) * self.EGO_DIM
            local_pos, local_vel = transform_to_local(pos, vel)
            obs[idx + 0] = local_pos[0]
            obs[idx + 1] = local_pos[1]
            obs[idx + 2] = local_vel[0]
            obs[idx + 3] = local_vel[1]
            obs[idx + 4] = heading - current_heading
        
        # Fill surrounding vehicle observations
        ego_total_dim = self.history_steps * self.EGO_DIM
        
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
                base_idx = ego_total_dim + v_idx * self.history_steps * self.VEHICLE_DIM + (hist_offset + i) * self.VEHICLE_DIM
                local_pos, local_vel = transform_to_local(pos, vel)
                obs[base_idx + 0] = local_pos[0]
                obs[base_idx + 1] = local_pos[1]
                obs[base_idx + 2] = local_vel[0]
                obs[base_idx + 3] = local_vel[1]
                obs[base_idx + 4] = heading - current_heading
        
        return obs
    
    def _build_trajectory_action(self, frames: List, current_idx: int) -> np.ndarray:
        """
        Build trajectory action (future waypoints).
        
        Action structure:
            future_steps × (rel_x, rel_y, rel_vx, rel_vy, heading)
            
            All coordinates in local (ego-centric) frame of current position.
        """
        action = np.zeros(self.action_dim, dtype=np.float32)
        
        current_frame = frames[current_idx]
        if not current_frame.ego_state:
            return action
        
        current_pos = np.array(current_frame.ego_state.position[:2])
        current_heading = current_frame.ego_state.heading
        cos_h, sin_h = np.cos(-current_heading), np.sin(-current_heading)
        
        n_waypoints = 0
        for idx in range(current_idx + 1, min(current_idx + self.future_steps + 1, len(frames))):
            frame = frames[idx]
            if frame.ego_state:
                pos = np.array(frame.ego_state.position[:2])
                vel = np.array(frame.ego_state.velocity[:2])
                heading = frame.ego_state.heading
                
                # Transform to local frame
                rel_pos = pos - current_pos
                local_x = rel_pos[0] * cos_h - rel_pos[1] * sin_h
                local_y = rel_pos[0] * sin_h + rel_pos[1] * cos_h
                local_vx = vel[0] * cos_h - vel[1] * sin_h
                local_vy = vel[0] * sin_h + vel[1] * cos_h
                
                base = n_waypoints * self.WAYPOINT_DIM
                action[base + 0] = local_x
                action[base + 1] = local_y
                action[base + 2] = local_vx
                action[base + 3] = local_vy
                action[base + 4] = heading - current_heading
                
                n_waypoints += 1
                if n_waypoints >= self.future_steps:
                    break
        
        # Extrapolate if necessary (copy last waypoint)
        if n_waypoints > 0 and n_waypoints < self.future_steps:
            last_base = (n_waypoints - 1) * self.WAYPOINT_DIM
            for i in range(n_waypoints, self.future_steps):
                base = i * self.WAYPOINT_DIM
                action[base:base + self.WAYPOINT_DIM] = action[last_base:last_base + self.WAYPOINT_DIM]
        
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
        except Exception:
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
                qa_json = json.dumps(self.qa_data)
                qa_grp.create_dataset('items', data=qa_json)
            
            # Save frame metadata
            if self.frame_metadata:
                meta_grp = f.create_group('frame_metadata')
                timestamps = np.array([m['timestamp'] for m in self.frame_metadata])
                is_sample = np.array([m['is_sample'] for m in self.frame_metadata])
                sample_indices = np.array([m['sample_index'] for m in self.frame_metadata])
                interp_ratios = np.array([m['interpolation_ratio'] for m in self.frame_metadata])
                ego_world_x = np.array([m['ego_world_x'] for m in self.frame_metadata], dtype=np.float64)
                ego_world_y = np.array([m['ego_world_y'] for m in self.frame_metadata], dtype=np.float64)
                ego_world_heading = np.array([m['ego_world_heading'] for m in self.frame_metadata], dtype=np.float64)
                
                meta_grp.create_dataset('timestamps', data=timestamps)
                meta_grp.create_dataset('is_sample', data=is_sample)
                meta_grp.create_dataset('sample_indices', data=sample_indices)
                meta_grp.create_dataset('interpolation_ratios', data=interp_ratios)
                meta_grp.create_dataset('ego_world_x', data=ego_world_x)
                meta_grp.create_dataset('ego_world_y', data=ego_world_y)
                meta_grp.create_dataset('ego_world_heading', data=ego_world_heading)
                
                # Scene names per step
                scene_names = [m.get('scene_name', '') for m in self.frame_metadata]
                dt_str = h5py.string_dtype()
                meta_grp.create_dataset('scene_names', data=scene_names, dtype=dt_str)
                
                # Camera image paths per step (6 cameras)
                camera_order = [
                    'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT',
                ]
                for cam in camera_order:
                    paths = [m.get('image_paths', {}).get(cam, '') for m in self.frame_metadata]
                    meta_grp.create_dataset(f'image_{cam}', data=paths, dtype=dt_str)
            
            # Save map data (per episode)
            if self.map_data_per_episode:
                map_grp = f.create_group('map_data')
                for ep_idx, map_ep in enumerate(self.map_data_per_episode):
                    if map_ep is not None:
                        ep_grp = map_grp.create_group(f'episode_{ep_idx}')
                        ep_grp.create_dataset('points', data=map_ep['points'], compression='gzip')
                        ep_grp.create_dataset('offsets', data=map_ep['offsets'])
                        ep_grp.create_dataset('types', data=map_ep['types'])
            
            # Save configuration metadata
            f.attrs['frequency'] = self.config.frequency.value
            f.attrs['include_qa'] = self.config.include_qa
            f.attrs['include_map'] = self.config.include_map
            f.attrs['map_radius'] = self.config.map_radius
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
            f.attrs['nuscenes_root'] = self.config.nuscenes_root
        
        if self.config.verbose:
            print(f"\nSaved dataset to {output_path}")
            print(f"  Total episodes: {self.total_episodes}")
            print(f"  Total steps: {self.total_steps}")
            if self.config.include_qa:
                print(f"  Total QA items: {self.total_qa_items}")
            if self.config.include_map:
                n_map_eps = sum(1 for m in self.map_data_per_episode if m is not None)
                total_map_feats = sum(len(m['types']) for m in self.map_data_per_episode if m is not None)
                print(f"  Map data: {total_map_feats} features across {n_map_eps} episodes")
            print(f"  Observation shape: ({self.total_steps}, {self.obs_dim})")
            print(f"  Action shape: ({self.total_steps}, {self.action_dim})")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        rewards = np.array(self.rewards) if self.rewards else np.array([0])
        costs = np.array(self.costs) if self.costs else np.array([0])
        
        return {
            'frequency': self.config.frequency.value,
            'include_qa': self.config.include_qa,
            'cost_type': self.config.cost_type.value,
            'history_sec': self.config.history_sec,
            'future_sec': self.config.future_sec,
            'history_steps': self.history_steps,
            'future_steps': self.future_steps,
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
        description="Trajectory-based offline RL data collection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Parameters:
  --frequency    Data frequency: original (~12Hz) or keyframe (2Hz)
  --qa           Include QA annotations (default: true, use --qa false to disable)
  --map          Include map features (default: true, use --map false to disable)
  --map_radius   Radius around ego trajectory to collect map features (default: 50m)
  --history_sec  Historical observation window (default: 0.5s)
  --future_sec   Future trajectory horizon (default: 2.0s)
  --cost_type    Cost function: none (MDP), collision, or ttc (CMDP)

Examples:
  # Original frequency (~12Hz) with QA + map (default)
  python -m meta_qa.scripts.data_collect.collect_data
  
  # Disable QA annotations
  python -m meta_qa.scripts.data_collect.collect_data --qa false
  
  # Disable map features
  python -m meta_qa.scripts.data_collect.collect_data --map false
  
  # Disable both QA and map
  python -m meta_qa.scripts.data_collect.collect_data --qa false --map false
  
  # Keyframe only (2Hz)
  python -m meta_qa.scripts.data_collect.collect_data --frequency keyframe
  
  # Keyframe with QA
  python -m meta_qa.scripts.data_collect.collect_data --frequency keyframe --qa
  
  # Custom trajectory windows
  python -m meta_qa.scripts.data_collect.collect_data \\
      --history_sec 1.0 --future_sec 3.0
  
  # CMDP with TTC cost
  python -m meta_qa.scripts.data_collect.collect_data --cost_type ttc
"""
    )
    
    # Frequency mode
    parser.add_argument("--frequency", type=str, default="original",
                       choices=["original", "keyframe"],
                       help="Data frequency: original (~12Hz) or keyframe (2Hz)")
    
    # QA flag
    parser.add_argument("--qa", type=lambda x: (str(x).lower() != 'false'), default=True,
                       help="Include QA annotations at keyframes (default: true, use --qa false to disable)")
    
    # Map flag
    parser.add_argument("--map", type=lambda x: (str(x).lower() != 'false'), default=True,
                       help="Include map features (default: true, use --map false to disable)")
    parser.add_argument("--map_radius", type=float, default=50.0,
                       help="Radius around ego trajectory to collect map features (default: 50m)")
    
    # Trajectory parameters
    parser.add_argument("--history_sec", type=float, default=0.5,
                       help="History window in seconds (default: 0.5)")
    parser.add_argument("--future_sec", type=float, default=2.0,
                       help="Future horizon in seconds (default: 2.0)")
    
    # Data paths
    parser.add_argument("--nuscenes_root", type=str, default=DEFAULT_NUSCENES_ROOT,
                       help="Path to NuScenes dataset root")
    parser.add_argument("--scenario_dir", type=str, default=DEFAULT_SCENARIO_DIR,
                       help="Path to ScenarioNet converted data")
    parser.add_argument("--qa_path", type=str, default=DEFAULT_QA_PATH,
                       help="Path to NuScenes QA JSON file or directory")
    
    # Scene selection
    parser.add_argument("--scene", type=str, default=None,
                       help="Specific scene to collect")
    parser.add_argument("--scenes", type=str, nargs="+", default=None,
                       help="List of scenes to collect")
    parser.add_argument("--max_scenes", type=int, default=None,
                       help="Maximum number of scenes")
    
    # Observation parameters
    parser.add_argument("--max_vehicles", type=int, default=10,
                       help="Maximum surrounding vehicles (default: 10)")
    parser.add_argument("--original_fps", type=float, default=12.0,
                       help="Original frame rate (default: 12.0)")
    
    # Cost (CMDP)
    parser.add_argument("--cost_type", type=str, default="none",
                       choices=["none", "collision", "ttc"],
                       help="Cost function type: none (MDP), collision, ttc")
    parser.add_argument("--ttc_threshold", type=float, default=3.0,
                       help="TTC threshold in seconds (default: 3.0)")
    
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
        frequency=FrequencyMode(args.frequency),
        include_qa=args.qa,
        include_map=args.map,
        map_radius=args.map_radius,
        nuscenes_root=args.nuscenes_root,
        scenario_dir=args.scenario_dir,
        qa_path=args.qa_path,
        history_sec=args.history_sec,
        future_sec=args.future_sec,
        original_fps=args.original_fps,
        max_vehicles=args.max_vehicles,
        cost_type=CostType(args.cost_type),
        ttc_threshold=args.ttc_threshold,
        verbose=not args.quiet,
    )
    
    # Set scenes
    if args.scene:
        config.scenes = [args.scene]
    elif args.scenes:
        config.scenes = args.scenes
    
    config.max_scenes = args.max_scenes
    
    # Create collector
    collector = TrajectoryDataCollector(config)
    
    # Collect data
    dataset = collector.collect()
    
    # Determine output path
    output_path = args.output
    if not output_path:
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Auto-generate filename
        freq_str = args.frequency
        qa_str = "_qa" if args.qa else ""
        map_str = "_map" if args.map else ""
        cost_str = f"_{args.cost_type}" if args.cost_type != "none" else ""
        hist_str = f"_h{args.history_sec}s" if args.history_sec != 0.5 else ""
        fut_str = f"_f{args.future_sec}s" if args.future_sec != 2.0 else ""
        
        filename = f"{freq_str}{qa_str}{map_str}{hist_str}{fut_str}{cost_str}.h5"
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
