#!/usr/bin/env python3
"""
Offline RL Dataset Collection Script.

This script collects expert trajectory data from ScenarioNet datasets and saves
them in MDP/CMDP format suitable for offline reinforcement learning.

Features:
    - Collects real expert trajectories from scenario data (nuPlan, nuScenes, Waymo, etc.)
    - Supports both MDP (reward only) and CMDP (reward + cost) formats
    - Saves in HDF5 format (recommended) or D4RL-compatible NPZ format
    - Extracts surrounding vehicle information
    - Multiple cost function options for safe RL

Output Format (HDF5):
    - observations: (N, obs_dim) - State observations
    - actions: (N, action_dim) - Trajectory actions (20 waypoints × 2 = 40 dims)
    - rewards: (N,) - Step rewards from MetaDrive
    - next_observations: (N, obs_dim) - Next state observations
    - terminals: (N,) - Episode termination flags
    - costs: (N,) - Safety costs (for CMDP, optional)
    - episode_starts: (num_episodes,) - Starting indices of each episode

Usage:
    # Collect MDP data (reward only)
    python -m meta_qa.scripts.data_collect.collect_offline_data \\
        --data_dir ~/scenarionet/exp_data/exp_nuplan_converted \\
        --num_scenarios 100 \\
        --output expert_nuplan.h5

    # Collect CMDP data with TTC cost
    python -m meta_qa.scripts.data_collect.collect_offline_data \\
        --data_dir ~/scenarionet/exp_data/exp_nuplan_converted \\
        --num_scenarios 100 \\
        --cost_type ttc \\
        --output expert_nuplan_cmdp.h5
"""

import os
import sys
import argparse
import numpy as np
from typing import Dict, Any, List, Optional, Tuple

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Import from cost module
from meta_qa.cost import (
    CostFunction,
    CostState,
    EgoState,
    SurroundingVehicle,
    CollisionCost,
    TTCCost,
)
from meta_qa.tools import SurroundingVehicleGetter
from meta_qa.data import (
    SurroundingVehicleInfo,
    save_dataset_hdf5,
    get_dataset_info,
)

# Default data directory
DEFAULT_DATA_DIR = os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes_converted")


class ExpertDataCollector:
    """
    Collects expert trajectory data from ScenarioNet scenarios.
    
    Uses ReplayEgoCarPolicy to replay real trajectories and collects
    state-action-reward-next_state tuples for offline RL.
    
    Observation format (ego + lane + surrounding vehicles):
        - Ego state: [vx, vy, heading, speed] (4 dims)
        - Lane info: [dist_to_left, dist_to_right, lane_width] (3 dims)
        - Per vehicle: [rel_x, rel_y, rel_vx, rel_vy, heading, distance] (6 dims)
        - Max 10 vehicles → 4 + 3 + 10*6 = 67 dims total
    """
    
    # Observation dimensions
    EGO_STATE_DIM = 4  # vx, vy, heading, speed
    LANE_INFO_DIM = 3  # dist_to_left, dist_to_right, lane_width
    PER_VEHICLE_DIM = 6  # rel_x, rel_y, rel_vx, rel_vy, heading, distance
    MAX_VEHICLES = 10
    OBS_DIM = EGO_STATE_DIM + LANE_INFO_DIM + MAX_VEHICLES * PER_VEHICLE_DIM  # 67
    
    def __init__(self, 
                 data_dir: str,
                 num_scenarios: int = 100,
                 cost_function: Optional[CostFunction] = None,
                 surrounding_getter: Optional[SurroundingVehicleGetter] = None):
        """
        Initialize the collector.
        
        Args:
            data_dir: Path to ScenarioNet dataset directory
            num_scenarios: Number of scenarios to load
            cost_function: Optional cost function for CMDP
            surrounding_getter: Optional getter for surrounding vehicle info
        """
        self.data_dir = data_dir
        self.num_scenarios = num_scenarios
        self.cost_function = cost_function
        self.surrounding_getter = surrounding_getter or SurroundingVehicleGetter(
            max_vehicles=10, detection_radius=50.0
        )
        
        # Data storage
        self.observations = []
        self.actions = []
        self.rewards = []
        self.next_observations = []
        self.terminals = []
        self.costs = []
        self.episode_starts = []
        self.episode_infos = []
        
        # Statistics
        self.total_steps = 0
        self.total_episodes = 0
    
    def _create_env(self):
        """Create MetaDrive ScenarioEnv with ReplayEgoCarPolicy."""
        from metadrive.envs.scenario_env import ScenarioEnv
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy
        
        config = dict(
            agent_policy=ReplayEgoCarPolicy,
            use_render=False,
            data_directory=self.data_dir,
            num_scenarios=self.num_scenarios,
            no_traffic=False,
            reactive_traffic=False,
        )
        
        return ScenarioEnv(config)
    
    def _build_observation(self, env, surrounding_info: SurroundingVehicleInfo) -> np.ndarray:
        """
        Build observation from ego state, lane info, and surrounding vehicle info.
        
        Observation format (67 dims):
            - Ego: [vx, vy, heading, speed] (4 dims)
            - Lane: [dist_to_left, dist_to_right, lane_width] (3 dims)
            - Vehicle i: [rel_x, rel_y, rel_vx, rel_vy, heading, distance] (6 dims each)
            - Padded with zeros if fewer than MAX_VEHICLES
        
        Args:
            env: MetaDrive environment
            surrounding_info: Surrounding vehicle information
            
        Returns:
            Observation array of shape (67,)
        """
        obs = np.zeros(self.OBS_DIM, dtype=np.float32)
        
        # Ego state
        ego = env.agent
        ego_vel = np.array([ego.velocity[0], ego.velocity[1]]) if hasattr(ego, 'velocity') else np.zeros(2)
        ego_speed = np.linalg.norm(ego_vel)
        ego_heading = ego.heading_theta
        
        obs[0] = ego_vel[0]  # vx
        obs[1] = ego_vel[1]  # vy
        obs[2] = ego_heading  # heading
        obs[3] = ego_speed   # speed
        
        # Lane info
        dist_to_left = getattr(ego, 'dist_to_left_side', 0.0)
        dist_to_right = getattr(ego, 'dist_to_right_side', 0.0)
        lane_width = 0.0
        if hasattr(ego, 'navigation') and ego.navigation:
            try:
                lane_width = ego.navigation.get_current_lane_width()
            except:
                lane_width = dist_to_left + dist_to_right
        
        obs[4] = dist_to_left   # distance to left lane line
        obs[5] = dist_to_right  # distance to right lane line
        obs[6] = lane_width     # lane width
        
        # Surrounding vehicles
        base_offset = self.EGO_STATE_DIM + self.LANE_INFO_DIM
        n_vehicles = min(surrounding_info.num_vehicles, self.MAX_VEHICLES)
        for i in range(n_vehicles):
            base_idx = base_offset + i * self.PER_VEHICLE_DIM
            obs[base_idx + 0] = surrounding_info.relative_positions[i, 0]  # rel_x
            obs[base_idx + 1] = surrounding_info.relative_positions[i, 1]  # rel_y
            obs[base_idx + 2] = surrounding_info.relative_velocities[i, 0]  # rel_vx
            obs[base_idx + 3] = surrounding_info.relative_velocities[i, 1]  # rel_vy
            obs[base_idx + 4] = surrounding_info.headings[i]  # heading
            obs[base_idx + 5] = surrounding_info.distances[i]  # distance
        
        return obs
    
    def _get_expert_trajectory(self, env) -> Optional[np.ndarray]:
        """
        Extract expert trajectory action from the current scenario.
        
        Returns:
            Expert trajectory as flattened array (40 dims for 20 waypoints)
        """
        try:
            ego = env.agent
            
            # Get scenario track data
            scenario_id = env.engine.current_seed
            current_map = env.engine.current_map
            
            if hasattr(current_map, 'get_scenario') and hasattr(current_map, '_scenario'):
                scenario = current_map._scenario
            elif hasattr(env.engine, 'data_manager') and hasattr(env.engine.data_manager, 'current_scenario'):
                scenario = env.engine.data_manager.current_scenario
            else:
                return None
            
            # Get ego track
            ego_track_id = scenario.get('metadata', {}).get('sdc_id', 'sdc')
            tracks = scenario.get('tracks', {})
            
            if ego_track_id not in tracks:
                for tid, track in tracks.items():
                    if track.get('type', '') == 'VEHICLE' and 'sdc' in str(tid).lower():
                        ego_track_id = tid
                        break
            
            if ego_track_id not in tracks:
                return None
            
            ego_track = tracks[ego_track_id]
            track_states = ego_track.get('state', {})
            positions = track_states.get('position', None)
            
            if positions is None:
                return None
            
            # Get current step and future waypoints
            current_step = env.engine.episode_step
            future_steps = min(20, len(positions) - current_step - 1)
            
            if future_steps < 5:
                return None
            
            # Extract future positions
            future_positions = positions[current_step + 1:current_step + 21]
            
            # Current ego state
            ego_pos = np.array([ego.position[0], ego.position[1]])
            ego_heading = ego.heading_theta
            
            # Transform to local frame
            cos_h = np.cos(-ego_heading)
            sin_h = np.sin(-ego_heading)
            
            local_waypoints = []
            for pos in future_positions:
                rel_x = pos[0] - ego_pos[0]
                rel_y = pos[1] - ego_pos[1]
                local_x = rel_x * cos_h - rel_y * sin_h
                local_y = rel_x * sin_h + rel_y * cos_h
                local_waypoints.extend([local_x, local_y])
            
            # Pad if necessary
            while len(local_waypoints) < 40:
                if len(local_waypoints) >= 2:
                    local_waypoints.extend([local_waypoints[-2], local_waypoints[-1]])
                else:
                    local_waypoints.extend([0.0, 0.0])
            
            return np.array(local_waypoints[:40], dtype=np.float32)
            
        except Exception as e:
            return None
    
    def collect(self, verbose: bool = True) -> Dict[str, np.ndarray]:
        """
        Collect expert data from all scenarios.
        
        Args:
            verbose: Print progress information
            
        Returns:
            Dictionary containing collected dataset
        """
        env = self._create_env()
        
        try:
            for scenario_idx in range(self.num_scenarios):
                if verbose and scenario_idx % 10 == 0:
                    print(f"Collecting scenario {scenario_idx + 1}/{self.num_scenarios}...")
                
                # Reset to specific scenario
                _, info = env.reset(seed=scenario_idx)
                done = False
                episode_reward = 0.0
                episode_cost = 0.0
                step_count = 0
                
                # Mark episode start
                self.episode_starts.append(len(self.observations))
                
                # Get initial surrounding info and build observation
                surrounding_info = self.surrounding_getter.get_surrounding_vehicles(env)
                obs = self._build_observation(env, surrounding_info)
                
                while not done:
                    # Get expert trajectory action
                    action = self._get_expert_trajectory(env)
                    
                    if action is None:
                        # Use dummy action if expert trajectory unavailable
                        action = np.zeros(40, dtype=np.float32)
                    
                    # Get ego state for cost computation
                    ego_state = {
                        'position': np.array(env.agent.position[:2]),
                        'velocity': np.array([
                            env.agent.velocity[0] if hasattr(env.agent, 'velocity') else 0.0,
                            env.agent.velocity[1] if hasattr(env.agent, 'velocity') else 0.0
                        ]),
                        'heading': env.agent.heading_theta
                    }
                    
                    # Step environment
                    _, reward, terminated, truncated, info = env.step(np.array([0.0, 0.0]))
                    
                    # Use replay_done to determine when replay is complete
                    # This is better than terminated/truncated for replay scenarios
                    # because it ignores arrive_dest, crash, etc. and focuses on
                    # whether the recorded trajectory data has been fully replayed
                    done = info.get("replay_done", False) or (terminated or truncated)
                    
                    # Get next surrounding info and build next observation
                    next_surrounding_info = self.surrounding_getter.get_surrounding_vehicles(env)
                    next_obs = self._build_observation(env, next_surrounding_info)
                    
                    # Compute cost if cost function provided
                    cost = 0.0
                    if self.cost_function is not None:
                        # Build CostState from ego and surrounding info
                        velocity = ego_state['velocity']
                        speed = np.linalg.norm(velocity)
                        ego = EgoState(
                            position=ego_state['position'],
                            velocity=velocity,
                            heading=ego_state['heading'],
                            speed=speed,
                        )
                        # Convert SurroundingVehicleInfo arrays to SurroundingVehicle list
                        surrounding = []
                        for i in range(surrounding_info.num_vehicles):
                            surrounding.append(SurroundingVehicle(
                                position=surrounding_info.positions[i],
                                velocity=surrounding_info.velocities[i],
                                heading=surrounding_info.headings[i],
                                distance=surrounding_info.distances[i],
                            ))
                        cost_state = CostState(ego=ego, surrounding=surrounding, info=info)
                        result = self.cost_function.compute(cost_state)
                        cost = result.value
                    
                    # Store transition
                    self.observations.append(obs)
                    self.actions.append(action)
                    self.rewards.append(float(reward))
                    self.next_observations.append(next_obs)
                    self.terminals.append(done)
                    self.costs.append(float(cost))
                    
                    episode_reward += reward
                    episode_cost += cost
                    step_count += 1
                    
                    # Update for next iteration
                    obs = next_obs
                    surrounding_info = next_surrounding_info
                
                # Store episode info
                self.episode_infos.append({
                    'scenario_idx': scenario_idx,
                    'total_reward': episode_reward,
                    'total_cost': episode_cost,
                    'steps': step_count,
                    'terminal_reason': info.get('episode_end_reason', 'unknown'),
                })
                
                self.total_episodes += 1
                self.total_steps += step_count
                
                if verbose:
                    print(f"  Scenario {scenario_idx}: {step_count} steps, "
                          f"reward={episode_reward:.2f}, cost={episode_cost:.2f}, "
                          f"reason={info.get('arrive_dest', False) and 'arrive_dest' or 'other'}")
        
        finally:
            env.close()
        
        # Convert to arrays
        dataset = {
            'observations': np.array(self.observations, dtype=np.float32),
            'actions': np.array(self.actions, dtype=np.float32),
            'rewards': np.array(self.rewards, dtype=np.float32),
            'next_observations': np.array(self.next_observations, dtype=np.float32),
            'terminals': np.array(self.terminals, dtype=np.bool_),
            'episode_starts': np.array(self.episode_starts, dtype=np.int64),
        }
        
        if self.cost_function is not None:
            dataset['costs'] = np.array(self.costs, dtype=np.float32)
        
        return dataset
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get collection statistics."""
        rewards = np.array(self.rewards) if self.rewards else np.array([0])
        costs = np.array(self.costs) if self.costs else np.array([0])
        
        return {
            'total_episodes': self.total_episodes,
            'total_steps': self.total_steps,
            'avg_episode_length': self.total_steps / max(1, self.total_episodes),
            'avg_reward': float(np.mean(rewards)),
            'std_reward': float(np.std(rewards)),
            'avg_cost': float(np.mean(costs)),
            'total_cost': float(np.sum(costs)),
        }


def create_cost_function(cost_type: str) -> Optional[CostFunction]:
    """Create cost function based on type string."""
    if cost_type == 'none' or cost_type is None:
        return None
    elif cost_type == 'collision':
        return CollisionCost()
    elif cost_type == 'ttc':
        return TTCCost(threshold=3.0)
    else:
        print(f"Warning: Unknown cost type '{cost_type}', using none")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Collect offline RL dataset from ScenarioNet data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Collect MDP dataset (reward only)
  python -m meta_qa.examples.collect_offline_data \\
      --data_dir ~/scenarionet/exp_data/exp_nuplan_converted \\
      --num_scenarios 100

  # Collect CMDP dataset with TTC cost
  python -m meta_qa.examples.collect_offline_data \\
      --data_dir ~/scenarionet/exp_data/exp_nuplan_converted \\
      --num_scenarios 100 \\
      --cost_type ttc

  # Specify output file
  python -m meta_qa.examples.collect_offline_data \\
      --data_dir ~/scenarionet/exp_data/exp_nuplan_converted \\
      --num_scenarios 100 \\
      --output my_dataset.h5
"""
    )
    
    parser.add_argument(
        '--data_dir', '-d',
        type=str,
        default=DEFAULT_DATA_DIR,
        help='Path to ScenarioNet dataset directory'
    )
    parser.add_argument(
        '--num_scenarios', '-n',
        type=int,
        default=100,
        help='Number of scenarios to collect (each scenario runs once)'
    )
    parser.add_argument(
        '--cost_type', '-c',
        type=str,
        choices=['none', 'collision', 'distance', 'ttc'],
        default='none',
        help='Cost function type for CMDP (none for MDP)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output file path (default: auto-generated based on settings)'
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['hdf5', 'npz'],
        default='hdf5',
        help='Output format (hdf5 recommended)'
    )
    
    args = parser.parse_args()
    
    # Default output directory
    default_output_dir = os.path.join(project_root, 'outputs', 'offline_data')
    
    # Determine output path
    if args.output is None:
        # Auto-generate filename
        data_type = 'cmdp' if args.cost_type != 'none' else 'mdp'
        cost_suffix = f'_{args.cost_type}' if args.cost_type != 'none' else ''
        ext = '.h5' if args.format == 'hdf5' else '.npz'
        filename = f'expert_{data_type}_{args.num_scenarios}scenarios{cost_suffix}{ext}'
        os.makedirs(default_output_dir, exist_ok=True)
        output_path = os.path.join(default_output_dir, filename)
    else:
        # User specified output path
        if os.path.isabs(args.output):
            output_path = args.output
        else:
            # Relative path - use as filename in default directory
            os.makedirs(default_output_dir, exist_ok=True)
            output_path = os.path.join(default_output_dir, os.path.basename(args.output))
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else default_output_dir, exist_ok=True)
    
    # Print configuration
    print("=" * 60)
    print("Offline RL Dataset Collection")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Num scenarios: {args.num_scenarios}")
    print(f"Cost type: {args.cost_type}")
    print(f"Format: {args.format}")
    print(f"Output: {output_path}")
    print("=" * 60)
    
    # Create cost function
    cost_function = create_cost_function(args.cost_type)
    
    # Create collector and collect data
    collector = ExpertDataCollector(
        data_dir=args.data_dir,
        num_scenarios=args.num_scenarios,
        cost_function=cost_function,
    )
    
    print("\nCollecting data...")
    dataset = collector.collect(verbose=True)
    
    # Get statistics
    stats = collector.get_statistics()
    print("\n" + "=" * 60)
    print("Collection Statistics")
    print("=" * 60)
    print(f"Total episodes: {stats['total_episodes']}")
    print(f"Total steps: {stats['total_steps']}")
    print(f"Avg episode length: {stats['avg_episode_length']:.2f}")
    print(f"Avg reward: {stats['avg_reward']:.4f}")
    if args.cost_type != 'none':
        print(f"Avg cost: {stats['avg_cost']:.4f}")
        print(f"Total cost: {stats['total_cost']:.4f}")
    
    # Save dataset
    print("\nSaving dataset...")
    
    metadata = {
        'data_source': os.path.basename(args.data_dir),
        'num_scenarios': args.num_scenarios,
        'cost_type': args.cost_type,
        'is_cmdp': args.cost_type != 'none',
        **stats
    }
    
    if args.format == 'hdf5':
        save_dataset_hdf5(dataset, output_path, metadata)
    else:
        np.savez_compressed(output_path, **dataset)
        print(f"Dataset saved to {output_path}")
    
    # Print dataset info
    print("\n" + "=" * 60)
    print("Dataset Information")
    print("=" * 60)
    for key, value in dataset.items():
        if isinstance(value, np.ndarray):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
    
    print("\nDone!")


if __name__ == '__main__':
    main()
