"""
Trajectory-based Environment Wrapper for MetaDrive.

This module wraps the ScenarioEnv to support trajectory-based actions
where the agent outputs future waypoints instead of direct control signals.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
import gymnasium as gym
from gymnasium import spaces

from meta_qa.core.config import (
    NUM_WAYPOINTS, TRAJECTORY_DT, MAX_LATERAL_OFFSET, 
    MAX_LONGITUDINAL_OFFSET, TRAJECTORY_HORIZON
)
from meta_qa.core.trajectory_tracker import TrajectoryTracker, VehicleState
from meta_qa.core.action_space import (
    TrajectoryActionSpace, TrajectoryActionSpaceNormalized,
    world_to_local_trajectory, local_to_world_trajectory,
    extract_trajectory_from_scenario
)


class TrajectoryEnv(gym.Wrapper):
    """
    Wrapper that converts MetaDrive's control-based action space
    to a trajectory-based action space.
    
    The agent outputs a sequence of future waypoints (2-second horizon),
    and a trajectory tracking controller converts them to control signals.
    """
    
    def __init__(self, env, 
                 num_waypoints: int = NUM_WAYPOINTS,
                 use_normalized_action: bool = True,
                 include_trajectory_in_obs: bool = True,
                 visualize_trajectory: bool = True,
                 use_ground_truth_trajectory: bool = False):
        """
        Initialize trajectory environment wrapper.
        
        Args:
            env: MetaDrive ScenarioEnv instance
            num_waypoints: Number of waypoints in trajectory action
            use_normalized_action: Whether to use normalized [-1,1] actions
            include_trajectory_in_obs: Whether to include GT trajectory in observation
            visualize_trajectory: Whether to visualize trajectories
            use_ground_truth_trajectory: Whether to use GT trajectory for training
        """
        super().__init__(env)
        
        self.num_waypoints = num_waypoints
        self.use_normalized_action = use_normalized_action
        self.include_trajectory_in_obs = include_trajectory_in_obs
        self.visualize_trajectory = visualize_trajectory
        self.use_ground_truth_trajectory = use_ground_truth_trajectory
        
        # Create action space
        if use_normalized_action:
            self.trajectory_action_space = TrajectoryActionSpaceNormalized(
                num_waypoints=num_waypoints,
                max_lateral=MAX_LATERAL_OFFSET,
                max_longitudinal=MAX_LONGITUDINAL_OFFSET
            )
        else:
            self.trajectory_action_space = TrajectoryActionSpace(
                num_waypoints=num_waypoints,
                max_lateral=MAX_LATERAL_OFFSET,
                max_longitudinal=MAX_LONGITUDINAL_OFFSET,
                output_format="flat"
            )
        
        # Override action space
        self.action_space = self.trajectory_action_space.get_gym_space()
        
        # Trajectory tracking controller (using MetaDrive's official PIDController)
        self.controller = TrajectoryTracker()
        
        # Visualizer (lazy import to avoid circular dependency)
        if visualize_trajectory:
            from meta_qa.tools.trajectory_vis import TrajectoryVisualizer
            self.visualizer = TrajectoryVisualizer()
        else:
            self.visualizer = None
        
        # State tracking
        self.current_trajectory = None  # Current predicted trajectory (local frame)
        self.current_trajectory_world = None  # Current trajectory in world frame
        self.gt_trajectory = None  # Ground truth trajectory
        self.trajectory_history = []  # History of executed trajectories
        self.current_step = 0
        
        # For sub-stepping (execute trajectory step by step)
        self.trajectory_step_idx = 0
        self.execute_every_n_steps = 1  # Execute new trajectory every N environment steps
        
        # Modify observation space if including trajectory
        if include_trajectory_in_obs:
            self._setup_observation_space()
    
    def _setup_observation_space(self):
        """Set up observation space with trajectory information."""
        original_obs_space = self.env.observation_space
        
        # Add future trajectory to observation
        # Trajectory: (num_waypoints, 2) flattened
        trajectory_dim = self.num_waypoints * 2
        
        if isinstance(original_obs_space, spaces.Box):
            # Extend the observation with trajectory
            new_low = np.concatenate([
                original_obs_space.low.flatten(),
                np.full(trajectory_dim, -100.0)  # Trajectory coordinates
            ])
            new_high = np.concatenate([
                original_obs_space.high.flatten(),
                np.full(trajectory_dim, 100.0)
            ])
            self.observation_space = spaces.Box(
                low=new_low, high=new_high, dtype=np.float32
            )
        elif isinstance(original_obs_space, spaces.Dict):
            # Add trajectory as a new key
            new_spaces = dict(original_obs_space.spaces)
            new_spaces['future_trajectory'] = spaces.Box(
                low=-100.0, high=100.0,
                shape=(self.num_waypoints, 2),
                dtype=np.float32
            )
            self.observation_space = spaces.Dict(new_spaces)
    
    def _get_vehicle_state(self) -> Tuple[np.ndarray, float, np.ndarray]:
        """
        Get current vehicle state from environment.
        
        Returns:
            Tuple of (position, heading, velocity)
        """
        try:
            vehicle = self.env.agent
            
            # Get position (x, y)
            position = np.array([vehicle.position[0], vehicle.position[1]])
            
            # Get heading
            heading = vehicle.heading_theta
            
            # Get velocity
            velocity = np.array([vehicle.velocity[0], vehicle.velocity[1]])
            
            return position, heading, velocity
        except Exception as e:
            print(f"Error getting vehicle state: {e}")
            return np.zeros(2), 0.0, np.zeros(2)
    
    def _get_ground_truth_trajectory(self) -> Optional[np.ndarray]:
        """
        Get ground truth future trajectory from scenario data.
        
        Returns:
            Trajectory in local frame (num_waypoints, 2) or None
        """
        try:
            # Access scenario data
            if hasattr(self.env, 'engine') and hasattr(self.env.engine, 'data_manager'):
                scenario = self.env.engine.data_manager.current_scenario
                if scenario is not None:
                    # Get ego trajectory
                    world_traj = extract_trajectory_from_scenario(
                        scenario, 'ego', self.current_step, self.num_waypoints
                    )
                    
                    if world_traj is not None:
                        position, heading, _ = self._get_vehicle_state()
                        local_traj = world_to_local_trajectory(world_traj, position, heading)
                        return local_traj
        except Exception as e:
            print(f"Error getting ground truth trajectory: {e}")
        
        return None
    
    def _augment_observation(self, obs: Any) -> Any:
        """
        Augment observation with trajectory information.
        
        Args:
            obs: Original observation
            
        Returns:
            Augmented observation
        """
        if not self.include_trajectory_in_obs:
            return obs
        
        # Get ground truth trajectory
        gt_traj = self._get_ground_truth_trajectory()
        
        if gt_traj is None:
            gt_traj = np.zeros((self.num_waypoints, 2), dtype=np.float32)
        else:
            # Pad if needed
            if len(gt_traj) < self.num_waypoints:
                padding = np.zeros((self.num_waypoints - len(gt_traj), 2))
                if len(gt_traj) > 0:
                    padding[:] = gt_traj[-1]
                gt_traj = np.vstack([gt_traj, padding])
        
        self.gt_trajectory = gt_traj
        
        if isinstance(obs, dict):
            obs['future_trajectory'] = gt_traj.astype(np.float32)
        else:
            # Flatten and concatenate
            obs_flat = obs.flatten() if isinstance(obs, np.ndarray) else np.array(obs).flatten()
            obs = np.concatenate([obs_flat, gt_traj.flatten()]).astype(np.float32)
        
        return obs
    
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset(**kwargs)
        
        # Reset state
        self.controller.reset()
        self.current_trajectory = None
        self.current_trajectory_world = None
        self.gt_trajectory = None
        self.trajectory_history = []
        self.current_step = 0
        self.trajectory_step_idx = 0
        
        # Augment observation
        obs = self._augment_observation(obs)
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute one environment step with trajectory action.
        
        Args:
            action: Trajectory action (normalized or unnormalized)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Convert action to trajectory
        if self.use_normalized_action:
            trajectory = self.trajectory_action_space.to_trajectory(action)
        else:
            trajectory = action
        
        # Reshape to (num_waypoints, 2)
        trajectory_2d = trajectory.reshape(-1, 2)
        self.current_trajectory = trajectory_2d
        
        # Get current vehicle state
        position, heading, velocity = self._get_vehicle_state()
        
        # Convert trajectory to world frame for visualization
        self.current_trajectory_world = local_to_world_trajectory(
            trajectory_2d, position, heading
        )
        
        # Compute control signals using trajectory tracker
        # TrajectoryTracker expects world frame trajectory and VehicleState
        vehicle_state = VehicleState(
            position=position,
            heading=heading,
            speed=np.linalg.norm(velocity),
            velocity=velocity,
        )
        steering, throttle = self.controller.track(
            self.current_trajectory_world, vehicle_state
        )
        
        # Create MetaDrive action [steering, throttle]
        metadrive_action = np.array([steering, throttle], dtype=np.float32)
        
        # Execute action in environment
        obs, reward, terminated, truncated, info = self.env.step(metadrive_action)
        
        # Update step counter
        self.current_step += 1
        
        # Store trajectory history
        self.trajectory_history.append({
            'step': self.current_step,
            'trajectory': self.current_trajectory_world.copy(),
            'position': position.copy(),
            'control': metadrive_action.copy()
        })
        
        # Augment observation
        obs = self._augment_observation(obs)
        
        # Add trajectory info to info dict
        info['trajectory_action'] = trajectory_2d
        info['trajectory_world'] = self.current_trajectory_world
        info['control_action'] = metadrive_action
        
        if self.gt_trajectory is not None:
            info['ground_truth_trajectory'] = self.gt_trajectory
        
        return obs, reward, terminated, truncated, info
    
    def render(self, *args, **kwargs):
        """Render with trajectory visualization."""
        # Get base render
        frame = self.env.render(*args, **kwargs)
        
        # Add trajectory visualization if enabled
        if self.visualize_trajectory and self.visualizer is not None:
            frame = self.visualizer.draw_on_frame(
                frame,
                predicted_trajectory=self.current_trajectory_world,
                ground_truth_trajectory=self._get_world_gt_trajectory(),
                vehicle_position=self._get_vehicle_state()[0],
                vehicle_heading=self._get_vehicle_state()[1]
            )
        
        return frame
    
    def _get_world_gt_trajectory(self) -> Optional[np.ndarray]:
        """Get ground truth trajectory in world frame."""
        if self.gt_trajectory is None:
            return None
        
        position, heading, _ = self._get_vehicle_state()
        return local_to_world_trajectory(self.gt_trajectory, position, heading)
    
    def get_trajectory_from_observation(self, obs: Any) -> Optional[np.ndarray]:
        """
        Extract trajectory from augmented observation.
        
        Args:
            obs: Augmented observation
            
        Returns:
            Trajectory (num_waypoints, 2) or None
        """
        if isinstance(obs, dict) and 'future_trajectory' in obs:
            return obs['future_trajectory']
        elif isinstance(obs, np.ndarray):
            traj_dim = self.num_waypoints * 2
            if len(obs) >= traj_dim:
                return obs[-traj_dim:].reshape(-1, 2)
        return None


class TrajectoryReplayEnv(TrajectoryEnv):
    """
    Environment for replaying scenarios with trajectory visualization.
    
    In this mode, the agent follows the ground truth trajectory,
    which is useful for data collection and visualization.
    """
    
    def __init__(self, env, **kwargs):
        kwargs['use_ground_truth_trajectory'] = True
        super().__init__(env, **kwargs)
    
    def step(self, action: Optional[np.ndarray] = None) -> Tuple[Any, float, bool, bool, Dict]:
        """
        Execute one step using ground truth trajectory.
        
        Args:
            action: Optional action (ignored if using GT trajectory)
            
        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        # Get ground truth trajectory
        gt_traj = self._get_ground_truth_trajectory()
        
        if gt_traj is not None and self.use_ground_truth_trajectory:
            # Use GT trajectory as action
            if self.use_normalized_action:
                # Normalize GT trajectory
                flat_traj = gt_traj.flatten()
                # Pad if needed
                if len(flat_traj) < self.num_waypoints * 2:
                    flat_traj = np.pad(flat_traj, (0, self.num_waypoints * 2 - len(flat_traj)))
                action = self.trajectory_action_space.from_trajectory(flat_traj)
            else:
                action = gt_traj.flatten()
        
        if action is None:
            # Default to straight trajectory
            if self.use_normalized_action:
                action = np.zeros(self.num_waypoints * 2, dtype=np.float32)
            else:
                action = self.trajectory_action_space.underlying.sample_straight()
        
        return super().step(action)


def create_trajectory_env(data_directory: str,
                         num_scenarios: int = 10,
                         use_render: bool = False,
                         render_mode: str = "top_down",
                         **kwargs) -> TrajectoryEnv:
    """
    Create a trajectory-based environment.
    
    Args:
        data_directory: Path to ScenarioNet dataset
        num_scenarios: Number of scenarios to load
        use_render: Whether to enable 3D rendering
        render_mode: Rendering mode ("top_down", "3D", etc.)
        **kwargs: Additional arguments for TrajectoryEnv
        
    Returns:
        TrajectoryEnv instance
    """
    from metadrive.envs.scenario_env import ScenarioEnv
    
    env_config = {
        "manual_control": False,
        "use_render": use_render,
        "data_directory": data_directory,
        "num_scenarios": num_scenarios,
        "reactive_traffic": False,
        "horizon": 1000,
        "vehicle_config": {
            "show_navi_mark": True,
            "no_wheel_friction": False,
        }
    }
    
    base_env = ScenarioEnv(env_config)
    return TrajectoryEnv(base_env, **kwargs)


def create_trajectory_replay_env(data_directory: str,
                                num_scenarios: int = 10,
                                use_render: bool = True,
                                **kwargs) -> TrajectoryReplayEnv:
    """
    Create a trajectory replay environment for visualization.
    
    Args:
        data_directory: Path to ScenarioNet dataset
        num_scenarios: Number of scenarios to load
        use_render: Whether to enable rendering
        **kwargs: Additional arguments for TrajectoryReplayEnv
        
    Returns:
        TrajectoryReplayEnv instance
    """
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    
    env_config = {
        "manual_control": False,
        "use_render": use_render,
        "agent_policy": ReplayEgoCarPolicy,
        "data_directory": data_directory,
        "num_scenarios": num_scenarios,
        "reactive_traffic": False,
        "horizon": 1000,
    }
    
    base_env = ScenarioEnv(env_config)
    return TrajectoryReplayEnv(base_env, **kwargs)
