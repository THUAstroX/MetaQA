"""
Trajectory Action Space for MetaDrive RL.

This module defines the action space where the agent outputs
future trajectory waypoints instead of direct control signals.
"""

import numpy as np
from gymnasium import spaces
from typing import Tuple, Optional
from meta_qa.core.config import (
    NUM_WAYPOINTS, TRAJECTORY_DT, TRAJECTORY_HORIZON,
    MAX_LATERAL_OFFSET, MAX_LONGITUDINAL_OFFSET
)


class TrajectoryActionSpace:
    """
    Custom action space for trajectory-based RL.
    
    The agent outputs a sequence of future waypoints (2-second horizon by default)
    as [x1, y1, x2, y2, ..., xN, yN] in the vehicle's local frame.
    
    Local frame convention:
    - x: forward direction (positive = forward)
    - y: lateral direction (positive = left)
    """
    
    def __init__(self, num_waypoints: int = NUM_WAYPOINTS,
                 max_lateral: float = MAX_LATERAL_OFFSET,
                 max_longitudinal: float = MAX_LONGITUDINAL_OFFSET,
                 output_format: str = "flat"):
        """
        Initialize trajectory action space.
        
        Args:
            num_waypoints: Number of waypoints in the trajectory
            max_lateral: Maximum lateral offset in meters
            max_longitudinal: Maximum longitudinal offset in meters
            output_format: "flat" for [x1,y1,x2,y2,...] or "2d" for [[x1,y1],[x2,y2],...]
        """
        self.num_waypoints = num_waypoints
        self.max_lateral = max_lateral
        self.max_longitudinal = max_longitudinal
        self.output_format = output_format
        self.dt = TRAJECTORY_DT
        self.horizon = TRAJECTORY_HORIZON
        
        # Create gymnasium Box space
        if output_format == "flat":
            # Flat array: [x1, y1, x2, y2, ...]
            low = np.zeros(num_waypoints * 2)
            high = np.zeros(num_waypoints * 2)
            
            for i in range(num_waypoints):
                # x (longitudinal) - always positive (forward)
                low[i * 2] = 0.0
                high[i * 2] = max_longitudinal * (i + 1)
                
                # y (lateral) - can be positive or negative
                low[i * 2 + 1] = -max_lateral
                high[i * 2 + 1] = max_lateral
            
            self.space = spaces.Box(low=low, high=high, dtype=np.float32)
        else:
            # 2D array: [[x1, y1], [x2, y2], ...]
            low = np.zeros((num_waypoints, 2))
            high = np.zeros((num_waypoints, 2))
            
            for i in range(num_waypoints):
                low[i, 0] = 0.0
                high[i, 0] = max_longitudinal * (i + 1)
                low[i, 1] = -max_lateral
                high[i, 1] = max_lateral
            
            self.space = spaces.Box(low=low, high=high, dtype=np.float32)
    
    def get_gym_space(self) -> spaces.Box:
        """Return the gymnasium Box space."""
        return self.space
    
    def sample(self) -> np.ndarray:
        """Sample a random trajectory from the action space."""
        return self.space.sample()
    
    def sample_straight(self, speed: float = 10.0) -> np.ndarray:
        """
        Sample a straight trajectory at given speed.
        
        Args:
            speed: Forward speed in m/s
            
        Returns:
            Trajectory array
        """
        if self.output_format == "flat":
            trajectory = np.zeros(self.num_waypoints * 2)
            for i in range(self.num_waypoints):
                trajectory[i * 2] = speed * self.dt * (i + 1)  # x
                trajectory[i * 2 + 1] = 0.0  # y
        else:
            trajectory = np.zeros((self.num_waypoints, 2))
            for i in range(self.num_waypoints):
                trajectory[i, 0] = speed * self.dt * (i + 1)
                trajectory[i, 1] = 0.0
        
        return trajectory.astype(np.float32)
    
    def sample_curve(self, speed: float = 10.0, curvature: float = 0.1) -> np.ndarray:
        """
        Sample a curved trajectory.
        
        Args:
            speed: Forward speed in m/s
            curvature: Curvature (1/radius), positive = left turn
            
        Returns:
            Trajectory array
        """
        if abs(curvature) < 1e-6:
            return self.sample_straight(speed)
        
        radius = 1.0 / curvature
        
        if self.output_format == "flat":
            trajectory = np.zeros(self.num_waypoints * 2)
            for i in range(self.num_waypoints):
                arc_length = speed * self.dt * (i + 1)
                angle = arc_length / abs(radius)
                
                # Position on arc
                x = abs(radius) * np.sin(angle)
                y = np.sign(curvature) * radius * (1 - np.cos(angle))
                
                trajectory[i * 2] = x
                trajectory[i * 2 + 1] = y
        else:
            trajectory = np.zeros((self.num_waypoints, 2))
            for i in range(self.num_waypoints):
                arc_length = speed * self.dt * (i + 1)
                angle = arc_length / abs(radius)
                
                x = abs(radius) * np.sin(angle)
                y = np.sign(curvature) * radius * (1 - np.cos(angle))
                
                trajectory[i, 0] = x
                trajectory[i, 1] = y
        
        return np.clip(trajectory, self.space.low, self.space.high).astype(np.float32)
    
    def to_2d(self, action: np.ndarray) -> np.ndarray:
        """
        Convert flat action to 2D array.
        
        Args:
            action: Flat action array [x1,y1,x2,y2,...]
            
        Returns:
            2D array [[x1,y1],[x2,y2],...]
        """
        if action.ndim == 2:
            return action
        return action.reshape(-1, 2)
    
    def to_flat(self, action: np.ndarray) -> np.ndarray:
        """
        Convert 2D action to flat array.
        
        Args:
            action: 2D array [[x1,y1],[x2,y2],...]
            
        Returns:
            Flat array [x1,y1,x2,y2,...]
        """
        if action.ndim == 1:
            return action
        return action.flatten()
    
    def normalize(self, action: np.ndarray) -> np.ndarray:
        """
        Normalize action to [-1, 1] range.
        
        Args:
            action: Raw action in original space
            
        Returns:
            Normalized action
        """
        low = self.space.low
        high = self.space.high
        
        # Avoid division by zero
        range_vals = high - low
        range_vals = np.where(range_vals < 1e-6, 1.0, range_vals)
        
        return 2.0 * (action - low) / range_vals - 1.0
    
    def denormalize(self, action: np.ndarray) -> np.ndarray:
        """
        Denormalize action from [-1, 1] to original space.
        
        Args:
            action: Normalized action in [-1, 1]
            
        Returns:
            Action in original space
        """
        low = self.space.low
        high = self.space.high
        
        return low + (action + 1.0) * 0.5 * (high - low)
    
    def contains(self, action: np.ndarray) -> bool:
        """Check if action is within the space bounds."""
        return self.space.contains(action)
    
    def clip(self, action: np.ndarray) -> np.ndarray:
        """Clip action to space bounds."""
        return np.clip(action, self.space.low, self.space.high)


class TrajectoryActionSpaceNormalized:
    """
    Normalized trajectory action space where actions are in [-1, 1].
    
    This is more suitable for neural network outputs.
    """
    
    def __init__(self, num_waypoints: int = NUM_WAYPOINTS,
                 max_lateral: float = MAX_LATERAL_OFFSET,
                 max_longitudinal: float = MAX_LONGITUDINAL_OFFSET):
        """
        Initialize normalized trajectory action space.
        
        Args:
            num_waypoints: Number of waypoints
            max_lateral: Maximum lateral offset
            max_longitudinal: Maximum longitudinal offset
        """
        self.num_waypoints = num_waypoints
        self.max_lateral = max_lateral
        self.max_longitudinal = max_longitudinal
        
        # The underlying unnormalized space
        self.underlying = TrajectoryActionSpace(
            num_waypoints=num_waypoints,
            max_lateral=max_lateral,
            max_longitudinal=max_longitudinal,
            output_format="flat"
        )
        
        # Normalized space: all in [-1, 1]
        self.space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(num_waypoints * 2,),
            dtype=np.float32
        )
    
    def get_gym_space(self) -> spaces.Box:
        """Return the normalized gymnasium Box space."""
        return self.space
    
    def sample(self) -> np.ndarray:
        """Sample a random normalized action."""
        return self.space.sample()
    
    def to_trajectory(self, normalized_action: np.ndarray) -> np.ndarray:
        """
        Convert normalized action to trajectory.
        
        Args:
            normalized_action: Action in [-1, 1]
            
        Returns:
            Trajectory in original space
        """
        return self.underlying.denormalize(normalized_action)
    
    def from_trajectory(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Convert trajectory to normalized action.
        
        Args:
            trajectory: Trajectory in original space
            
        Returns:
            Normalized action in [-1, 1]
        """
        return self.underlying.normalize(trajectory)


def extract_trajectory_from_scenario(scenario_data: dict, 
                                    vehicle_id: str,
                                    current_step: int,
                                    num_waypoints: int = NUM_WAYPOINTS) -> Tuple[Optional[np.ndarray], bool]:
    """
    Extract future trajectory from scenario data with truncation handling.
    
    Args:
        scenario_data: Scenario description dict
        vehicle_id: ID of the vehicle (usually 'ego')
        current_step: Current time step
        num_waypoints: Number of future waypoints to extract
        
    Returns:
        trajectory: Trajectory array (N, 2) in world frame, or None if not available
        is_complete: True if trajectory has full num_waypoints, False if truncated
    """
    try:
        tracks = scenario_data.get('tracks', {})
        vehicle = tracks.get(vehicle_id, None)
        
        if vehicle is None:
            return None, False
        
        state = vehicle.get('state', {})
        positions = state.get('position', None)
        valid = state.get('valid', None)
        
        if positions is None:
            return None, False
        
        # Extract future positions
        trajectory = []
        for i in range(current_step + 1, min(current_step + 1 + num_waypoints, len(positions))):
            if valid is not None and not valid[i]:
                break
            trajectory.append(positions[i][:2])  # Only x, y
        
        if len(trajectory) < 2:
            return None, False
        
        # Check if trajectory is complete (full 2 seconds)
        is_complete = len(trajectory) >= num_waypoints
        
        return np.array(trajectory, dtype=np.float32), is_complete
    
    except Exception as e:
        print(f"Error extracting trajectory: {e}")
        return None, False


def pad_trajectory(trajectory: np.ndarray, 
                   num_waypoints: int = NUM_WAYPOINTS,
                   pad_mode: str = "last") -> np.ndarray:
    """
    Pad truncated trajectory to full length.
    
    Args:
        trajectory: Truncated trajectory (M, 2) where M < num_waypoints
        num_waypoints: Target number of waypoints
        pad_mode: Padding mode:
            - "last": Repeat last position
            - "extrapolate": Linear extrapolation from last two points
            - "zero": Zero padding (not recommended)
            
    Returns:
        Padded trajectory (num_waypoints, 2)
    """
    if trajectory is None or len(trajectory) == 0:
        return np.zeros((num_waypoints, 2), dtype=np.float32)
    
    current_len = len(trajectory)
    
    if current_len >= num_waypoints:
        return trajectory[:num_waypoints]
    
    # Need to pad
    padded = np.zeros((num_waypoints, 2), dtype=np.float32)
    padded[:current_len] = trajectory
    
    if pad_mode == "last":
        # Repeat last position
        padded[current_len:] = trajectory[-1]
        
    elif pad_mode == "extrapolate":
        # Linear extrapolation
        if current_len >= 2:
            direction = trajectory[-1] - trajectory[-2]
            for i in range(current_len, num_waypoints):
                padded[i] = trajectory[-1] + direction * (i - current_len + 1)
        else:
            padded[current_len:] = trajectory[-1]
            
    elif pad_mode == "zero":
        pass  # Already zero-filled
    
    return padded


def create_trajectory_mask(trajectory_length: int, 
                           num_waypoints: int = NUM_WAYPOINTS) -> np.ndarray:
    """
    Create a mask indicating valid waypoints in a trajectory.
    
    Args:
        trajectory_length: Actual length of trajectory (before padding)
        num_waypoints: Total number of waypoints
        
    Returns:
        Boolean mask (num_waypoints,) where True = valid waypoint
    """
    mask = np.zeros(num_waypoints, dtype=bool)
    mask[:min(trajectory_length, num_waypoints)] = True
    return mask


def world_to_local_trajectory(trajectory: np.ndarray,
                             position: np.ndarray,
                             heading: float) -> np.ndarray:
    """
    Convert trajectory from world frame to local (vehicle) frame.
    
    Args:
        trajectory: Trajectory in world frame (N, 2)
        position: Vehicle position in world frame (2,)
        heading: Vehicle heading in radians
        
    Returns:
        Trajectory in local frame (N, 2)
    """
    # Translate to vehicle origin
    translated = trajectory - position
    
    # Rotate to vehicle frame
    cos_h = np.cos(-heading)
    sin_h = np.sin(-heading)
    
    rotation = np.array([[cos_h, -sin_h],
                        [sin_h, cos_h]])
    
    local_trajectory = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        local_trajectory[i] = rotation @ translated[i]
    
    return local_trajectory


def local_to_world_trajectory(trajectory: np.ndarray,
                             position: np.ndarray,
                             heading: float) -> np.ndarray:
    """
    Convert trajectory from local (vehicle) frame to world frame.
    
    Args:
        trajectory: Trajectory in local frame (N, 2)
        position: Vehicle position in world frame (2,)
        heading: Vehicle heading in radians
        
    Returns:
        Trajectory in world frame (N, 2)
    """
    # Rotate to world frame
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    
    rotation = np.array([[cos_h, -sin_h],
                        [sin_h, cos_h]])
    
    world_trajectory = np.zeros_like(trajectory)
    for i in range(len(trajectory)):
        world_trajectory[i] = rotation @ trajectory[i] + position
    
    return world_trajectory
