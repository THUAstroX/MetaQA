"""
Trajectory Interpolation Module.

This module provides interpolation utilities for ScenarioNet trajectory data
to match the 12Hz frame rate of NuScenes raw images.

Key features:
- Linear interpolation for position, velocity
- Spherical linear interpolation (SLERP) for rotation/heading
- Supports both ego vehicle and surrounding vehicles
- Maintains consistency with ScenarioNet data structure
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import pickle


@dataclass
class InterpolatedState:
    """
    Interpolated vehicle state at a specific timestamp.
    
    Attributes:
        timestamp: Interpolated timestamp
        position: (x, y) world coordinates
        velocity: (vx, vy) world velocities
        heading: Heading angle in radians
        length: Vehicle length
        width: Vehicle width
        is_keyframe: Whether this state is at a keyframe
        keyframe_index: Index of the keyframe this state belongs to
        interpolation_ratio: Position between keyframes [0, 1]
    """
    timestamp: float
    position: np.ndarray  # (2,) or (3,)
    velocity: np.ndarray  # (2,)
    heading: float
    length: float = 4.5
    width: float = 1.8
    is_keyframe: bool = False
    keyframe_index: int = -1
    interpolation_ratio: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "position": self.position.tolist(),
            "velocity": self.velocity.tolist(),
            "heading": float(self.heading),
            "length": self.length,
            "width": self.width,
            "is_keyframe": self.is_keyframe,
            "keyframe_index": self.keyframe_index,
            "interpolation_ratio": self.interpolation_ratio,
        }


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi]."""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def interpolate_angle(angle1: float, angle2: float, ratio: float) -> float:
    """
    Interpolate between two angles using shortest path.
    
    Args:
        angle1: Start angle in radians
        angle2: End angle in radians
        ratio: Interpolation ratio [0, 1]
        
    Returns:
        Interpolated angle in radians
    """
    # Normalize angles
    angle1 = normalize_angle(angle1)
    angle2 = normalize_angle(angle2)
    
    # Find shortest path
    diff = angle2 - angle1
    if diff > np.pi:
        diff -= 2 * np.pi
    elif diff < -np.pi:
        diff += 2 * np.pi
    
    result = angle1 + ratio * diff
    return normalize_angle(result)


def linear_interpolate(
    v1: np.ndarray,
    v2: np.ndarray,
    ratio: float
) -> np.ndarray:
    """Linear interpolation between two vectors."""
    return v1 + ratio * (v2 - v1)


class TrajectoryInterpolator:
    """
    Interpolates ScenarioNet trajectory data to higher frame rates.
    
    This class takes trajectory data from ScenarioNet (typically at 10Hz)
    and interpolates it to match NuScenes 12Hz frame timestamps.
    """
    
    def __init__(
        self,
        scenario_path: str,
        target_fps: float = 12.0,
    ):
        """
        Initialize the interpolator.
        
        Args:
            scenario_path: Path to ScenarioNet scenario .pkl file
            target_fps: Target frame rate for interpolation
        """
        self.scenario_path = scenario_path
        self.target_fps = target_fps
        
        # Load scenario data
        self.scenario_data = None
        self.tracks: Dict[str, Dict] = {}
        self.ego_track: Optional[Dict] = None
        self.metadata: Dict = {}
        
        self._loaded = False
    
    def load(self) -> "TrajectoryInterpolator":
        """Load scenario data from file."""
        with open(self.scenario_path, 'rb') as f:
            self.scenario_data = pickle.load(f)
        
        # Extract tracks
        self.tracks = self.scenario_data.get('tracks', {})
        self.metadata = self.scenario_data.get('metadata', {})
        
        # Find ego vehicle track
        for track_id, track in self.tracks.items():
            if track.get('type', '').lower() in ['vehicle', 'ego', 'sdc']:
                # Check if this is the ego vehicle (usually marked or first vehicle)
                if track.get('object_type', '') == 'VEHICLE' or track_id == 'ego':
                    # ScenarioNet usually marks ego with specific id
                    pass
        
        # Get the SDC (self-driving car) track ID from metadata
        sdc_id = self.metadata.get('sdc_id', self.metadata.get('current_agent_id'))
        if sdc_id and sdc_id in self.tracks:
            self.ego_track = self.tracks[sdc_id]
        elif self.tracks:
            # Fallback: use the first vehicle track
            for track_id, track in self.tracks.items():
                if track.get('type', 'VEHICLE') == 'VEHICLE':
                    self.ego_track = track
                    break
        
        self._loaded = True
        return self
    
    def get_track_state_at_step(
        self,
        track: Dict,
        step: int
    ) -> Optional[InterpolatedState]:
        """
        Get track state at a specific simulation step.
        
        Args:
            track: Track data dictionary
            step: Simulation step index
            
        Returns:
            InterpolatedState at the given step, or None if out of bounds
        """
        states = track.get('state', {})
        
        # Get position
        position = states.get('position', np.array([]))
        if len(position) <= step:
            return None
        
        pos = position[step]
        
        # Get velocity
        velocity = states.get('velocity', np.zeros((len(position), 2)))
        vel = velocity[step] if len(velocity) > step else np.zeros(2)
        
        # Get heading
        heading = states.get('heading', np.zeros(len(position)))
        h = heading[step] if len(heading) > step else 0.0
        
        # Get dimensions
        length = track.get('length', states.get('length', [4.5]))[0] if isinstance(track.get('length', states.get('length', [4.5])), (list, np.ndarray)) else track.get('length', 4.5)
        width = track.get('width', states.get('width', [1.8]))[0] if isinstance(track.get('width', states.get('width', [1.8])), (list, np.ndarray)) else track.get('width', 1.8)
        
        return InterpolatedState(
            timestamp=step / 10.0,  # ScenarioNet is typically 10Hz
            position=np.array(pos[:2]) if len(pos) >= 2 else np.zeros(2),
            velocity=np.array(vel[:2]) if len(vel) >= 2 else np.zeros(2),
            heading=float(h),
            length=float(length),
            width=float(width),
            is_keyframe=True,  # Original data points
            keyframe_index=step // 5,  # Every 5 steps at 10Hz = 2Hz keyframe
            interpolation_ratio=0.0,
        )
    
    def interpolate_state(
        self,
        state1: InterpolatedState,
        state2: InterpolatedState,
        ratio: float,
        target_timestamp: float,
    ) -> InterpolatedState:
        """
        Interpolate between two states.
        
        Args:
            state1: Start state
            state2: End state
            ratio: Interpolation ratio [0, 1]
            target_timestamp: Target timestamp for the interpolated state
            
        Returns:
            Interpolated state
        """
        return InterpolatedState(
            timestamp=target_timestamp,
            position=linear_interpolate(state1.position, state2.position, ratio),
            velocity=linear_interpolate(state1.velocity, state2.velocity, ratio),
            heading=interpolate_angle(state1.heading, state2.heading, ratio),
            length=state1.length,  # Constant
            width=state1.width,    # Constant
            is_keyframe=False,
            keyframe_index=state1.keyframe_index,
            interpolation_ratio=ratio,
        )
    
    def interpolate_track_at_time(
        self,
        track: Dict,
        timestamp: float,
        source_fps: float = 10.0,
    ) -> Optional[InterpolatedState]:
        """
        Interpolate track state at a specific timestamp.
        
        Args:
            track: Track data dictionary
            timestamp: Target timestamp in seconds
            source_fps: Frame rate of source data (ScenarioNet = 10Hz)
            
        Returns:
            Interpolated state at the given timestamp
        """
        # Convert timestamp to step indices
        exact_step = timestamp * source_fps
        step1 = int(np.floor(exact_step))
        step2 = int(np.ceil(exact_step))
        
        # Get states at boundary steps
        state1 = self.get_track_state_at_step(track, step1)
        if state1 is None:
            return None
        
        if step1 == step2:
            return state1
        
        state2 = self.get_track_state_at_step(track, step2)
        if state2 is None:
            return state1  # Use state1 if state2 doesn't exist
        
        # Interpolate
        ratio = exact_step - step1
        return self.interpolate_state(state1, state2, ratio, timestamp)
    
    def get_ego_state_at_time(
        self,
        timestamp: float,
    ) -> Optional[InterpolatedState]:
        """Get interpolated ego vehicle state at a specific timestamp."""
        if not self._loaded:
            self.load()
        
        if self.ego_track is None:
            return None
        
        return self.interpolate_track_at_time(self.ego_track, timestamp)
    
    def get_all_vehicle_states_at_time(
        self,
        timestamp: float,
        max_vehicles: int = 20,
    ) -> List[Tuple[str, InterpolatedState]]:
        """
        Get interpolated states for all vehicles at a specific timestamp.
        
        Args:
            timestamp: Target timestamp in seconds
            max_vehicles: Maximum number of vehicles to return
            
        Returns:
            List of (track_id, state) tuples
        """
        if not self._loaded:
            self.load()
        
        results = []
        for track_id, track in self.tracks.items():
            # Only include vehicles
            track_type = track.get('type', 'VEHICLE')
            if track_type not in ['VEHICLE', 'vehicle', 'Vehicle']:
                continue
            
            state = self.interpolate_track_at_time(track, timestamp)
            if state is not None:
                results.append((track_id, state))
            
            if len(results) >= max_vehicles:
                break
        
        return results
    
    def generate_12hz_trajectory(
        self,
        track: Dict,
        duration: float,
        source_fps: float = 10.0,
    ) -> List[InterpolatedState]:
        """
        Generate a complete 12Hz trajectory for a track.
        
        Args:
            track: Track data dictionary
            duration: Total duration in seconds
            source_fps: Frame rate of source data
            
        Returns:
            List of interpolated states at 12Hz
        """
        states = []
        dt = 1.0 / self.target_fps
        t = 0.0
        
        while t <= duration:
            state = self.interpolate_track_at_time(track, t, source_fps)
            if state is not None:
                states.append(state)
            t += dt
        
        return states
    
    def get_scenario_duration(self) -> float:
        """Get the total duration of the scenario in seconds."""
        if not self._loaded:
            self.load()
        
        max_steps = 0
        for track in self.tracks.values():
            states = track.get('state', {})
            position = states.get('position', [])
            if len(position) > max_steps:
                max_steps = len(position)
        
        # ScenarioNet is typically 10Hz
        return max_steps / 10.0


class ScenarioTrajectoryMatcher:
    """
    Matches NuScenes 12Hz frame data with ScenarioNet trajectories.
    
    This class bridges the gap between:
    - NuScenes 12Hz frame timestamps (from nuscenes_12hz.py)
    - ScenarioNet 10Hz trajectory data
    
    It provides interpolated trajectory data for each 12Hz frame.
    """
    
    def __init__(
        self,
        scenario_path: str,
        frame_timestamps: List[int],  # In microseconds
    ):
        """
        Initialize the matcher.
        
        Args:
            scenario_path: Path to ScenarioNet scenario file
            frame_timestamps: List of frame timestamps from NuScenes (microseconds)
        """
        self.interpolator = TrajectoryInterpolator(scenario_path)
        self.frame_timestamps = frame_timestamps
        
        # Normalize timestamps to seconds, starting from 0
        if frame_timestamps:
            self.base_timestamp = frame_timestamps[0]
            self.normalized_times = [
                (t - self.base_timestamp) / 1e6 
                for t in frame_timestamps
            ]
        else:
            self.base_timestamp = 0
            self.normalized_times = []
    
    def load(self) -> "ScenarioTrajectoryMatcher":
        """Load scenario data."""
        self.interpolator.load()
        return self
    
    def get_ego_state_at_frame(
        self,
        frame_idx: int
    ) -> Optional[InterpolatedState]:
        """Get interpolated ego state at a specific frame index."""
        if frame_idx >= len(self.normalized_times):
            return None
        
        t = self.normalized_times[frame_idx]
        return self.interpolator.get_ego_state_at_time(t)
    
    def get_all_states_at_frame(
        self,
        frame_idx: int,
        max_vehicles: int = 20,
    ) -> List[Tuple[str, InterpolatedState]]:
        """Get all vehicle states at a specific frame index."""
        if frame_idx >= len(self.normalized_times):
            return []
        
        t = self.normalized_times[frame_idx]
        return self.interpolator.get_all_vehicle_states_at_time(t, max_vehicles)
    
    def generate_all_ego_states(self) -> List[InterpolatedState]:
        """Generate ego states for all frames."""
        states = []
        for idx in range(len(self.normalized_times)):
            state = self.get_ego_state_at_frame(idx)
            if state:
                states.append(state)
        return states


def compute_interpolation_info(
    frame_timestamp: int,  # microseconds
    keyframe_timestamps: List[int],  # microseconds
) -> Tuple[int, int, float]:
    """
    Compute interpolation information for a frame timestamp.
    
    Args:
        frame_timestamp: Timestamp of the frame to interpolate
        keyframe_timestamps: List of keyframe timestamps
        
    Returns:
        Tuple of (prev_kf_idx, next_kf_idx, interpolation_ratio)
    """
    if not keyframe_timestamps:
        return 0, 0, 0.0
    
    # Handle edge cases
    if frame_timestamp <= keyframe_timestamps[0]:
        return 0, 0, 0.0
    if frame_timestamp >= keyframe_timestamps[-1]:
        n = len(keyframe_timestamps) - 1
        return n, n, 1.0
    
    # Find boundary keyframes
    for i in range(len(keyframe_timestamps) - 1):
        t0 = keyframe_timestamps[i]
        t1 = keyframe_timestamps[i + 1]
        
        if t0 <= frame_timestamp <= t1:
            ratio = (frame_timestamp - t0) / (t1 - t0) if t1 != t0 else 0.0
            return i, i + 1, ratio
    
    return 0, 0, 0.0
