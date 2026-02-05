"""
Trajectory Tracking Controller.

Reference: MetaDrive official PIDController implementation
https://github.com/metadriverse/metadrive/blob/main/metadrive/component/vehicle/PID_controller.py

"""

import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

# Use MetaDrive's official functions
from metadrive.component.vehicle.PID_controller import PIDController
from metadrive.utils.math import wrap_to_pi, clip, norm


@dataclass
class VehicleParams:
    """
    Vehicle dynamics parameters extracted from MetaDrive vehicle.
    
    These parameters vary by vehicle type (S, M, L, XL, Default, etc.)
    and affect how trajectory → action conversion should behave.
    """
    # Geometry
    length: float = 4.51          # meters
    width: float = 1.85           # meters  
    wheelbase: float = 2.47       # meters (front + rear wheelbase)
    
    # Dynamics
    max_steering_deg: float = 40.0  # degrees, max steering angle
    max_speed_km_h: float = 80.0    # km/h
    mass: float = 1100.0            # kg
    
    # Force limits
    max_engine_force: float = 800.0   # N
    max_brake_force: float = 150.0    # N
    wheel_friction: float = 0.9
    
    @property
    def max_steering_rad(self) -> float:
        """Max steering angle in radians."""
        return np.deg2rad(self.max_steering_deg)
    
    @property
    def max_speed_m_s(self) -> float:
        """Max speed in m/s."""
        return self.max_speed_km_h / 3.6
    
    @property
    def min_turn_radius(self) -> float:
        """Minimum turning radius based on max steering and wheelbase."""
        return self.wheelbase / np.tan(self.max_steering_rad)
    
    @classmethod
    def from_vehicle(cls, vehicle) -> "VehicleParams":
        """
        Extract parameters from a MetaDrive vehicle object.
        
        Works with any vehicle type: DefaultVehicle, SVehicle, MVehicle, etc.
        """
        # Get dynamics parameters (includes max_steering, forces, friction)
        dynamics = vehicle.get_dynamics_parameters()
        
        # Compute wheelbase from front + rear
        wheelbase = getattr(vehicle, 'FRONT_WHEELBASE', 1.05) + getattr(vehicle, 'REAR_WHEELBASE', 1.42)
        
        return cls(
            length=vehicle.LENGTH,
            width=vehicle.WIDTH,
            wheelbase=wheelbase,
            max_steering_deg=dynamics.get('max_steering', 40.0),
            max_speed_km_h=vehicle.max_speed_km_h,
            mass=dynamics.get('mass', 1100.0),
            max_engine_force=dynamics.get('max_engine_force', 800.0),
            max_brake_force=dynamics.get('max_brake_force', 150.0),
            wheel_friction=dynamics.get('wheel_friction', 0.9),
        )
    
    @classmethod
    def default(cls) -> "VehicleParams":
        """Return default vehicle parameters (DefaultVehicle)."""
        return cls()


@dataclass
class VehicleState:
    """Vehicle state for trajectory tracking."""
    position: np.ndarray      # (2,) world frame [x, y]
    heading: float            # radians
    speed: float              # m/s
    velocity: Optional[np.ndarray] = None  # (2,) world frame [vx, vy]
    params: Optional[VehicleParams] = None  # Vehicle dynamics parameters
    
    @classmethod
    def from_env(cls, env, include_params: bool = True) -> "VehicleState":
        """
        Create from MetaDrive environment.
        
        Args:
            env: MetaDrive environment (can be wrapped)
            include_params: Whether to extract vehicle dynamics parameters
        """
        base_env = env
        while hasattr(base_env, "env"):
            base_env = base_env.env
        
        ego = base_env.agent
        velocity = np.array([ego.velocity[0], ego.velocity[1]])
        
        # Extract vehicle parameters if requested
        params = VehicleParams.from_vehicle(ego) if include_params else None
        
        return cls(
            position=np.array([ego.position[0], ego.position[1]]),
            heading=ego.heading_theta,
            speed=norm(velocity[0], velocity[1]),
            velocity=velocity,
            params=params,
        )


class TrajectoryTracker:
    """
    Trajectory tracking controller following MetaDrive's official approach.
    
    Uses dual PID for lateral control:
        - heading_pid: Controls heading error
        - lateral_pid: Controls lateral offset
    
    Formula (from MetaDrive IDMPolicy):
        long, lat = target_lane.local_coordinates(ego_position)
        lane_heading = target_lane.heading_theta_at(long + lookahead)
        
        steering = heading_pid(-wrap_to_pi(lane_heading - vehicle_heading))
                 + lateral_pid(-lat)
    
    Args:
        heading_pid_params: (k_p, k_i, k_d) for heading PID, default from MetaDrive
        lateral_pid_params: (k_p, k_i, k_d) for lateral PID, default from MetaDrive
        lookahead_distance: Distance ahead to compute target heading (m)
        k_speed: Proportional gain for speed control
        
    Usage:
        tracker = TrajectoryTracker()
        state = VehicleState.from_env(env)
        steering, throttle = tracker.track(trajectory, state)
    """
    
    # Default PID parameters from MetaDrive IDMPolicy
    # Reference: metadrive/policy/idm_policy.py line 224-234
    DEFAULT_HEADING_PID = (1.7, 0.01, 3.5)
    DEFAULT_LATERAL_PID = (0.3, 0.002, 0.05)
    
    def __init__(
        self,
        heading_pid_params: Tuple[float, float, float] = None,
        lateral_pid_params: Tuple[float, float, float] = None,
        lookahead_distance: float = 1.0,
        k_speed: float = 0.5,
    ):
        # Initialize PIDs using MetaDrive's PIDController
        hp = heading_pid_params or self.DEFAULT_HEADING_PID
        lp = lateral_pid_params or self.DEFAULT_LATERAL_PID
        
        self.heading_pid = PIDController(hp[0], hp[1], hp[2])
        self.lateral_pid = PIDController(lp[0], lp[1], lp[2])
        
        self.lookahead_distance = lookahead_distance
        self.k_speed = k_speed
    
    def reset(self) -> None:
        """Reset controller state."""
        self.heading_pid.reset()
        self.lateral_pid.reset()
    
    # Low-speed control parameters
    LOW_SPEED_THRESHOLD = 1.0   # m/s, below this speed starts to attenuate
    MIN_SPEED_THRESHOLD = 0.01  # m/s, below this considered stopped
    
    def _compute_speed_attenuation(self, target_speed: float, current_speed: float) -> float:
        """
        Compute smooth attenuation factor for low-speed situations.
        
        Uses a smooth sigmoid-like transition instead of hard thresholds.
        Factor ranges from 0 (stopped) to 1 (normal driving).
        
        Reference speed uses max of target and current to handle:
        - Stopping: target→0, current still moving → use current for smooth brake
        - Starting: current=0, target>0 → use target for smooth acceleration
        """
        ref_speed = max(target_speed, current_speed)
        
        if ref_speed < self.MIN_SPEED_THRESHOLD:
            return 0.0
        elif ref_speed >= self.LOW_SPEED_THRESHOLD:
            return 1.0
        else:
            # Smooth cubic interpolation (ease-in-out)
            t = (ref_speed - self.MIN_SPEED_THRESHOLD) / (self.LOW_SPEED_THRESHOLD - self.MIN_SPEED_THRESHOLD)
            # Smoothstep: 3t² - 2t³ gives smooth transition with zero derivative at endpoints
            return t * t * (3.0 - 2.0 * t)
    
    def track(
        self,
        trajectory: np.ndarray,
        state: VehicleState,
        target_speed: Optional[float] = None,
    ) -> Tuple[float, float]:
        """
        Compute control commands to track trajectory.
        
        Args:
            trajectory: Target trajectory (N, 2) in world coordinates
            state: Current vehicle state
            target_speed: Target speed (m/s), None = estimate from trajectory
            
        Returns:
            Tuple of (steering, throttle) in [-1, 1]
        """
        if len(trajectory) < 2:
            return 0.0, 0.0
        
        # === Step 1: Estimate target speed ===
        if target_speed is None:
            target_speed = self._estimate_target_speed(trajectory)
        
        # === Step 2: Compute speed-based attenuation factor ===
        # Smooth transition for low-speed situations
        speed_factor = self._compute_speed_attenuation(target_speed, state.speed)
        
        # === Step 3: Convert to local coordinates (like lane.local_coordinates) ===
        long, lat = self._get_local_coordinates(trajectory, state.position)
        
        # === Step 4: Get target heading at lookahead point ===
        target_heading = self._get_heading_at(trajectory, long + self.lookahead_distance)
        
        # === Step 5: Compute steering (MetaDrive's approach) ===
        heading_error = wrap_to_pi(target_heading - state.heading)
        
        steering = self.heading_pid.get_result(-heading_error)
        steering += self.lateral_pid.get_result(-lat)
        steering = clip(steering, -1.0, 1.0)
        
        # Apply speed-based attenuation to steering
        # At low speeds, steering has little effect and can cause oscillation
        steering *= speed_factor
        
        # === Step 6: Compute throttle ===
        speed_error = target_speed - state.speed
        throttle = self.k_speed * speed_error
        throttle = clip(throttle, -1.0, 1.0)
        
        # Apply attenuation to throttle as well for smooth stopping
        # But preserve braking ability when vehicle is still moving
        if throttle >= 0:
            # Accelerating: attenuate based on speed factor
            throttle *= speed_factor
        else:
            # Braking: allow full brake, but reduce if already nearly stopped
            if state.speed < self.MIN_SPEED_THRESHOLD:
                throttle = 0.0  # Already stopped, no need to brake
        
        return float(steering), float(throttle)
    
    def _get_local_coordinates(
        self,
        trajectory: np.ndarray,
        position: np.ndarray,
    ) -> Tuple[float, float]:
        """
        Convert position to local (longitudinal, lateral) coordinates on trajectory.
        
        Similar to MetaDrive's lane.local_coordinates().
        
        Returns:
            (longitudinal, lateral) where:
                - longitudinal: distance along trajectory
                - lateral: signed perpendicular distance (positive = left)
        """
        # Find closest point on trajectory
        distances = np.linalg.norm(trajectory - position, axis=1)
        closest_idx = np.argmin(distances)
        
        # Compute cumulative arc length up to closest point
        arc_lengths = np.zeros(len(trajectory))
        for i in range(1, len(trajectory)):
            diff = trajectory[i] - trajectory[i-1]
            arc_lengths[i] = arc_lengths[i-1] + norm(diff[0], diff[1])
        longitudinal = arc_lengths[closest_idx]
        
        # Compute lateral offset (signed)
        if closest_idx < len(trajectory) - 1:
            segment = trajectory[closest_idx + 1] - trajectory[closest_idx]
        elif closest_idx > 0:
            segment = trajectory[closest_idx] - trajectory[closest_idx - 1]
        else:
            segment = np.array([1.0, 0.0])
        
        to_position = position - trajectory[closest_idx]
        
        # Cross product gives signed lateral distance
        # Positive = position is to the left of the path
        cross = segment[0] * to_position[1] - segment[1] * to_position[0]
        lateral = cross / (norm(segment[0], segment[1]) + 1e-6)
        
        return longitudinal, lateral
    
    def _get_heading_at(
        self,
        trajectory: np.ndarray,
        longitudinal: float,
    ) -> float:
        """
        Get trajectory heading at given longitudinal position.
        
        Similar to MetaDrive's lane.heading_theta_at().
        """
        # Compute arc lengths
        arc_lengths = np.zeros(len(trajectory))
        for i in range(1, len(trajectory)):
            diff = trajectory[i] - trajectory[i-1]
            arc_lengths[i] = arc_lengths[i-1] + norm(diff[0], diff[1])
        
        total_length = arc_lengths[-1]
        longitudinal = clip(longitudinal, 0, total_length)
        
        # Find segment containing this longitudinal position
        idx = np.searchsorted(arc_lengths, longitudinal) - 1
        idx = int(clip(idx, 0, len(trajectory) - 2))
        
        # Compute heading from segment
        segment = trajectory[idx + 1] - trajectory[idx]
        heading = np.arctan2(segment[1], segment[0])
        
        return heading
    
    def _estimate_target_speed(
        self,
        trajectory: np.ndarray,
        dt: float = 0.1,
    ) -> float:
        """Estimate target speed from trajectory."""
        if len(trajectory) < 2:
            return 0.0  # Not enough points, assume stopped
        
        # Total trajectory length
        total_dist = np.sum(np.linalg.norm(np.diff(trajectory, axis=0), axis=1))
        
        # Assume trajectory covers N * dt seconds
        duration = len(trajectory) * dt
        
        target_speed = total_dist / max(duration, 0.1)
        
        # Allow very low speeds (including near-zero for stopped vehicles)
        # Only clip upper bound, allow natural zero/low speeds
        return clip(target_speed, 0.0, 20.0)
    
    def track_with_info(
        self,
        trajectory: np.ndarray,
        state: VehicleState,
        target_speed: Optional[float] = None,
    ) -> Tuple[float, float, Dict[str, Any]]:
        """
        Track trajectory and return detailed info for debugging.
        
        Returns:
            Tuple of (steering, throttle, info_dict)
        """
        if len(trajectory) < 2:
            return 0.0, 0.0, {"error": "trajectory too short"}
        
        # Get local coordinates
        long, lat = self._get_local_coordinates(trajectory, state.position)
        
        # Get target heading
        target_heading = self._get_heading_at(trajectory, long + self.lookahead_distance)
        heading_error = wrap_to_pi(target_heading - state.heading)
        
        # Compute control
        steering, throttle = self.track(trajectory, state, target_speed)
        
        # Estimate target speed if not provided
        estimated_target_speed = target_speed if target_speed is not None else self._estimate_target_speed(trajectory)
        
        # Compute speed attenuation factor for debugging
        speed_factor = self._compute_speed_attenuation(estimated_target_speed, state.speed)
        
        info = {
            "steering": steering,
            "throttle": throttle,
            "longitudinal": long,
            "lateral": lat,
            "heading_error": heading_error,
            "target_heading": target_heading,
            "vehicle_heading": state.heading,
            "speed": state.speed,
            "target_speed": estimated_target_speed,
            "speed_factor": speed_factor,  # Shows how much attenuation is applied
        }
        
        # Include vehicle parameters if available
        if state.params is not None:
            info["vehicle_params"] = {
                "length": state.params.length,
                "width": state.params.width,
                "wheelbase": state.params.wheelbase,
                "max_steering_deg": state.params.max_steering_deg,
                "max_speed_km_h": state.params.max_speed_km_h,
                "mass": state.params.mass,
            }
        
        return steering, throttle, info
