"""
Configuration parameters for trajectory-based RL module.

MetaDrive Vehicle Model Reference:
- Uses modified bicycle model for kinematics
- Action space: [steering, throttle_brake] both in [-1, 1]
- steering: actual_steering_angle = action * max_steering (degrees)
- throttle_brake: engine_force = action * max_engine_force (N) when >= 0
                  brake_force = |action| * max_brake_force (N) when < 0

Note: Vehicle-specific parameters (dimensions, max_steering, forces, etc.) 
are now dynamically extracted from the MetaDrive vehicle at runtime.
See VehicleParams in trajectory_tracker.py for details.

MetaDrive Vehicle Types (for reference):
- SVehicle:  length=4.3m,  max_steering=50°, mass=800kg
- MVehicle:  length=4.6m,  max_steering=45°, mass=1200kg
- Default:   length=4.51m, max_steering=40°, mass=1100kg
- LVehicle:  length=4.87m, max_steering=40°, mass=1300kg
- XLVehicle: length=5.74m, max_steering=35°, mass=1600kg
"""

import numpy as np

# ==============================================================================
# Trajectory parameters
# ==============================================================================
TRAJECTORY_HORIZON = 2.0  # seconds, prediction horizon
TRAJECTORY_DT = 0.1  # seconds, time step for waypoints (matching MetaDrive default)
NUM_WAYPOINTS = int(TRAJECTORY_HORIZON / TRAJECTORY_DT)  # 20 waypoints for 2 seconds

# ==============================================================================
# Controller parameters - PID gains for trajectory tracking
# ==============================================================================
# These are tuned for MetaDrive's physics and work across vehicle types.
# Reference: MetaDrive IDMPolicy uses similar values.

# PID gains for heading control (used in TrajectoryTracker)
# From MetaDrive: heading_pid = (1.7, 0.01, 3.5)
PID_HEADING_KP = 1.7
PID_HEADING_KI = 0.01
PID_HEADING_KD = 3.5

# PID gains for lateral offset control
# From MetaDrive: lateral_pid = (0.3, 0.002, 0.05)
PID_LATERAL_KP = 0.3
PID_LATERAL_KI = 0.002
PID_LATERAL_KD = 0.05

# Speed control gain
K_SPEED = 0.5  # Proportional gain for speed error → throttle

# Lookahead distance for heading calculation
LOOKAHEAD_DISTANCE = 1.0  # meters

# ==============================================================================
# Action space bounds (for RL trajectory prediction)
# ==============================================================================
# Maximum trajectory offset per timestep (for feasibility checking)
MAX_LATERAL_OFFSET = 4.0  # meters per timestep (lane change capability)
MAX_LONGITUDINAL_OFFSET = 3.0  # meters per timestep

# ==============================================================================
# Visualization settings
# ==============================================================================
TRAJECTORY_COLOR_PREDICTED = (0, 255, 0)  # Green for predicted trajectory
TRAJECTORY_COLOR_ACTUAL = (0, 0, 255)  # Blue for actual trajectory
TRAJECTORY_COLOR_GROUND_TRUTH = (255, 0, 0)  # Red for ground truth
TRAJECTORY_LINE_WIDTH = 2
WAYPOINT_RADIUS = 3

