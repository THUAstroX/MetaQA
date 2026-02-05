"""
MetaDrive Trajectory Action Demo

This script demonstrates:
1. Using MetaDrive ScenarioEnv to render full scene (map, surrounding vehicles)
2. Displaying ego trajectory with future waypoints
3. Showing control signals (steering, throttle/brake)
4. Proper trajectory truncation when remaining trajectory < 2 seconds

Usage:
    # Top-down view (default)
    python -m meta_qa.scripts.demo_vis.trajectory_action_demo
    
    # 3D view
    python -m meta_qa.scripts.demo_vis.trajectory_action_demo --mode 3d
    
    # Custom data directory
    python -m meta_qa.scripts.demo_vis.trajectory_action_demo --data_dir /path/to/data
    
    # Custom trajectory horizon
    python -m meta_qa.scripts.demo_vis.trajectory_action_demo --horizon 3.0
"""

import os
import sys
import numpy as np
from typing import Optional, List, Dict, Tuple

# Add paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Data directory (default)
DEFAULT_DATA_DIR = os.path.join(
    project_root,
    "dataset", "Scenario_Data", "exp_nuscenes_converted"
)

# Trajectory parameters
NUM_WAYPOINTS = 20  # 20 waypoints for 2 seconds
TRAJECTORY_DT = 0.1  # 0.1s interval
TRAJECTORY_HORIZON = 2.0  # 2 seconds


def draw_trajectory_on_frame(frame: np.ndarray,
                             trajectory: np.ndarray,
                             ego_pos: np.ndarray,
                             ego_heading: float,
                             env,
                             is_complete: bool = True,
                             steering: float = 0.0,
                             throttle: float = 0.0,
                             horizon: float = 2.0,
                             frenet_info: Optional[dict] = None) -> np.ndarray:
    """
    Draw future trajectory (action waypoints) on the top-down frame.
    
    Args:
        frame: Top-down render frame (H, W, 3)
        trajectory: Future trajectory in world coordinates (N, 2)
        ego_pos: Ego vehicle position (world coordinates)
        ego_heading: Ego vehicle heading
        env: MetaDrive environment (to get renderer info)
        is_complete: Whether trajectory is complete
        steering: Current steering value
        throttle: Current throttle/brake value
        horizon: Trajectory horizon in seconds
        
    Returns:
        Frame with trajectory drawn
    """
    import cv2
    
    if trajectory is None or len(trajectory) < 2:
        return frame
    
    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2
    
    # Get the scaling from the top-down renderer
    try:
        renderer = env.engine.top_down_renderer
        if renderer is not None:
            # Get the canvas and scaling info
            scaling = renderer.scaling if hasattr(renderer, 'scaling') else 1.0
            # The renderer uses pygame coordinates
            # pixels_per_meter is typically scaling
            pixels_per_meter = scaling
        else:
            pixels_per_meter = 5.0  # Default fallback
    except:
        pixels_per_meter = 5.0  # Default fallback
    
    points = []
    for wp in trajectory:
        # World to relative (ego-centered)
        rel_x = wp[0] - ego_pos[0]
        rel_y = wp[1] - ego_pos[1]
        
        # Convert to image coordinates
        # MetaDrive uses: x = east, y = north in world
        # Image: origin top-left, x right, y down
        img_x = center_x + rel_x * pixels_per_meter
        img_y = center_y - rel_y * pixels_per_meter  # Flip y
        
        # Clamp to frame bounds
        img_x = max(0, min(w - 1, int(img_x)))
        img_y = max(0, min(h - 1, int(img_y)))
        
        points.append((img_x, img_y))
    
    # Draw trajectory line
    if len(points) >= 2:
        # Color: green if complete, orange if truncated
        if is_complete:
            color = (0, 255, 0)  # Green (BGR)
        else:
            color = (0, 165, 255)  # Orange (BGR)
        
        # Draw line segments (thinner line)
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], color, 1, cv2.LINE_AA)
        
        # Draw waypoint circles (smaller)
        for i, pt in enumerate(points):
            # Larger circle for every 5th waypoint
            radius = 3 if i % 5 == 0 else 2
            cv2.circle(frame, pt, radius, color, -1, cv2.LINE_AA)
        
        # Draw arrow at the end to show direction
        if len(points) >= 2:
            cv2.arrowedLine(frame, points[-2], points[-1], color, 1, cv2.LINE_AA, tipLength=0.3)
    
    # Add legend text (thinner font)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.45
    font_thickness = 1
    
    # Line 1: Waypoints info
    if is_complete:
        text1 = f"Action: {len(trajectory)} pts ({horizon:.1f}s)"
        text_color = (0, 255, 0)
    else:
        text1 = f"Action: {len(trajectory)} pts (truncated)"
        text_color = (0, 165, 255)
    cv2.putText(frame, text1, (10, 20), font, font_scale, text_color, font_thickness, cv2.LINE_AA)
    
    # Line 2: Steering
    steer_text = f"Steering: {steering:+.3f}"
    steer_color = (0, 255, 0)  # Green
    cv2.putText(frame, steer_text, (10, 40), font, font_scale, steer_color, font_thickness, cv2.LINE_AA)
    
    # Line 3: Throttle/Brake
    if throttle >= 0:
        throttle_text = f"Throttle: {throttle:.3f}"
        throttle_color = (0, 255, 0)  # Green
    else:
        throttle_text = f"Brake: {abs(throttle):.3f}"
        throttle_color = (0, 0, 255)  # Red
    cv2.putText(frame, throttle_text, (10, 60), font, font_scale, throttle_color, font_thickness, cv2.LINE_AA)
    
    # Line 4: Frenet info (if available)
    if frenet_info is not None:
        s_val = frenet_info.get('s', 0)
        d_val = frenet_info.get('d', 0)
        speed_s = frenet_info.get('speed_s', 0)
        frenet_text = f"Frenet: s={s_val:.1f}m d={d_val:.2f}m v_s={speed_s:.1f}m/s"
        cv2.putText(frame, frenet_text, (10, 80), font, font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA)
    
    return frame


def get_future_trajectory(env, num_waypoints: int = NUM_WAYPOINTS) -> Tuple[np.ndarray, bool]:
    """
    Get future trajectory from replay data with proper truncation.
    
    Args:
        env: MetaDrive environment
        num_waypoints: Number of waypoints to extract
        
    Returns:
        trajectory: Future trajectory (num_waypoints, 2) or truncated
        is_complete: Whether trajectory is complete (2 seconds)
    """
    ego = env.agent
    
    # Try to get trajectory from the scenario data
    try:
        # Method 1: Get from scenario manager
        if hasattr(env, 'engine') and hasattr(env.engine, 'data_manager'):
            scenario = env.engine.data_manager.current_scenario
            if scenario is not None:
                tracks = scenario.get('tracks', {})
                sdc_id = scenario.get('metadata', {}).get('sdc_id', 'sdc')
                
                # Find ego track
                ego_track = tracks.get(sdc_id) or tracks.get('ego') or tracks.get('sdc')
                
                if ego_track is None:
                    # Try to find by type
                    for track_id, track in tracks.items():
                        if track.get('type', '') == 'VEHICLE':
                            ego_track = track
                            break
                
                if ego_track is not None:
                    state = ego_track.get('state', {})
                    positions = state.get('position')
                    
                    if positions is not None:
                        # Get current step
                        current_step = env.engine.episode_step if hasattr(env.engine, 'episode_step') else 0
                        
                        # Extract future trajectory
                        future_start = current_step + 1
                        future_end = current_step + 1 + num_waypoints
                        
                        if future_end <= len(positions):
                            future_traj = positions[future_start:future_end, :2]
                            return np.array(future_traj), True
                        else:
                            remaining_traj = positions[future_start:, :2]
                            if len(remaining_traj) >= 2:
                                return np.array(remaining_traj), False
        
        # Method 2: Try to get from ego's navigation or policy
        if hasattr(ego, 'expert_takeover') and hasattr(ego, 'before_step'):
            # ReplayPolicy stores trajectory internally
            pass
            
    except Exception as e:
        pass  # Fall back to default
    
    # Default: return current position (will trigger brake)
    return np.array([[ego.position[0], ego.position[1]]]), False


def compute_control_from_trajectory(trajectory: np.ndarray, 
                                    current_pos: np.ndarray,
                                    current_heading: float,
                                    current_velocity: np.ndarray,
                                    is_complete: bool) -> Tuple[float, float, Optional[dict]]:
    """
    Compute control signals from trajectory using TrajectoryTracker.
    
    Uses MetaDrive's official PIDController for trajectory tracking.
    
    Args:
        trajectory: Future trajectory waypoints (in world frame)
        current_pos: Current position
        current_heading: Current heading
        current_velocity: Current velocity
        is_complete: Whether trajectory is complete (True = full 2s, False = truncated)
        
    Returns:
        steering: Steering signal [-1, 1]
        throttle: Throttle/brake signal [-1, 1]
        info: Debug info (or None)
    """
    if len(trajectory) < 2:
        # Not enough waypoints - stop
        return 0.0, -0.5, None  # Gentle brake
    
    # NOTE: Even if trajectory is truncated (is_complete=False), we should still
    # use the available waypoints to infer the intended speed, rather than
    # forcing a brake. The trajectory data still contains valid speed info!
    
    frenet_info = None
    
    # Use TrajectoryTracker (MetaDrive's official PIDController approach)
    from meta_qa.core.trajectory_tracker import TrajectoryTracker, VehicleState
    
    tracker = TrajectoryTracker()
    
    # Create vehicle state - params will be added by caller if needed
    vehicle_state = VehicleState(
        position=np.array(current_pos),
        heading=current_heading,
        speed=np.linalg.norm(current_velocity),
        velocity=np.array(current_velocity),
    )
    
    steering, throttle = tracker.track(trajectory, vehicle_state)
    
    # If trajectory is truncated and very short, reduce speed
    if not is_complete and len(trajectory) < 5:
        remaining_time = len(trajectory) * 0.1  # seconds
        if remaining_time < 0.5:
            blend_factor = remaining_time / 0.5
            throttle = throttle * blend_factor - 0.2 * (1 - blend_factor)
    
    return float(steering), float(throttle), frenet_info


def print_vehicle_info(env):
    """Print vehicle parameters extracted from the environment."""
    from meta_qa.core.trajectory_tracker import VehicleState
    
    state = VehicleState.from_env(env, include_params=True)
    if state.params:
        print("\n" + "=" * 50)
        print("Vehicle Parameters (extracted from environment)")
        print("=" * 50)
        print(f"  Length:          {state.params.length:.2f} m")
        print(f"  Width:           {state.params.width:.2f} m")
        print(f"  Wheelbase:       {state.params.wheelbase:.2f} m")
        print(f"  Max steering:    {state.params.max_steering_deg}°")
        print(f"  Min turn radius: {state.params.min_turn_radius:.2f} m")
        print(f"  Max speed:       {state.params.max_speed_km_h} km/h")
        print(f"  Mass:            {state.params.mass} kg")
        print(f"  Max engine:      {state.params.max_engine_force} N")
        print(f"  Wheel friction:  {state.params.wheel_friction}")
        print("=" * 50 + "\n")


def run_metadrive_demo(data_dir: str = DEFAULT_DATA_DIR):
    """
    Run MetaDrive demo with full scene visualization.
    
    Args:
        data_dir: Path to ScenarioNet dataset directory
    """
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    
    print("=" * 60)
    print("MetaDrive Full Scene Visualization Demo")
    print(f"Data directory: {data_dir}")
    print("=" * 60)
    
    # Create output directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(project_root, "outputs", "vis")
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure environment
    config = dict(
        agent_policy=ReplayEgoCarPolicy,  # Use replay policy
        use_render=True,  # Enable rendering
        data_directory=data_dir,
        num_scenarios=10,
        sequential_seed=True,
        
        # Rendering settings
        window_size=(1200, 800),
        
        # Camera settings
        vehicle_config=dict(
            show_navi_mark=False,
            show_line_to_navi_mark=False,
        ),
        
        # Show other vehicles
        no_traffic=False,
        
        # Top-down view settings
        top_down_show_real_size=True,
    )
    
    print("\n[1/3] Initializing MetaDrive environment...")
    env = ScenarioEnv(config)
    
    print("\n[2/3] Running replay with trajectory visualization...")
    
    # Storage for recording
    trajectory_history = []
    control_history = []
    position_history = []
    reward_history = []
    
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        
        # Print vehicle parameters on first reset
        print_vehicle_info(env)
        
        print("\nPress ESC to exit, or wait for scenario to complete.\n")
        print("-" * 60)
        print(f"{'Step':>5} | {'Position':>20} | {'Steering':>10} | {'Throttle':>10} | {'Complete':>8}")
        print("-" * 60)
        
        while not done:
            # Get ego state
            ego = env.agent
            current_pos = np.array(ego.position[:2])
            current_heading = ego.heading_theta
            current_vel = np.array(ego.velocity[:2]) if hasattr(ego, 'velocity') else np.array([0, 0])
            
            # Get future trajectory with truncation handling
            future_traj, is_complete = get_future_trajectory(env, NUM_WAYPOINTS)
            
            # Compute control signals using Frenet coordinates
            steering, throttle, frenet_info = compute_control_from_trajectory(
                future_traj, current_pos, current_heading, current_vel, is_complete
            )
            
            # Record data
            position_history.append(current_pos.copy())
            trajectory_history.append(future_traj.copy())
            control_history.append((steering, throttle))
            
            # Print status every 10 steps
            if step_count % 10 == 0:
                complete_str = "Yes" if is_complete else f"No({len(future_traj)})"
                print(f"{step_count:>5} | ({current_pos[0]:>8.1f}, {current_pos[1]:>8.1f}) | "
                      f"{steering:>10.4f} | {throttle:>10.4f} | {complete_str:>8}")
            
            # Use replay action (follow original trajectory)
            # In replay mode, action is ignored, vehicle follows recorded trajectory
            action = np.array([steering, throttle])
            
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Use replay_done to determine when replay is complete
            # This is better than terminated/truncated for replay scenarios
            # because it ignores arrive_dest, crash, etc. and focuses on
            # whether the recorded trajectory data has been fully replayed
            if info.get("replay_done", False):
                print(f"\n  Replay completed at step {step_count}")
                done = True
            
            # Track reward
            reward_history.append(reward)
            
            # Render
            env.render(mode="topdown")
            
            step_count += 1
            
            # Safety limit
            if step_count > 500:
                print("\nReached step limit.")
                break
        
        print("-" * 60)
        print(f"\nScenario completed! Total steps: {step_count}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        env.close()
    
    # Generate visualization
    print("\n[3/3] Generating trajectory and control visualization...")
    generate_visualization(position_history, control_history, output_dir, reward_history=reward_history)
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print(f"Output directory: {output_dir}")
    print("=" * 60)


def generate_visualization(position_history: List[np.ndarray],
                          control_history: List[Tuple[float, float]],
                          output_dir: str,
                          reward_history: Optional[List[float]] = None):
    """
    Generate trajectory, control signal, and reward visualization.
    """
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('TkAgg')
    
    positions = np.array(position_history)
    steering = np.array([c[0] for c in control_history])
    throttle = np.array([c[1] for c in control_history])
    time_steps = np.arange(len(steering)) * 0.1
    
    # Determine layout based on whether we have reward data
    if reward_history is not None and len(reward_history) > 0:
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.flatten()
    else:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
    
    fig.suptitle('Trajectory, Control & Reward (MetaDrive Replay)', fontsize=14, fontweight='bold')
    
    # 1. Bird's Eye View
    ax1 = axes[0]
    ax1.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Ego Trajectory')
    ax1.scatter(positions[0, 0], positions[0, 1], c='green', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(positions[-1, 0], positions[-1, 1], c='red', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_title("Bird's Eye View")
    ax1.legend()
    ax1.axis('equal')
    ax1.grid(True, alpha=0.3)
    
    # 2. Steering Signal
    ax2 = axes[1]
    ax2.plot(time_steps, steering, 'b-', linewidth=1.5)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.fill_between(time_steps, steering, 0, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Steering [-1, 1]')
    ax2.set_title('Steering Signal')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Throttle/Brake Signal
    ax3 = axes[2] if reward_history is not None and len(reward_history) > 0 else axes[2]
    ax3.plot(time_steps, throttle, 'g-', linewidth=1.5)
    ax3.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    throttle_pos = np.maximum(throttle, 0)
    throttle_neg = np.minimum(throttle, 0)
    ax3.fill_between(time_steps, throttle_pos, 0, alpha=0.3, color='green', label='Throttle')
    ax3.fill_between(time_steps, throttle_neg, 0, alpha=0.3, color='red', label='Brake')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Throttle/Brake [-1, 1]')
    ax3.set_title('Throttle/Brake Signal')
    ax3.set_ylim(-1.1, 1.1)
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 4. Speed
    ax4 = axes[3]
    if len(positions) > 1:
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / 0.1
        speeds = np.append(speeds, speeds[-1])  # Pad to match length
    else:
        speeds = np.array([0.0])
    ax4.plot(time_steps, speeds, 'purple', linewidth=1.5)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('Speed (m/s)')
    ax4.set_title('Speed Curve')
    ax4.grid(True, alpha=0.3)
    avg_speed = np.mean(speeds) if len(speeds) > 0 else 0.0
    ax4.axhline(y=avg_speed, color='orange', linestyle='--', alpha=0.7, label=f'Avg: {avg_speed:.1f} m/s')
    ax4.legend()
    
    # 5. Reward (if available)
    if reward_history is not None and len(reward_history) > 0:
        rewards = np.array(reward_history)
        reward_time = np.arange(len(rewards)) * 0.1
        
        # Per-step reward
        ax5 = axes[4]
        ax5.plot(reward_time, rewards, 'orange', linewidth=1.5)
        ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        reward_pos = np.maximum(rewards, 0)
        reward_neg = np.minimum(rewards, 0)
        ax5.fill_between(reward_time, reward_pos, 0, alpha=0.3, color='green')
        ax5.fill_between(reward_time, reward_neg, 0, alpha=0.3, color='red')
        ax5.set_xlabel('Time (s)')
        ax5.set_ylabel('Reward')
        ax5.set_title(f'Step Reward (avg={np.mean(rewards):.4f})')
        ax5.grid(True, alpha=0.3)
        
        # Cumulative reward
        ax6 = axes[5]
        cumulative_reward = np.cumsum(rewards)
        ax6.plot(reward_time, cumulative_reward, 'darkgreen', linewidth=2)
        ax6.fill_between(reward_time, cumulative_reward, 0, alpha=0.2, color='green')
        ax6.set_xlabel('Time (s)')
        ax6.set_ylabel('Cumulative Reward')
        ax6.set_title(f'Cumulative Reward (total={cumulative_reward[-1]:.4f})')
        ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "metadrive_trajectory.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    
    plt.show()


def run_topdown_demo(data_dir: str = DEFAULT_DATA_DIR, horizon: float = 2.0):
    """
    Run top-down view demo (no 3D rendering, faster).
    
    Args:
        data_dir: Path to ScenarioNet dataset directory
        horizon: Trajectory horizon in seconds
    """
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    
    # Calculate number of waypoints based on horizon
    num_waypoints = int(horizon / TRAJECTORY_DT)
    
    print("=" * 60)
    print("MetaDrive Top-Down View Demo")
    print(f"Data directory: {data_dir}")
    print(f"Trajectory Horizon: {horizon}s ({num_waypoints} waypoints)")
    print("=" * 60)
    
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    output_dir = os.path.join(project_root, "outputs", "vis")
    os.makedirs(output_dir, exist_ok=True)

    # Random scenario selection
    import random
    random_seed = random.randint(0, 9)
    print(f"\nRandomly selected scenario index: {random_seed}")
    
    # Configure environment - top-down only
    config = dict(
        agent_policy=ReplayEgoCarPolicy,  # Use replay policy
        use_render=False,  # No 3D rendering
        data_directory=data_dir,
        num_scenarios=1,  # Only load 1 scenario
        start_scenario_index=random_seed,  # Start from random scenario
        no_traffic=False,
    )
    
    print("\n[1/2] Initializing MetaDrive environment (top-down mode)...")
    env = ScenarioEnv(config)
    
    print("\n[2/2] Running replay...")
    
    # Storage
    frames = []
    position_history = []
    control_history = []
    reward_history = []  # Track rewards
    
    try:
        obs, info = env.reset()
        done = False
        step_count = 0
        valid_step_count = 0  # Count of steps with complete trajectory
        skipped_count = 0     # Count of skipped steps (incomplete trajectory)
        total_reward = 0.0    # Cumulative reward
        
        # Print vehicle parameters on first reset
        print_vehicle_info(env)
        
        print("\n" + "-" * 50)
        print("Press ESC or close window to quit.\n")
        
        # Use pygame for real-time display
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("MetaDrive Top-Down - Trajectory Action")
        clock = pygame.time.Clock()
        
        running = True
        while not done and running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            ego = env.agent
            current_pos = np.array(ego.position[:2])
            current_heading = ego.heading_theta
            current_vel = np.array(ego.velocity[:2]) if hasattr(ego, 'velocity') else np.array([0, 0])
            
            # Get trajectory and compute control using Frenet coordinates
            future_traj, is_complete = get_future_trajectory(env, num_waypoints)
            
            # Skip data points where trajectory is incomplete (not enough for full horizon)
            # This ensures training data quality - only use complete trajectory samples
            if is_complete:
                steering, throttle, frenet_info = compute_control_from_trajectory(
                    future_traj, current_pos, current_heading, current_vel, is_complete
                )
                
                # Record data only for complete trajectories
                position_history.append(current_pos.copy())
                control_history.append((steering, throttle))
                valid_step_count += 1
            else:
                # Trajectory incomplete - skip this data point
                # Still show visualization but mark as skipped
                steering, throttle, frenet_info = 0.0, 0.0, None
            
            # Get top-down frame - larger film_size = zoom in closer
            try:
                frame = env.render(mode="topdown",
                                  film_size=(4000, 4000),
                                  screen_size=(800, 800),
                                  draw_contour=True)
                
                if frame is not None and isinstance(frame, np.ndarray):
                    # Draw future trajectory (action) on the frame with Frenet info
                    frame_with_traj = draw_trajectory_on_frame(
                        frame.copy(), 
                        future_traj, 
                        current_pos,
                        current_heading,
                        env=env,
                        is_complete=is_complete,
                        steering=steering,
                        throttle=throttle,
                        horizon=horizon,
                        frenet_info=frenet_info
                    )
                    frames.append(frame_with_traj)
                    
                    # Display with pygame (need to transpose for pygame surface)
                    # frame is (H, W, 3) RGB, pygame needs (W, H) surface
                    surf = pygame.surfarray.make_surface(frame_with_traj.swapaxes(0, 1))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    
            except Exception as e:
                if step_count == 0:
                    print(f"Warning: Could not get top-down frame: {e}")
            
            # Print progress every 20 steps
            if step_count % 20 == 0:
                status = "OK" if is_complete else "SKIP"
                print(f"Step {step_count}: pos=({current_pos[0]:.1f}, {current_pos[1]:.1f}), "
                      f"steering={steering:.3f}, throttle={throttle:.3f}, [{status}]")
            
            # Track skipped steps
            if not is_complete:
                skipped_count += 1
            
            action = np.array([steering, throttle])
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Use replay_done to determine when replay is complete
            # This is better than terminated/truncated for replay scenarios
            # because it ignores arrive_dest, crash, etc. and focuses on
            # whether the recorded trajectory data has been fully replayed
            if info.get("replay_done", False):
                print(f"\n  Replay completed at step {step_count}")
                done = True
            
            # Track reward
            reward_history.append(reward)
            total_reward += reward
            
            step_count += 1
            
            # Control frame rate
            clock.tick(30)  # 30 FPS
            
            if step_count > 300:
                break
        
        print("-" * 50)
        print(f"\nCompleted! Total steps: {step_count}")
        print(f"Valid samples (complete trajectory): {valid_step_count}")
        print(f"Skipped samples (incomplete trajectory): {skipped_count}")
        
        # Reward statistics
        if reward_history:
            rewards = np.array(reward_history)
            print(f"\n--- Reward Statistics ---")
            print(f"Total reward: {total_reward:.4f}")
            print(f"Average reward per step: {np.mean(rewards):.4f}")
            print(f"Reward std: {np.std(rewards):.4f}")
            print(f"Min reward: {np.min(rewards):.4f}")
            print(f"Max reward: {np.max(rewards):.4f}")
            print(f"Positive rewards: {np.sum(rewards > 0)} steps")
            print(f"Negative rewards: {np.sum(rewards < 0)} steps")
            print(f"Zero rewards: {np.sum(rewards == 0)} steps")
        
    finally:
        pygame.quit()
        env.close()
    
    # Save GIF
    if frames:
        from PIL import Image
        
        gif_path = os.path.join(output_dir, "metadrive_topdown.gif")
        print(f"\nSaving GIF ({len(frames)} frames)...")
        
        pil_frames = [Image.fromarray(f) for f in frames]
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=100,
            loop=0
        )
        print(f"GIF saved to: {gif_path}")
    
    # Generate control visualization
    generate_visualization(position_history, control_history, output_dir, reward_history=reward_history)
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='MetaDrive Trajectory Action Demo')
    parser.add_argument('--data_dir', '-d', type=str, default=DEFAULT_DATA_DIR,
                       help='Path to ScenarioNet dataset directory')
    parser.add_argument('--mode', type=str, default='topdown',
                       choices=['3d', 'topdown'],
                       help='Rendering mode: 3d (full 3D) or topdown (top-down view)')
    parser.add_argument('--horizon', type=float, default=2.0,
                       help='Trajectory horizon in seconds (default: 2.0)')
    
    args = parser.parse_args()
    
    # Update global parameters based on args
    TRAJECTORY_HORIZON = args.horizon
    NUM_WAYPOINTS = int(TRAJECTORY_HORIZON / TRAJECTORY_DT)
    
    if args.mode == '3d':
        run_metadrive_demo(data_dir=args.data_dir)
    else:
        run_topdown_demo(data_dir=args.data_dir, horizon=args.horizon)
