#!/usr/bin/env python3
"""
Unified Scenario Demo - Trajectory Visualization with Surrounding Vehicles.

This script provides unified scenario visualization combining:
1. Ego trajectory with future waypoints (action visualization)
2. Surrounding vehicle detection and tracking
3. Control signal computation (steering, throttle)
4. Statistics and analysis

Modes:
    - topdown:  Top-down BEV visualization with trajectory and surrounding info
    - 3d:       Full 3D MetaDrive rendering
    - stats:    Generate statistics without interactive display

Features:
    - Ego trajectory with future waypoints
    - Surrounding vehicle positions, velocities, headings
    - Distance-based coloring (close=red, medium=orange, far=cyan)
    - Control signal display (steering, throttle/brake)
    - Statistics plotting (vehicle count, distances, positions)

Usage:
    # Top-down view with trajectory and surrounding vehicles
    python -m meta_qa.scripts.demo_vis.scenario_demo
    
    # 3D view
    python -m meta_qa.scripts.demo_vis.scenario_demo --mode 3d
    
    # Custom trajectory horizon
    python -m meta_qa.scripts.demo_vis.scenario_demo --horizon 3.0
    
    # Custom detection radius
    python -m meta_qa.scripts.demo_vis.scenario_demo --detection-radius 40
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, List, Dict, Tuple, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Default paths
DEFAULT_DATA_DIR = os.path.join(
    project_root,
    "dataset", "Scenario_Data", "exp_nuscenes_converted"
)
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "outputs", "vis")

# Trajectory parameters
TRAJECTORY_DT = 0.1  # 0.1s interval
DEFAULT_HORIZON = 2.0  # 2 seconds


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Scenario Demo with trajectory and surrounding vehicle visualization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Top-down view (default)
    python -m meta_qa.scripts.demo_vis.scenario_demo
    
    # 3D view
    python -m meta_qa.scripts.demo_vis.scenario_demo --mode 3d
    
    # Custom horizon and detection radius
    python -m meta_qa.scripts.demo_vis.scenario_demo --horizon 3.0 --detection-radius 40
"""
    )
    
    # Mode
    parser.add_argument("--mode", type=str, default="topdown",
                       choices=["topdown", "3d", "stats"],
                       help="Visualization mode")
    
    # Data path
    parser.add_argument("--data-dir", "-d", type=str, default=DEFAULT_DATA_DIR,
                       help="Path to ScenarioNet dataset directory")
    
    # Scenario selection
    parser.add_argument("--scenario-index", type=int, default=None,
                       help="Specific scenario index (default: random)")
    parser.add_argument("--num-scenarios", type=int, default=1,
                       help="Number of scenarios to load")
    
    # Trajectory parameters
    parser.add_argument("--horizon", type=float, default=DEFAULT_HORIZON,
                       help="Trajectory horizon in seconds (default: 2.0)")
    
    # Surrounding vehicle parameters
    parser.add_argument("--detection-radius", type=float, default=50.0,
                       help="Detection radius for surrounding vehicles in meters (default: 50)")
    parser.add_argument("--max-vehicles", type=int, default=10,
                       help="Maximum surrounding vehicles to track (default: 10)")
    
    # Output options
    parser.add_argument("--save-gif", action="store_false", default=True,
                       help="Save visualization as GIF")
    parser.add_argument("--plot-stats", action="store_true", default=True,
                       help="Plot surrounding vehicle statistics")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory")
    
    # Display options
    parser.add_argument("--max-steps", type=int, default=300,
                       help="Maximum simulation steps")
    
    return parser.parse_args()


def draw_frame_overlay(frame: np.ndarray,
                       ego_pos: np.ndarray,
                       ego_heading: float,
                       ego_vel: np.ndarray,
                       trajectory: Optional[np.ndarray],
                       is_complete: bool,
                       steering: float,
                       throttle: float,
                       surrounding_info,
                       env,
                       step: int,
                       horizon: float = 2.0) -> np.ndarray:
    """
    Draw comprehensive overlay on top-down frame including:
    - Ego trajectory (future waypoints)
    - Surrounding vehicle information
    - Control signals
    
    Args:
        frame: Top-down render frame (H, W, 3)
        ego_pos: Ego vehicle position (world coordinates)
        ego_heading: Ego vehicle heading
        ego_vel: Ego vehicle velocity
        trajectory: Future trajectory waypoints
        is_complete: Whether trajectory is complete
        steering: Current steering value
        throttle: Current throttle value
        surrounding_info: SurroundingVehicleInfo object
        env: MetaDrive environment
        step: Current step number
        horizon: Trajectory horizon in seconds
        
    Returns:
        Frame with overlay drawn
    """
    import cv2
    
    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2
    
    # Get scaling from renderer
    try:
        renderer = env.engine.top_down_renderer
        pixels_per_meter = renderer.scaling if renderer and hasattr(renderer, 'scaling') else 5.0
    except:
        pixels_per_meter = 5.0
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.4
    font_thickness = 1
    
    # === Draw Ego Info (top-left) ===
    ego_speed = np.linalg.norm(ego_vel)
    cv2.putText(frame, f"Step: {step}", (10, 15), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(frame, f"Ego Pos: ({ego_pos[0]:.1f}, {ego_pos[1]:.1f})", (10, 30), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(frame, f"Ego Vel: ({ego_vel[0]:.1f}, {ego_vel[1]:.1f}) m/s", (10, 45), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(frame, f"Ego Speed: {ego_speed:.1f} m/s", (10, 60), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(frame, f"Ego Heading: {np.degrees(ego_heading):.1f} deg", (10, 75), font, font_scale, (0, 255, 0), font_thickness)
    
    # === Draw Trajectory (Action) ===
    if trajectory is not None and len(trajectory) >= 2:
        # Convert waypoints to image coordinates
        points = []
        for wp in trajectory:
            rel_x = wp[0] - ego_pos[0]
            rel_y = wp[1] - ego_pos[1]
            img_x = int(center_x + rel_x * pixels_per_meter)
            img_y = int(center_y - rel_y * pixels_per_meter)
            img_x = max(0, min(w - 1, img_x))
            img_y = max(0, min(h - 1, img_y))
            points.append((img_x, img_y))
        
        # Color: green if complete, orange if truncated
        traj_color = (0, 255, 0) if is_complete else (0, 165, 255)
        
        # Draw trajectory line
        for i in range(len(points) - 1):
            cv2.line(frame, points[i], points[i + 1], traj_color, 1, cv2.LINE_AA)
        
        # Draw waypoint circles
        for i, pt in enumerate(points):
            radius = 3 if i % 5 == 0 else 2
            cv2.circle(frame, pt, radius, traj_color, -1, cv2.LINE_AA)
        
        # Arrow at end
        if len(points) >= 2:
            cv2.arrowedLine(frame, points[-2], points[-1], traj_color, 1, cv2.LINE_AA, tipLength=0.3)
    
    # === Draw Control Info ===
    y_offset = 95
    
    # Trajectory info
    if trajectory is not None:
        if is_complete:
            text = f"Action: {len(trajectory)} pts ({horizon:.1f}s)"
            color = (0, 255, 0)
        else:
            text = f"Action: {len(trajectory)} pts (truncated)"
            color = (0, 165, 255)
        cv2.putText(frame, text, (10, y_offset), font, font_scale, color, font_thickness, cv2.LINE_AA)
        y_offset += 15
    
    # Steering
    steer_text = f"Steering: {steering:+.3f}"
    cv2.putText(frame, steer_text, (10, y_offset), font, font_scale, (0, 255, 0), font_thickness, cv2.LINE_AA)
    y_offset += 15
    
    # Throttle/Brake
    if throttle >= 0:
        throttle_text = f"Throttle: {throttle:.3f}"
        throttle_color = (0, 255, 0)
    else:
        throttle_text = f"Brake: {abs(throttle):.3f}"
        throttle_color = (0, 0, 255)
    cv2.putText(frame, throttle_text, (10, y_offset), font, font_scale, throttle_color, font_thickness, cv2.LINE_AA)
    y_offset += 20
    
    # === Draw Surrounding Vehicles ===
    if surrounding_info is not None:
        cv2.putText(frame, f"Surrounding: {surrounding_info.num_vehicles}", 
                   (10, y_offset), font, font_scale, (255, 255, 0), font_thickness)
        
        for i in range(surrounding_info.num_vehicles):
            veh_pos = surrounding_info.positions[i]
            veh_vel = surrounding_info.velocities[i]
            distance = surrounding_info.distances[i]
            
            # Convert to image coordinates
            rel_x = veh_pos[0] - ego_pos[0]
            rel_y = veh_pos[1] - ego_pos[1]
            img_x = int(center_x + rel_x * pixels_per_meter)
            img_y = int(center_y - rel_y * pixels_per_meter)
            img_x = max(20, min(w - 20, img_x))
            img_y = max(20, min(h - 20, img_y))
            
            # Color based on distance
            if distance < 10:
                color = (0, 0, 255)  # Red - close
            elif distance < 25:
                color = (0, 165, 255)  # Orange - medium
            else:
                color = (255, 255, 0)  # Cyan - far
            
            # Draw vehicle marker
            cv2.circle(frame, (img_x, img_y), 8, color, 2)
            cv2.putText(frame, str(i), (img_x - 3, img_y + 3), font, 0.3, color, 1)
            
            # Draw velocity vector
            veh_speed = np.linalg.norm(veh_vel)
            if veh_speed > 0.5:
                vel_scale = 0.5
                vel_end_x = int(img_x + veh_vel[0] * vel_scale * pixels_per_meter)
                vel_end_y = int(img_y - veh_vel[1] * vel_scale * pixels_per_meter)
                cv2.arrowedLine(frame, (img_x, img_y), (vel_end_x, vel_end_y), color, 1, tipLength=0.3)
            
            # Draw info box for first few vehicles (right side)
            if i < 5:
                info_y = 15 + i * 80
                info_x = w - 200
                
                cv2.rectangle(frame, (info_x - 5, info_y - 12), (w - 5, info_y + 58), (50, 50, 50), -1)
                cv2.rectangle(frame, (info_x - 5, info_y - 12), (w - 5, info_y + 58), color, 1)
                
                cv2.putText(frame, f"Vehicle {i}:", (info_x, info_y), font, font_scale, color, font_thickness)
                cv2.putText(frame, f"  Dist: {distance:.1f}m", (info_x, info_y + 12), font, font_scale, (255, 255, 255), font_thickness)
                cv2.putText(frame, f"  Pos: ({veh_pos[0]:.1f}, {veh_pos[1]:.1f})", (info_x, info_y + 24), font, font_scale, (255, 255, 255), font_thickness)
                
                rel_pos = surrounding_info.relative_positions[i]
                cv2.putText(frame, f"  Local: ({rel_pos[0]:.1f}, {rel_pos[1]:.1f})", (info_x, info_y + 36), font, font_scale, (200, 200, 255), font_thickness)
                cv2.putText(frame, f"  Vel: ({veh_vel[0]:.1f}, {veh_vel[1]:.1f})", (info_x, info_y + 48), font, font_scale, (255, 255, 255), font_thickness)
    
    # Draw detection radius circle
    radius_pixels = int(50 * pixels_per_meter)
    cv2.circle(frame, (center_x, center_y), radius_pixels, (100, 100, 100), 1)
    
    # Draw coordinate axes
    axis_len = 30
    cv2.arrowedLine(frame, (center_x, center_y), 
                   (center_x + int(axis_len * np.cos(-ego_heading)), 
                    center_y + int(axis_len * np.sin(-ego_heading))), 
                   (0, 255, 0), 2, tipLength=0.2)
    cv2.putText(frame, "X", (center_x + int(axis_len * np.cos(-ego_heading)) + 5, 
                             center_y + int(axis_len * np.sin(-ego_heading))), 
               font, 0.3, (0, 255, 0), 1)
    
    cv2.arrowedLine(frame, (center_x, center_y),
                   (center_x + int(axis_len * np.cos(-ego_heading + np.pi/2)),
                    center_y + int(axis_len * np.sin(-ego_heading + np.pi/2))),
                   (255, 0, 0), 2, tipLength=0.2)
    cv2.putText(frame, "Y", (center_x + int(axis_len * np.cos(-ego_heading + np.pi/2)) + 5,
                             center_y + int(axis_len * np.sin(-ego_heading + np.pi/2))),
               font, 0.3, (255, 0, 0), 1)
    
    return frame


def get_future_trajectory(env, num_waypoints: int) -> Tuple[np.ndarray, bool]:
    """
    Get future trajectory from replay data.
    
    Returns:
        trajectory: Future trajectory waypoints
        is_complete: Whether trajectory is complete
    """
    ego = env.agent
    
    try:
        if hasattr(env, 'engine') and hasattr(env.engine, 'data_manager'):
            scenario = env.engine.data_manager.current_scenario
            if scenario is not None:
                tracks = scenario.get('tracks', {})
                sdc_id = scenario.get('metadata', {}).get('sdc_id', 'sdc')
                
                ego_track = tracks.get(sdc_id) or tracks.get('ego') or tracks.get('sdc')
                
                if ego_track is None:
                    for track_id, track in tracks.items():
                        if track.get('type', '') == 'VEHICLE':
                            ego_track = track
                            break
                
                if ego_track is not None:
                    state = ego_track.get('state', {})
                    positions = state.get('position')
                    
                    if positions is not None:
                        current_step = env.engine.episode_step if hasattr(env.engine, 'episode_step') else 0
                        
                        future_start = current_step + 1
                        future_end = current_step + 1 + num_waypoints
                        
                        if future_end <= len(positions):
                            future_traj = positions[future_start:future_end, :2]
                            return np.array(future_traj), True
                        else:
                            remaining_traj = positions[future_start:, :2]
                            if len(remaining_traj) >= 2:
                                return np.array(remaining_traj), False
    except Exception:
        pass
    
    return np.array([[ego.position[0], ego.position[1]]]), False


def compute_control_from_trajectory(trajectory: np.ndarray, 
                                    current_pos: np.ndarray,
                                    current_heading: float,
                                    current_velocity: np.ndarray,
                                    is_complete: bool) -> Tuple[float, float]:
    """Compute control signals from trajectory."""
    if len(trajectory) < 2:
        return 0.0, -0.5
    
    try:
        from meta_qa.core.trajectory_tracker import TrajectoryTracker, VehicleState
        
        tracker = TrajectoryTracker()
        vehicle_state = VehicleState(
            position=np.array(current_pos),
            heading=current_heading,
            speed=np.linalg.norm(current_velocity),
            velocity=np.array(current_velocity),
        )
        
        steering, throttle = tracker.track(trajectory, vehicle_state)
        
        if not is_complete and len(trajectory) < 5:
            remaining_time = len(trajectory) * 0.1
            if remaining_time < 0.5:
                blend_factor = remaining_time / 0.5
                throttle = throttle * blend_factor - 0.2 * (1 - blend_factor)
        
        return float(steering), float(throttle)
    except:
        return 0.0, 0.0


def plot_statistics(position_history: List[np.ndarray],
                   control_history: List[Tuple[float, float]],
                   surrounding_history: List,
                   reward_history: List[float],
                   output_dir: str):
    """Plot comprehensive statistics."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    fig.suptitle('Scenario Statistics - Trajectory & Surrounding Vehicles', fontsize=14, fontweight='bold')
    
    # 1. Bird's Eye View
    ax1 = axes[0, 0]
    if position_history:
        positions = np.array(position_history)
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
    ax2 = axes[0, 1]
    if control_history:
        steering = np.array([c[0] for c in control_history])
        time_steps = np.arange(len(steering)) * 0.1
        ax2.plot(time_steps, steering, 'b-', linewidth=1.5)
        ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax2.fill_between(time_steps, steering, 0, alpha=0.3)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Steering [-1, 1]')
    ax2.set_title('Steering Signal')
    ax2.set_ylim(-1.1, 1.1)
    ax2.grid(True, alpha=0.3)
    
    # 3. Throttle/Brake Signal
    ax3 = axes[0, 2]
    if control_history:
        throttle = np.array([c[1] for c in control_history])
        time_steps = np.arange(len(throttle)) * 0.1
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
    
    # 4. Number of Surrounding Vehicles
    ax4 = axes[1, 0]
    if surrounding_history:
        num_vehicles = [info.num_vehicles for info in surrounding_history]
        steps = list(range(len(num_vehicles)))
        ax4.plot(steps, num_vehicles, 'b-', linewidth=1.5)
        ax4.set_xlabel('Step')
        ax4.set_ylabel('Number of Vehicles')
        ax4.set_title('Number of Surrounding Vehicles')
        ax4.grid(True, alpha=0.3)
    
    # 5. Minimum Distance
    ax5 = axes[1, 1]
    if surrounding_history:
        min_distances = []
        for info in surrounding_history:
            if info.num_vehicles > 0:
                min_distances.append(np.min(info.distances))
            else:
                min_distances.append(np.nan)
        steps = list(range(len(min_distances)))
        ax5.plot(steps, min_distances, 'r-', linewidth=1.5)
        ax5.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5m')
        ax5.axhline(y=10, color='yellow', linestyle='--', alpha=0.7, label='10m')
    ax5.set_xlabel('Step')
    ax5.set_ylabel('Distance (m)')
    ax5.set_title('Minimum Distance to Surrounding Vehicle')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. Distance Distribution
    ax6 = axes[1, 2]
    if surrounding_history:
        all_distances = []
        for info in surrounding_history:
            if info.num_vehicles > 0:
                all_distances.extend(info.distances.tolist())
        if all_distances:
            ax6.hist(all_distances, bins=30, color='green', alpha=0.7, edgecolor='black')
            ax6.axvline(x=np.mean(all_distances), color='red', linestyle='--', label=f'Mean: {np.mean(all_distances):.1f}m')
    ax6.set_xlabel('Distance (m)')
    ax6.set_ylabel('Frequency')
    ax6.set_title('Distance Distribution')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    # 7. Relative Positions Scatter
    ax7 = axes[2, 0]
    if surrounding_history:
        all_rel_x, all_rel_y = [], []
        for info in surrounding_history:
            if info.num_vehicles > 0:
                all_rel_x.extend(info.relative_positions[:, 0].tolist())
                all_rel_y.extend(info.relative_positions[:, 1].tolist())
        if all_rel_x:
            ax7.scatter(all_rel_x, all_rel_y, alpha=0.3, s=5, c='blue')
            ax7.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
            ax7.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
            ax7.scatter([0], [0], c='red', s=100, marker='*', label='Ego', zorder=5)
    ax7.set_xlabel('Relative X (forward) [m]')
    ax7.set_ylabel('Relative Y (left) [m]')
    ax7.set_title('Relative Positions (Ego Frame)')
    ax7.set_xlim(-50, 50)
    ax7.set_ylim(-30, 30)
    ax7.legend()
    ax7.grid(True, alpha=0.3)
    
    # 8. Speed
    ax8 = axes[2, 1]
    if position_history and len(position_history) > 1:
        positions = np.array(position_history)
        speeds = np.linalg.norm(np.diff(positions, axis=0), axis=1) / 0.1
        speeds = np.append(speeds, speeds[-1])
        time_steps = np.arange(len(speeds)) * 0.1
        ax8.plot(time_steps, speeds, 'purple', linewidth=1.5)
        ax8.axhline(y=np.mean(speeds), color='orange', linestyle='--', alpha=0.7, label=f'Avg: {np.mean(speeds):.1f} m/s')
    ax8.set_xlabel('Time (s)')
    ax8.set_ylabel('Speed (m/s)')
    ax8.set_title('Speed Curve')
    ax8.legend()
    ax8.grid(True, alpha=0.3)
    
    # 9. Cumulative Reward
    ax9 = axes[2, 2]
    if reward_history:
        rewards = np.array(reward_history)
        cumulative_reward = np.cumsum(rewards)
        time_steps = np.arange(len(cumulative_reward)) * 0.1
        ax9.plot(time_steps, cumulative_reward, 'darkgreen', linewidth=2)
        ax9.fill_between(time_steps, cumulative_reward, 0, alpha=0.2, color='green')
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Cumulative Reward')
    ax9.set_title(f'Cumulative Reward (total={cumulative_reward[-1]:.4f})' if reward_history else 'Cumulative Reward')
    ax9.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    save_path = os.path.join(output_dir, "scenario_statistics.png")
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Statistics saved to: {save_path}")
    
    plt.show()


def run_demo(args):
    """Run the unified scenario demo."""
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    from meta_qa.tools import SurroundingVehicleGetter
    from meta_qa.tools import SurroundingVehicleInfo
    import glob
    import random
    
    # Calculate waypoints
    num_waypoints = int(args.horizon / TRAJECTORY_DT)
    
    # Count scenarios
    scenario_files = glob.glob(os.path.join(args.data_dir, "*.pkl"))
    total_scenarios = len(scenario_files)
    
    # Select scenario
    if args.scenario_index is not None:
        start_index = args.scenario_index
    else:
        start_index = random.randint(0, max(0, total_scenarios - 1))
    
    print("=" * 60)
    print("Unified Scenario Demo")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Selected scenario: {start_index}")
    print(f"Trajectory horizon: {args.horizon}s ({num_waypoints} waypoints)")
    print(f"Detection radius: {args.detection_radius}m")
    print(f"Max vehicles: {args.max_vehicles}")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Configure environment
    config = dict(
        agent_policy=ReplayEgoCarPolicy,
        use_render=(args.mode == "3d"),
        data_directory=args.data_dir,
        num_scenarios=args.num_scenarios,
        start_scenario_index=start_index,
        no_traffic=False,
        reactive_traffic=False,
    )
    
    if args.mode == "3d":
        config["window_size"] = (1200, 800)
    
    print("\nInitializing environment...")
    env = ScenarioEnv(config)
    
    # Create surrounding getter
    surrounding_getter = SurroundingVehicleGetter(
        max_vehicles=args.max_vehicles,
        detection_radius=args.detection_radius,
        include_parked=False
    )
    
    # Storage
    frames = []
    position_history = []
    control_history = []
    surrounding_history = []
    reward_history = []
    
    # Setup display
    if args.mode == "topdown":
        import pygame
        pygame.init()
        screen = pygame.display.set_mode((800, 800))
        pygame.display.set_caption("Scenario Demo - Trajectory & Surrounding")
        clock = pygame.time.Clock()
    
    try:
        obs, info = env.reset()
        done = False
        step = 0
        
        print("\nPress ESC to exit, SPACE to pause/resume\n")
        print("-" * 60)
        
        paused = False
        running = True
        
        while not done and running and step < args.max_steps:
            # Handle events
            if args.mode == "topdown":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        elif event.key == pygame.K_SPACE:
                            paused = not paused
                            print("PAUSED" if paused else "RESUMED")
                
                if paused:
                    clock.tick(10)
                    continue
            
            # Get ego state
            ego = env.agent
            ego_pos = np.array([ego.position[0], ego.position[1]])
            ego_heading = ego.heading_theta
            ego_vel = np.array([ego.velocity[0], ego.velocity[1]]) if hasattr(ego, 'velocity') else np.array([0, 0])
            
            # Get trajectory
            trajectory, is_complete = get_future_trajectory(env, num_waypoints)
            
            # Compute control
            steering, throttle = compute_control_from_trajectory(
                trajectory, ego_pos, ego_heading, ego_vel, is_complete
            )
            
            # Get surrounding info
            surrounding_info = surrounding_getter.get_surrounding_vehicles(env)
            
            # Record data
            position_history.append(ego_pos.copy())
            control_history.append((steering, throttle))
            surrounding_history.append(surrounding_info)
            
            # Render and display
            if args.mode == "topdown":
                try:
                    frame = env.render(mode="topdown",
                                      film_size=(4000, 4000),
                                      screen_size=(800, 800),
                                      draw_contour=True)
                    
                    if frame is not None and isinstance(frame, np.ndarray):
                        frame_with_overlay = draw_frame_overlay(
                            frame.copy(),
                            ego_pos, ego_heading, ego_vel,
                            trajectory, is_complete,
                            steering, throttle,
                            surrounding_info, env, step,
                            horizon=args.horizon
                        )
                        frames.append(frame_with_overlay)
                        
                        # Display
                        import pygame
                        surf = pygame.surfarray.make_surface(frame_with_overlay.swapaxes(0, 1))
                        screen.blit(surf, (0, 0))
                        pygame.display.flip()
                except Exception as e:
                    if step == 0:
                        print(f"Warning: Could not render frame: {e}")
            elif args.mode == "3d":
                env.render(mode="topdown")
            
            # Print progress
            if step % 20 == 0:
                status = "OK" if is_complete else "TRUNC"
                print(f"Step {step}: pos=({ego_pos[0]:.1f}, {ego_pos[1]:.1f}), "
                      f"surr={surrounding_info.num_vehicles}, [{status}]")
            
            # Step
            action = np.array([steering, throttle])
            obs, reward, terminated, truncated, info = env.step(action)
            reward_history.append(reward)
            
            # Check replay completion (NOT arrive_dest or other termination)
            # We use replay_done as the ending criterion
            if info.get("replay_done", False):
                print(f"\nReplay completed at step {step}")
                done = True
            elif terminated or truncated:
                # Log but don't end - continue until replay_done
                term_reason = info.get("arrive_dest", False) and "arrive_dest" or \
                             info.get("crash", False) and "crash" or \
                             info.get("out_of_road", False) and "out_of_road" or "unknown"
                if step % 20 == 0:  # Don't spam
                    print(f"  [Note] terminated={terminated}, reason={term_reason}, continuing replay...")
                # Reset environment state if needed but keep replaying
                # Don't set done = True
            
            step += 1
            
            if args.mode == "topdown":
                clock.tick(30)
        
        print("-" * 60)
        print(f"\nCompleted! Total steps: {step}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        if args.mode == "topdown":
            import pygame
            pygame.quit()
        env.close()
    
    # Save GIF
    if args.save_gif and frames:
        from PIL import Image
        
        gif_path = os.path.join(args.output_dir, "scenario_demo.gif")
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
    
    # Plot statistics
    if args.plot_stats and (position_history or surrounding_history):
        plot_statistics(
            position_history, control_history, 
            surrounding_history, reward_history,
            args.output_dir
        )
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


def main():
    args = parse_args()
    
    # Convert relative paths
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.join(project_root, args.data_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    run_demo(args)


if __name__ == "__main__":
    main()
