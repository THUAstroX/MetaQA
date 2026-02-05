"""
Visualize Surrounding Vehicle Information.

This script renders the scenario replay and draws surrounding vehicle 
information on the frame to verify the data collection is correct.

Usage:
    # Default
    python -m meta_qa.scripts.demo_vis.scenario_info_demo
    
    # Custom data directory
    python -m meta_qa.scripts.demo_vis.scenario_info_demo --data_dir /path/to/data
    
    # Save as GIF
    python -m meta_qa.scripts.demo_vis.scenario_info_demo --save_gif
    
    # Custom settings
    python -m meta_qa.scripts.demo_vis.scenario_info_demo --num_scenarios 5 --detection_radius 30
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, List, Dict, Tuple

# Add paths
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from meta_qa.tools import SurroundingVehicleGetter
from meta_qa.data import SurroundingVehicleInfo

# Data directory (default)
DATA_DIR = os.path.join(
    project_root,
    "dataset", "Scenario_Data", "exp_nuscenes_converted"
)


def draw_surrounding_info_on_frame(frame: np.ndarray,
                                    ego_pos: np.ndarray,
                                    ego_heading: float,
                                    ego_vel: np.ndarray,
                                    surrounding_info: SurroundingVehicleInfo,
                                    env,
                                    step: int) -> np.ndarray:
    """
    Draw surrounding vehicle information on the top-down frame.
    
    Args:
        frame: Top-down render frame (H, W, 3)
        ego_pos: Ego vehicle position (world coordinates)
        ego_heading: Ego vehicle heading
        ego_vel: Ego vehicle velocity
        surrounding_info: SurroundingVehicleInfo object
        env: MetaDrive environment
        step: Current step number
        
    Returns:
        Frame with surrounding vehicle info drawn
    """
    import cv2
    
    h, w = frame.shape[:2]
    center_x = w // 2
    center_y = h // 2
    
    # Get scaling from renderer
    try:
        renderer = env.engine.top_down_renderer
        if renderer is not None:
            pixels_per_meter = renderer.scaling if hasattr(renderer, 'scaling') else 5.0
        else:
            pixels_per_meter = 5.0
    except:
        pixels_per_meter = 5.0
    
    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.35
    font_thickness = 1
    
    # Draw ego vehicle info (top-left corner)
    ego_speed = np.linalg.norm(ego_vel)
    cv2.putText(frame, f"Step: {step}", (10, 15), font, font_scale, (255, 255, 255), font_thickness)
    cv2.putText(frame, f"Ego Pos: ({ego_pos[0]:.1f}, {ego_pos[1]:.1f})", (10, 30), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(frame, f"Ego Vel: ({ego_vel[0]:.1f}, {ego_vel[1]:.1f}) m/s", (10, 45), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(frame, f"Ego Speed: {ego_speed:.1f} m/s", (10, 60), font, font_scale, (0, 255, 0), font_thickness)
    cv2.putText(frame, f"Ego Heading: {np.degrees(ego_heading):.1f} deg", (10, 75), font, font_scale, (0, 255, 0), font_thickness)
    
    # Draw surrounding vehicles count
    cv2.putText(frame, f"Surrounding Vehicles: {surrounding_info.num_vehicles}", (10, 95), font, font_scale, (255, 255, 0), font_thickness)
    
    # Draw each surrounding vehicle
    for i in range(surrounding_info.num_vehicles):
        # World position
        veh_pos = surrounding_info.positions[i]
        veh_vel = surrounding_info.velocities[i]
        veh_heading = surrounding_info.headings[i]
        distance = surrounding_info.distances[i]
        
        # Relative position (in ego's local frame)
        rel_pos = surrounding_info.relative_positions[i]
        rel_vel = surrounding_info.relative_velocities[i]
        
        # Convert world position to image coordinates
        rel_x = veh_pos[0] - ego_pos[0]
        rel_y = veh_pos[1] - ego_pos[1]
        
        img_x = int(center_x + rel_x * pixels_per_meter)
        img_y = int(center_y - rel_y * pixels_per_meter)  # Flip y
        
        # Clamp to frame bounds
        img_x = max(20, min(w - 20, img_x))
        img_y = max(20, min(h - 20, img_y))
        
        # Color based on distance (red=close, yellow=medium, blue=far)
        if distance < 10:
            color = (0, 0, 255)  # Red - close
        elif distance < 25:
            color = (0, 165, 255)  # Orange - medium
        else:
            color = (255, 255, 0)  # Cyan - far
        
        # Draw vehicle marker (circle with ID)
        cv2.circle(frame, (img_x, img_y), 8, color, 2)
        cv2.putText(frame, str(i), (img_x - 3, img_y + 3), font, 0.3, color, 1)
        
        # Draw velocity vector
        veh_speed = np.linalg.norm(veh_vel)
        if veh_speed > 0.5:
            vel_scale = 0.5  # Scale for visualization
            vel_end_x = int(img_x + veh_vel[0] * vel_scale * pixels_per_meter)
            vel_end_y = int(img_y - veh_vel[1] * vel_scale * pixels_per_meter)
            cv2.arrowedLine(frame, (img_x, img_y), (vel_end_x, vel_end_y), color, 1, tipLength=0.3)
        
        # Draw info box for first few vehicles (on the right side)
        if i < 5:
            info_y = 15 + i * 80
            info_x = w - 200
            
            # Background box
            cv2.rectangle(frame, (info_x - 5, info_y - 12), (w - 5, info_y + 58), (50, 50, 50), -1)
            cv2.rectangle(frame, (info_x - 5, info_y - 12), (w - 5, info_y + 58), color, 1)
            
            # Vehicle info
            cv2.putText(frame, f"Vehicle {i}:", (info_x, info_y), font, font_scale, color, font_thickness)
            cv2.putText(frame, f"  Dist: {distance:.1f}m", (info_x, info_y + 12), font, font_scale, (255, 255, 255), font_thickness)
            cv2.putText(frame, f"  World: ({veh_pos[0]:.1f}, {veh_pos[1]:.1f})", (info_x, info_y + 24), font, font_scale, (255, 255, 255), font_thickness)
            cv2.putText(frame, f"  Local: ({rel_pos[0]:.1f}, {rel_pos[1]:.1f})", (info_x, info_y + 36), font, font_scale, (200, 200, 255), font_thickness)
            cv2.putText(frame, f"  Vel: ({veh_vel[0]:.1f}, {veh_vel[1]:.1f})", (info_x, info_y + 48), font, font_scale, (255, 255, 255), font_thickness)
    
    # Draw detection radius circle
    radius_pixels = int(30 * pixels_per_meter)  # 30m default radius
    cv2.circle(frame, (center_x, center_y), radius_pixels, (100, 100, 100), 1)
    
    # Draw coordinate axes at ego position
    axis_len = 30
    # X axis (forward) - green
    cv2.arrowedLine(frame, (center_x, center_y), 
                   (center_x + int(axis_len * np.cos(-ego_heading)), 
                    center_y + int(axis_len * np.sin(-ego_heading))), 
                   (0, 255, 0), 2, tipLength=0.2)
    cv2.putText(frame, "X", (center_x + int(axis_len * np.cos(-ego_heading)) + 5, 
                             center_y + int(axis_len * np.sin(-ego_heading))), 
               font, 0.3, (0, 255, 0), 1)
    
    # Y axis (left) - blue  
    cv2.arrowedLine(frame, (center_x, center_y),
                   (center_x + int(axis_len * np.cos(-ego_heading + np.pi/2)),
                    center_y + int(axis_len * np.sin(-ego_heading + np.pi/2))),
                   (255, 0, 0), 2, tipLength=0.2)
    cv2.putText(frame, "Y", (center_x + int(axis_len * np.cos(-ego_heading + np.pi/2)) + 5,
                             center_y + int(axis_len * np.sin(-ego_heading + np.pi/2))),
               font, 0.3, (255, 0, 0), 1)
    
    return frame


def print_surrounding_info(step: int, 
                           ego_pos: np.ndarray,
                           ego_vel: np.ndarray,
                           surrounding_info: SurroundingVehicleInfo):
    """Print surrounding vehicle info to console."""
    ego_speed = np.linalg.norm(ego_vel)
    
    print(f"\n{'='*60}")
    print(f"Step {step}")
    print(f"{'='*60}")
    print(f"Ego Position: ({ego_pos[0]:.2f}, {ego_pos[1]:.2f}) m")
    print(f"Ego Velocity: ({ego_vel[0]:.2f}, {ego_vel[1]:.2f}) m/s, Speed: {ego_speed:.2f} m/s")
    print(f"Surrounding Vehicles: {surrounding_info.num_vehicles}")
    print("-" * 60)
    
    for i in range(surrounding_info.num_vehicles):
        print(f"\n  Vehicle {i}:")
        print(f"    Distance:     {surrounding_info.distances[i]:.2f} m")
        print(f"    World Pos:    ({surrounding_info.positions[i][0]:.2f}, {surrounding_info.positions[i][1]:.2f})")
        print(f"    World Vel:    ({surrounding_info.velocities[i][0]:.2f}, {surrounding_info.velocities[i][1]:.2f})")
        print(f"    Local Pos:    ({surrounding_info.relative_positions[i][0]:.2f}, {surrounding_info.relative_positions[i][1]:.2f})")
        print(f"    Local Vel:    ({surrounding_info.relative_velocities[i][0]:.2f}, {surrounding_info.relative_velocities[i][1]:.2f})")
        print(f"    Heading:      {np.degrees(surrounding_info.headings[i]):.1f} deg")


def run_visualization(args):
    """Run the surrounding vehicle visualization."""
    from metadrive.envs.scenario_env import ScenarioEnv
    from metadrive.policy.replay_policy import ReplayEgoCarPolicy
    import random
    import glob
    
    # Count available scenarios
    scenario_files = glob.glob(os.path.join(args.data_dir, "*.pkl"))
    total_scenarios = len(scenario_files)
    
    # Random scenario selection
    if total_scenarios > 0:
        start_index = random.randint(0, total_scenarios - 1)
        print(f"Randomly selected scenario index: {start_index} / {total_scenarios}")
    else:
        start_index = 0
    
    print("=" * 60)
    print("Surrounding Vehicle Visualization")
    print("=" * 60)
    print(f"Data directory: {args.data_dir}")
    print(f"Total scenarios available: {total_scenarios}")
    print(f"Starting scenario index: {start_index}")
    print(f"Detection radius: {args.detection_radius}m")
    print(f"Max vehicles: {args.max_vehicles}")
    print("=" * 60)
    
    # Create environment
    config = dict(
        agent_policy=ReplayEgoCarPolicy,
        use_render=False,
        data_directory=args.data_dir,
        num_scenarios=args.num_scenarios,
        start_scenario_index=start_index,
        no_traffic=False,
        reactive_traffic=False,
    )
    
    print("\nCreating environment...")
    env = ScenarioEnv(config)
    
    # Create surrounding vehicle getter
    surrounding_getter = SurroundingVehicleGetter(
        max_vehicles=args.max_vehicles,
        detection_radius=args.detection_radius,
        include_parked=False
    )
    
    # Storage for recording
    frames = []
    surrounding_history = []
    
    # Use pygame for display
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((800, 800))
    pygame.display.set_caption("Surrounding Vehicle Visualization")
    clock = pygame.time.Clock()
    
    try:
        obs, info = env.reset()
        done = False
        step = 0
        
        print("\nPress ESC to exit, SPACE to pause/resume, P to print current info")
        print("-" * 60)
        
        paused = False
        running = True
        
        while not done and running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_SPACE:
                        paused = not paused
                        print("PAUSED" if paused else "RESUMED")
                    elif event.key == pygame.K_p:
                        # Print current surrounding info
                        ego = env.agent
                        ego_pos = np.array([ego.position[0], ego.position[1]])
                        ego_vel = np.array([ego.velocity[0], ego.velocity[1]])
                        surrounding_info = surrounding_getter.get_surrounding_vehicles(env)
                        print_surrounding_info(step, ego_pos, ego_vel, surrounding_info)
            
            if paused:
                clock.tick(10)
                continue
            
            # Get ego state
            ego = env.agent
            ego_pos = np.array([ego.position[0], ego.position[1]])
            ego_heading = ego.heading_theta
            ego_vel = np.array([ego.velocity[0], ego.velocity[1]])
            
            # Get surrounding vehicle info
            surrounding_info = surrounding_getter.get_surrounding_vehicles(env)
            surrounding_history.append(surrounding_info)
            
            # Get top-down frame
            try:
                frame = env.render(mode="topdown",
                                  film_size=(4000, 4000),
                                  screen_size=(800, 800),
                                  draw_contour=True)
                
                if frame is not None and isinstance(frame, np.ndarray):
                    # Draw surrounding vehicle info
                    frame_with_info = draw_surrounding_info_on_frame(
                        frame.copy(),
                        ego_pos, ego_heading, ego_vel,
                        surrounding_info, env, step
                    )
                    frames.append(frame_with_info)
                    
                    # Display with pygame
                    surf = pygame.surfarray.make_surface(frame_with_info.swapaxes(0, 1))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    
            except Exception as e:
                if step == 0:
                    print(f"Warning: Could not get frame: {e}")
            
            # Print progress every 20 steps
            if step % 20 == 0:
                print(f"Step {step}: Ego pos=({ego_pos[0]:.1f}, {ego_pos[1]:.1f}), "
                      f"Surrounding vehicles: {surrounding_info.num_vehicles}")
            
            # Step environment
            action = np.array([0.0, 0.0])  # ReplayPolicy ignores action
            obs, reward, terminated, truncated, info = env.step(action)
            step += 1
            
            # Use replay_done to determine when replay is complete
            # This is better than terminated/truncated for replay scenarios
            # because it ignores arrive_dest, crash, etc. and focuses on
            # whether the recorded trajectory data has been fully replayed
            if info.get("replay_done", False):
                print(f"  Replay completed at step {step}")
                done = True
            
            clock.tick(30)
        
        print("-" * 60)
        print(f"\nCompleted! Total steps: {step}")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    finally:
        pygame.quit()
        env.close()
    
    # Save GIF if requested
    if args.save_gif and frames:
        from PIL import Image
        
        output_dir = os.path.join(project_root, "outputs", "vis")
        os.makedirs(output_dir, exist_ok=True)
        gif_path = os.path.join(output_dir, "surrounding_vehicles.gif")
        
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
    
    # Plot surrounding vehicle statistics
    if args.plot_stats and surrounding_history:
        plot_surrounding_stats(surrounding_history, output_dir if args.save_gif else None)


def plot_surrounding_stats(surrounding_history: List[SurroundingVehicleInfo], output_dir: Optional[str] = None):
    """Plot statistics of surrounding vehicles over time."""
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    
    steps = list(range(len(surrounding_history)))
    num_vehicles = [info.num_vehicles for info in surrounding_history]
    
    # Min distance over time
    min_distances = []
    for info in surrounding_history:
        if info.num_vehicles > 0:
            min_distances.append(np.min(info.distances))
        else:
            min_distances.append(np.nan)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Surrounding Vehicle Statistics Over Time', fontsize=14)
    
    # Number of vehicles
    ax1 = axes[0, 0]
    ax1.plot(steps, num_vehicles, 'b-', linewidth=1.5)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Number of Vehicles')
    ax1.set_title('Number of Surrounding Vehicles')
    ax1.grid(True, alpha=0.3)
    
    # Minimum distance
    ax2 = axes[0, 1]
    ax2.plot(steps, min_distances, 'r-', linewidth=1.5)
    ax2.axhline(y=5, color='orange', linestyle='--', alpha=0.7, label='5m threshold')
    ax2.axhline(y=10, color='yellow', linestyle='--', alpha=0.7, label='10m threshold')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance (m)')
    ax2.set_title('Minimum Distance to Surrounding Vehicle')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Distribution of distances (histogram)
    ax3 = axes[1, 0]
    all_distances = []
    for info in surrounding_history:
        if info.num_vehicles > 0:
            all_distances.extend(info.distances.tolist())
    if all_distances:
        ax3.hist(all_distances, bins=30, color='green', alpha=0.7, edgecolor='black')
        ax3.axvline(x=np.mean(all_distances), color='red', linestyle='--', label=f'Mean: {np.mean(all_distances):.1f}m')
        ax3.set_xlabel('Distance (m)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Distances')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Relative positions scatter plot
    ax4 = axes[1, 1]
    all_rel_x = []
    all_rel_y = []
    for info in surrounding_history:
        if info.num_vehicles > 0:
            all_rel_x.extend(info.relative_positions[:, 0].tolist())
            all_rel_y.extend(info.relative_positions[:, 1].tolist())
    if all_rel_x:
        ax4.scatter(all_rel_x, all_rel_y, alpha=0.3, s=5, c='blue')
        ax4.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='-', alpha=0.5)
        ax4.set_xlabel('Relative X (forward) [m]')
        ax4.set_ylabel('Relative Y (left) [m]')
        ax4.set_title('Relative Positions (Ego Frame)')
        ax4.set_xlim(-50, 50)
        ax4.set_ylim(-30, 30)
        ax4.grid(True, alpha=0.3)
        # Mark ego at origin
        ax4.scatter([0], [0], c='red', s=100, marker='*', label='Ego', zorder=5)
        ax4.legend()
    
    plt.tight_layout()
    
    if output_dir:
        save_path = os.path.join(output_dir, "surrounding_stats.png")
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Statistics plot saved to: {save_path}")
    
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize surrounding vehicle information")
    parser.add_argument(
        "--data_dir", "-d",
        type=str,
        default=DATA_DIR,
        help="Path to ScenarioNet dataset"
    )
    parser.add_argument(
        "--num_scenarios", "-n",
        type=int,
        default=1,
        help="Number of scenarios to load"
    )
    parser.add_argument(
        "--detection_radius",
        type=float,
        default=50.0,
        help="Detection radius for surrounding vehicles (meters)"
    )
    parser.add_argument(
        "--max_vehicles",
        type=int,
        default=10,
        help="Maximum surrounding vehicles to track"
    )
    parser.add_argument(
        "--save_gif",
        action="store_true",
        default=True,
        help="Save visualization as GIF"
    )
    parser.add_argument(
        "--plot_stats",
        action="store_true",
        default=True,
        help="Plot surrounding vehicle statistics"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_visualization(args)
