"""
Trajectory Visualization for MetaDrive.

This module provides visualization utilities for displaying
predicted trajectories, ground truth trajectories, and vehicle paths.
"""

import numpy as np
from typing import Optional, List, Tuple, Dict
from meta_qa.core.config import (
    TRAJECTORY_COLOR_PREDICTED, TRAJECTORY_COLOR_ACTUAL,
    TRAJECTORY_COLOR_GROUND_TRUTH, TRAJECTORY_LINE_WIDTH,
    WAYPOINT_RADIUS
)


class TrajectoryVisualizer:
    """
    Visualizer for trajectories in MetaDrive.
    
    Supports both 2D top-down view and 3D rendering.
    """
    
    def __init__(self,
                 predicted_color: Tuple[int, int, int] = TRAJECTORY_COLOR_PREDICTED,
                 actual_color: Tuple[int, int, int] = TRAJECTORY_COLOR_ACTUAL,
                 ground_truth_color: Tuple[int, int, int] = TRAJECTORY_COLOR_GROUND_TRUTH,
                 line_width: int = TRAJECTORY_LINE_WIDTH,
                 waypoint_radius: int = WAYPOINT_RADIUS):
        """
        Initialize trajectory visualizer.
        
        Args:
            predicted_color: RGB color for predicted trajectory
            actual_color: RGB color for actual trajectory
            ground_truth_color: RGB color for ground truth trajectory
            line_width: Width of trajectory lines
            waypoint_radius: Radius of waypoint circles
        """
        self.predicted_color = predicted_color
        self.actual_color = actual_color
        self.ground_truth_color = ground_truth_color
        self.line_width = line_width
        self.waypoint_radius = waypoint_radius
        
        # History of actual positions
        self.position_history: List[np.ndarray] = []
        self.max_history_length = 100
    
    def reset(self):
        """Reset visualizer state."""
        self.position_history = []
    
    def update_position_history(self, position: np.ndarray):
        """
        Update position history.
        
        Args:
            position: Current vehicle position (2,)
        """
        self.position_history.append(position.copy())
        
        # Limit history length
        if len(self.position_history) > self.max_history_length:
            self.position_history.pop(0)
    
    def world_to_screen(self, world_pos: np.ndarray,
                       camera_center: np.ndarray,
                       scale: float,
                       screen_size: Tuple[int, int]) -> Tuple[int, int]:
        """
        Convert world coordinates to screen coordinates.
        
        Args:
            world_pos: Position in world coordinates (2,)
            camera_center: Camera center in world coordinates (2,)
            scale: Pixels per meter
            screen_size: Screen size (width, height)
            
        Returns:
            Screen coordinates (x, y)
        """
        # Offset from camera center
        offset = world_pos - camera_center
        
        # Convert to screen coordinates
        screen_x = int(screen_size[0] / 2 + offset[0] * scale)
        screen_y = int(screen_size[1] / 2 - offset[1] * scale)  # Y is flipped
        
        return screen_x, screen_y
    
    def draw_trajectory_pygame(self, surface,
                               trajectory: np.ndarray,
                               camera_center: np.ndarray,
                               scale: float,
                               color: Tuple[int, int, int],
                               draw_waypoints: bool = True):
        """
        Draw trajectory using pygame.
        
        Args:
            surface: Pygame surface to draw on
            trajectory: Trajectory in world coordinates (N, 2)
            camera_center: Camera center in world coordinates
            scale: Pixels per meter
            color: RGB color tuple
            draw_waypoints: Whether to draw waypoint circles
        """
        try:
            import pygame
        except ImportError:
            return
        
        if trajectory is None or len(trajectory) < 2:
            return
        
        screen_size = surface.get_size()
        
        # Convert trajectory points to screen coordinates
        screen_points = []
        for point in trajectory:
            screen_point = self.world_to_screen(point, camera_center, scale, screen_size)
            screen_points.append(screen_point)
        
        # Draw lines connecting waypoints
        if len(screen_points) >= 2:
            pygame.draw.lines(surface, color, False, screen_points, self.line_width)
        
        # Draw waypoint circles
        if draw_waypoints:
            for point in screen_points:
                pygame.draw.circle(surface, color, point, self.waypoint_radius)
    
    def draw_on_frame(self, frame: np.ndarray,
                     predicted_trajectory: Optional[np.ndarray] = None,
                     ground_truth_trajectory: Optional[np.ndarray] = None,
                     vehicle_position: Optional[np.ndarray] = None,
                     vehicle_heading: Optional[float] = None,
                     camera_center: Optional[np.ndarray] = None,
                     scale: float = 10.0) -> np.ndarray:
        """
        Draw trajectories on a rendered frame.
        
        Args:
            frame: Rendered frame as numpy array (H, W, 3)
            predicted_trajectory: Predicted trajectory in world coords (N, 2)
            ground_truth_trajectory: Ground truth trajectory in world coords (N, 2)
            vehicle_position: Current vehicle position (2,)
            vehicle_heading: Current vehicle heading in radians
            camera_center: Camera center for coordinate transform
            scale: Pixels per meter
            
        Returns:
            Frame with trajectories drawn
        """
        if frame is None:
            return frame
        
        # Update position history
        if vehicle_position is not None:
            self.update_position_history(vehicle_position)
        
        # Use vehicle position as camera center if not provided
        if camera_center is None and vehicle_position is not None:
            camera_center = vehicle_position
        elif camera_center is None:
            camera_center = np.zeros(2)
        
        # Make a copy to avoid modifying original
        frame = frame.copy()
        
        # Draw using OpenCV (faster than pygame for image manipulation)
        try:
            import cv2
            
            screen_size = (frame.shape[1], frame.shape[0])
            
            # Draw position history (actual path)
            if len(self.position_history) >= 2:
                history_points = []
                for pos in self.position_history:
                    screen_point = self.world_to_screen(pos, camera_center, scale, screen_size)
                    history_points.append(screen_point)
                
                # Draw actual path
                pts = np.array(history_points, dtype=np.int32)
                cv2.polylines(frame, [pts], False, self.actual_color, self.line_width)
            
            # Draw ground truth trajectory
            if ground_truth_trajectory is not None and len(ground_truth_trajectory) >= 2:
                gt_points = []
                for point in ground_truth_trajectory:
                    screen_point = self.world_to_screen(point, camera_center, scale, screen_size)
                    gt_points.append(screen_point)
                
                pts = np.array(gt_points, dtype=np.int32)
                cv2.polylines(frame, [pts], False, self.ground_truth_color, self.line_width)
                
                # Draw waypoints
                for point in gt_points:
                    cv2.circle(frame, point, self.waypoint_radius, self.ground_truth_color, -1)
            
            # Draw predicted trajectory
            if predicted_trajectory is not None and len(predicted_trajectory) >= 2:
                pred_points = []
                for point in predicted_trajectory:
                    screen_point = self.world_to_screen(point, camera_center, scale, screen_size)
                    pred_points.append(screen_point)
                
                pts = np.array(pred_points, dtype=np.int32)
                cv2.polylines(frame, [pts], False, self.predicted_color, self.line_width + 1)
                
                # Draw waypoints
                for i, point in enumerate(pred_points):
                    # Use different size for first and last waypoints
                    radius = self.waypoint_radius + 2 if i == 0 or i == len(pred_points) - 1 else self.waypoint_radius
                    cv2.circle(frame, point, radius, self.predicted_color, -1)
            
            # Draw vehicle position
            if vehicle_position is not None:
                vehicle_screen = self.world_to_screen(vehicle_position, camera_center, scale, screen_size)
                cv2.circle(frame, vehicle_screen, self.waypoint_radius + 3, (255, 255, 0), 2)
                
                # Draw heading arrow
                if vehicle_heading is not None:
                    arrow_length = 15
                    arrow_end = (
                        int(vehicle_screen[0] + arrow_length * np.cos(vehicle_heading)),
                        int(vehicle_screen[1] - arrow_length * np.sin(vehicle_heading))
                    )
                    cv2.arrowedLine(frame, vehicle_screen, arrow_end, (255, 255, 0), 2)
            
            # Add legend
            legend_y = 30
            cv2.putText(frame, "Predicted", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.predicted_color, 1)
            cv2.putText(frame, "Ground Truth", (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.ground_truth_color, 1)
            cv2.putText(frame, "Actual Path", (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.actual_color, 1)
            
        except ImportError:
            # Fallback: just return the frame without visualization
            pass
        
        return frame
    
    def create_trajectory_plot(self,
                              predicted_trajectories: List[np.ndarray],
                              ground_truth_trajectories: List[np.ndarray],
                              actual_positions: List[np.ndarray],
                              save_path: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Create a plot of trajectories using matplotlib.
        
        Args:
            predicted_trajectories: List of predicted trajectories
            ground_truth_trajectories: List of ground truth trajectories
            actual_positions: List of actual vehicle positions
            save_path: Optional path to save the figure
            
        Returns:
            Plot as numpy array (H, W, 3) or None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Non-interactive backend
            
            fig, ax = plt.subplots(1, 1, figsize=(10, 10))
            
            # Plot predicted trajectories (with transparency for older ones)
            for i, traj in enumerate(predicted_trajectories):
                if traj is not None and len(traj) >= 2:
                    alpha = 0.3 + 0.7 * (i / max(1, len(predicted_trajectories) - 1))
                    ax.plot(traj[:, 0], traj[:, 1], 'g-', alpha=alpha, linewidth=1)
            
            # Plot ground truth trajectories
            for i, traj in enumerate(ground_truth_trajectories):
                if traj is not None and len(traj) >= 2:
                    alpha = 0.3 + 0.7 * (i / max(1, len(ground_truth_trajectories) - 1))
                    ax.plot(traj[:, 0], traj[:, 1], 'r-', alpha=alpha, linewidth=1)
            
            # Plot actual path
            if len(actual_positions) >= 2:
                positions = np.array(actual_positions)
                ax.plot(positions[:, 0], positions[:, 1], 'b-', linewidth=2, label='Actual')
            
            # Labels and legend
            ax.set_xlabel('X (m)')
            ax.set_ylabel('Y (m)')
            ax.set_title('Trajectory Comparison')
            ax.axis('equal')
            ax.grid(True, alpha=0.3)
            ax.legend(['Predicted', 'Ground Truth', 'Actual'])
            
            if save_path:
                plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            # Convert to numpy array
            fig.canvas.draw()
            img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            plt.close(fig)
            return img
            
        except ImportError:
            print("Matplotlib not available for trajectory plotting")
            return None


class TrajectoryRecorder:
    """
    Records trajectory data during simulation for analysis.
    """
    
    def __init__(self):
        """Initialize trajectory recorder."""
        self.records: List[Dict] = []
        self.episode_count = 0
    
    def new_episode(self):
        """Start a new episode."""
        self.episode_count += 1
    
    def record(self, step: int,
              predicted_trajectory: Optional[np.ndarray],
              ground_truth_trajectory: Optional[np.ndarray],
              vehicle_position: np.ndarray,
              vehicle_heading: float,
              vehicle_velocity: np.ndarray,
              control_action: np.ndarray):
        """
        Record a single timestep.
        
        Args:
            step: Current step number
            predicted_trajectory: Predicted trajectory (N, 2)
            ground_truth_trajectory: Ground truth trajectory (N, 2)
            vehicle_position: Vehicle position (2,)
            vehicle_heading: Vehicle heading in radians
            vehicle_velocity: Vehicle velocity (2,)
            control_action: Control action [steering, throttle]
        """
        record = {
            'episode': self.episode_count,
            'step': step,
            'predicted_trajectory': predicted_trajectory.copy() if predicted_trajectory is not None else None,
            'ground_truth_trajectory': ground_truth_trajectory.copy() if ground_truth_trajectory is not None else None,
            'position': vehicle_position.copy(),
            'heading': vehicle_heading,
            'velocity': vehicle_velocity.copy(),
            'control': control_action.copy()
        }
        self.records.append(record)
    
    def compute_metrics(self) -> Dict:
        """
        Compute trajectory prediction metrics.
        
        Returns:
            Dictionary of metrics
        """
        if len(self.records) == 0:
            return {}
        
        # Collect valid predictions
        ade_list = []  # Average Displacement Error
        fde_list = []  # Final Displacement Error
        
        for record in self.records:
            pred = record['predicted_trajectory']
            gt = record['ground_truth_trajectory']
            
            if pred is not None and gt is not None:
                # Ensure same length
                min_len = min(len(pred), len(gt))
                if min_len >= 2:
                    pred = pred[:min_len]
                    gt = gt[:min_len]
                    
                    # Compute errors
                    errors = np.linalg.norm(pred - gt, axis=1)
                    ade_list.append(np.mean(errors))
                    fde_list.append(errors[-1])
        
        metrics = {}
        if len(ade_list) > 0:
            metrics['ADE'] = np.mean(ade_list)
            metrics['ADE_std'] = np.std(ade_list)
        if len(fde_list) > 0:
            metrics['FDE'] = np.mean(fde_list)
            metrics['FDE_std'] = np.std(fde_list)
        
        metrics['num_predictions'] = len(ade_list)
        
        return metrics
    
    def save(self, path: str):
        """Save records to file."""
        import pickle
        with open(path, 'wb') as f:
            pickle.dump({
                'records': self.records,
                'metrics': self.compute_metrics()
            }, f)
    
    def load(self, path: str):
        """Load records from file."""
        import pickle
        with open(path, 'rb') as f:
            data = pickle.load(f)
            self.records = data['records']


def make_trajectory_gif(frames: List[np.ndarray],
                       output_path: str,
                       fps: int = 10):
    """
    Create a GIF from trajectory visualization frames.
    
    Args:
        frames: List of frames as numpy arrays
        output_path: Path to save the GIF
        fps: Frames per second
    """
    try:
        from PIL import Image
        
        if len(frames) == 0:
            return
        
        imgs = [Image.fromarray(frame) for frame in frames]
        duration = int(1000 / fps)
        imgs[0].save(
            output_path,
            save_all=True,
            append_images=imgs[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {output_path}")
    except ImportError:
        print("PIL not available for GIF creation")
