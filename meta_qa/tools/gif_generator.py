"""
GIF Generator for QA-enhanced scenario visualization.

This module generates animated GIFs showing synchronized:
- MetaDrive BEV (Bird's Eye View) replay
- QA annotations for each keyframe
- 6-camera images in 3x2 grid layout

Layout:
    ┌─────────────────────────────────────────────────────────────┐
    │  MetaDrive BEV      │                                       │
    │  (left top)         │  QA Panel (full height)               │
    │                     │  - Frame info                         │
    ├─────────────────────┤  - Scene description                  │
    │  Camera Grid (3x2)  │  - All Q&A items                      │
    │  [FL] [F ] [FR]     │  - Full question text                 │
    │  [BL] [B ] [BR]     │  - Full answer text                   │
    └─────────────────────┴───────────────────────────────────────┘
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import imageio
    HAS_IMAGEIO = True
except ImportError:
    HAS_IMAGEIO = False

from .qa_vis import render_qa_panel, render_camera_grid


@dataclass
class GIFConfig:
    """Configuration for GIF generation."""
    
    # Output dimensions (increased to show more QA content)
    width: int = 1200
    height: int = 900
    
    # Layout ratios (new layout: BEV top-left, cameras bottom-left, QA right)
    left_width_ratio: float = 0.5   # Left panel (BEV + cameras) takes 50% width
    bev_height_ratio: float = 0.5   # BEV takes 50% of left panel height
    
    # Colors (RGB)
    bg_color: tuple = (30, 30, 40)
    border_color: tuple = (80, 80, 100)
    header_color: tuple = (70, 130, 180)
    
    @property
    def left_width(self) -> int:
        """Width of left panel (BEV + cameras)."""
        return int(self.width * self.left_width_ratio)
    
    @property
    def qa_width(self) -> int:
        """Width of QA panel (right side, full height)."""
        return self.width - self.left_width
    
    @property
    def bev_height(self) -> int:
        """Height of BEV area (top-left)."""
        return int(self.height * self.bev_height_ratio)
    
    @property
    def camera_height(self) -> int:
        """Height of camera area (bottom-left)."""
        return self.height - self.bev_height
    
    # Keep these for backward compatibility
    @property
    def bev_width(self) -> int:
        return self.left_width
    
    @property
    def panel_width(self) -> int:
        return self.qa_width
    
    @property
    def qa_height(self) -> int:
        return self.height


class GIFGenerator:
    """
    Generates animated GIFs from MetaDrive scenarios with QA annotations.
    
    Features:
    - MetaDrive top-down BEV rendering
    - 6-camera 3x2 grid layout
    - QA annotation panel
    - Synchronized with NuScenes keyframes (2Hz)
    """
    
    def __init__(
        self,
        qa_loader: Any,
        matcher: Any,
        config: Optional[GIFConfig] = None,
    ):
        """
        Initialize GIF generator.
        
        Args:
            qa_loader: NuScenesQALoader instance
            matcher: ScenarioQAMatcher instance
            config: GIF configuration
        """
        self.qa_loader = qa_loader
        self.matcher = matcher
        self.config = config or GIFConfig()
    
    def generate(
        self,
        scene_name: str,
        output_path: str,
        fps: int = 2,
        max_frames: Optional[int] = None,
    ) -> bool:
        """
        Generate GIF for a scene.
        
        Args:
            scene_name: Scene to generate GIF for
            output_path: Output GIF path
            fps: Frames per second (default 2 = NuScenes keyframe rate)
            max_frames: Maximum frames (None for all keyframes)
            
        Returns:
            True if successful
        """
        if not HAS_PIL:
            print("Error: Pillow required. Install with: pip install Pillow")
            return False
        if not HAS_IMAGEIO:
            print("Error: imageio required. Install with: pip install imageio")
            return False
        
        # Import MetaDrive
        try:
            from metadrive.envs.scenario_env import ScenarioEnv
            from metadrive.policy.replay_policy import ReplayEgoCarPolicy
        except ImportError:
            print("Error: MetaDrive required. Install with: pip install metadrive-simulator")
            return False
        
        # Get scene QA data
        scene_qa = self.qa_loader.get_qa_for_scene(scene_name)
        if not scene_qa:
            print(f"Error: No QA data for scene {scene_name}")
            return False
        
        # Find scenario file
        scenario_path = self.matcher.scene_to_scenario_path.get(scene_name)
        if scenario_path is None:
            print(f"Error: Scenario file not found for {scene_name}")
            return False
        
        # Get scenario directory and index
        scenario_subdir = os.path.dirname(scenario_path)
        scenario_files = sorted([
            f for f in os.listdir(scenario_subdir)
            if f.endswith('.pkl') and f.startswith('sd_')
        ])
        scenario_idx = 0
        for i, f in enumerate(scenario_files):
            if os.path.basename(scenario_path) == f:
                scenario_idx = i
                break
        
        print(f"Scene: {scene_name}")
        print(f"  Directory: {os.path.basename(scenario_subdir)}")
        print(f"  Scenario index: {scenario_idx}/{len(scenario_files)}")
        
        # Configure MetaDrive environment
        # Note: num_scenarios should be set such that start_scenario_index + num_scenarios <= total_scenarios
        # We only need to load 1 scenario at a time, so set num_scenarios=1
        config = dict(
            agent_policy=ReplayEgoCarPolicy,
            use_render=False,
            data_directory=scenario_subdir,
            num_scenarios=1,  # Only load the single scenario we need
            start_scenario_index=scenario_idx,
            no_traffic=False,
        )
        
        print(f"Initializing MetaDrive...")
        env = ScenarioEnv(config)
        
        try:
            obs, info = env.reset()
            
            # Calculate keyframe count
            num_keyframes = len(scene_qa.sample_order)
            if max_frames:
                num_keyframes = min(num_keyframes, max_frames)
            
            print(f"Generating {num_keyframes} frames at {fps} FPS...")
            
            frames = []
            step = 0
            keyframe_idx = 0
            
            cfg = self.config
            
            while keyframe_idx < num_keyframes:
                # Check if current step is a keyframe (every 5 steps for 10Hz -> 2Hz)
                is_keyframe = (step % 5 == 0)
                
                if is_keyframe and keyframe_idx < num_keyframes:
                    # Get MetaDrive top-down frame (square aspect ratio)
                    try:
                        bev_size = min(cfg.bev_width, cfg.height)
                        metadrive_frame = env.render(
                            mode="topdown",
                            film_size=(4000, 4000),
                            screen_size=(bev_size, bev_size),
                            draw_contour=True
                        )
                    except:
                        metadrive_frame = None
                    
                    # Get QA data and image paths for this keyframe
                    sample_qa = None
                    image_paths = {}
                    
                    if keyframe_idx < len(scene_qa.sample_order):
                        sample_token = scene_qa.sample_order[keyframe_idx]
                        sample_qa = scene_qa.samples.get(sample_token)
                        
                        # Get all 6 camera paths
                        for camera in ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                                       'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']:
                            image_paths[camera] = self.qa_loader.get_image_path(sample_token, camera)
                    
                    # Compose final frame
                    frame = self._compose_frame(
                        metadrive_frame=metadrive_frame,
                        sample_qa=sample_qa,
                        image_paths=image_paths,
                        description=scene_qa.description,
                        frame_idx=keyframe_idx,
                        total_frames=num_keyframes,
                    )
                    
                    frames.append(np.array(frame))
                    
                    if (keyframe_idx + 1) % 10 == 0:
                        print(f"  Generated {keyframe_idx + 1}/{num_keyframes} frames...")
                    
                    keyframe_idx += 1
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step([0, 0])
                step += 1
                
                # Use replay_done to determine when replay is complete
                # This is better than terminated/truncated for replay scenarios
                # because it ignores arrive_dest, crash, etc. and focuses on
                # whether the recorded trajectory data has been fully replayed
                if info.get("replay_done", False):
                    print(f"  Replay completed at step {step}")
                    break
            
            # Save GIF with proper duration control
            # Use fps parameter (frames per second) instead of duration
            # For imageio >= 2.9, use fps parameter directly
            duration_per_frame = 1.0 / fps
            try:
                # Try newer imageio API with fps
                imageio.mimsave(output_path, frames, fps=fps, loop=0)
            except TypeError:
                # Fallback to duration parameter (in seconds)
                imageio.mimsave(output_path, frames, duration=duration_per_frame, loop=0)
            
            print(f"\n✓ Saved GIF: {output_path}")
            print(f"  Dimensions: {cfg.width}x{cfg.height}")
            print(f"  Frames: {len(frames)}")
            print(f"  Duration: {len(frames) * duration_per_frame:.1f}s ({fps} FPS)")
            
            return True
            
        finally:
            env.close()
    
    def _compose_frame(
        self,
        metadrive_frame: Optional[np.ndarray],
        sample_qa: Optional[Any],
        image_paths: Dict[str, Optional[str]],
        description: str,
        frame_idx: int,
        total_frames: int,
    ) -> Image.Image:
        """
        Compose a single frame with BEV, cameras, and QA panel.
        
        New Layout:
            [BEV (top-left)     | QA Panel (right, full height)]
            [Cameras (bot-left) |                              ]
        """
        cfg = self.config
        final = Image.new('RGB', (cfg.width, cfg.height), cfg.bg_color)
        
        # Top-left: MetaDrive BEV
        if metadrive_frame is not None and isinstance(metadrive_frame, np.ndarray):
            bev_img = Image.fromarray(metadrive_frame)
            # Resize BEV to fit top-left area (keep square aspect ratio)
            bev_size = min(cfg.left_width, cfg.bev_height)
            if bev_img.size != (bev_size, bev_size):
                bev_img = bev_img.resize((bev_size, bev_size), Image.Resampling.LANCZOS)
            # Center horizontally and vertically within BEV area
            x_offset = (cfg.left_width - bev_size) // 2
            y_offset = (cfg.bev_height - bev_size) // 2
            final.paste(bev_img, (x_offset, y_offset))
        else:
            self._draw_placeholder(final, 0, 0, cfg.left_width, cfg.bev_height, "MetaDrive BEV")
        
        # Bottom-left: Camera grid (3x2)
        camera_img = render_camera_grid(
            image_paths=image_paths,
            width=cfg.left_width,
            height=cfg.camera_height,
            bg_color=cfg.bg_color,
        )
        final.paste(camera_img, (0, cfg.bev_height))
        
        # Right side: QA panel (full height)
        qa_img = render_qa_panel(
            sample_qa=sample_qa,
            description=description,
            width=cfg.qa_width,
            height=cfg.height,  # Full height for QA
            frame_idx=frame_idx,
            total_frames=total_frames,
            bg_color=cfg.bg_color,
        )
        final.paste(qa_img, (cfg.left_width, 0))
        
        # Draw borders
        draw = ImageDraw.Draw(final)
        # Vertical border between left panels and QA
        draw.line(
            [(cfg.left_width, 0), (cfg.left_width, cfg.height)],
            fill=cfg.border_color, width=2
        )
        # Horizontal border between BEV and cameras (left side only)
        draw.line(
            [(0, cfg.bev_height), (cfg.left_width, cfg.bev_height)],
            fill=cfg.border_color, width=2
        )
        
        return final
    
    def _draw_placeholder(
        self,
        img: Image.Image,
        x: int,
        y: int,
        width: int,
        height: int,
        text: str,
    ):
        """Draw a placeholder rectangle with text."""
        draw = ImageDraw.Draw(img)
        draw.rectangle([x, y, x + width, y + height], fill=(40, 40, 50))
        
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
        
        text_x = x + width // 2 - len(text) * 5
        text_y = y + height // 2
        draw.text((text_x, text_y), text, fill=(150, 150, 150), font=font)
