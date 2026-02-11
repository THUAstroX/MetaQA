#!/usr/bin/env python3
"""
Unified Replay Demo - GIF Generator and Console Demo.

This script provides unified replay functionality combining:
1. QA-enhanced replay with MetaDrive BEV and camera images
2. Original frequency replay with QA annotations
3. Console-based frame-by-frame inspection

Modes:
    - gif:     Generate GIF with MetaDrive BEV, camera grid, and QA panel
    - console: Interactive console demo showing frame info
    - list:    List available scenes

Usage:
    # List available scenes
    python -m meta_qa.scripts.demo_vis.replay_demo --list-scenes
    
    # Generate GIF (default)
    python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --output demo.gif
    
    # Console demo
    python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --mode console
    
    # Samples only (keyframes at 2Hz)
    python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --samples-only
    
    # Custom FPS and frame limit
    python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --fps 5 --max-frames 50
"""

import os
import sys
import argparse
import numpy as np
from typing import Optional, List, Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Default paths
DEFAULT_NUSCENES_ROOT = os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes", "v1.0-mini")
DEFAULT_SCENARIO_DIR = os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes_converted")
DEFAULT_QA_DIR = os.path.join(project_root, "dataset", "QA_Data")
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "outputs", "vis")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Replay Demo with QA annotations and MetaDrive BEV",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available scenes
    python -m meta_qa.scripts.demo_vis.replay_demo --list-scenes
    
    # Generate GIF for a scene
    python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --output demo.gif
    
    # Console demo
    python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --mode console
    
    # Custom settings
    python -m meta_qa.scripts.demo_vis.replay_demo \\
        --scene scene-0061 --width 1600 --height 1000 --fps 5
"""
    )
    
    # Mode
    parser.add_argument("--mode", type=str, default="gif",
                       choices=["gif", "console", "list"],
                       help="Demo mode: gif (generate GIF), console (text output), list (list scenes)")
    
    # Data paths
    parser.add_argument("--nuscenes-root", type=str, default=DEFAULT_NUSCENES_ROOT,
                       help="Path to NuScenes dataset root")
    parser.add_argument("--scenario-dir", type=str, default=DEFAULT_SCENARIO_DIR,
                       help="Path to ScenarioNet converted data")
    parser.add_argument("--qa-dir", type=str, default=DEFAULT_QA_DIR,
                       help="Path to NuScenes-QA data directory")
    
    # Scene selection
    parser.add_argument("--scene", type=str, default=None,
                       help="Scene to replay (e.g., scene-0061)")
    parser.add_argument("--list-scenes", action="store_true",
                       help="List available scenes and exit")
    
    # Display options
    parser.add_argument("--samples-only", action="store_true",
                       help="Show only samples (2Hz keyframes)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to process")
    parser.add_argument("--fps", type=int, default=12,
                       help="Display/GIF frame rate (default: 12)")
    
    # GIF options
    parser.add_argument("--width", type=int, default=1200,
                       help="GIF output width (default: 1200)")
    parser.add_argument("--height", type=int, default=900,
                       help="GIF output height (default: 900)")
    
    # Output
    parser.add_argument("--output", type=str, default=None,
                       help="Output GIF path (optional)")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR,
                       help="Output directory")
    
    # NuScenes version
    parser.add_argument("--version", type=str, default="v1.0-mini",
                       help="NuScenes version")
    
    # Verbosity
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    return parser.parse_args()


def list_available_scenes(replay_env, qa_loader=None):
    """List all available scenes with QA info."""
    print("\n" + "=" * 70)
    print("Available Scenes")
    print("=" * 70)
    
    scenes = replay_env.nuscenes_processor.scenes
    
    for i, (token, scene) in enumerate(scenes.items(), 1):
        scene_name = scene['name']
        description = scene.get('description', 'No description')[:50]
        
        # Get QA count if available
        qa_count = 0
        if qa_loader:
            scene_qa = qa_loader.get_qa_for_scene(scene_name)
            if scene_qa:
                qa_count = scene_qa.total_qa_count
        
        print(f"{i:2d}. {scene_name}")
        print(f"    Description: {description}")
        if qa_count > 0:
            print(f"    QA Items: {qa_count}")
        print()
    
    print("-" * 70)
    print(f"Total: {len(scenes)} scenes")


def display_frame_info(frame_info, show_qa: bool = True):
    """Display frame information to console."""
    print(f"\n{'='*60}")
    print(f"Frame {frame_info.frame_idx:4d} | ", end="")
    
    if frame_info.is_sample:
        print(f"SAMPLE {frame_info.sample_index:2d} | ", end="")
    else:
        print(f"Sweep (ratio: {frame_info.interpolation_ratio:.2f}) | ", end="")
    
    print(f"t={frame_info.timestamp/1e6:.3f}s")
    
    # Show ego state
    if frame_info.ego_state:
        ego = frame_info.ego_state
        print(f"  Ego: pos=({ego.position[0]:.2f}, {ego.position[1]:.2f}), "
              f"vel=({ego.velocity[0]:.2f}, {ego.velocity[1]:.2f}), "
              f"heading={ego.heading:.2f}rad")
    
    # Show surrounding vehicles
    if frame_info.surrounding_states:
        print(f"  Surrounding vehicles: {len(frame_info.surrounding_states)}")
    
    # Show images
    if frame_info.image_paths:
        cams = list(frame_info.image_paths.keys())
        print(f"  Cameras: {', '.join(cams)}")
    
    # Show QA at samples
    if show_qa and frame_info.is_sample and frame_info.has_qa:
        print(f"\n  QA Items ({len(frame_info.qa_items)}):")
        for i, qa in enumerate(frame_info.qa_items[:3]):  # Show first 3
            q = qa['question'][:60] + "..." if len(qa['question']) > 60 else qa['question']
            print(f"    Q{i+1}: {q}")
            print(f"    A{i+1}: {qa['answer']}")
        if len(frame_info.qa_items) > 3:
            print(f"    ... and {len(frame_info.qa_items) - 3} more QA items")


def run_console_demo(replay_env, args):
    """Run interactive console demo."""
    if not replay_env.load_scene(args.scene):
        print(f"Failed to load scene: {args.scene}")
        return
    
    scene_info = replay_env.get_scene_info()
    
    print("\n" + "=" * 60)
    print(f"Scene: {scene_info['scene_name']}")
    print("=" * 60)
    print(f"Duration: {scene_info['duration_seconds']:.2f}s")
    print(f"Total frames: {scene_info['num_frames']}")
    print(f"Samples (keyframes): {scene_info['num_samples']}")
    print(f"Actual FPS: {scene_info['actual_fps']:.2f}")
    print(f"Has trajectory: {scene_info['has_trajectory']}")
    print(f"Has QA: {scene_info['has_qa']} ({scene_info['qa_count']} items)")
    print("=" * 60)
    
    # Iterate through frames
    frame_count = 0
    max_frames = args.max_frames or scene_info['num_frames']
    
    if args.samples_only:
        iterator = replay_env.iterate_samples()
    else:
        iterator = replay_env.iterate_frames()
    
    for frame_info in iterator:
        if frame_count >= max_frames:
            break
        
        display_frame_info(frame_info)
        frame_count += 1
    
    print(f"\n{'='*60}")
    print(f"Displayed {frame_count} frames")


def generate_demo_gif(replay_env, args):
    """
    Generate demo GIF with MetaDrive BEV, camera grid, and QA panel.
    
    Layout:
        +---------------------------+----------------------+
        |  MetaDrive BEV (top-left) |                      |
        |                           |   QA Panel           |
        +---------------------------+   - Frame info       |
        |  Camera Grid (3x2)        |   - Ego state        |
        |  [FL] [F ] [FR]           |   - QA items         |
        |  [BL] [B ] [BR]           |                      |
        +---------------------------+----------------------+
    """
    try:
        from PIL import Image, ImageDraw, ImageFont
        import imageio
    except ImportError:
        print("Error: Pillow and imageio required for GIF generation")
        print("Install with: pip install Pillow imageio")
        return False
    
    # Try to import MetaDrive
    HAS_METADRIVE = False
    try:
        from metadrive.envs.scenario_env import ScenarioEnv
        from metadrive.policy.replay_policy import ReplayEgoCarPolicy
        HAS_METADRIVE = True
    except ImportError:
        print("Warning: MetaDrive not available, BEV will show placeholder")
    
    # Load scene
    if not replay_env.load_scene(args.scene):
        print(f"Failed to load scene: {args.scene}")
        return False
    
    scene_info = replay_env.get_scene_info()
    print(f"\nGenerating GIF for {args.scene}")
    print(f"  Total frames: {scene_info['num_frames']}")
    print(f"  Samples (keyframes): {scene_info['num_samples']}")
    print(f"  Duration: {scene_info['duration_seconds']:.2f}s")
    
    # === Phase 1: Pre-collect MetaDrive BEV frames ===
    bev_frames_cache = []
    METADRIVE_DT = 0.1  # 10Hz
    
    if HAS_METADRIVE:
        scenario_path = replay_env._find_scenario_file(args.scene)
        if scenario_path:
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
            
            print(f"  MetaDrive scenario: {os.path.basename(scenario_path)}")
            
            config = dict(
                agent_policy=ReplayEgoCarPolicy,
                use_render=False,
                data_directory=scenario_subdir,
                num_scenarios=1,
                start_scenario_index=scenario_idx,
                no_traffic=False,
            )
            
            metadrive_env = None
            try:
                metadrive_env = ScenarioEnv(config)
                metadrive_env.reset()
                print("  MetaDrive initialized, pre-collecting BEV frames...")
                
                metadrive_time = 0.0
                scene_duration = scene_info['duration_seconds']
                
                # Initial frame
                try:
                    bev_frame = metadrive_env.render(
                        mode="topdown",
                        film_size=(4000, 4000),
                        screen_size=(400, 400),
                        draw_contour=True
                    )
                    bev_frames_cache.append((metadrive_time, bev_frame.copy()))
                except:
                    pass
                
                # Collect BEV frames
                replay_done = False
                while not replay_done and metadrive_time < scene_duration + METADRIVE_DT:
                    obs, reward, terminated, truncated, info = metadrive_env.step([0, 0])
                    metadrive_time += METADRIVE_DT
                    
                    try:
                        bev_frame = metadrive_env.render(
                            mode="topdown",
                            film_size=(4000, 4000),
                            screen_size=(400, 400),
                            draw_contour=True
                        )
                        bev_frames_cache.append((metadrive_time, bev_frame.copy()))
                    except:
                        pass
                    
                    replay_done = info.get("replay_done", False)
                    if replay_done:
                        print(f"  MetaDrive replay completed at t={metadrive_time:.2f}s")
                
                # Fill remaining time with last frame
                if bev_frames_cache and metadrive_time < scene_duration:
                    last_bev = bev_frames_cache[-1][1]
                    while metadrive_time < scene_duration + METADRIVE_DT:
                        metadrive_time += METADRIVE_DT
                        bev_frames_cache.append((metadrive_time, last_bev))
                
                print(f"  Pre-collected {len(bev_frames_cache)} BEV frames")
                
            except Exception as e:
                print(f"  Warning: MetaDrive init failed: {e}")
            finally:
                if metadrive_env:
                    metadrive_env.close()
    
    # Helper to find nearest BEV frame
    def get_bev_frame_at_time(time_sec):
        if not bev_frames_cache:
            return None
        
        best_idx = 0
        best_diff = abs(bev_frames_cache[0][0] - time_sec)
        
        for i, (t, _) in enumerate(bev_frames_cache):
            diff = abs(t - time_sec)
            if diff < best_diff:
                best_diff = diff
                best_idx = i
            elif diff > best_diff:
                break
        
        return bev_frames_cache[best_idx][1]
    
    # === Phase 2: Generate GIF frames ===
    frames = []
    max_frames = args.max_frames or scene_info['num_frames']
    first_timestamp = None
    
    print("  Generating GIF frames...")
    
    for frame_info in replay_env.iterate_frames():
        if len(frames) >= max_frames:
            break
        
        if first_timestamp is None:
            first_timestamp = frame_info.timestamp
        
        nuscenes_time = (frame_info.timestamp - first_timestamp) / 1e6
        
        if args.samples_only and not frame_info.is_sample:
            continue
        
        bev_frame = get_bev_frame_at_time(nuscenes_time)
        
        frame_img = create_frame_image(
            frame_info, replay_env, args, bev_frame
        )
        if frame_img:
            frames.append(np.array(frame_img))
        
        if len(frames) % 20 == 0:
            print(f"    Processed {len(frames)} frames... (t={nuscenes_time:.2f}s)")
    
    if not frames:
        print("No frames generated")
        return False
    
    # Save GIF
    output_path = args.output
    if not output_path:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{args.scene}_replay.gif")
    
    print(f"\nSaving GIF with {len(frames)} frames at {args.fps} FPS...")
    try:
        imageio.mimsave(output_path, frames, fps=args.fps, loop=0)
    except TypeError:
        imageio.mimsave(output_path, frames, duration=1000/args.fps/1000, loop=0)
    print(f"Saved to: {output_path}")
    
    return True


def create_frame_image(frame_info, replay_env, args, bev_frame=None):
    """Create a single frame image for the GIF."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return None
    
    # Dimensions - left side 65%, right QA panel 35%
    width, height = args.width, args.height
    left_width = int(width * 0.65)
    qa_width = width - left_width
    bev_height = int(height * 0.5)
    camera_height = height - bev_height
    
    # Create base image
    img = Image.new('RGB', (width, height), (30, 30, 40))
    draw = ImageDraw.Draw(img)
    
    # Fonts
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        font_small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 11)
        font_large = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    except:
        font = ImageFont.load_default()
        font_small = font
        font_large = font
    
    # === Top-left: MetaDrive BEV ===
    if bev_frame is not None and isinstance(bev_frame, np.ndarray):
        bev_img = Image.fromarray(bev_frame)
        bev_size = min(left_width, bev_height)
        if bev_img.size != (bev_size, bev_size):
            bev_img = bev_img.resize((bev_size, bev_size), Image.Resampling.LANCZOS)
        x_offset = (left_width - bev_size) // 2
        y_offset = (bev_height - bev_size) // 2
        img.paste(bev_img, (x_offset, y_offset))
    else:
        draw.rectangle([2, 2, left_width - 2, bev_height - 2], fill=(40, 40, 50))
        draw.text((left_width // 2 - 60, bev_height // 2 - 10), 
                 "MetaDrive BEV", fill=(150, 150, 150), font=font)
    
    # === Bottom-left: Camera Grid (3x2) ===
    camera_order = [
        ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT'],
        ['CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT'],
    ]
    
    cam_w = left_width // 3
    cam_h = camera_height // 2
    
    for row_idx, row in enumerate(camera_order):
        for col_idx, cam in enumerate(row):
            cx = col_idx * cam_w
            cy = bev_height + row_idx * cam_h
            
            if cam in frame_info.image_paths:
                img_path = replay_env.get_image_path(frame_info.frame_idx, cam)
                if img_path and os.path.exists(img_path):
                    try:
                        cam_img = Image.open(img_path)
                        
                        # Maintain aspect ratio
                        orig_w, orig_h = cam_img.size
                        target_w, target_h = cam_w - 4, cam_h - 4
                        
                        # Calculate scaling to fit within target size
                        scale = min(target_w / orig_w, target_h / orig_h)
                        new_w = int(orig_w * scale)
                        new_h = int(orig_h * scale)
                        
                        cam_img = cam_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        
                        # Center the image
                        paste_x = cx + 2 + (target_w - new_w) // 2
                        paste_y = cy + 2 + (target_h - new_h) // 2
                        
                        # Draw consistent background first
                        draw.rectangle([cx + 2, cy + 2, cx + cam_w - 2, cy + cam_h - 2], 
                                     fill=(30, 30, 40))
                        img.paste(cam_img, (paste_x, paste_y))
                        
                        label = cam.replace('CAM_', '').replace('_', ' ')
                        draw.text((cx + 5, cy + 5), label, fill=(255, 255, 255), font=font_small)
                    except:
                        draw.rectangle([cx + 2, cy + 2, cx + cam_w - 2, cy + cam_h - 2], 
                                     fill=(50, 50, 60))
                        label = cam.replace('CAM_', '')
                        draw.text((cx + 10, cy + 10), label, fill=(150, 150, 150), font=font_small)
            else:
                draw.rectangle([cx + 2, cy + 2, cx + cam_w - 2, cy + cam_h - 2], 
                             fill=(50, 50, 60))
                label = cam.replace('CAM_', '')
                draw.text((cx + 10, cy + 10), label, fill=(150, 150, 150), font=font_small)
    
    # === Right side: QA Panel ===
    panel_x = left_width
    
    # Draw borders
    draw.line([(left_width, 0), (left_width, height)], fill=(80, 80, 100), width=2)
    draw.line([(0, bev_height), (left_width, bev_height)], fill=(80, 80, 100), width=2)
    
    # Frame info header
    y = 10
    header = f"Frame {frame_info.frame_idx}"
    if frame_info.is_sample:
        header += f" [SAMPLE {frame_info.sample_index}]"
        draw.text((panel_x + 10, y), header, fill=(100, 200, 255), font=font_large)
    else:
        header += f" [sweep]"
        draw.text((panel_x + 10, y), header, fill=(180, 180, 180), font=font_large)
    y += 30
    
    # Timestamp
    t_sec = frame_info.timestamp / 1e6
    draw.text((panel_x + 10, y), f"Time: {t_sec:.3f}s", fill=(200, 200, 200), font=font)
    y += 20
    
    # Interpolation info
    if not frame_info.is_sample:
        draw.text((panel_x + 10, y), 
                 f"Interpolation: {frame_info.interpolation_ratio:.2%}", 
                 fill=(150, 200, 150), font=font)
        y += 20
    
    # Scene description
    if replay_env.scene_original_data and replay_env.scene_original_data.description:
        y += 5
        desc = replay_env.scene_original_data.description
        if len(desc) > 60:
            desc = desc[:57] + "..."
        draw.text((panel_x + 10, y), desc, fill=(180, 180, 200), font=font_small)
        y += 20
    
    # Ego state
    if frame_info.ego_state:
        y += 10
        draw.text((panel_x + 10, y), "Ego Vehicle:", fill=(100, 200, 255), font=font)
        y += 18
        ego = frame_info.ego_state
        draw.text((panel_x + 20, y), 
                 f"Position: ({ego.position[0]:.1f}, {ego.position[1]:.1f})", 
                 fill=(180, 180, 180), font=font_small)
        y += 15
        draw.text((panel_x + 20, y), 
                 f"Velocity: ({ego.velocity[0]:.1f}, {ego.velocity[1]:.1f}) m/s", 
                 fill=(180, 180, 180), font=font_small)
        y += 15
        draw.text((panel_x + 20, y), 
                 f"Heading: {ego.heading:.2f} rad", 
                 fill=(180, 180, 180), font=font_small)
        y += 25
    
    # Surrounding vehicles
    if frame_info.surrounding_states:
        draw.text((panel_x + 10, y), 
                 f"Surrounding: {len(frame_info.surrounding_states)} vehicles", 
                 fill=(200, 200, 100), font=font)
        y += 25
    
    # QA section
    if frame_info.is_sample and frame_info.has_qa:
        y += 5
        draw.line([(panel_x + 10, y), (width - 10, y)], fill=(80, 80, 100))
        y += 10
        
        draw.text((panel_x + 10, y), 
                 f"QA ({len(frame_info.qa_items)} items):", 
                 fill=(255, 200, 100), font=font)
        y += 22
        
        max_qa_display = 10
        for i, qa in enumerate(frame_info.qa_items[:max_qa_display]):
            if y > height - 60:
                draw.text((panel_x + 10, y), "...", fill=(150, 150, 150), font=font)
                break
            
            # Question
            q = qa['question']
            max_q_len = (qa_width - 30) // 6
            if len(q) > max_q_len:
                q = q[:max_q_len - 3] + "..."
            draw.text((panel_x + 15, y), f"Q: {q}", fill=(200, 200, 220), font=font_small)
            y += 16
            
            # Answer
            a = str(qa['answer'])
            max_a_len = (qa_width - 30) // 6
            if len(a) > max_a_len:
                a = a[:max_a_len - 3] + "..."
            draw.text((panel_x + 15, y), f"A: {a}", fill=(150, 220, 150), font=font_small)
            y += 20
        
        if len(frame_info.qa_items) > max_qa_display:
            draw.text((panel_x + 15, y), 
                     f"... and {len(frame_info.qa_items) - max_qa_display} more", 
                     fill=(150, 150, 150), font=font_small)
    
    # Footer
    draw.line([(0, height - 25), (width, height - 25)], fill=(60, 60, 80))
    scene_name = replay_env.current_scene_name or "Unknown"
    fps_info = f"{replay_env.scene_original_data.actual_fps:.1f}" if replay_env.scene_original_data else "~12"
    footer = f"Scene: {scene_name} | {fps_info} FPS | Frame {frame_info.frame_idx}"
    draw.text((10, height - 20), footer, fill=(150, 150, 170), font=font_small)
    
    return img


def main():
    args = parse_args()
    
    # Convert relative paths to absolute
    if not os.path.isabs(args.nuscenes_root):
        args.nuscenes_root = os.path.join(project_root, args.nuscenes_root)
    if not os.path.isabs(args.scenario_dir):
        args.scenario_dir = os.path.join(project_root, args.scenario_dir)
    if not os.path.isabs(args.qa_dir):
        args.qa_dir = os.path.join(project_root, args.qa_dir)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.join(project_root, args.output_dir)
    
    # Handle list-scenes shortcut
    if args.list_scenes:
        args.mode = "list"
    
    print("=" * 60)
    print("Unified Replay Demo")
    print("=" * 60)
    
    # Load QA data
    qa_loader = None
    try:
        from meta_qa.tools.qa_loader import NuScenesQALoader
        qa_loader = NuScenesQALoader(
            qa_data_dir=args.qa_dir,
            nuscenes_dataroot=args.nuscenes_root,
            version=args.version,
        )
        qa_loader.load()
        print(f"Loaded QA data: {len(qa_loader.qa_items)} items")
    except Exception as e:
        print(f"Warning: Could not load QA data: {e}")
    
    # Create replay environment
    from meta_qa.tools.replay_original import ReplayOriginalEnv
    
    replay_env = ReplayOriginalEnv(
        nuscenes_dataroot=args.nuscenes_root,
        scenario_dir=args.scenario_dir,
        qa_loader=qa_loader,
        version=args.version,
    )
    replay_env.load()
    
    # Execute based on mode
    if args.mode == "list":
        list_available_scenes(replay_env, qa_loader)
        return
    
    # Require scene for other modes
    if not args.scene:
        print("\nNo scene specified. Use --scene <name> or --list-scenes")
        list_available_scenes(replay_env, qa_loader)
        return
    
    if args.mode == "console":
        run_console_demo(replay_env, args)
    else:  # gif mode
        generate_demo_gif(replay_env, args)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
