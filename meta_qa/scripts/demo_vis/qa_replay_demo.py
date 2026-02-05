#!/usr/bin/env python3
"""
QA-Enhanced Scenario Replay Demo - GIF Generator.

This script generates animated GIFs that integrate:
1. MetaDrive scenario replay (from ScenarioNet converted data)
2. NuScenes-QA question-answer annotations
3. Original NuScenes camera images (6 cameras in 3x2 grid)

Usage:
    # List available scenes with QA data
    python -m meta_qa.scripts.demo_vis.qa_replay_demo --list-scenes
    
    # Generate GIF with 6-camera layout
    python -m meta_qa.scripts.demo_vis.qa_replay_demo \
        --scene scene-0916 \
        --output outputs/vis/demo.gif
    
    # Custom FPS and frame limit
    python -m meta_qa.scripts.demo_vis.qa_replay_demo \
        --scene scene-0916 \
        --output outputs/vis/demo.gif \
        --fps 1 --max-frames 30
"""

import os
import sys
import argparse
from typing import List

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

from meta_qa.tools import (
    NuScenesQALoader,
    ScenarioQAMatcher,
    GIFGenerator,
    GIFConfig,
)


# Default paths (relative to project root)
DEFAULT_QA_DIR = os.path.join(project_root, "dataset", "QA_Data")
DEFAULT_NUSCENES_DIR = os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes", "v1.0-mini")
DEFAULT_SCENARIO_DIR = os.path.join(project_root, "dataset", "Scenario_Data", "exp_nuscenes_converted")
DEFAULT_OUTPUT_DIR = os.path.join(project_root, "outputs", "vis")


def list_available_scenes(matcher: ScenarioQAMatcher):
    """List scenes that have both QA data and scenario data."""
    matching_scenes = matcher.get_matching_scenes()
    
    if not matching_scenes:
        print("No scenes found with both QA data and scenario data.")
        return
    
    print(f"\n{'='*60}")
    print(f"Available Scenes: {len(matching_scenes)}")
    print(f"{'='*60}\n")
    
    for i, scene_name in enumerate(sorted(matching_scenes), 1):
        scene_qa = matcher.qa_loader.get_qa_for_scene(scene_name)
        
        print(f"{i:2d}. {scene_name}")
        if scene_qa:
            print(f"    Description: {scene_qa.description}")
            print(f"    Keyframes: {len(scene_qa.sample_order)}")
            print(f"    QA Items: {scene_qa.total_qa_count}")
        print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate QA-enhanced scenario replay GIFs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # List available scenes
    python -m meta_qa.scripts.demo_vis.qa_replay_demo --list-scenes
    
    # Generate GIF for a scene
    python -m meta_qa.scripts.demo_vis.qa_replay_demo \\
        --scene scene-0916 \\
        --output outputs/vis/scene-0916.gif
    
    # Generate with custom settings
    python -m meta_qa.scripts.demo_vis.qa_replay_demo \\
        --scene scene-0916 \\
        --output outputs/vis/scene-0916.gif \\
        --width 1600 --height 800 \\
        --fps 5 --max-frames 30
        """,
    )
    
    # Data paths
    parser.add_argument("--qa-dir", type=str, default=DEFAULT_QA_DIR,
                       help="Path to QA data directory")
    parser.add_argument("--nuscenes-dir", type=str, default=DEFAULT_NUSCENES_DIR,
                       help="Path to NuScenes data directory")
    parser.add_argument("--scenario-dir", type=str, default=DEFAULT_SCENARIO_DIR,
                       help="Path to ScenarioNet converted directory")
    
    # Scene selection
    parser.add_argument("--scene", type=str, default=None,
                       help="Scene name to generate GIF for (e.g., scene-0916)")
    parser.add_argument("--list-scenes", action="store_true",
                       help="List available scenes and exit")
    
    # Output options
    parser.add_argument("--output", type=str, default=None,
                       help="Output GIF path (default: outputs/vis/{scene_name}.gif)")
    parser.add_argument("--width", type=int, default=1200,
                       help="GIF output width (default: 1200)")
    parser.add_argument("--height", type=int, default=900,
                       help="GIF output height (default: 900)")
    parser.add_argument("--fps", type=int, default=2,
                       help="Target frame rate (default: 2)")
    parser.add_argument("--max-frames", type=int, default=None,
                       help="Maximum frames to generate (default: all)")
    
    args = parser.parse_args()
    
    # Load QA data
    print("Loading QA data...")
    qa_loader = NuScenesQALoader(
        qa_data_dir=args.qa_dir,
        nuscenes_dataroot=args.nuscenes_dir,
    ).load()
    
    # Create matcher
    print("Creating scenario matcher...")
    matcher = ScenarioQAMatcher(qa_loader, args.scenario_dir)
    
    # List scenes mode
    if args.list_scenes:
        list_available_scenes(matcher)
        return
    
    # Select scene
    matching_scenes = matcher.get_matching_scenes()
    if not matching_scenes:
        print("Error: No scenes found with both QA data and scenario data.")
        return
    
    if not args.scene:
        print("\nError: Please specify --scene or use --list-scenes to see available scenes")
        print(f"\nAvailable scenes: {', '.join(sorted(matching_scenes)[:5])}...")
        return
    
    scene_name = args.scene
    if scene_name not in matching_scenes:
        print(f"Error: Scene '{scene_name}' not found or missing QA data.")
        print(f"\nUse --list-scenes to see available scenes.")
        return
    
    # Set default output path if not specified
    if not args.output:
        os.makedirs(DEFAULT_OUTPUT_DIR, exist_ok=True)
        args.output = os.path.join(DEFAULT_OUTPUT_DIR, f"{scene_name}.gif")
    
    # Create output directory
    output_dir = os.path.dirname(args.output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Generate GIF
    print(f"\n{'='*60}")
    print(f"Generating GIF for scene: {scene_name}")
    print(f"{'='*60}")
    
    config = GIFConfig(
        width=args.width,
        height=args.height,
    )
    
    generator = GIFGenerator(qa_loader, matcher, config)
    
    print(f"\nOutput: {args.output}")
    print(f"Size: {args.width}x{args.height}")
    print(f"FPS: {args.fps}")
    if args.max_frames:
        print(f"Max frames: {args.max_frames}")
    print()
    
    generator.generate(
        scene_name=scene_name,
        output_path=args.output,
        fps=args.fps,
        max_frames=args.max_frames,
    )
    
    print(f"\n✓ GIF saved to: {args.output}")


if __name__ == "__main__":
    main()
