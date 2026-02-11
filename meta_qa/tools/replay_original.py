"""
Original Frequency Replay Environment for NuScenes Data.

This module provides a replay environment that operates at original camera 
frequency (~12Hz) from NuScenes, while synchronizing with:
- ScenarioNet trajectory data (interpolated from 10Hz)
- NuScenes-QA annotations (at 2Hz samples/keyframes)

Terminology:
- "original frequency": The native sensor capture rate (~12Hz for cameras)
- "sample": NuScenes keyframe with annotations (2Hz)
- "sweep": Inter-sample frames at original frequency
- "frame": Any frame (sample or sweep)

The replay provides:
- Frame-by-frame control at original frequency
- Interpolated trajectory states
- QA data at samples (2Hz keyframes)
- Synchronized camera images
"""

import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import numpy as np

from .nuscenes_original import NuScenesOriginalProcessor, SceneOriginalData, FrameData
from .interpolation import (
    TrajectoryInterpolator,
    ScenarioTrajectoryMatcher,
    InterpolatedState,
)


@dataclass
class FrameOriginalInfo:
    """
    Complete information for a single frame at original frequency.
    
    Combines:
    - NuScenes frame data (images, timestamps)
    - Interpolated trajectory states
    - QA data (if at sample/keyframe)
    """
    # Frame identification
    frame_idx: int
    timestamp: int  # microseconds
    is_sample: bool  # True if this is a 2Hz keyframe/sample
    sample_index: int  # Index of current or previous sample
    interpolation_ratio: float
    
    # NuScenes data
    image_paths: Dict[str, str] = field(default_factory=dict)
    sample_token: Optional[str] = None
    
    # Trajectory data
    ego_state: Optional[InterpolatedState] = None
    surrounding_states: List[Tuple[str, InterpolatedState]] = field(default_factory=list)
    
    # QA data (only at samples)
    qa_items: List[Dict] = field(default_factory=list)
    has_qa: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_idx": self.frame_idx,
            "timestamp": self.timestamp,
            "is_sample": self.is_sample,
            "sample_index": self.sample_index,
            "interpolation_ratio": self.interpolation_ratio,
            "image_paths": self.image_paths,
            "sample_token": self.sample_token,
            "ego_state": self.ego_state.to_dict() if self.ego_state else None,
            "surrounding_count": len(self.surrounding_states),
            "qa_count": len(self.qa_items),
            "has_qa": self.has_qa,
        }



class ReplayOriginalEnv:
    """
    Original frequency replay environment for NuScenes scenarios.
    
    This environment provides step-by-step replay at original camera frequency
    (~12Hz), with:
    - Interpolated trajectory data from ScenarioNet (10Hz → original)
    - Synchronized camera images from NuScenes
    - QA annotations at 2Hz samples (keyframes)
    
    Usage:
        env = ReplayOriginalEnv(
            nuscenes_dataroot="path/to/nuscenes",
            scenario_dir="path/to/scenarionet",
            qa_loader=qa_loader,  # Optional NuScenesQALoader
        )
        env.load_scene("scene-0061")
        
        for frame_info in env.iterate_frames():
            if frame_info.is_sample and frame_info.has_qa:
                # Process QA at samples (2Hz keyframes)
                pass
            # Process all frames at original frequency
            ego_state = frame_info.ego_state
            images = frame_info.image_paths
    """
    
    def __init__(
        self,
        nuscenes_dataroot: str,
        scenario_dir: str,
        qa_loader: Any = None,  # NuScenesQALoader
        version: str = "v1.0-mini",
        original_fps: float = 12.0,
    ):
        """
        Initialize the original frequency replay environment.
        
        Args:
            nuscenes_dataroot: Root directory of NuScenes dataset
            scenario_dir: Directory containing ScenarioNet converted scenarios
            qa_loader: Optional NuScenesQALoader for QA annotations
            version: NuScenes version
            original_fps: Expected original frame rate (nominal ~12Hz)
        """
        self.nuscenes_dataroot = nuscenes_dataroot
        self.scenario_dir = scenario_dir
        self.qa_loader = qa_loader
        self.version = version
        self.original_fps = original_fps
        
        # Processors
        self.nuscenes_processor = NuScenesOriginalProcessor(
            nuscenes_dataroot, version, original_fps
        )
        
        # Current scene data
        self.current_scene_name: Optional[str] = None
        self.scene_original_data: Optional[SceneOriginalData] = None
        self.trajectory_matcher: Optional[ScenarioTrajectoryMatcher] = None
        self.scene_qa_data: Optional[Any] = None  # SceneQAData
        
        # Replay state
        self.current_frame_idx: int = 0
        
        self._loaded = False
    
    def load(self) -> "ReplayOriginalEnv":
        """Load base data (NuScenes metadata)."""
        self.nuscenes_processor.load()
        if self.qa_loader and not self.qa_loader._loaded:
            self.qa_loader.load()
        self._loaded = True
        return self
    
    def _find_scenario_file(self, scene_name: str) -> Optional[str]:
        """Find the ScenarioNet file for a scene."""
        # ScenarioNet files are named like: sd_nuscenes_v1.0-mini_scene-0061.pkl
        # They may be in subdirectories
        
        import glob
        pattern = os.path.join(
            self.scenario_dir, 
            "**", 
            f"sd_nuscenes_*_{scene_name}.pkl"
        )
        
        matches = glob.glob(pattern, recursive=True)
        if matches:
            return matches[0]
        
        # Try alternative patterns
        alt_pattern = os.path.join(
            self.scenario_dir,
            "**",
            f"*{scene_name}*.pkl"
        )
        matches = glob.glob(alt_pattern, recursive=True)
        if matches:
            return matches[0]
        
        return None
    
    def load_scene(self, scene_name: str) -> bool:
        """
        Load a specific scene for replay.
        
        Args:
            scene_name: Name of the scene (e.g., "scene-0061")
            
        Returns:
            True if scene loaded successfully
        """
        if not self._loaded:
            self.load()
        
        print(f"\n=== Loading scene: {scene_name} ===")
        
        # Load original frequency NuScenes data
        self.scene_original_data = self.nuscenes_processor.get_scene_by_name(scene_name)
        if self.scene_original_data is None:
            print(f"Error: Could not load NuScenes data for {scene_name}")
            return False
        
        print(f"  Original frames: {self.scene_original_data.num_frames}")
        print(f"  Samples (keyframes): {self.scene_original_data.num_samples}")
        
        # Load ScenarioNet trajectory data
        scenario_path = self._find_scenario_file(scene_name)
        if scenario_path:
            print(f"  Scenario file: {os.path.basename(scenario_path)}")
            
            # Get frame timestamps for interpolation
            frame_timestamps = [f.timestamp for f in self.scene_original_data.frames]
            
            self.trajectory_matcher = ScenarioTrajectoryMatcher(
                scenario_path, frame_timestamps
            )
            self.trajectory_matcher.load()
        else:
            print(f"  Warning: No ScenarioNet file found for {scene_name}")
            self.trajectory_matcher = None
        
        # Load QA data
        if self.qa_loader:
            self.scene_qa_data = self.qa_loader.get_qa_for_scene(scene_name)
            if self.scene_qa_data:
                print(f"  QA items: {self.scene_qa_data.total_qa_count}")
            else:
                print(f"  No QA data for this scene")
        
        self.current_scene_name = scene_name
        self.current_frame_idx = 0
        
        return True
    
    def reset(self) -> Optional[FrameOriginalInfo]:
        """Reset to the first frame of the current scene."""
        self.current_frame_idx = 0
        return self.get_current_frame()
    
    def get_current_frame(self) -> Optional[FrameOriginalInfo]:
        """Get information for the current frame."""
        if self.scene_original_data is None:
            return None
        
        return self.get_frame(self.current_frame_idx)
    
    def get_frame(self, frame_idx: int) -> Optional[FrameOriginalInfo]:
        """
        Get complete information for a specific frame.
        
        Args:
            frame_idx: Frame index (0-based)
            
        Returns:
            FrameOriginalInfo with all data, or None if out of bounds
        """
        if self.scene_original_data is None:
            return None
        
        frame_data = self.scene_original_data.get_frame_by_index(frame_idx)
        if frame_data is None:
            return None
        
        # Get trajectory data
        ego_state = None
        surrounding_states = []
        
        if self.trajectory_matcher:
            ego_state = self.trajectory_matcher.get_ego_state_at_frame(frame_idx)
            surrounding_states = self.trajectory_matcher.get_all_states_at_frame(
                frame_idx, max_vehicles=20
            )
        
        # Get QA data (only at samples/keyframes)
        qa_items = []
        has_qa = False
        
        if frame_data.is_keyframe and self.scene_qa_data:
            sample_token = frame_data.sample_token
            if sample_token in self.scene_qa_data.samples:
                sample_qa = self.scene_qa_data.samples[sample_token]
                qa_items = [q.to_dict() for q in sample_qa.qa_items]
                has_qa = len(qa_items) > 0
        
        return FrameOriginalInfo(
            frame_idx=frame_idx,
            timestamp=frame_data.timestamp,
            is_sample=frame_data.is_keyframe,
            sample_index=frame_data.keyframe_index,
            interpolation_ratio=frame_data.interpolation_ratio,
            image_paths=frame_data.image_paths,
            sample_token=frame_data.sample_token,
            ego_state=ego_state,
            surrounding_states=surrounding_states,
            qa_items=qa_items,
            has_qa=has_qa,
        )
    
    def step(self) -> Tuple[Optional[FrameOriginalInfo], bool]:
        """
        Advance to the next frame.
        
        Returns:
            Tuple of (frame_info, done)
        """
        if self.scene_original_data is None:
            return None, True
        
        self.current_frame_idx += 1
        
        if self.current_frame_idx >= self.scene_original_data.num_frames:
            return None, True
        
        return self.get_current_frame(), False
    
    def iterate_frames(self):
        """
        Generator to iterate through all frames at original frequency.
        
        Yields:
            FrameOriginalInfo for each frame
        """
        self.reset()
        
        while True:
            frame_info = self.get_current_frame()
            if frame_info is None:
                break
            
            yield frame_info
            
            _, done = self.step()
            if done:
                break
    
    def iterate_samples(self):
        """
        Generator to iterate through only samples (2Hz keyframes).
        
        Yields:
            FrameOriginalInfo for each sample
        """
        for frame_info in self.iterate_frames():
            if frame_info.is_sample:
                yield frame_info
    
    def get_image_path(
        self,
        frame_idx: int,
        camera: str = "CAM_FRONT"
    ) -> Optional[str]:
        """
        Get the full image path for a frame and camera.
        
        Args:
            frame_idx: Frame index
            camera: Camera name
            
        Returns:
            Full path to the image file
        """
        frame = self.get_frame(frame_idx)
        if frame is None:
            return None
        
        relative_path = frame.image_paths.get(camera)
        if relative_path is None:
            return None
        
        return os.path.join(self.nuscenes_dataroot, relative_path)
    
    @property
    def num_frames(self) -> int:
        """Total number of frames in the current scene."""
        if self.scene_original_data:
            return self.scene_original_data.num_frames
        return 0
    
    @property
    def num_samples(self) -> int:
        """Number of samples (2Hz keyframes) in the current scene."""
        if self.scene_original_data:
            return self.scene_original_data.num_samples
        return 0
    
    @property
    def duration(self) -> float:
        """Duration of the current scene in seconds."""
        if self.scene_original_data:
            return self.scene_original_data.duration_seconds
        return 0.0
    
    def get_scene_info(self) -> Dict[str, Any]:
        """Get summary information about the current scene."""
        if self.scene_original_data is None:
            return {}
        
        return {
            "scene_name": self.current_scene_name,
            "num_frames": self.num_frames,
            "num_samples": self.num_samples,
            "duration_seconds": self.duration,
            "actual_fps": self.scene_original_data.actual_fps,
            "original_fps": self.original_fps,
            "has_trajectory": self.trajectory_matcher is not None,
            "has_qa": self.scene_qa_data is not None,
            "qa_count": self.scene_qa_data.total_qa_count if self.scene_qa_data else 0,
        }

