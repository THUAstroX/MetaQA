"""
NuScenes Original Frequency Data Processor.

This module processes NuScenes data at the original camera frequency (~12Hz)
instead of the 2Hz sample/keyframe rate. It extracts all sweep images and their
timestamps, determines sample boundaries, and enables high-frequency data replay.

Key concepts:
- Samples (Keyframes): 2Hz annotated frames with ground truth annotations
- Sweeps: ~12Hz raw sensor data between samples  
- sample_data: Contains all sensor readings (both samples and sweeps)

Based on nuScenes-H from ASAP (https://github.com/JeffWang987/ASAP)
"""

import os
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np


@dataclass
class FrameData:
    """
    Data for a single frame (either keyframe or sweep).
    
    Attributes:
        token: Unique frame identifier (from sample or sample_data)
        timestamp: Frame timestamp in microseconds
        is_keyframe: Whether this is an annotated keyframe
        keyframe_index: Index of the keyframe this frame belongs to (or is)
        prev_keyframe_token: Token of previous keyframe (None if this is first)
        next_keyframe_token: Token of next keyframe (None if this is last or is keyframe)
        interpolation_ratio: Position between keyframes [0, 1] (0 = at prev keyframe)
        image_paths: Dict of camera type to image file path
        sample_token: Associated sample token (keyframe token)
        ego_pose_token: Token for ego pose data
        calibrated_sensor_token: Token for calibrated sensor data
    """
    token: str
    timestamp: int  # microseconds
    is_keyframe: bool = False
    keyframe_index: int = -1
    prev_keyframe_token: Optional[str] = None
    next_keyframe_token: Optional[str] = None
    interpolation_ratio: float = 0.0
    image_paths: Dict[str, str] = field(default_factory=dict)
    sample_token: Optional[str] = None
    ego_pose_token: Optional[str] = None
    calibrated_sensor_token: Optional[str] = None
    
    @property
    def frame_id(self) -> str:
        """Generate unique frame ID for interpolated data."""
        if self.is_keyframe:
            return self.token
        else:
            # For sweeps, create ID from sample_token + index
            return f"{self.sample_token}_{int(self.interpolation_ratio * 6)}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "token": self.token,
            "timestamp": self.timestamp,
            "is_keyframe": self.is_keyframe,
            "keyframe_index": self.keyframe_index,
            "prev_keyframe_token": self.prev_keyframe_token,
            "next_keyframe_token": self.next_keyframe_token,
            "interpolation_ratio": self.interpolation_ratio,
            "image_paths": self.image_paths,
            "sample_token": self.sample_token,
            "ego_pose_token": self.ego_pose_token,
            "calibrated_sensor_token": self.calibrated_sensor_token,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FrameData":
        """Create from dictionary."""
        return cls(
            token=data["token"],
            timestamp=data["timestamp"],
            is_keyframe=data.get("is_keyframe", False),
            keyframe_index=data.get("keyframe_index", -1),
            prev_keyframe_token=data.get("prev_keyframe_token"),
            next_keyframe_token=data.get("next_keyframe_token"),
            interpolation_ratio=data.get("interpolation_ratio", 0.0),
            image_paths=data.get("image_paths", {}),
            sample_token=data.get("sample_token"),
            ego_pose_token=data.get("ego_pose_token"),
            calibrated_sensor_token=data.get("calibrated_sensor_token"),
        )


@dataclass
class SceneOriginalData:
    """
    Original frequency data for a complete scene.
    
    Contains all frames (samples and sweeps) ordered by timestamp,
    with sample boundary information for interpolation.
    """
    scene_name: str
    scene_token: str
    description: str = ""
    
    # All frames ordered by timestamp
    frames: List[FrameData] = field(default_factory=list)
    
    # Sample (keyframe) indices for quick lookup
    sample_indices: List[int] = field(default_factory=list)
    sample_tokens: List[str] = field(default_factory=list)
    
    # Frame rate information
    original_fps: float = 12.0  # Original sensor frame rate
    actual_fps: float = 0.0     # Computed actual frame rate
    sample_fps: float = 2.0     # Sample (keyframe) rate
    
    @property
    def num_frames(self) -> int:
        return len(self.frames)
    
    @property
    def num_samples(self) -> int:
        return len(self.sample_indices)
    
    # Alias for backward compatibility
    @property
    def num_keyframes(self) -> int:
        return self.num_samples
    
    @property
    def keyframe_indices(self) -> List[int]:
        return self.sample_indices
    
    @property
    def keyframe_tokens(self) -> List[str]:
        return self.sample_tokens
    
    @property
    def duration_seconds(self) -> float:
        """Scene duration in seconds."""
        if len(self.frames) < 2:
            return 0.0
        return (self.frames[-1].timestamp - self.frames[0].timestamp) / 1e6
    
    def get_frame_by_index(self, idx: int) -> Optional[FrameData]:
        """Get frame by index."""
        if 0 <= idx < len(self.frames):
            return self.frames[idx]
        return None
    
    def get_sample_by_index(self, sample_idx: int) -> Optional[FrameData]:
        """Get sample (keyframe) by sample index."""
        if 0 <= sample_idx < len(self.sample_indices):
            frame_idx = self.sample_indices[sample_idx]
            return self.frames[frame_idx]
        return None
    
    def get_frames_between_samples(
        self, sample_idx: int
    ) -> List[FrameData]:
        """Get all frames between sample sample_idx and sample_idx+1."""
        if sample_idx < 0 or sample_idx >= len(self.sample_indices) - 1:
            return []
        
        start = self.sample_indices[sample_idx]
        end = self.sample_indices[sample_idx + 1]
        return self.frames[start:end + 1]
    
    def find_sample_boundary(self, timestamp: int) -> Tuple[int, int, float]:
        """
        Find the sample boundary for a given timestamp.
        
        Returns:
            Tuple of (prev_sample_idx, next_sample_idx, interpolation_ratio)
        """
        # Find which samples this timestamp falls between
        for i in range(len(self.sample_indices) - 1):
            sample_start = self.frames[self.sample_indices[i]]
            sample_end = self.frames[self.sample_indices[i + 1]]
            
            if sample_start.timestamp <= timestamp <= sample_end.timestamp:
                ratio = (timestamp - sample_start.timestamp) / (sample_end.timestamp - sample_start.timestamp)
                return i, i + 1, ratio
        
        # Handle edge cases
        if timestamp < self.frames[self.sample_indices[0]].timestamp:
            return 0, 0, 0.0
        return len(self.sample_indices) - 1, len(self.sample_indices) - 1, 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scene_name": self.scene_name,
            "scene_token": self.scene_token,
            "description": self.description,
            "frames": [f.to_dict() for f in self.frames],
            "sample_indices": self.sample_indices,
            "sample_tokens": self.sample_tokens,
            "original_fps": self.original_fps,
            "actual_fps": self.actual_fps,
            "sample_fps": self.sample_fps,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SceneOriginalData":
        """Create from dictionary."""
        scene = cls(
            scene_name=data["scene_name"],
            scene_token=data["scene_token"],
            description=data.get("description", ""),
            frames=[FrameData.from_dict(f) for f in data["frames"]],
            sample_indices=data.get("sample_indices", data.get("keyframe_indices", [])),
            sample_tokens=data.get("sample_tokens", data.get("keyframe_tokens", [])),
            original_fps=data.get("original_fps", data.get("target_fps", 12.0)),
            actual_fps=data.get("actual_fps", 0.0),
            sample_fps=data.get("sample_fps", data.get("keyframe_fps", 2.0)),
        )
        return scene



class NuScenesOriginalProcessor:
    """
    Processor for extracting original frequency frame data from NuScenes dataset.
    
    NuScenes captures sensor data at original frequency (~12Hz for cameras),
    but only provides annotations at sample rate (2Hz keyframes).
    
    This processor:
    1. Loads NuScenes metadata (samples, sample_data, scenes, etc.)
    2. For each scene, extracts all sweep images at original frequency
    3. Determines sample (keyframe) boundaries and interpolation ratios
    4. Builds SceneOriginalData structures for downstream use
    
    Terminology:
    - "sample": NuScenes keyframe at 2Hz with annotations
    - "sweep": Inter-sample frames at original ~12Hz frequency
    - "frame": Any frame (sample or sweep)
    """
    
    CAMERA_TYPES = [
        'CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT',
        'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'
    ]
    
    # Primary camera for determining frame sequence
    PRIMARY_CAMERA = 'CAM_FRONT'
    
    def __init__(
        self,
        nuscenes_dataroot: str,
        version: str = "v1.0-mini",
        original_fps: float = 12.0,
    ):
        """
        Initialize the original frequency processor.
        
        Args:
            nuscenes_dataroot: Root directory of NuScenes dataset
            version: NuScenes version
            original_fps: Expected original frame rate (nominal ~12Hz)
        """
        self.nuscenes_dataroot = nuscenes_dataroot
        self.version = version
        self.original_fps = original_fps
        
        # NuScenes metadata
        self.samples: Dict[str, Dict] = {}
        self.sample_data: Dict[str, Dict] = {}
        self.scenes: Dict[str, Dict] = {}
        self.sensors: Dict[str, Dict] = {}
        self.ego_poses: Dict[str, Dict] = {}
        self.calibrated_sensors: Dict[str, Dict] = {}
        
        # Processed data
        self.scene_original_data: Dict[str, SceneOriginalData] = {}
        
        self._loaded = False
    
    def load(self) -> "NuScenesOriginalProcessor":
        """Load NuScenes metadata. Returns self for chaining."""
        self._load_metadata()
        self._loaded = True
        return self
    
    def _find_metadata_dir(self) -> str:
        """Find the metadata directory."""
        possible_dirs = [
            os.path.join(self.nuscenes_dataroot, self.version),
            os.path.join(self.nuscenes_dataroot, self.version, self.version),
            self.nuscenes_dataroot,
        ]
        
        for path in possible_dirs:
            if os.path.exists(os.path.join(path, "sample.json")):
                return path
        
        raise FileNotFoundError(
            f"Could not find NuScenes metadata in: {possible_dirs}"
        )
    
    def _load_metadata(self):
        """Load all NuScenes metadata files."""
        meta_dir = self._find_metadata_dir()
        print(f"Loading NuScenes metadata from: {meta_dir}")
        
        # Load samples (keyframes)
        with open(os.path.join(meta_dir, "sample.json"), 'r') as f:
            samples_list = json.load(f)
            self.samples = {s['token']: s for s in samples_list}
        
        # Load sample_data (all sensor readings including sweeps)
        with open(os.path.join(meta_dir, "sample_data.json"), 'r') as f:
            sample_data_list = json.load(f)
            self.sample_data = {sd['token']: sd for sd in sample_data_list}
        
        # Load scenes
        with open(os.path.join(meta_dir, "scene.json"), 'r') as f:
            scenes_list = json.load(f)
            self.scenes = {s['token']: s for s in scenes_list}
        
        # Load sensors
        with open(os.path.join(meta_dir, "sensor.json"), 'r') as f:
            sensors_list = json.load(f)
            self.sensors = {s['token']: s for s in sensors_list}
        
        # Load ego_pose
        with open(os.path.join(meta_dir, "ego_pose.json"), 'r') as f:
            ego_poses_list = json.load(f)
            self.ego_poses = {ep['token']: ep for ep in ego_poses_list}
        
        # Load calibrated_sensor
        with open(os.path.join(meta_dir, "calibrated_sensor.json"), 'r') as f:
            calibrated_sensors_list = json.load(f)
            self.calibrated_sensors = {cs['token']: cs for cs in calibrated_sensors_list}
        
        # Build sample -> sensor data mapping
        # NuScenes sample.json doesn't contain 'data' field directly
        # We need to build it from sample_data.json
        self._build_sample_data_mapping()
        
        print(f"  Samples (keyframes): {len(self.samples)}")
        print(f"  Sample_data (all readings): {len(self.sample_data)}")
        print(f"  Scenes: {len(self.scenes)}")
    
    def get_available_scenes(self) -> List[str]:
        """
        Get list of available scene names.
        
        Returns:
            List of scene names (e.g., ["scene-0061", "scene-0103", ...])
        """
        if not self._loaded:
            self.load()
        
        return sorted([scene['name'] for scene in self.scenes.values()])
    
    def get_scene_token_by_name(self, scene_name: str) -> Optional[str]:
        """
        Get scene token by scene name.
        
        Args:
            scene_name: Scene name (e.g., "scene-0061")
            
        Returns:
            Scene token or None if not found
        """
        if not self._loaded:
            self.load()
        
        for token, scene in self.scenes.items():
            if scene['name'] == scene_name:
                return token
        return None
    
    def _build_sample_data_mapping(self):
        """
        Build mapping from sample_token to sensor data tokens.
        
        NuScenes sample.json doesn't have 'data' field - we need to 
        construct it from sample_data.json by finding all keyframe
        sample_data entries for each sample.
        """
        # For each sample, collect all sample_data tokens grouped by sensor
        for sd in self.sample_data.values():
            if sd.get('is_key_frame', False):
                sample_token = sd['sample_token']
                if sample_token in self.samples:
                    # Initialize data dict if needed
                    if 'data' not in self.samples[sample_token]:
                        self.samples[sample_token]['data'] = {}
                    
                    # Get sensor channel name
                    channel = self._get_sensor_channel(sd)
                    if channel:
                        self.samples[sample_token]['data'][channel] = sd['token']
    
    def _get_sensor_channel(self, sample_data_entry: Dict) -> str:
        """Get sensor channel name from sample_data entry."""
        cs_token = sample_data_entry['calibrated_sensor_token']
        cs = self.calibrated_sensors.get(cs_token)
        if cs:
            sensor = self.sensors.get(cs['sensor_token'])
            if sensor:
                return sensor['channel']
        return ""
    
    def process_scene(self, scene_token: str) -> SceneOriginalData:
        """
        Process a scene to extract original frequency frame data.
        
        Args:
            scene_token: Token of the scene to process
            
        Returns:
            SceneOriginalData with all frames at original frequency
        """
        if not self._loaded:
            self.load()
        
        scene = self.scenes[scene_token]
        scene_name = scene['name']
        
        print(f"Processing scene: {scene_name}")
        
        # Step 1: Get all samples (keyframes) for this scene
        sample_list = []
        sample_token = scene['first_sample_token']
        while sample_token:
            sample = self.samples.get(sample_token)
            if sample:
                sample_list.append(sample)
                sample_token = sample.get('next', '')
            else:
                break
        
        print(f"  Found {len(sample_list)} samples (keyframes)")
        
        # Step 2: For each sample, get the primary camera sample_data
        # and traverse to get all sweep frames
        all_frames: List[FrameData] = []
        sample_indices: List[int] = []
        sample_tokens: List[str] = []
        
        for sample_idx, sample in enumerate(sample_list):
            # Get primary camera data token for this sample
            primary_cam_token = sample['data'].get(self.PRIMARY_CAMERA)
            if not primary_cam_token:
                continue
            
            # Get all image paths for this sample
            sample_images = {}
            for cam in self.CAMERA_TYPES:
                cam_token = sample['data'].get(cam)
                if cam_token and cam_token in self.sample_data:
                    sample_images[cam] = self.sample_data[cam_token]['filename']
            
            # Create sample (keyframe) entry
            primary_sd = self.sample_data[primary_cam_token]
            sample_frame = FrameData(
                token=sample['token'],
                timestamp=sample['timestamp'],
                is_keyframe=True,
                keyframe_index=sample_idx,
                prev_keyframe_token=sample_list[sample_idx - 1]['token'] if sample_idx > 0 else None,
                next_keyframe_token=sample_list[sample_idx + 1]['token'] if sample_idx < len(sample_list) - 1 else None,
                interpolation_ratio=0.0,
                image_paths=sample_images,
                sample_token=sample['token'],
                ego_pose_token=primary_sd.get('ego_pose_token'),
                calibrated_sensor_token=primary_sd.get('calibrated_sensor_token'),
            )
            
            sample_indices.append(len(all_frames))
            sample_tokens.append(sample['token'])
            all_frames.append(sample_frame)
            
            # Get sweep frames until next sample
            if sample_idx < len(sample_list) - 1:
                next_sample = sample_list[sample_idx + 1]
                next_timestamp = next_sample['timestamp']
                
                # Traverse sample_data chain from primary camera
                sd_token = primary_sd.get('next', '')
                sweep_count = 0
                
                while sd_token:
                    sd = self.sample_data.get(sd_token)
                    if not sd:
                        break
                    
                    # Check if we've reached or passed the next sample
                    if sd['is_key_frame'] or sd['timestamp'] >= next_timestamp:
                        break
                    
                    # Verify this is from the primary camera
                    if self._get_sensor_channel(sd) != self.PRIMARY_CAMERA:
                        sd_token = sd.get('next', '')
                        continue
                    
                    # Calculate interpolation ratio
                    ratio = (sd['timestamp'] - sample['timestamp']) / (next_timestamp - sample['timestamp'])
                    ratio = max(0.0, min(1.0, ratio))
                    
                    # Get all camera images at this timestamp (approximately)
                    sweep_images = self._get_synchronized_camera_images(sd['timestamp'], sample, next_sample)
                    
                    # Create sweep frame
                    sweep_frame = FrameData(
                        token=sd['token'],
                        timestamp=sd['timestamp'],
                        is_keyframe=False,
                        keyframe_index=sample_idx,
                        prev_keyframe_token=sample['token'],
                        next_keyframe_token=next_sample['token'],
                        interpolation_ratio=ratio,
                        image_paths=sweep_images,
                        sample_token=sample['token'],
                        ego_pose_token=sd.get('ego_pose_token'),
                        calibrated_sensor_token=sd.get('calibrated_sensor_token'),
                    )
                    
                    all_frames.append(sweep_frame)
                    sweep_count += 1
                    sd_token = sd.get('next', '')
                
                # print(f"    Sample {sample_idx}: {sweep_count} sweeps")
        
        # Calculate actual FPS
        if len(all_frames) >= 2:
            duration_us = all_frames[-1].timestamp - all_frames[0].timestamp
            actual_fps = (len(all_frames) - 1) * 1e6 / duration_us if duration_us > 0 else 0
        else:
            actual_fps = 0.0
        
        # Create scene data
        scene_data = SceneOriginalData(
            scene_name=scene_name,
            scene_token=scene_token,
            description=scene.get('description', ''),
            frames=all_frames,
            sample_indices=sample_indices,
            sample_tokens=sample_tokens,
            original_fps=self.original_fps,
            actual_fps=actual_fps,
            sample_fps=len(sample_list) / (all_frames[-1].timestamp - all_frames[0].timestamp) * 1e6 if len(all_frames) >= 2 else 2.0,
        )
        
        print(f"  Total frames: {scene_data.num_frames}")
        print(f"  Actual FPS: {scene_data.actual_fps:.2f}")
        print(f"  Duration: {scene_data.duration_seconds:.2f}s")
        
        # Cache result
        self.scene_original_data[scene_name] = scene_data
        
        return scene_data
    
    def _get_synchronized_camera_images(
        self,
        target_timestamp: int,
        prev_sample: Dict,
        next_sample: Dict,
    ) -> Dict[str, str]:
        """
        Get camera images closest to the target timestamp.
        
        For sweep frames, we need to find the closest camera images
        from each camera's sample_data chain.
        """
        images = {}
        
        for cam in self.CAMERA_TYPES:
            # Start from previous keyframe's camera data
            cam_token = prev_sample['data'].get(cam)
            if not cam_token:
                continue
            
            best_sd = None
            best_diff = float('inf')
            
            # Traverse to find closest timestamp
            while cam_token:
                sd = self.sample_data.get(cam_token)
                if not sd:
                    break
                
                # Check if past next keyframe
                if sd['is_key_frame'] and sd['sample_token'] == next_sample['token']:
                    break
                
                diff = abs(sd['timestamp'] - target_timestamp)
                if diff < best_diff:
                    best_diff = diff
                    best_sd = sd
                
                # If we've passed the target, we can stop
                if sd['timestamp'] > target_timestamp and best_sd:
                    break
                
                cam_token = sd.get('next', '')
            
            if best_sd:
                images[cam] = best_sd['filename']
        
        return images
    
    def process_all_scenes(self) -> Dict[str, SceneOriginalData]:
        """Process all scenes in the dataset."""
        if not self._loaded:
            self.load()
        
        for scene_token in self.scenes:
            self.process_scene(scene_token)
        
        return self.scene_original_data
    
    def get_scene_by_name(self, scene_name: str) -> Optional[SceneOriginalData]:
        """Get processed original frequency data for a scene by name."""
        if scene_name in self.scene_original_data:
            return self.scene_original_data[scene_name]
        
        # Try to find and process the scene
        for token, scene in self.scenes.items():
            if scene['name'] == scene_name:
                return self.process_scene(token)
        
        return None
    
    def save_processed_data(self, output_path: str):
        """Save all processed scene data to JSON."""
        data = {
            scene_name: scene_data.to_dict()
            for scene_name, scene_data in self.scene_original_data.items()
        }
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(data, f)
        
        print(f"Saved original frequency data for {len(data)} scenes to {output_path}")
    
    @classmethod
    def load_processed_data(cls, path: str) -> Dict[str, SceneOriginalData]:
        """Load previously processed scene data from JSON."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        return {
            scene_name: SceneOriginalData.from_dict(scene_dict)
            for scene_name, scene_dict in data.items()
        }

