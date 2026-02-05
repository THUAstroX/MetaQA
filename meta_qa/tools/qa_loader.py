"""
NuScenes-QA Data Loader.

This module provides utilities for loading and indexing NuScenes-QA data,
and matching it with ScenarioNet converted scene data.
"""

import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
import pickle


@dataclass
class QAItem:
    """Single QA item from NuScenes-QA dataset."""
    sample_token: str
    question: str
    answer: str
    template_type: str
    num_hop: int
    split: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "sample_token": self.sample_token,
            "question": self.question,
            "answer": self.answer,
            "template_type": self.template_type,
            "num_hop": self.num_hop,
            "split": self.split,
        }


@dataclass
class SampleQAData:
    """All QA items associated with a single sample (frame)."""
    sample_token: str
    scene_token: Optional[str] = None
    scene_name: Optional[str] = None
    timestamp: Optional[int] = None
    qa_items: List[QAItem] = field(default_factory=list)
    image_paths: Dict[str, str] = field(default_factory=dict)
    
    def __len__(self) -> int:
        return len(self.qa_items)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sample_token": self.sample_token,
            "scene_token": self.scene_token,
            "scene_name": self.scene_name,
            "timestamp": self.timestamp,
            "qa_items": [q.to_dict() for q in self.qa_items],
            "image_paths": self.image_paths,
        }


@dataclass  
class SceneQAData:
    """All QA data associated with a single scene."""
    scene_name: str
    scene_token: str
    description: str = ""
    samples: Dict[str, SampleQAData] = field(default_factory=dict)
    sample_order: List[str] = field(default_factory=list)  # Ordered sample tokens
    
    def __len__(self) -> int:
        return len(self.samples)
    
    @property
    def total_qa_count(self) -> int:
        return sum(len(s) for s in self.samples.values())
    
    def get_sample_by_index(self, idx: int) -> Optional[SampleQAData]:
        """Get sample by temporal index."""
        if 0 <= idx < len(self.sample_order):
            return self.samples.get(self.sample_order[idx])
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "scene_name": self.scene_name,
            "scene_token": self.scene_token,
            "description": self.description,
            "samples": {k: v.to_dict() for k, v in self.samples.items()},
            "sample_order": self.sample_order,
        }


class NuScenesQALoader:
    """
    Loader for NuScenes-QA dataset that integrates with NuScenes metadata.
    
    This loader:
    1. Loads QA data from JSON files
    2. Indexes QA items by sample_token
    3. Links samples to scenes using NuScenes metadata
    4. Provides image path lookup for each sample
    """
    
    CAMERA_TYPES = [
        'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_FRONT',
        'CAM_BACK_LEFT', 'CAM_BACK_RIGHT', 'CAM_BACK'
    ]
    
    def __init__(
        self,
        qa_data_dir: str,
        nuscenes_dataroot: str,
        version: str = "v1.0-mini",
        split: str = "all",
    ):
        """
        Initialize the QA loader.
        
        Args:
            qa_data_dir: Directory containing NuScenes_QA_*.json files
            nuscenes_dataroot: Root directory of NuScenes dataset
            version: NuScenes version (e.g., 'v1.0-mini', 'v1.0-trainval')
            split: QA split to load ('train', 'val', or 'all' for both)
        """
        self.qa_data_dir = qa_data_dir
        self.nuscenes_dataroot = nuscenes_dataroot
        self.version = version
        self.split = split
        
        # Data storage
        self.qa_items: List[QAItem] = []
        self.sample_to_qa: Dict[str, SampleQAData] = {}
        self.scene_to_data: Dict[str, SceneQAData] = {}
        
        # NuScenes metadata
        self.samples: Dict[str, Dict] = {}
        self.scenes: Dict[str, Dict] = {}
        self.sample_data: List[Dict] = []
        
        # Mappings
        self.sample_to_scene: Dict[str, str] = {}  # sample_token -> scene_token
        self.scene_token_to_name: Dict[str, str] = {}  # scene_token -> scene_name
        self.sample_to_images: Dict[str, Dict[str, str]] = {}  # sample_token -> {cam: path}
        
        self._loaded = False
    
    def load(self) -> "NuScenesQALoader":
        """Load all data. Returns self for chaining."""
        self._load_nuscenes_metadata()
        self._load_qa_data()
        self._build_scene_data()
        self._loaded = True
        return self
    
    def _load_nuscenes_metadata(self):
        """Load NuScenes metadata files."""
        # Try different path patterns
        possible_meta_dirs = [
            os.path.join(self.nuscenes_dataroot, self.version),  # v1.0-mini/v1.0-mini
            os.path.join(self.nuscenes_dataroot, self.version, self.version),  # old pattern
            self.nuscenes_dataroot,  # direct path
        ]
        
        meta_dir = None
        for path in possible_meta_dirs:
            if os.path.exists(os.path.join(path, "sample.json")):
                meta_dir = path
                break
        
        if meta_dir is None:
            raise FileNotFoundError(
                f"Could not find NuScenes metadata. Tried: {possible_meta_dirs}"
            )
        
        # Load samples
        with open(os.path.join(meta_dir, "sample.json"), 'r') as f:
            samples_list = json.load(f)
            self.samples = {s['token']: s for s in samples_list}
            for s in samples_list:
                self.sample_to_scene[s['token']] = s['scene_token']
        
        # Load scenes
        with open(os.path.join(meta_dir, "scene.json"), 'r') as f:
            scenes_list = json.load(f)
            self.scenes = {s['token']: s for s in scenes_list}
            self.scene_token_to_name = {s['token']: s['name'] for s in scenes_list}
        
        # Load sample_data for image paths
        with open(os.path.join(meta_dir, "sample_data.json"), 'r') as f:
            self.sample_data = json.load(f)
        
        # Build sample -> images mapping
        # Each sample_data entry corresponds to one sensor (camera) data
        for sd in self.sample_data:
            if sd.get('is_key_frame', False):
                sample_tok = sd['sample_token']
                filename = sd['filename']
                
                # Determine which camera this file belongs to
                for cam in self.CAMERA_TYPES:
                    if cam in filename:
                        if sample_tok not in self.sample_to_images:
                            self.sample_to_images[sample_tok] = {}
                        self.sample_to_images[sample_tok][cam] = filename
                        break
        
        print(f"Loaded NuScenes metadata: {len(self.samples)} samples, {len(self.scenes)} scenes")
    
    def _load_qa_data(self):
        """Load QA data from JSON file(s)."""
        # Determine which splits to load
        if self.split == "all":
            splits_to_load = ["train", "val"]
        else:
            splits_to_load = [self.split]
        
        total_loaded = 0
        for split_name in splits_to_load:
            qa_file = os.path.join(self.qa_data_dir, f"NuScenes_QA_{split_name}.json")
            
            if not os.path.exists(qa_file):
                print(f"Warning: QA file not found: {qa_file}")
                continue
            
            with open(qa_file, 'r') as f:
                data = json.load(f)
            
            # Process questions
            count = 0
            for q in data['questions']:
                qa_item = QAItem(
                    sample_token=q['sample_token'],
                    question=q['question'],
                    answer=str(q['answer']),
                    template_type=q.get('template_type', 'unknown'),
                    num_hop=q.get('num_hop', 0),
                    split=q.get('split', split_name),
                )
                self.qa_items.append(qa_item)
                
                # Group by sample_token
                if qa_item.sample_token not in self.sample_to_qa:
                    self.sample_to_qa[qa_item.sample_token] = SampleQAData(
                        sample_token=qa_item.sample_token
                    )
                self.sample_to_qa[qa_item.sample_token].qa_items.append(qa_item)
                count += 1
            
            print(f"  Loaded {count} QA items from {split_name} split")
            total_loaded += count
        
        print(f"Loaded {total_loaded} QA items for {len(self.sample_to_qa)} samples")
    
    def _build_scene_data(self):
        """Build scene-level data structures."""
        # First, enrich sample data with scene info and images
        for sample_token, sample_qa in self.sample_to_qa.items():
            if sample_token in self.sample_to_scene:
                scene_token = self.sample_to_scene[sample_token]
                sample_qa.scene_token = scene_token
                sample_qa.scene_name = self.scene_token_to_name.get(scene_token)
                
                # Add timestamp
                if sample_token in self.samples:
                    sample_qa.timestamp = self.samples[sample_token].get('timestamp')
                
                # Add image paths
                if sample_token in self.sample_to_images:
                    sample_qa.image_paths = self.sample_to_images[sample_token]
        
        # Group samples by scene
        scene_samples: Dict[str, List[Tuple[int, str]]] = defaultdict(list)
        for sample_token, sample_qa in self.sample_to_qa.items():
            if sample_qa.scene_name:
                timestamp = sample_qa.timestamp or 0
                scene_samples[sample_qa.scene_name].append((timestamp, sample_token))
        
        # Create SceneQAData objects
        for scene_name, samples_list in scene_samples.items():
            # Sort by timestamp
            samples_list.sort(key=lambda x: x[0])
            sample_order = [tok for _, tok in samples_list]
            
            # Find scene token and description
            scene_token = None
            description = ""
            for tok, scene in self.scenes.items():
                if scene['name'] == scene_name:
                    scene_token = tok
                    description = scene.get('description', '')
                    break
            
            self.scene_to_data[scene_name] = SceneQAData(
                scene_name=scene_name,
                scene_token=scene_token or "",
                description=description,
                samples={tok: self.sample_to_qa[tok] for tok in sample_order},
                sample_order=sample_order,
            )
        
        print(f"Built scene data for {len(self.scene_to_data)} scenes")
    
    def get_qa_for_sample(self, sample_token: str) -> Optional[SampleQAData]:
        """Get QA data for a specific sample."""
        return self.sample_to_qa.get(sample_token)
    
    def get_qa_for_scene(self, scene_name: str) -> Optional[SceneQAData]:
        """Get all QA data for a scene."""
        return self.scene_to_data.get(scene_name)
    
    def get_matching_scenes(self) -> List[str]:
        """Get list of scenes that have QA data."""
        return list(self.scene_to_data.keys())
    
    def get_image_path(
        self, 
        sample_token: str, 
        camera: str = "CAM_FRONT",
        absolute: bool = True
    ) -> Optional[str]:
        """
        Get image path for a sample and camera.
        
        Args:
            sample_token: The sample token
            camera: Camera type (e.g., 'CAM_FRONT')
            absolute: If True, return absolute path
            
        Returns:
            Path to image file or None
        """
        images = self.sample_to_images.get(sample_token, {})
        rel_path = images.get(camera)
        
        if rel_path is None:
            return None
        
        if absolute:
            # Try different base paths
            possible_paths = [
                os.path.join(self.nuscenes_dataroot, rel_path),  # dataroot/samples/...
                os.path.join(self.nuscenes_dataroot, self.version, rel_path),  # old pattern
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
            # Return first option even if not found (for error reporting)
            return possible_paths[0]
        return rel_path
    
    def get_all_images_for_sample(
        self, 
        sample_token: str,
        absolute: bool = True
    ) -> Dict[str, str]:
        """Get all camera images for a sample."""
        result = {}
        for cam in self.CAMERA_TYPES:
            path = self.get_image_path(sample_token, cam, absolute)
            if path:
                result[cam] = path
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the loaded data."""
        if not self._loaded:
            return {"error": "Data not loaded"}
        
        # Count QA by type
        type_counts = defaultdict(int)
        for qa in self.qa_items:
            type_counts[qa.template_type] += 1
        
        return {
            "total_qa_items": len(self.qa_items),
            "total_samples_with_qa": len(self.sample_to_qa),
            "total_scenes_with_qa": len(self.scene_to_data),
            "qa_type_distribution": dict(type_counts),
            "scenes": list(self.scene_to_data.keys()),
        }
    
    def save_index(self, output_path: str):
        """Save the index to a pickle file for faster loading."""
        data = {
            "sample_to_qa": {k: v.to_dict() for k, v in self.sample_to_qa.items()},
            "scene_to_data": {k: v.to_dict() for k, v in self.scene_to_data.items()},
            "sample_to_images": self.sample_to_images,
            "sample_to_scene": self.sample_to_scene,
            "scene_token_to_name": self.scene_token_to_name,
        }
        with open(output_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved index to {output_path}")


class ScenarioQAMatcher:
    """
    Matches QA data with ScenarioNet converted scenario data.
    
    This class bridges the gap between:
    - NuScenes-QA data (indexed by sample_token)
    - ScenarioNet scenarios (indexed by scene_name like 'scene-0061')
    
    It also handles the temporal mapping between:
    - NuScenes samples (keyframes at ~2Hz)
    - ScenarioNet time steps (at 10Hz after interpolation)
    """
    
    def __init__(
        self,
        qa_loader: NuScenesQALoader,
        scenario_dir: str,
    ):
        """
        Initialize the matcher.
        
        Args:
            qa_loader: Loaded NuScenesQALoader instance
            scenario_dir: Directory containing ScenarioNet converted scenarios
        """
        self.qa_loader = qa_loader
        self.scenario_dir = scenario_dir
        
        # Load scenario metadata
        self.scenarios: Dict[str, Dict] = {}
        self.scene_to_scenario_path: Dict[str, str] = {}
        
        self._load_scenarios()
    
    def _load_scenarios(self):
        """Load scenario metadata from converted directory."""
        # Find all scenario pkl files
        for subdir in os.listdir(self.scenario_dir):
            subdir_path = os.path.join(self.scenario_dir, subdir)
            if os.path.isdir(subdir_path):
                for fname in os.listdir(subdir_path):
                    if fname.endswith('.pkl') and fname.startswith('sd_'):
                        pkl_path = os.path.join(subdir_path, fname)
                        
                        # Extract scene name from filename
                        # e.g., 'sd_nuscenes_v1.0-mini_scene-0061.pkl' -> 'scene-0061'
                        parts = fname.replace('.pkl', '').split('_')
                        scene_name = parts[-1]  # Last part is scene name
                        
                        self.scene_to_scenario_path[scene_name] = pkl_path
        
        print(f"Found {len(self.scene_to_scenario_path)} scenario files")
    
    def get_matching_scenes(self) -> List[str]:
        """Get scenes that have both QA data and scenario data."""
        qa_scenes = set(self.qa_loader.get_matching_scenes())
        scenario_scenes = set(self.scene_to_scenario_path.keys())
        return list(qa_scenes.intersection(scenario_scenes))
    
    def load_scenario(self, scene_name: str) -> Optional[Dict]:
        """Load a scenario by scene name."""
        pkl_path = self.scene_to_scenario_path.get(scene_name)
        if pkl_path is None:
            return None
        
        with open(pkl_path, 'rb') as f:
            return pickle.load(f)
    
    def get_qa_for_timestep(
        self,
        scene_name: str,
        timestep: int,
        sample_rate: float = 0.1,
        keyframe_rate: float = 0.5,
    ) -> Optional[SampleQAData]:
        """
        Get QA data for a specific timestep in a scenario.
        
        Since NuScenes keyframes are at ~2Hz (0.5s interval) and
        ScenarioNet runs at 10Hz (0.1s interval), we need to map
        timesteps to the nearest keyframe.
        
        Args:
            scene_name: Scene name (e.g., 'scene-0061')
            timestep: ScenarioNet timestep index
            sample_rate: ScenarioNet sample rate (default 0.1s = 10Hz)
            keyframe_rate: NuScenes keyframe rate (default 0.5s = 2Hz)
            
        Returns:
            SampleQAData for the nearest keyframe, or None
        """
        scene_qa = self.qa_loader.get_qa_for_scene(scene_name)
        if scene_qa is None or len(scene_qa.sample_order) == 0:
            return None
        
        # Calculate which keyframe index this timestep corresponds to
        time = timestep * sample_rate
        keyframe_idx = int(time / keyframe_rate)
        
        # Clamp to valid range
        keyframe_idx = max(0, min(keyframe_idx, len(scene_qa.sample_order) - 1))
        
        return scene_qa.get_sample_by_index(keyframe_idx)
    
    def create_enhanced_scenario(
        self,
        scene_name: str,
        output_path: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Create an enhanced scenario with QA annotations embedded.
        
        This adds a 'qa_annotations' field to the scenario metadata,
        containing QA data indexed by keyframe.
        
        Args:
            scene_name: Scene name to enhance
            output_path: Optional path to save enhanced scenario
            
        Returns:
            Enhanced scenario dict, or None if scene not found
        """
        # Load base scenario
        scenario = self.load_scenario(scene_name)
        if scenario is None:
            print(f"Scenario not found: {scene_name}")
            return None
        
        # Get QA data
        scene_qa = self.qa_loader.get_qa_for_scene(scene_name)
        if scene_qa is None:
            print(f"No QA data for scene: {scene_name}")
            return scenario
        
        # Add QA annotations to metadata
        qa_annotations = {
            "scene_name": scene_name,
            "total_qa_count": scene_qa.total_qa_count,
            "keyframe_count": len(scene_qa.sample_order),
            "samples": {},
        }
        
        for idx, sample_token in enumerate(scene_qa.sample_order):
            sample_qa = scene_qa.samples[sample_token]
            qa_annotations["samples"][sample_token] = {
                "keyframe_index": idx,
                "qa_items": [q.to_dict() for q in sample_qa.qa_items],
                "image_paths": sample_qa.image_paths,
                "timestamp": sample_qa.timestamp,
            }
        
        # Add to scenario metadata
        if "metadata" not in scenario:
            scenario["metadata"] = {}
        scenario["metadata"]["qa_annotations"] = qa_annotations
        
        # Save if output path provided
        if output_path:
            with open(output_path, 'wb') as f:
                pickle.dump(scenario, f)
            print(f"Saved enhanced scenario to {output_path}")
        
        return scenario


def main():
    """Test the QA loader."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test NuScenes-QA Loader")
    parser.add_argument("--qa_dir", type=str, 
                        default="./dataset/QA_Data",
                        help="Path to QA data directory")
    parser.add_argument("--nuscenes_dir", type=str,
                        default="./dataset/Scenario_Data/exp_nuscenes/v1.0-mini",
                        help="Path to NuScenes data directory")
    parser.add_argument("--scenario_dir", type=str,
                        default="./dataset/Scenario_Data/exp_nuscenes_converted",
                        help="Path to ScenarioNet converted directory")
    parser.add_argument("--version", type=str, default="v1.0-mini")
    
    args = parser.parse_args()
    
    # Load QA data
    print("Loading QA data...")
    qa_loader = NuScenesQALoader(
        qa_data_dir=args.qa_dir,
        nuscenes_dataroot=args.nuscenes_dir,
        version=args.version,
    ).load()
    
    # Print stats
    stats = qa_loader.get_stats()
    print("\n=== QA Data Statistics ===")
    print(f"Total QA items: {stats['total_qa_items']}")
    print(f"Samples with QA: {stats['total_samples_with_qa']}")
    print(f"Scenes with QA: {stats['total_scenes_with_qa']}")
    print(f"Scenes: {stats['scenes']}")
    print(f"QA types: {stats['qa_type_distribution']}")
    
    # Test matcher
    print("\n=== Testing Scenario Matcher ===")
    matcher = ScenarioQAMatcher(qa_loader, args.scenario_dir)
    matching_scenes = matcher.get_matching_scenes()
    print(f"Scenes with both QA and scenario data: {matching_scenes}")
    
    # Test QA lookup
    if matching_scenes:
        test_scene = matching_scenes[0]
        print(f"\n=== Sample QA for {test_scene} ===")
        
        scene_qa = qa_loader.get_qa_for_scene(test_scene)
        if scene_qa:
            print(f"Total QA items: {scene_qa.total_qa_count}")
            print(f"Sample count: {len(scene_qa.samples)}")
            
            # Show first sample's QA
            first_sample = scene_qa.get_sample_by_index(0)
            if first_sample:
                print(f"\nFirst sample ({first_sample.sample_token}):")
                print(f"  Images: {list(first_sample.image_paths.keys())}")
                for qa in first_sample.qa_items[:3]:
                    print(f"  Q: {qa.question}")
                    print(f"  A: {qa.answer}")
                    print()


if __name__ == "__main__":
    main()
