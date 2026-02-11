"""
MetaQA Tools Module.

Utility tools for data processing, visualization, and extraction.

Components:
    - structures: MDP/CMDP transition data classes
    - qa_loader: NuScenes-QA data loader & scene matcher
    - nuscenes_original: Original frequency (~12Hz) frame extraction
    - interpolation: Trajectory interpolation (10Hz → ~12Hz)
    - replay_original: Original frequency replay environment
    - trajectory_vis: Trajectory visualization (OpenCV/Matplotlib)
    - surrounding: Surrounding vehicle extraction from MetaDrive
"""

from .structures import (
    MDPTransition,
    CMDPTransition,
)
from .nuscenes_original import (
    FrameData,
    SceneOriginalData,
    NuScenesOriginalProcessor,
)
from .interpolation import (
    InterpolatedState,
    TrajectoryInterpolator,
    ScenarioTrajectoryMatcher,
    interpolate_angle,
    linear_interpolate,
    compute_interpolation_info,
)
from .replay_original import (
    FrameOriginalInfo,
    ReplayOriginalEnv,
)
from .qa_loader import (
    QAItem,
    SampleQAData,
    SceneQAData,
    NuScenesQALoader,
    ScenarioQAMatcher,
)
from .trajectory_vis import TrajectoryVisualizer
from .surrounding import SurroundingVehicleGetter, SurroundingVehicleInfo

__all__ = [
    # Structures
    'MDPTransition',
    'CMDPTransition',
    'SurroundingVehicleInfo',
    # Original frequency NuScenes
    'FrameData',
    'SceneOriginalData',
    'NuScenesOriginalProcessor',
    # Interpolation
    'InterpolatedState',
    'TrajectoryInterpolator',
    'ScenarioTrajectoryMatcher',
    'interpolate_angle',
    'linear_interpolate',
    'compute_interpolation_info',
    # Original frequency Replay
    'FrameOriginalInfo',
    'ReplayOriginalEnv',
    # QA Data Loading
    'QAItem',
    'SampleQAData',
    'SceneQAData',
    'NuScenesQALoader',
    'ScenarioQAMatcher',
    # Trajectory Visualization
    'TrajectoryVisualizer',
    # Surrounding Vehicle
    'SurroundingVehicleGetter',
]
