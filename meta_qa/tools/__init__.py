"""
MetaQA Tools Module.

Utility tools for visualization, data loading, and extraction.
- QA data loading (NuScenes-QA)
- QA visualization (PIL and Pygame)
- GIF generation for scenario replay
- Trajectory visualization
- Surrounding vehicle extraction
"""

from .gif_generator import GIFGenerator, GIFConfig
from .qa_vis import (
    QARenderer,
    QADisplayItem,
    QAInfoPanel,
    ImagePanel,
    IntegratedQAVisualizer,
    render_qa_panel,
    render_camera_grid,
    CAMERA_LAYOUT,
    CAMERA_LABELS,
)
from .trajectory_vis import TrajectoryVisualizer
from .surrounding import SurroundingVehicleGetter
from .qa_loader import (
    QAItem,
    SampleQAData,
    SceneQAData,
    NuScenesQALoader,
    ScenarioQAMatcher,
)

__all__ = [
    # QA Data Loading
    'QAItem',
    'SampleQAData', 
    'SceneQAData',
    'NuScenesQALoader',
    'ScenarioQAMatcher',
    # QA Visualization
    'QARenderer',
    'QADisplayItem',
    'QAInfoPanel',
    'ImagePanel',
    'IntegratedQAVisualizer',
    'render_qa_panel',
    'render_camera_grid',
    'CAMERA_LAYOUT',
    'CAMERA_LABELS',
    # GIF Generation
    'GIFGenerator',
    'GIFConfig',
    # Trajectory Visualization
    'TrajectoryVisualizer',
    # Surrounding Vehicle
    'SurroundingVehicleGetter',
]
