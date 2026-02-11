# MetaQA

**MetaQA** integrates [MetaDrive](https://github.com/metadriverse/metadrive)/[ScenarioNet](https://github.com/metadriverse/scenarionet) converted data with [NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA) question-answering annotations for multi-modal autonomous driving research.

---

## Architecture

```
NuScenes (~12Hz)           ScenarioNet (10Hz)         NuScenes-QA (2Hz)
  ┌──────────┐             ┌────────────────┐           ┌──────────┐
  │ samples/ │             │ .pkl scenario  │           │ QA JSON  │
  │ sweeps/  │             │ trajectories   │           │ per      │
  │ camera   │             │ (ego + traffic)│           │ sample   │
  └────┬─────┘             └──────┬─────────┘           └────┬─────┘
       │                          │                          │
       ▼                          ▼                          ▼
  NuScenesOriginal         TrajectoryInter-          NuScenesQA-Loader
  Processor                polator (~12Hz)                   │
       │                          │                          │
       └───────────────┬──────────┘                          │
                       ▼                                     │
                    ReplayOriginalEnv  ◄─────────────────────┘
                     (synchronized)
                            │
                            │
                            ▼
        collect_data (collect offline RL data) 
        replay_demo & scenario_demo
```

**Data Flow:**
- **NuScenes Original Processor**: Extracts all frames (samples + sweeps) at ~12Hz with timestamps
- **Trajectory Interpolator**: Interpolates ScenarioNet 10Hz trajectories to match ~12Hz timestamps
- **QA Loader**: Maps NuScenes-QA annotations to sample tokens
- **ReplayOriginalEnv**: Synchronizes all three data sources for frame-by-frame replay
- **collect_data.py**: Collects offline RL datasets (observations, actions, rewards, costs, QA)
- **replay_demo.py**: Generates GIFs with MetaDrive BEV + 6-camera grid + QA panel
- **scenario_demo.py**: Real-time visualization with trajectory + surrounding vehicles

---

## Demo

<p align="center">
  <img src="outputs/vis/demo.gif" alt="MetaQA Demo - QA-enhanced scenario replay with 6-camera view" width="100%">
</p>

*Integrated visualization showing MetaDrive BEV, 6-camera grid, and QA annotations*

---

## Installation

```bash
# Same as scenarionet (https://github.com/metadriverse/scenarionet)
```

---

## Data Preparation

Place your data in the `dataset/` directory:

```
dataset/
├── QA_Data/
│   ├── NuScenes_QA_train.json    # NuScenes-QA annotations (train split)
│   └── NuScenes_QA_val.json      # NuScenes-QA annotations (val split)
└── Scenario_Data/
    ├── exp_nuscenes/
    │   └── v1.0-mini/            # Original NuScenes dataset
    │       ├── samples/          # Keyframe images (2Hz)
    │       ├── sweeps/           # Inter-keyframe images (~12Hz)
    │       ├── v1.0-mini/        # Metadata JSONs
    │       └── ...
    └── exp_nuscenes_converted/   # ScenarioNet converted scenarios
        ├── exp_nuscenes_converted_0/
        ├── exp_nuscenes_converted_1/
        └── ...
```

---

## Project Structure

```
MetaQA/
├── meta_qa/
│   ├── core/                       # Core RL components
│   │   ├── config.py              # Configuration constants (PID gains, colors, etc.)
│   │   ├── env.py                 # TrajectoryEnv (trajectory action → control)
│   │   ├── action_space.py        # Trajectory action space definition
│   │   └── trajectory_tracker.py  # PID-based trajectory tracking controller
│   │
│   ├── cost/                       # Cost functions for Safe RL (CMDP)
│   │   ├── base.py                # Abstract base cost function
│   │   ├── collision.py           # Binary collision cost
│   │   ├── ttc.py                 # Time-to-Collision cost
│   │   └── kinematic.py           # Kinematic stability cost
│   │
│   ├── tools/                      # Data processing, visualization & utilities
│   │   ├── structures.py          # MDP/CMDP data classes
│   │   ├── qa_loader.py           # NuScenes-QA data loader & scene matcher
│   │   ├── nuscenes_original.py   # Original frequency (~12Hz) frame extraction
│   │   ├── interpolation.py       # Trajectory interpolation (10Hz → ~12Hz)
│   │   ├── replay_original.py     # Original frequency replay environment
│   │   ├── trajectory_vis.py      # Trajectory visualization (OpenCV/Matplotlib)
│   │   └── surrounding.py         # Surrounding vehicle extraction from MetaDrive
│   │
│   └── scripts/                    # Runnable scripts
│       ├── demo_vis/              # Visualization demos
│       │   ├── replay_demo.py     # GIF/console replay with BEV + cameras + QA
│       │   └── scenario_demo.py   # Trajectory + surrounding vehicle visualization
│       └── data_collect/          # Data collection
│           ├── collect_data.py    # Unified data collection (all modes)
│           └── read_data.py       # Dataset inspection & visualization
│
├── dataset/                        # Input data
└── outputs/                        # Output directory
    ├── offline_data/              # Collected datasets (.h5)
    └── vis/                       # Visualizations (.gif, .png)
```

---

## Scripts Usage

### 1. Replay Demo (Main Demo ⭐)

Generate animated GIFs with synchronized MetaDrive BEV, 6-camera grid, and QA annotations.

```bash
# List available scenes with QA data
python -m meta_qa.scripts.demo_vis.replay_demo --list-scenes

# Generate GIF for a scene
python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --output outputs/vis/demo.gif

# Console demo (text output)
python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --mode console

# Samples only (keyframes at 2Hz)
python -m meta_qa.scripts.demo_vis.replay_demo --scene scene-0061 --samples-only \
    --fps 2 --output outputs/vis/demo.gif

# Custom FPS, resolution, and frame limit
python -m meta_qa.scripts.demo_vis.replay_demo \
    --scene scene-0061 --width 1600 --height 1000 --fps 5 --max-frames 50
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | gif | Mode: `gif` / `console` / `list` |
| `--scene` | None | Scene name (e.g., `scene-0061`) |
| `--list-scenes` | — | List available scenes and exit |
| `--output` | auto | Output GIF path |
| `--width` | 1200 | GIF width in pixels |
| `--height` | 900 | GIF height in pixels |
| `--fps` | 12 | Frame rate |
| `--samples-only` | False | Show only 2Hz keyframes |
| `--max-frames` | None | Max frames to process |

---

### 2. Scenario Demo

Visualize ego trajectory with surrounding vehicles, control signals, and statistics.

```bash
# Top-down view (default)
python -m meta_qa.scripts.demo_vis.scenario_demo

# 3D view
python -m meta_qa.scripts.demo_vis.scenario_demo --mode 3d

# Custom trajectory horizon and detection radius
python -m meta_qa.scripts.demo_vis.scenario_demo --horizon 3.0 --detection-radius 40
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | topdown | View mode: `topdown` / `3d` / `stats` |
| `--horizon` | 2.0 | Trajectory horizon in seconds |
| `--detection-radius` | 50.0 | Surrounding vehicle detection radius (m) |
| `--max-vehicles` | 10 | Max surrounding vehicles to track |
| `--save-gif` | True | Save visualization as GIF |
| `--plot-stats` | True | Plot statistics after completion |
| `--scenario-index` | random | Specific scenario index |

---

### 3. Collect Data (Unified)

Unified data collection script supporting multiple modes.

```bash
# Original frequency (~12Hz) without QA
python -m meta_qa.scripts.data_collect.collect_data --mode original

# Original frequency with QA annotations
python -m meta_qa.scripts.data_collect.collect_data --mode original_qa

# Keyframe only (2Hz)
python -m meta_qa.scripts.data_collect.collect_data --mode keyframe

# Keyframe with QA
python -m meta_qa.scripts.data_collect.collect_data --mode keyframe_qa

# Trajectory-based observation/action
python -m meta_qa.scripts.data_collect.collect_data \
    --mode trajectory --history_sec 0.5 --future_sec 2.0

# CMDP with TTC cost
python -m meta_qa.scripts.data_collect.collect_data --mode original --cost_type ttc

# Specific scenes
python -m meta_qa.scripts.data_collect.collect_data \
    --mode original_qa --scenes scene-0061 scene-0103
```

**Collection Modes:**
| Mode | Frequency | QA | Description |
|------|-----------|-----|-------------|
| `original` | ~12Hz | ✗ | All frames at original sensor rate |
| `original_qa` | ~12Hz | ✓ | All frames + QA at keyframes |
| `keyframe` | 2Hz | ✗ | Keyframes only |
| `keyframe_qa` | 2Hz | ✓ | Keyframes + QA |
| `trajectory` | ~12Hz | ✗ | Trajectory-based obs/action (history + future) |

**Key Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--mode` | original | Collection mode (see above) |
| `--cost_type` | none | Cost type: `none` / `collision` / `ttc` |
| `--output` | auto | Output HDF5 file path |
| `--scenes` | all | Specific scene names |
| `--history_sec` | 0.5 | Trajectory history window (trajectory mode) |
| `--future_sec` | 2.0 | Trajectory future window (trajectory mode) |

**Output Format (HDF5):**
- `observations`: (N, obs_dim) — State observations
- `actions`: (N, 40) — Trajectory actions (20 waypoints × 2)
- `rewards`: (N,) — Step rewards
- `terminals`: (N,) — Episode termination flags
- `costs`: (N,) — Safety costs (CMDP only)
- `qa_data`: QA annotations (QA modes only)
- `metadata`: Collection parameters

---

### 4. Read/Inspect Data

Inspect collected datasets.

```bash
# Show dataset info
python -m meta_qa.scripts.data_collect.read_data \
    --file outputs/offline_data/original_qa.h5

# List available datasets
python -m meta_qa.scripts.data_collect.read_data --list

# Show QA data
python -m meta_qa.scripts.data_collect.read_data \
    --file outputs/offline_data/original_qa.h5 --show_qa --max_qa 5

# Visualize episode trajectory
python -m meta_qa.scripts.data_collect.read_data \
    --file outputs/offline_data/original_qa.h5 --visualize --episode 5

# Show more sample data points
python -m meta_qa.scripts.data_collect.read_data \
    --file outputs/offline_data/original_qa.h5 --samples 10
```

---

## Python API

### Load QA Data

```python
from meta_qa.tools import NuScenesQALoader, ScenarioQAMatcher

# Load QA annotations
loader = NuScenesQALoader(
    qa_data_dir="dataset/QA_Data",
    nuscenes_dataroot="dataset/Scenario_Data/exp_nuscenes/v1.0-mini"
).load()

# Match to scenario
matcher = ScenarioQAMatcher(loader, "dataset/Scenario_Data/exp_nuscenes_converted")
scene_qa = matcher.get_scene_qa("scene-0916")

# Iterate QA items
for sample_token, sample in scene_qa.samples.items():
    print(f"Sample: {sample_token}")
    for qa in sample.qa_items:
        print(f"  Q: {qa.question}")
        print(f"  A: {qa.answer}")
        print(f"  Type: {qa.template_type}")
```

### Trajectory RL Environment

```python
from meta_qa.core import TrajectoryEnv
from metadrive.envs.scenario_env import ScenarioEnv

# Create environment with trajectory-based actions
base_env = ScenarioEnv(config={
    "data_directory": "dataset/Scenario_Data/exp_nuscenes_converted",
    "use_render": True
})
env = TrajectoryEnv(base_env)

obs, info = env.reset()

# Action: 20 waypoints × 2 (x, y) = 40 dimensions
action = env.action_space.sample()
obs, reward, done, truncated, info = env.step(action)

env.close()
```

### Original Frequency Replay

```python
from meta_qa.tools import ReplayOriginalEnv, NuScenesQALoader

# Load QA data (optional)
qa_loader = NuScenesQALoader(
    qa_data_dir="dataset/QA_Data",
    nuscenes_dataroot="dataset/Scenario_Data/exp_nuscenes/v1.0-mini"
).load()

# Create original frequency replay environment
env = ReplayOriginalEnv(
    nuscenes_dataroot="dataset/Scenario_Data/exp_nuscenes/v1.0-mini",
    scenario_dir="dataset/Scenario_Data/exp_nuscenes_converted",
    qa_loader=qa_loader,
).load()

# Load a scene
env.load_scene("scene-0061")
print(f"Total frames: {env.num_frames} (original frequency ~12Hz)")
print(f"Samples (keyframes): {env.num_samples} (2Hz)")

# Iterate through all frames at original frequency
for frame_info in env.iterate_frames():
    print(f"Frame {frame_info.frame_idx}: sample={frame_info.is_sample}")
    
    # Interpolated trajectory state at this frame's timestamp
    if frame_info.ego_state:
        print(f"  Position: {frame_info.ego_state.position}")
        print(f"  Velocity: {frame_info.ego_state.velocity}")
    
    # QA only available at samples (2Hz keyframes)
    if frame_info.is_sample and frame_info.has_qa:
        for qa in frame_info.qa_items:
            print(f"  Q: {qa['question']}")
            print(f"  A: {qa['answer']}")
    
    # Camera images (all 6 cameras)
    for cam, path in frame_info.image_paths.items():
        print(f"  {cam}: {path}")
```


---

## Acknowledgments

This project builds upon the excellent work of:

- [MetaDrive](https://github.com/metadriverse/metadrive)
- [ScenarioNet](https://github.com/metadriverse/scenarionet)
- [NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA)
- [ASAP](https://github.com/JeffWang987/ASAP)

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
