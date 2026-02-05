# MetaQA

**MetaQA** integrates [MetaDrive](https://github.com/metadriverse/metadrive)/[ScenarioNet](https://github.com/metadriverse/scenarionet) converted data with [NuScenes-QA](https://github.com/qiantianwen/NuScenes-QA) question-answering annotations for multi-modal autonomous driving research.

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
    │       ├── samples/          # Keyframe images
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
│   ├── core/                    # Core components
│   │   ├── config.py           # Configuration parameters
│   │   ├── env.py              # TrajectoryEnv (trajectory-based action space)
│   │   ├── action_space.py     # Trajectory action space definition
│   │   └── trajectory_tracker.py
│   │
│   ├── tools/                   # Utility modules
│   │   ├── qa_loader.py        # NuScenes-QA data loader & matcher
│   │   ├── qa_vis.py           # QA visualization (PIL/Pygame)
│   │   ├── gif_generator.py    # GIF generator with 6-camera layout
│   │   ├── trajectory_vis.py   # Trajectory visualization
│   │   └── surrounding.py      # Surrounding vehicle extraction
│   │
│   ├── data/                    # Data I/O
│   │   ├── structures.py       # MDP/CMDP data classes
│   │   └── io.py               # HDF5/NPZ save/load utilities
│   │
│   ├── cost/                    # Cost functions for Safe RL
│   │   ├── collision.py        # Collision-based cost
│   │   ├── ttc.py              # Time-to-Collision cost
│   │   └── kinematic.py        # Kinematic stability cost
│   │
│   └── scripts/                 # Runnable scripts
│       ├── demo_vis/           # Visualization demos
│       │   ├── qa_replay_demo.py
│       │   ├── trajectory_action_demo.py
│       │   └── scenario_info_demo.py
│       └── data_collect/       # Data collection
│           ├── collect_offline_data.py
│           ├── collect_qa_data.py
│           └── read_offline_data.py
│
├── dataset/                     # Input data
└── outputs/                     # Output directory
    ├── offline_data/           # Collected datasets (.h5, .npz)
    └── vis/                    # Visualizations (.gif, .png)
```

---

## Scripts Usage

### 1. QA Replay Demo (Main Demo ⭐)

Generate animated GIFs with synchronized QA annotations and camera images.

```bash
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
    --fps 1 --max-frames 20
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--scene` | None | Scene name (e.g., scene-0916) |
| `--qa-dir` | dataset/QA_Data | Path to QA data |
| `--nuscenes-dir` | dataset/Scenario_Data/exp_nuscenes/v1.0-mini | NuScenes path |
| `--scenario-dir` | dataset/Scenario_Data/exp_nuscenes_converted | Converted scenarios |
| `--list-scenes` | - | List available scenes and exit |
| `--output` | outputs/vis/{scene_name}.gif | Output GIF path |
| `--width` | 1200 | GIF width in pixels |
| `--height` | 1200 | GIF height in pixels |
| `--max-frames` | None | Max frames (None = all) |
| `--fps` | 2 | Target frame rate |

---

### 2. Trajectory Action Demo

Visualize trajectory-based actions with future waypoints and control signals.

```bash
# Default top-down view
python -m meta_qa.scripts.demo_vis.trajectory_action_demo

# 3D view mode
python -m meta_qa.scripts.demo_vis.trajectory_action_demo --mode 3d

# Custom trajectory horizon (seconds)
python -m meta_qa.scripts.demo_vis.trajectory_action_demo --horizon 3.0

# Custom data directory
python -m meta_qa.scripts.demo_vis.trajectory_action_demo \
    --data_dir /path/to/scenario_data
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | dataset/Scenario_Data/exp_nuscenes_converted | Scenario data path |
| `--mode` | topdown | View mode: topdown/3d |
| `--horizon` | 2.0 | Trajectory horizon in seconds |

---

### 3. Scenario Info Demo

Visualize surrounding vehicle information (positions, velocities, distances).

```bash
# Basic visualization
python -m meta_qa.scripts.demo_vis.scenario_info_demo

# Multiple scenarios with custom detection radius
python -m meta_qa.scripts.demo_vis.scenario_info_demo \
    --num_scenarios 3 \
    --detection_radius 10
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | dataset/Scenario_Data/exp_nuscenes_converted | Scenario data path |
| `--num_scenarios` | 1 | Number of scenarios to visualize |
| `--detection_radius` | 50.0 | Detection radius in meters |
| `--max_vehicles` | 10 | Max surrounding vehicles to track |
| `--save_gif` | True | Save visualization as GIF |
| `--plot_stats` | True | Plot surrounding vehicle statistics |

---

### 4. Collect Offline Data

Collect expert trajectory data for offline reinforcement learning.

```bash
# Collect MDP data (reward only)
python -m meta_qa.scripts.data_collect.collect_offline_data \
    --data_dir dataset/Scenario_Data/exp_nuscenes_converted \
    --num_scenarios 10 \
    --output outputs/offline_data/expert_mdp.h5

# Collect CMDP data with TTC cost
python -m meta_qa.scripts.data_collect.collect_offline_data \
    --data_dir dataset/Scenario_Data/exp_nuscenes_converted \
    --num_scenarios 10 \
    --cost_type ttc \
    --output outputs/offline_data/expert_cmdp_ttc.h5
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--data_dir` | (required) | Path to ScenarioNet dataset |
| `--num_scenarios` | 100 | Number of scenarios to collect |
| `--cost_type` | none | Cost type: none/collision/distance/ttc |
| `--output` | auto-generated | Output file path |
| `--format` | hdf5 | Output format: hdf5/npz |

**Output Format (HDF5):**
- `observations`: (N, obs_dim) - State observations
- `actions`: (N, 40) - Trajectory actions (20 waypoints × 2)
- `rewards`: (N,) - Step rewards
- `next_observations`: (N, obs_dim) - Next states
- `terminals`: (N,) - Episode termination flags
- `costs`: (N,) - Safety costs (CMDP only)
- `episode_starts`: Starting indices per episode

---

### 5. Collect QA-Enhanced Data

Collect trajectory data with QA annotations for multi-modal learning.

```bash
python -m meta_qa.scripts.data_collect.collect_qa_data \
    --scenario-dir dataset/Scenario_Data/exp_nuscenes_converted \
    --qa-dir dataset/QA_Data \
    --nuscenes-dir dataset/Scenario_Data/exp_nuscenes/v1.0-mini \
    --output outputs/offline_data/qa_enhanced.h5

# Specific scenes only
python -m meta_qa.scripts.data_collect.collect_qa_data \
    --scenes scene-0553 \
    --output outputs/offline_data/qa_subset.h5

# Include camera image paths (optional, increases dataset size)
python -m meta_qa.scripts.data_collect.collect_qa_data \
    --scenario-dir dataset/Scenario_Data/exp_nuscenes_converted \
    --qa-dir dataset/QA_Data \
    --nuscenes-dir dataset/Scenario_Data/exp_nuscenes/v1.0-mini \
    --output outputs/offline_data/qa_with_images.h5 \
    --include-image-paths

# Custom number of trajectory waypoints (default: 20)
python -m meta_qa.scripts.data_collect.collect_qa_data \
    --scenario-dir dataset/Scenario_Data/exp_nuscenes_converted \
    --qa-dir dataset/QA_Data \
    --nuscenes-dir dataset/Scenario_Data/exp_nuscenes/v1.0-mini \
    --output outputs/offline_data/qa_10waypoints.h5 \
    --num-waypoints 10
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--scenario-dir` | dataset/Scenario_Data/exp_nuscenes_converted | Converted scenarios |
| `--qa-dir` | dataset/QA_Data | QA annotations path |
| `--nuscenes-dir` | dataset/Scenario_Data/exp_nuscenes/v1.0-mini | Original NuScenes |
| `--output` | outputs/offline_data/qa_enhanced_dataset.h5 | Output file |
| `--format` | hdf5 | Output format: hdf5/json |
| `--scenes` | None | Specific scenes (default: all) |
| `--include-image-paths` | False | Include 6-camera image paths |
| `--num-waypoints` | 20 | Number of future waypoints |

---

### 6. Read Offline Data

Inspect collected datasets.

```bash
# List available datasets
python -m meta_qa.scripts.data_collect.read_offline_data --list

# Inspect a specific dataset
python -m meta_qa.scripts.data_collect.read_offline_data \
    --file outputs/offline_data/expert_mdp.h5

# Show more sample data points
python -m meta_qa.scripts.data_collect.read_offline_data \
    --file outputs/offline_data/expert_mdp.h5 \
    --samples 10
```

**Arguments:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--file` | None | Dataset file to inspect |
| `--list` | False | List available datasets |
| `--samples` | 3 | Number of sample points to show |

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

## Acknowledgments

This project builds upon the excellent work of:

MetaDrive(https://github.com/metadriverse/metadrive)

ScenarioNet(https://github.com/metadriverse/scenarionet)

NuScenes-QA(https://github.com/qiantianwen/NuScenes-QA) 


---

## License

MIT
