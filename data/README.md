# Dataset Directory Structure

This directory should contain the CompoSuite offline reinforcement learning datasets.

## Directory Structure

The expected structure is:

```
data/
├── expert-{robot}-offline-comp-data/
│   ├── {ROBOT}_{OBJECT}_{OBSTACLE}_{SUBTASK}/
│   │   └── data.hdf5
│   ├── {ROBOT}_{OBJECT}_{OBSTACLE}_{SUBTASK}/
│   │   └── data.hdf5
│   └── ...
```

## Folder Naming Convention

### Dataset Type Folders
- Format: `{dataset_type}-{robot}-offline-comp-data`
- `dataset_type`: One of `expert`, `medium`, `random`, `medium-replay-subsampled`
- `robot`: Lowercase robot name (e.g., `iiwa`, `jaco`, `kinova3`, `panda`)

Examples:
- `expert-iiwa-offline-comp-data/`
- `expert-jaco-offline-comp-data/`
- `expert-kinova3-offline-comp-data/`
- `expert-panda-offline-comp-data/`

### Task Folders
- Format: `{ROBOT}_{OBJECT}_{OBSTACLE}_{SUBTASK}`
- `ROBOT`: Uppercase robot name (e.g., `IIWA`, `JACO`, `KINOVA3`, `PANDA`)
- `OBJECT`: Object type (e.g., `Box`, `Dumbbell`, `Hollowbox`, `Plate`)
- `OBSTACLE`: Obstacle type (e.g., `None`, `GoalWall`, `ObjectDoor`, `ObjectWall`)
- `SUBTASK`: Subtask type (e.g., `PickPlace`, `Push`, `Shelf`, `Trashcan`)

Examples:
- `IIWA_Box_GoalWall_PickPlace/`
- `IIWA_Box_ObjectDoor_Push/`
- `IIWA_Dumbbell_None_Shelf/`

## Data File Format

Each task folder must contain a `data.hdf5` file with the following keys:
- `observations`: State observations (shape: [1000000, state_dim])
- `actions`: Actions taken (shape: [1000000, action_dim])
- `rewards`: Rewards received (shape: [1000000])
- `successes`: Success indicators (shape: [1000000])
- `terminals`: Terminal flags (shape: [1000000])
- `timeouts`: Timeout flags (shape: [1000000])


## Example Structure

```
data/
├── expert-iiwa-offline-comp-data/
│   ├── IIWA_Box_GoalWall_PickPlace/
│   │   └── data.hdf5
│   ├── IIWA_Box_ObjectDoor_Push/
│   │   └── data.hdf5
│   ├── IIWA_Dumbbell_None_Shelf/
│   │   └── data.hdf5
│   └── ...
├── expert-jaco-offline-comp-data/
│   └── ...
└── expert-kinova3-offline-comp-data/
    └── ...
```

## Downloading Datasets

The datasets can be downloaded from [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.9cnp5hqps). We only need to download the expert datsets.

After downloading, you may need to unzip the data to match the expected structure above. The downloaded data is typically organized by robot arm.

## Usage in Code

The datasets are loaded using functions from `diffusion.utils`:

```python
from diffusion.utils import load_single_composuite_dataset

# Load a single dataset
dataset = load_single_composuite_dataset(
    base_path="data/",  # Path to this data directory
    dataset_type="expert",
    robot="IIWA",
    obj="Box",
    obst="GoalWall",
    task="PickPlace"
)
```

The function will automatically construct the path:
```
data/expert-iiwa-offline-comp-data/IIWA_Box_GoalWall_PickPlace/data.hdf5
```

