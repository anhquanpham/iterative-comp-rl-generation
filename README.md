# Iterative Compositional Data Generation for Robot Control

This repository contains the official implementation of Iterative Compositional Data Generation (ICDG), introduced in "Iterative Compositional Data Generation for Robot Control" (Pham et al.). ICDG is a self-improving generative framework for robotic manipulation that uses a semantic compositional diffusion transformer to synthesize high-quality training data for unseen tasks.

Robotic manipulation domains often contain a combinatorial number of possible tasks, arising from combinations of different robots, objects, obstacles, and objectives. Collecting real demonstrations for all combinations is prohibitively expensive. ICDG leverages the underlying compositional structure of these domains to generalize far beyond the tasks it has been trained on, enabling large-scale capability growth from limited real data.

<p align="center">
  <img width="90%" src="https://github.com/user-attachments/assets/c3cf0f25-b6e1-4b13-aaf6-6ab2334e1bfe" />
</p>

## Key Contributions

- **Semantic Compositional Diffusion Transformer**:  
  Factorizes each transition into robot-, object-, obstacle-, and objective-specific components and learns their interactions through attention, enabling strong compositional generalization.

- **Zero-Shot Generation**:  
  Generates full state–action–next-state transitions for new task combinations that were never observed in real data.

- **Iterative Self-Improvement**:  
  Synthetic data is evaluated using offline RL; only high-quality, policy-validated transitions are added back into the training pool, allowing the model to continuously refine itself without additional real data collection.

- **Data Efficiency and Generalization**:  
  Trained on real data from approximately 20 percent of possible task combinations, ICDG generates useful data for the remaining tasks and ultimately solves nearly all held-out tasks.

- **Emergent Compositional Structure**:  
  Attention patterns and intervention tests reveal that the model recovers meaningful task-factor dependencies, despite no hand-crafted structure being imposed.




## Setup

### Prerequisites

- Python 3.9.6
- CUDA-capable GPU (for training diffusion models and policies)
- SLURM cluster access (for running experiments)

### Installation

1. Create a Python 3.9.6 virtual environment:
```bash
python3.9 -m venv venv
source venv/bin/activate  # On Linux/Mac
# or
venv\Scripts\activate  # On Windows
```

2. Install dependencies from `requirements.txt`:
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**Note**: The `requirements.txt` includes an editable install of CompoSuite from a specific git commit for reproducibility:
```
-e git+https://github.com/Lifelong-ML/CompoSuite.git@1fa36f67f31aeccc9ef75748bfc797960e044a86#egg=composuite
```

3. Set up the data directory:
   - Download expert datasets from [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.9cnp5hqps)
   - Organize the data according to the structure described in `data/README.md`
   - Only expert datasets are needed for this project

## Usage

### Automated Iterative Compositional Data Generation

The main pipeline implements the iterative self-improvement procedure from the paper (see Figure 1). The process consists of:

1. **Compositional Diffusion Training**: Train the semantic compositional diffusion transformer on N expert datasets + M high-quality synthetic datasets from previous iterations
2. **Zero-shot Data Generation**: Generate synthetic transitions for all remaining task combinations (All combinations - N - M)
3. **Offline RL Validation**: Train policies on synthetic data and evaluate performance via offline RL
4. **Quality-based Filtering**: 
   - **Good datasets**: Added to training set for next iteration (M synthetic datasets)
   - **Bad datasets**: Removed from future generation cycles
5. **Iteration**: Repeat until convergence or max iterations reached

**Run the pipeline:**
```bash
python3 -u -m scripts.automated_iterative_diffusion_dits_iiwa \
    --max_iterations 5 \
    --num_train 14 \
    --diffusion_seed 0 \
    --curriculum_seed 0 \
    2>&1 | tee iterative_diffusion_0_dits_iiwa.out
```

**Key arguments:**
- `--max_iterations`: Maximum number of iterations to run
- `--num_train`: Number of training tasks (14 for IIWA subset)
- `--diffusion_seed`: Random seed for diffusion model training
- `--curriculum_seed`: Random seed for curriculum schedule generation
- `--success_threshold`: Success rate threshold for good tasks (default: 0.8)
- `--threshold_reduction_amount`: Amount to reduce threshold by when no good tasks found (default: 0.1)
- `--threshold_reduction_cycle`: Number of consecutive iterations with no good tasks before reducing threshold (default: 1)
- `--min_threshold`: Minimum threshold value (default: 0.5)

**Output:**
- Diffusion models: `results/augmented_{iteration}/diffusion/`
- Synthetic data: `results/augmented_{iteration}/diffusion/{model_name}/{task}/samples_0.npz`
- Policy checkpoints: `results/augmented_{iteration}/policies/`
- Analysis logs: `scripts/policies_slurm_logs/`
- Best test task dataset: `results/best_testtask_dataset/`

### Semantic + Compositional RL Baseline with Transformer Policy

Run the transformer TD3+BC multitask baseline for comparison:

```bash
python3 -u -m scripts.run_transformer_baseline_pipeline \
    --num_train 14 \
    --seeds 10 11 12 13 14 \
    --memory 50 \
    --time 24 \
    2>&1 | tee multitask_Trans_OfflineRL_iwa_seed2.out
```

**Key arguments:**
- `--num_train`: Number of training tasks (14 for IIWA subset)
- `--seeds`: List of random seeds to run (e.g., `10 11 12 13 14`)
- `--memory`: Memory per job in GB (default: 50)
- `--time`: Time limit per job in hours (default: 24)
- `--max_timesteps`: Maximum training timesteps (default: 50000)
- `--batch_size`: Batch size (default: 1792)

**Output:**
- Model checkpoints: `results/transformer_baseline/seed_{seed}/`
- Results CSV: `results/transformer_baseline/transformer_baseline_results.csv`
- Training logs: `scripts/transformer_baseline_logs/`

## Project Structure

```
.
├── data/                          # Dataset directory (see data/README.md)
├── results/                       # Experiment results
│   ├── augmented_{iteration}/     # Iterative diffusion results
│   └── transformer_baseline/      # Transformer baseline results
├── scripts/                       # Main scripts
│   ├── automated_iterative_diffusion_dits_iiwa.py  # Main pipeline
│   ├── run_transformer_baseline_pipeline.py        # Transformer baseline
│   ├── train_augmented_diffusion.py               # Diffusion training
│   ├── train_augmented_policy.py                  # Policy training
│   └── generate_augmented_data_dits.py            # Data generation
├── diffusion/                     # Diffusion model code
├── corl/                          # Offline RL algorithms (TD3-BC, IQL)
├── config/                        # Configuration files
└── requirements.txt               # Python dependencies
```

## Key Features

- **Semantic Compositional Architecture**: Diffusion transformer with factorized components (robot, object, obstacle, objective)
- **Iterative Self-Improvement**: Each iteration uses validated high-quality synthetic tasks to improve the diffusion model
- **Zero-shot Generation**: Generates data for unseen task combinations without additional training
- **Automatic Retry**: Failed jobs are automatically retried with increased resources
- **Curriculum Learning**: Component-specific curriculum filtering for iterations 5+ (optional)
- **Adaptive Threshold**: Success threshold automatically reduces if no good tasks are found
- **Comprehensive Logging**: Detailed logs and CSV analysis files for each iteration

## Configuration

Default paths are set in the script configuration classes. Modify these in the scripts if needed:
- `base_path`: Project root directory
- `data_path`: Path to expert datasets
- `results_path`: Path to save results
- `tasks_path`: Path to task list JSON files

## Citation

If you use this code, please cite:

```bibtex
@article{pham2026iterative,
  title={Iterative Compositional Data Generation for Robot Control},
  author={Pham, Anh-Quan and Hussing, Marcel and Patankar, Shubhankar P. and Bassett, Dani S. and Mendez-Mendez, Jorge and Eaton, Eric},
  year={2026}
}
```

**Related Resources:**
- CompoSuite Benchmark: [GitHub](https://github.com/Lifelong-ML/CompoSuite)
- Datasets: [Dryad](https://datadryad.org/stash/dataset/doi:10.5061/dryad.9cnp5hqps)

## Contact

For inquiries, please contact [Anh-Quan Pham](https://anhquanpham.github.io/).

