import os
import subprocess
from typing import List, Dict
from offline_compositional_rl_datasets.utils.data_utils import *

# Base configuration
BASE_PATH = '/home/quanpham/iterative-comp-rl-generation'
TASK_LIST_PATH = f'{BASE_PATH}/offline_compositional_rl_datasets/_train_test_splits'
SCRIPT_PATH = f'{BASE_PATH}/scripts/train_policy.py'
DATA_PATH = f'{BASE_PATH}/data'
SYNTH_DATA_PATH = f'{BASE_PATH}/results/diffusion'
RESULTS_PATH = f'{BASE_PATH}/results/policies'

# Default job parameters
DEFAULT_CONFIG = {
    'memory': 16,
    'time': 1,
    'seeds': [0, 1, 2, 3, 4],  # multiple RL seeds
    'denoiser': 'monolithic',
    'num_train': 56, 98, 140, 182, 224
    'diffusion_training_run': 1,
    'task_list_seeds': [0, 1, 2, 3, 4],  # multiple task list seeds
    'algorithm': 'td3_bc',
    'dataset_type': 'synthetic'
}


def create_job_name(config: Dict) -> str:
    """Create a verbose job name from config parameters."""

    algorithm = config['algorithm']
    seed = config['seed']
    denoiser = config['denoiser']
    task_list_seed = config['task_list_seed']
    num_train = config['num_train']
    diffusion_training_run = config['diffusion_training_run']
    robot = config['robot']
    obj = config['obj']
    obst = config['obst']
    subtask = config['subtask']
    dataset_type = config['dataset_type']

    if dataset_type == 'synthetic':
        job_name = (
            f"{algorithm}_{seed}_"
            f"{denoiser}_{task_list_seed}_"
            f"{num_train}_{diffusion_training_run}_"
            f"{robot}_{obj}_{obst}_{subtask}"
        )
    else:
        job_name = (
            f"{algorithm}_{seed}_"
            f"tl_{task_list_seed}_"
            f"{robot}_{obj}_{obst}_{subtask}"
        )
    
    return job_name
    
def generate_job_configs():

    job_configs = []
    task_list_seeds = DEFAULT_CONFIG['task_list_seeds']
    rl_seeds = DEFAULT_CONFIG['seeds']
    
    for task_list_seed in task_list_seeds:
        _, _, _, task_list = get_task_list(
            TASK_LIST_PATH,
            DEFAULT_CONFIG['dataset_type'],
            'default',  # experiment_type
            None,  # holdout element
            task_list_seed
        )
        
        for robot, obj, obst, subtask in task_list:
            for seed in rl_seeds:
                config = DEFAULT_CONFIG.copy()
                config['seed'] = seed
                config['task_list_seed'] = task_list_seed
                config.update({
                    'robot': robot,
                    'obj': obj,
                    'obst': obst,
                    'subtask': subtask
                })
                job_configs.append(config)
    
    return job_configs

def generate_script(config: Dict) -> str:
    robot = config['robot']
    obj = config['obj']
    obst = config['obst']
    subtask = config['subtask']
    memory = config['memory']
    time = config['time']
    seed = config['seed']
    denoiser = config['denoiser']
    num_train = config['num_train']
    diffusion_training_run = config['diffusion_training_run']
    task_list_seed = config['task_list_seed']
    algorithm = config['algorithm']
    dataset_type = config['dataset_type']

    job_name = create_job_name(config)
    
    # Script header
    script = (
        f'#!/bin/bash\n'
        f'#SBATCH --job-name={job_name}\n'
        f'#SBATCH --output=policies_slurm_2/%j_{job_name}.out\n'
        f'#SBATCH --mem={memory}G\n'
        f'#SBATCH --gpus=1\n'
        f'#SBATCH --cpus-per-gpu=8\n'
        f'#SBATCH --time={time}:00:00\n'
        f'#SBATCH --partition=eaton-compute\n'
        f'#SBATCH --qos=ee-med\n'
        f'\n'
        f'source {BASE_PATH}/venv/bin/activate\n'
        f'\n'
    )
    
    # Python command
    python_cmd = (
        f'python {SCRIPT_PATH} \\\n'
        f'    --base_agent_data_path {DATA_PATH} \\\n'
        f'    --base_synthetic_data_path {SYNTH_DATA_PATH} \\\n'
        f'    --base_results_folder {RESULTS_PATH} \\\n'
        f'    --dataset_type {dataset_type} \\\n'
        f'    --robot {robot} \\\n'
        f'    --obj {obj} \\\n'
        f'    --obst {obst} \\\n'
        f'    --subtask {subtask} \\\n'
        f'    --algorithm {algorithm} \\\n'
        f'    --seed {seed} \\\n'
        f'    --denoiser {denoiser} \\\n'
        f'    --task_list_seed {task_list_seed} \\\n'
        f'    --num_train {num_train} \\\n'
        f'    --diffusion_training_run {diffusion_training_run}\n'
    )
    script += python_cmd

    return script

def submit_jobs(configs: List[Dict]):
    """Generate and submit all jobs."""
    os.makedirs('policies_slurm_2', exist_ok=True)
    for i, config in enumerate(configs):
        job_name = create_job_name(config)
        script_path = f'job_{job_name}.sh'
        script_content = generate_script(config)
        with open(script_path, 'w') as f:
            f.write(script_content)
        print(f'Submitting job {i+1}/{len(configs)}: {job_name}')
        subprocess.run(["sbatch", script_path])
        os.remove(script_path)

if __name__ == '__main__':
    job_configs = generate_job_configs()
    submit_jobs(job_configs)
    print(f'Submitted {len(job_configs)} jobs.')

