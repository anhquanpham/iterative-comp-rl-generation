import os
import subprocess
from typing import List, Dict

job_configs = [
    {'denoiser': 'monolithic', 'num_train': 56, 'diffusion_run': 1, 'seed': 0, 'memory': 256, 'time': 48},
    {'denoiser': 'monolithic', 'num_train': 56, 'diffusion_run': 1, 'seed': 1, 'memory': 256, 'time': 48},
    {'denoiser': 'monolithic', 'num_train': 56, 'diffusion_run': 1, 'seed': 2, 'memory': 256, 'time': 48},
    {'denoiser': 'monolithic', 'num_train': 56, 'diffusion_run': 1, 'seed': 3, 'memory': 256, 'time': 48},
    {'denoiser': 'monolithic', 'num_train': 56, 'diffusion_run': 1, 'seed': 4, 'memory': 256, 'time': 48}
]

BASE_PATH = '/home/quanpham/iterative-comp-rl-generation'
SCRIPT_PATH = f'{BASE_PATH}/scripts/generate_data.py'
CONFIG_PATH = f'{BASE_PATH}/config/diffusion.gin'
DATA_PATH = f'{BASE_PATH}/data'
RESULTS_PATH = f'{BASE_PATH}/results/diffusion'
TASKS_PATH = f'{BASE_PATH}/offline_compositional_rl_datasets/_train_test_splits'


def generate_script(config: Dict) -> str:

    denoiser = config['denoiser'] 
    num_train = config['num_train']
    seed = config['seed']
    memory = config['memory']
    time = config['time']
    diffusion_run = config['diffusion_run']
    # Script header
    script = (
        f'#!/bin/bash\n'
        f'#SBATCH --job-name=diffusion_training\n'
        f'#SBATCH --output=slurm/%j_generate_{denoiser}_train{num_train}_seed{seed}.out\n'
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
        f'    --base_data_path {DATA_PATH} \\\n'
        f'    --base_results_folder {RESULTS_PATH} \\\n'
        f'    --gin_config_files {CONFIG_PATH} \\\n'
        f'    --denoiser {denoiser} \\\n'
        f'    --task_list_path {TASKS_PATH} \\\n'
        f'    --num_train {num_train} \\\n'
        f'    --dataset_type expert \\\n'
        f'    --experiment_type default \\\n'
        f'    --seed {seed} \\\n'
        f'    --run {diffusion_run}\n'
    )
    script += python_cmd

    return script


def submit_jobs(configs: List[Dict]):
    """Generate and submit all jobs."""
    os.makedirs('slurm', exist_ok=True)
    for i, config in enumerate(configs):
        script_content = generate_script(config)
        script_path = f'job_{config["num_train"]}_{config["seed"]}.sh'
        with open(script_path, 'w') as f:
            f.write(script_content)
        print(f'Submitting job {i+1}/{len(configs)}: generate{config["num_train"]}_seed{config["seed"]}')
        subprocess.run(["sbatch", script_path])
        os.remove(script_path)

if __name__ == '__main__':
    submit_jobs(job_configs)
    print(f'Submitted {len(job_configs)} jobs.')