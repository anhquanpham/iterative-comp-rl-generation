#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=slurm/%j_monolithic_train56_seed9.out
#SBATCH --mem=224G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=24:00:00

source /home/quanpham/first_3.9.6/bin/activate

python /home/quanpham/iterative-comp-rl-generation/scripts/train_diffusion.py \
    --base_data_path /home/quanpham/iterative-comp-rl-generation/data \
    --base_results_folder /home/quanpham/iterative-comp-rl-generation/results/diffusion \
    --gin_config_files /home/quanpham/iterative-comp-rl-generation/config/diffusion.gin \
    --denoiser monolithic \
    --task_list_path /home/quanpham/iterative-comp-rl-generation/offline_compositional_rl_datasets/_train_test_splits \
    --num_train 56 \
    --dataset_type expert \
    --experiment_type default \
    --seed 9

