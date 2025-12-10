#!/bin/bash
#SBATCH --job-name=policy_training
#SBATCH --output=slurm/%j.out
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time=4:00:00

source /home/quanpham/first_3.9.6/bin/activate

python /home/quanpham/iterative-comp-rl-generation/scripts/train_policy.py \
    --base_agent_data_path /home/quanpham/iterative-comp-rl-generation/data \
    --base_synthetic_data_path /home/quanpham/iterative-comp-rl-generation/results/diffusion \
    --base_results_folder /home/quanpham/iterative-comp-rl-generation/results/policies \
    --dataset_type synthetic \
    --robot IIWA \
    --obj Box \
    --obst ObjectDoor \
    --subtask Push \
    --algorithm iql \
    --seed 0 \
    --denoiser monolithic \
    --task_list_seed 0 \
    --num_train 56 \
    --diffusion_training_run 1

