# Train diffusion model on CompoSuite transitions.
import argparse
import pathlib
from typing import Tuple

import gin
import gym
import numpy as np
import torch
import wandb

from diffusion.elucidated_diffusion import Trainer
from diffusion.norm import MinMaxNormalizer
from diffusion.utils import *


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-replay-v2')
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=[])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])

    # wandb config
    parser.add_argument('--wandb-project', type=str, default="offline-rl-diffusion")
    parser.add_argument('--wandb-entity', type=str, default="")
    parser.add_argument('--wandb-group', type=str, default="diffusion_training")
    
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_samples', action='store_true', default=True)
    parser.add_argument('--save_num_samples', type=int, default=int(5e6))
    parser.add_argument('--save_file_name', type=str, default='5m_samples.npz')
    parser.add_argument('--load_checkpoint', action='store_true')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    # Set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    # Create the environment and dataset.
    env = gym.make(args.dataset)
    inputs = make_inputs(env)
    inputs = torch.from_numpy(inputs).float()
    dataset = torch.utils.data.TensorDataset(inputs)

    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    with open(results_folder / 'config.gin', 'w') as f:
        f.write(gin.config_str())

    # Create the diffusion model and trainer.
    diffusion = construct_diffusion_model(inputs=inputs)
    trainer = Trainer(
        diffusion,
        dataset,
        results_folder=args.results_folder,
    )

    if not args.load_checkpoint:
        # Initialize logging.
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            group=args.wandb_group,
            name=args.results_folder.split('/')[-1],
        )
        # Train model.
        trainer.train()
    else:
        trainer.ema.to(trainer.accelerator.device)
        # Load the last checkpoint.
        trainer.load(milestone=trainer.train_num_steps)

    # Generate samples and save them.
    if args.save_samples:
        generator = SimpleDiffusionGenerator(
            env=env,
            ema_model=trainer.ema.ema_model,
        )
        observations, actions, rewards, next_observations, terminals = generator.sample(
            num_samples=args.save_num_samples,
        )
        np.savez_compressed(
            results_folder / args.save_file_name,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
        )
