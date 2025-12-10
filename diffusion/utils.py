"""
Utilities for diffusion.
Code adapted from https://github.com/conglu1997/SynthER
"""

from typing import Optional, List, Union, Tuple
from itertools import product
import os
import h5py
import gin
import composuite
import matplotlib.pyplot as plt
import gym
import numpy as np
from tqdm import tqdm
import torch
from torch import nn

# GIN-required imports.
from diffusion.denoiser_network import ResidualMLPDenoiser, CompositionalResidualMLPDenoiser
from diffusion.dit_denoiser import DiT1D_models
from diffusion.elucidated_diffusion import ElucidatedDiffusion
from diffusion.norm import normalizer_factory, MinMaxNormalizer

@gin.configurable
def load_single_composuite_dataset(base_path, dataset_type, robot, obj, obst, task):

    keys = ["observations", "actions", "rewards", "successes", "terminals", "timeouts"]
    dataset_folder = f"{dataset_type}-{robot.lower()}-offline-comp-data"
    data_path = os.path.join(
        base_path, dataset_folder, f"{robot}_{obj}_{obst}_{task}", "data.hdf5"
    )

    data_dict = {}

    with h5py.File(data_path, "r") as dataset_file:
        for k in keys:
            assert k in dataset_file, f"Key {k} not found in dataset"
            data_dict[k] = dataset_file[k][:]
            assert len(data_dict[k]) == 1000000, f"Key {k} has wrong length"

    return data_dict


@gin.configurable
def load_multiple_composuite_datasets(base_path, dataset_type, robots, objs, obsts, tasks):

    combinations = list(product(robots, objs, obsts, tasks))
    datasets = []
    for robot, obj, obst, task in tqdm(combinations, desc="Loading data"):
        datasets.append(load_single_composuite_dataset(base_path, dataset_type, robot, obj, obst, task))

    return datasets


def load_single_synthetic_dataset(base_path, robot, obj, obst, task):
    data_path = os.path.join(base_path, f"{robot}_{obj}_{obst}_{task}", "samples_0.npz")
    return dict(np.load(data_path))


def load_multiple_synthetic_datasets(base_path, robots, objs, obsts, tasks):

    tuples = []
    for robot in robots:
        for obj in objs:
            for obst in obsts:
                for task in tasks:
                    tuples.append((robot, obj, obst, task))
                    
    datasets = []

    for robot, obj, obst, task in tqdm(tuples, desc="Loading data"):
        datasets.append(load_single_synthetic_dataset(base_path, robot, obj, obst, task))

    return datasets


@gin.configurable
def transitions_dataset(dataset):
    """
    https://github.com/Farama-Foundation/D4RL/blob/89141a689b0353b0dac3da5cba60da4b1b16254d/d4rl/__init__.py#L69
    """

    N = dataset['rewards'].shape[0]

    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    terminal_ = []

    for i in range(N - 1):

        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])
        final_timestep = dataset['timeouts'][i]
        terminal = done_bool or final_timestep

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        terminal_.append(terminal)

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(terminal_),
    }


@gin.configurable
def make_inputs(dataset, modelled_terminals=True):

    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']

    num_samples = obs.shape[0]
    input_dim = obs.shape[1] + actions.shape[1] + next_obs.shape[1] + 1 + (1 if modelled_terminals else 0)
    inputs = np.empty((num_samples, input_dim), dtype=np.float32)
    
    inputs[:, :obs.shape[1]] = obs
    inputs[:, obs.shape[1]:obs.shape[1] + actions.shape[1]] = actions
    inputs[:, obs.shape[1] + actions.shape[1]] = rewards
    inputs[:, obs.shape[1] + actions.shape[1] + 1:obs.shape[1] + actions.shape[1] + 1 + next_obs.shape[1]] = next_obs
    
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs[:, -1] = terminals
    
    return inputs


def remove_indicator_vectors(modality_dims, dataset):

    dataset = dataset.copy()

    start_index = sum([dim[0] for key, dim in modality_dims.items() if key in ['object-state', 'obstacle-state', 'goal-state']])
    end_index = start_index + sum([dim[0] for key, dim in modality_dims.items() if key in ['object_id', 'robot_id', 'obstacle_id', 'subtask_id']])

    def remove_indicator_dims(data, start, end):
        indicators_ = data[:, start:end]
        data_ = np.delete(data, np.arange(start, end), axis=1)
        return data_, indicators_

    observations, obs_indicators = remove_indicator_dims(dataset['observations'], start_index, end_index)
    dataset['observations'] = observations
    next_observations, _ = remove_indicator_dims(dataset['next_observations'], start_index, end_index)
    dataset['next_observations'] = next_observations
    
    return dataset, obs_indicators


def get_task_indicator(robot, obj, obst, task):

    env = composuite.make(robot, obj, obst, task, use_task_id_obs=True, ignore_done=False)
    dims = env.modality_dims
    start_index = sum([dim[0] for key, dim in dims.items() if key in ['object-state', 'obstacle-state', 'goal-state']])
    end_index = start_index + sum([dim[0] for key, dim in dims.items() if key in ['object_id', 'robot_id', 'obstacle_id', 'subtask_id']])
    obs = env.reset()
    task_indicator = obs[start_index:end_index]

    return task_indicator


def identify_special_dimensions(data):

    integer_dims = []
    constant_dims = []
    
    for i in range(data.shape[1]):
        column = data[:, i]
        if np.all(np.equal(column, np.round(column))):
            integer_dims.append(i)
        elif np.all(column == column[0]):
            constant_dims.append(i)
    
    return integer_dims, constant_dims


def process_special_dimensions(synthetic_dataset, integer_dims, constant_dims):

    processed_dataset = {k: v.copy() for k, v in synthetic_dataset.items()}
    
    for key in ['observations', 'next_observations']:
        # Round integer dimensions
        if integer_dims:
            processed_dataset[key][:, integer_dims] = np.round(
                synthetic_dataset[key][:, integer_dims]
            )
        
        # Round constant dimensions to 2 decimal places
        if constant_dims:
            processed_dataset[key][:, constant_dims] = np.round(
                synthetic_dataset[key][:, constant_dims], 
                decimals=2
            )
    
    return processed_dataset


# Convert diffusion samples back to (s, a, r, s') format.
@gin.configurable
def split_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
):
    # Compute dimensions from env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs


@gin.configurable
def construct_diffusion_model(
        inputs: torch.Tensor,
        normalizer_type: str,
        denoiser: str,
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
        # DiT-specific parameters (defaults match diffusion_dit.gin - 25.8M params)
        dit_hidden_size: int = 416,
        dit_depth: int = 8,
        dit_num_heads: int = 8,
        dit_patch_size: int = 4,
        dit_mlp_ratio: float = 4.0,
        dit_dropout: float = 0.0,
        dit_use_semantic_patching: bool = True,  # Set to False for standard patching
) -> ElucidatedDiffusion:
    event_dim = inputs.shape[1]

    if denoiser == 'compositional':
        model = CompositionalResidualMLPDenoiser(d_in=event_dim, cond_dim=cond_dim)
    elif denoiser in DiT1D_models:
        # Use predefined DiT model
        print(f"Using predefined DiT model: {denoiser}")
        model = DiT1D_models[denoiser](
            d_in=event_dim,
            cond_dim=cond_dim,
            patch_size=dit_patch_size,
            dropout=dit_dropout,
            use_semantic_patching=dit_use_semantic_patching,
        )
    elif denoiser.startswith('dit'):
        # Custom DiT configuration
        patching_type = "Semantic" if dit_use_semantic_patching else "Standard"
        print(f"Using custom DiT model with hidden_size={dit_hidden_size}, depth={dit_depth}, "
              f"num_heads={dit_num_heads} | Patching: {patching_type} (patch_size={dit_patch_size})")
        from diffusion.dit_denoiser import DiT1D
        model = DiT1D(
            d_in=event_dim,
            hidden_size=dit_hidden_size,
            depth=dit_depth,
            num_heads=dit_num_heads,
            mlp_ratio=dit_mlp_ratio,
            patch_size=dit_patch_size,
            cond_dim=cond_dim,
            dropout=dit_dropout,
            use_semantic_patching=dit_use_semantic_patching,
        )
    else:
        # Default to monolithic MLP
        model = ResidualMLPDenoiser(d_in=event_dim, cond_dim=cond_dim)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        event_shape=[event_dim],
    )


@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
            self,
            env: gym.Env,
            ema_model,
            num_sample_steps: int = 128,
            sample_batch_size: int = 100000,
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')

    def sample(
            self,
            num_samples: int,
            cond=None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'

        if cond is not None:
            cond = torch.from_numpy(cond).float().to(self.diffusion.device)
            cond = cond.unsqueeze(0).expand(self.sample_batch_size, -1)

        num_batches = num_samples // self.sample_batch_size
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []

        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}.')
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
                cond=cond
            )
            sampled_outputs = sampled_outputs.cpu().numpy()

            # Split samples into (s, a, r, s') format
            transitions = split_diffusion_samples(sampled_outputs, self.env)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        return observations, actions, rewards, next_observations, terminals


