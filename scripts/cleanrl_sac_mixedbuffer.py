# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
from __future__ import annotations

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import gym as old_gym  # Import old gym for CompoSuite compatibility
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# --- ADD THESE IMPORTS ---
import composuite
from diffusion.utils import load_single_synthetic_dataset

# from cleanrl_utils.buffers import ReplayBuffer
##############################################################################################
# Copyright notice
#
# This file contains code adapted from stable-baselines3
# (https://github.com/DLR-RM/stable-baselines3/blob/master/stable_baselines3/common/buffers.py)
# licensed under the MIT License.
#
# Copyright (c) 2019-2023 Antonin Raffin, Ashley Hill, Anssi Kanervisto,
# Maximilian Ernestus, Rinu Boney, Pavan Goli, and other contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.



import warnings
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gym import spaces

try:
    # Check memory used by replay buffer when possible
    import psutil
except ImportError:
    psutil = None


__all__ = [
    "BaseBuffer",
    # "RolloutBuffer",
    "ReplayBuffer",
    "RolloutBufferSamples",
    "ReplayBufferSamples",
]


class RolloutBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    old_values: th.Tensor
    old_log_prob: th.Tensor
    advantages: th.Tensor
    returns: th.Tensor


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


def get_action_dim(action_space: spaces.Space) -> int:
    """
    Get the dimension of the action space.

    :param action_space:
    :return:
    """
    if isinstance(action_space, spaces.Box):
        return int(np.prod(action_space.shape))
    elif isinstance(action_space, spaces.Discrete):
        # Action is an int
        return 1
    elif isinstance(action_space, spaces.MultiDiscrete):
        # Number of discrete actions
        return int(len(action_space.nvec))
    elif isinstance(action_space, spaces.MultiBinary):
        # Number of binary actions
        assert isinstance(
            action_space.n, int
        ), f"Multi-dimensional MultiBinary({action_space.n}) action space is not supported. You can flatten it instead."
        return int(action_space.n)
    else:
        raise NotImplementedError(f"{action_space} action space is not supported")


def get_obs_shape(
    observation_space: spaces.Space,
) -> tuple[int, ...] | dict[str, tuple[int, ...]]:
    """
    Get the shape of the observation (useful for the buffers).

    :param observation_space:
    :return:
    """
    if isinstance(observation_space, spaces.Box):
        return observation_space.shape
    elif isinstance(observation_space, spaces.Discrete):
        # Observation is an int
        return (1,)
    elif isinstance(observation_space, spaces.MultiDiscrete):
        # Number of discrete features
        return (int(len(observation_space.nvec)),)
    elif isinstance(observation_space, spaces.MultiBinary):
        # Number of binary features
        return observation_space.shape
    elif isinstance(observation_space, spaces.Dict):
        return {key: get_obs_shape(subspace) for (key, subspace) in observation_space.spaces.items()}  # type: ignore[misc]

    else:
        raise NotImplementedError(f"{observation_space} observation space is not supported")


def get_device(device: th.device | str = "auto") -> th.device:
    """
    Retrieve PyTorch device.
    It checks that the requested device is available first.
    For now, it supports only cpu and cuda.
    By default, it tries to use the gpu.

    :param device: One for 'auto', 'cuda', 'cpu'
    :return: Supported Pytorch device
    """
    # Cuda by default
    if device == "auto":
        device = "cuda"
    # Force conversion to th.device
    device = th.device(device)

    # Cuda not available
    if device.type == th.device("cuda").type and not th.cuda.is_available():
        return th.device("cpu")

    return device


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
        to which the values will be converted
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space
        self.obs_shape = get_obs_shape(observation_space)  # type: ignore[assignment]

        self.action_dim = get_action_dim(action_space)
        self.pos = 0
        self.full = False
        self.device = get_device(device)
        self.n_envs = n_envs

    @staticmethod
    def swap_and_flatten(arr: np.ndarray) -> np.ndarray:
        """
        Swap and then flatten axes 0 (buffer_size) and 1 (n_envs)
        to convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
        to [n_steps * n_envs, ...] (which maintain the order)

        :param arr:
        :return:
        """
        shape = arr.shape
        if len(shape) < 3:
            shape = (*shape, 1)
        return arr.swapaxes(0, 1).reshape(shape[0] * shape[1], *shape[2:])

    def size(self) -> int:
        """
        :return: The current size of the buffer
        """
        if self.full:
            return self.buffer_size
        return self.pos

    def add(self, *args, **kwargs) -> None:
        """
        Add elements to the buffer.
        """
        raise NotImplementedError()

    def extend(self, *args, **kwargs) -> None:
        """
        Add a new batch of transitions to the buffer
        """
        # Do a for loop along the batch axis
        for data in zip(*args):
            self.add(*data)

    def reset(self) -> None:
        """
        Reset the buffer.
        """
        self.pos = 0
        self.full = False

    def sample(self, batch_size: int):
        """
        :param batch_size: Number of element to sample
        :return:
        """
        upper_bound = self.buffer_size if self.full else self.pos
        batch_inds = np.random.randint(0, upper_bound, size=batch_size)
        return self._get_samples(batch_inds)

    @abstractmethod
    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples | RolloutBufferSamples:
        """
        :param batch_inds:
        :return:
        """
        raise NotImplementedError()

    def to_torch(self, array: np.ndarray, copy: bool = True) -> th.Tensor:
        """
        Convert a numpy array to a PyTorch tensor.
        Note: it copies the data by default

        :param array:
        :param copy: Whether to copy or not the data (may be useful to avoid changing things
            by reference). This argument is inoperative if the device is not the CPU.
        :return:
        """
        if copy:
            return th.tensor(array, device=self.device)
        return th.as_tensor(array, device=self.device)


class ReplayBuffer(BaseBuffer):
    """
    Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    observations: np.ndarray
    next_observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    timeouts: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
        handle_timeout_termination: bool = True,
    ):
        super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)

        # Adjust buffer size
        self.buffer_size = max(buffer_size // n_envs, 1)

        # Check that the replay buffer can fit into the memory
        if psutil is not None:
            mem_available = psutil.virtual_memory().available

        # there is a bug if both optimize_memory_usage and handle_timeout_termination are true
        # see https://github.com/DLR-RM/stable-baselines3/issues/934
        if optimize_memory_usage and handle_timeout_termination:
            raise ValueError(
                "ReplayBuffer does not support optimize_memory_usage = True "
                "and handle_timeout_termination = True simultaneously."
            )
        self.optimize_memory_usage = optimize_memory_usage

        self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        if not optimize_memory_usage:
            # When optimizing memory, `observations` contains also the next observation
            self.next_observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=observation_space.dtype)

        self.actions = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self._maybe_cast_dtype(action_space.dtype)
        )

        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.dones = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        # Handle timeouts termination properly if needed
        # see https://github.com/DLR-RM/stable-baselines3/issues/284
        self.handle_timeout_termination = handle_timeout_termination
        self.timeouts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)

        if psutil is not None:
            total_memory_usage: float = (
                self.observations.nbytes + self.actions.nbytes + self.rewards.nbytes + self.dones.nbytes
            )

            if not optimize_memory_usage:
                total_memory_usage += self.next_observations.nbytes

            if total_memory_usage > mem_available:
                # Convert to GB
                total_memory_usage /= 1e9
                mem_available /= 1e9
                warnings.warn(
                    "This system does not have apparently enough memory to store the complete "
                    f"replay buffer {total_memory_usage:.2f}GB > {mem_available:.2f}GB"
                )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        # Reshape needed when using multiple envs with discrete observations
        # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
        if isinstance(self.observation_space, spaces.Discrete):
            obs = obs.reshape((self.n_envs, *self.obs_shape))
            next_obs = next_obs.reshape((self.n_envs, *self.obs_shape))

        # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
        action = action.reshape((self.n_envs, self.action_dim))

        # Copy to avoid modification by reference
        self.observations[self.pos] = np.array(obs)

        if self.optimize_memory_usage:
            self.observations[(self.pos + 1) % self.buffer_size] = np.array(next_obs)
        else:
            self.next_observations[self.pos] = np.array(next_obs)

        self.actions[self.pos] = np.array(action)
        self.rewards[self.pos] = np.array(reward)
        self.dones[self.pos] = np.array(done)

        if self.handle_timeout_termination:
            self.timeouts[self.pos] = np.array([info.get("TimeLimit.truncated", False) for info in infos])

        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True
            self.pos = 0

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        """
        Sample elements from the replay buffer.
        Custom sampling when using memory efficient variant,
        as we should not sample the element with index `self.pos`
        See https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274

        :param batch_size: Number of element to sample
        :return:
        """
        if not self.optimize_memory_usage:
            return super().sample(batch_size=batch_size)
        # Do not sample the element with index `self.pos` as the transitions is invalid
        # (we use only one array to store `obs` and `next_obs`)
        if self.full:
            batch_inds = (np.random.randint(1, self.buffer_size, size=batch_size) + self.pos) % self.buffer_size
        else:
            batch_inds = np.random.randint(0, self.pos, size=batch_size)
        return self._get_samples(batch_inds)

    def _get_samples(self, batch_inds: np.ndarray) -> ReplayBufferSamples:
        # Sample randomly the env idx
        env_indices = np.random.randint(0, high=self.n_envs, size=(len(batch_inds),))

        if self.optimize_memory_usage:
            next_obs = self.observations[(batch_inds + 1) % self.buffer_size, env_indices, :]
        else:
            next_obs = self.next_observations[batch_inds, env_indices, :]

        data = (
            self.observations[batch_inds, env_indices, :],
            self.actions[batch_inds, env_indices, :],
            next_obs,
            # Only use dones that are not due to timeouts
            # deactivated by default (timeouts is initialized as an array of False)
            (self.dones[batch_inds, env_indices] * (1 - self.timeouts[batch_inds, env_indices])).reshape(-1, 1),
            self.rewards[batch_inds, env_indices].reshape(-1, 1),
        )
        return ReplayBufferSamples(*tuple(map(self.to_torch, data)))

    @staticmethod
    def _maybe_cast_dtype(dtype: np.typing.DTypeLike) -> np.typing.DTypeLike:
        """
        Cast `np.float64` action datatype to `np.float32`,
        keep the others dtype unchanged.
        See GH#1572 for more information.

        :param dtype: The original action space dtype
        :return: ``np.float32`` if the dtype was float64,
            the original dtype otherwise.
        """
        if dtype == np.float64:
            return np.float32
        return dtype


# class OfflineReplayBuffer(ReplayBuffer):
#     """
#     A replay buffer initialized with offline data that doesn't get updated during training.
#     """
#     def __init__(
#         self,
#         synthetic_dataset: dict,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         device: th.device | str = "auto",
#         n_envs: int = 1,
#     ):
#         # Initialize with the size of the synthetic dataset
#         buffer_size = len(synthetic_dataset["observations"])
#         super().__init__(
#             buffer_size=buffer_size,
#             observation_space=observation_space,
#             action_space=action_space,
#             device=device,
#             n_envs=n_envs,
#             optimize_memory_usage=False,  # Don't optimize memory since we won't be adding more data
#             handle_timeout_termination=False,
#         )
        
#         # Load the synthetic dataset directly into the buffer
#         self.observations = synthetic_dataset["observations"]
#         self.next_observations = synthetic_dataset["next_observations"]
#         self.actions = synthetic_dataset["actions"]
#         self.rewards = synthetic_dataset["rewards"].reshape(-1, 1)
#         self.dones = synthetic_dataset["terminals"].reshape(-1, 1)
        
#         # Set buffer as full since we loaded all data
#         self.full = True
#         self.pos = 0  # Position doesn't matter since we won't be adding new data

#     def add(self, *args, **kwargs):
#         # Override add to prevent modifications to offline buffer
#         pass

class OfflineReplayBuffer(ReplayBuffer):
    """
    A replay buffer initialized with offline data that doesn't get updated during training.
    """
    def __init__(
        self,
        synthetic_dataset: dict,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
    ):
        # Initialize with the size of the synthetic dataset
        buffer_size = len(synthetic_dataset["observations"])
        super().__init__(
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=False,  # offline only
            handle_timeout_termination=False,
        )
        # Alias for readability
        bs = buffer_size
        obs = synthetic_dataset["observations"]          # shape (bs, *obs_shape)
        next_obs = synthetic_dataset["next_observations"]
        acts = synthetic_dataset["actions"]               # shape (bs, action_dim)
        rews = synthetic_dataset["rewards"]               # shape (bs,)
        dones = synthetic_dataset["terminals"]            # shape (bs,)

        # Copy into the pre-allocated 3D buffers: (bs, n_envs, ...)
        # For n_envs==1, index at env slot 0
        self.observations[:bs, 0, :]      = obs
        self.next_observations[:bs, 0, :] = next_obs
        self.actions[:bs, 0, :]           = acts
        # rewards and dones are 2D arrays (bs, n_envs)
        self.rewards[:bs, 0]              = rews
        self.dones[:bs, 0]                = dones

        # Mark buffer as full since we've loaded all transitions
        self.full = True
        # pos doesn't matter since offline buffer is read-only
        self.pos = 0

    def add(self, *args, **kwargs):
        # Prevent any runtime additions to the offline buffer
        pass


class MixedReplayBuffer:
    """
    A buffer that samples from both online and offline replay buffers with a specified mix ratio.
    """
    def __init__(
        self,
        online_buffer: ReplayBuffer,
        offline_buffer: OfflineReplayBuffer,
        device: th.device | str = "auto",
        mix_ratio: float = 0.5,  # Ratio of samples to take from offline buffer (0.0 to 1.0)
    ):
        self.online_buffer = online_buffer
        self.offline_buffer = offline_buffer
        self.device = device
        self.mix_ratio = mix_ratio

    def sample(self, batch_size: int) -> ReplayBufferSamples:
        # Calculate number of samples from each buffer
        offline_samples = int(batch_size * self.mix_ratio)
        online_samples = batch_size - offline_samples

        if online_samples > 0 and self.online_buffer.pos > 0:
            # Sample from online buffer
            online_data = self.online_buffer.sample(online_samples)
            
            if offline_samples > 0:
                # Sample from offline buffer and combine
                offline_data = self.offline_buffer.sample(offline_samples)
                
                # Concatenate the samples
                return ReplayBufferSamples(
                    observations=torch.cat([online_data.observations, offline_data.observations]),
                    actions=torch.cat([online_data.actions, offline_data.actions]),
                    next_observations=torch.cat([online_data.next_observations, offline_data.next_observations]),
                    dones=torch.cat([online_data.dones, offline_data.dones]),
                    rewards=torch.cat([online_data.rewards, offline_data.rewards]),
                )
            return online_data
        else:
            # If no online samples available or requested, sample only from offline
            return self.offline_buffer.sample(batch_size)

    def add(self, *args, **kwargs):
        # Add only to the online buffer
        self.online_buffer.add(*args, **kwargs)

    @property
    def pos(self):
        # Return the position of the online buffer
        return self.online_buffer.pos

    def set_mix_ratio(self, mix_ratio: float):
        """Update the mix ratio between offline and online data."""
        self.mix_ratio = max(0.0, min(1.0, mix_ratio))


# class RolloutBuffer(BaseBuffer):
#     """
#     Rollout buffer used in on-policy algorithms like A2C/PPO.
#     It corresponds to ``buffer_size`` transitions collected
#     using the current policy.
#     This experience will be discarded after the policy update.
#     In order to use PPO objective, we also store the current value of each state
#     and the log probability of each taken action.

#     The term rollout here refers to the model-free notion and should not
#     be used with the concept of rollout used in model-based RL or planning.
#     Hence, it is only involved in policy and value function training but not action selection.

#     :param buffer_size: Max number of element in the buffer
#     :param observation_space: Observation space
#     :param action_space: Action space
#     :param device: PyTorch device
#     :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
#         Equivalent to classic advantage when set to 1.
#     :param gamma: Discount factor
#     :param n_envs: Number of parallel environments
#     """

#     observations: np.ndarray
#     actions: np.ndarray
#     rewards: np.ndarray
#     advantages: np.ndarray
#     returns: np.ndarray
#     episode_starts: np.ndarray
#     log_probs: np.ndarray
#     values: np.ndarray

#     def __init__(
#         self,
#         buffer_size: int,
#         observation_space: spaces.Space,
#         action_space: spaces.Space,
#         device: th.device | str = "auto",
#         gae_lambda: float = 1,
#         gamma: float = 0.99,
#         n_envs: int = 1,
#     ):
#         super().__init__(buffer_size, observation_space, action_space, device, n_envs=n_envs)
#         self.gae_lambda = gae_lambda
#         self.gamma = gamma
#         self.generator_ready = False
#         self.reset()

#     def reset(self) -> None:
#         self.observations = np.zeros((self.buffer_size, self.n_envs, *self.obs_shape), dtype=np.float32)
#         self.actions = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=np.float32)
#         self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.log_probs = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
#         self.generator_ready = False
#         super().reset()

#     def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
#         """
#         Post-processing step: compute the lambda-return (TD(lambda) estimate)
#         and GAE(lambda) advantage.

#         Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
#         to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
#         where R is the sum of discounted reward with value bootstrap
#         (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

#         The TD(lambda) estimator has also two special cases:
#         - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
#         - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

#         For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

#         :param last_values: state value estimation for the last step (one for each env)
#         :param dones: if the last step was a terminal step (one bool for each env).
#         """
#         # Convert to numpy
#         last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

#         last_gae_lam = 0
#         for step in reversed(range(self.buffer_size)):
#             if step == self.buffer_size - 1:
#                 next_non_terminal = 1.0 - dones.astype(np.float32)
#                 next_values = last_values
#             else:
#                 next_non_terminal = 1.0 - self.episode_starts[step + 1]
#                 next_values = self.values[step + 1]
#             delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
#             last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
#             self.advantages[step] = last_gae_lam
#         # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
#         # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
#         self.returns = self.advantages + self.values

#     def add(
#         self,
#         obs: np.ndarray,
#         action: np.ndarray,
#         reward: np.ndarray,
#         episode_start: np.ndarray,
#         value: th.Tensor,
#         log_prob: th.Tensor,
#     ) -> None:
#         """
#         :param obs: Observation
#         :param action: Action
#         :param reward:
#         :param episode_start: Start of episode signal.
#         :param value: estimated value of the current state
#             following the current policy.
#         :param log_prob: log probability of the action
#             following the current policy.
#         """
#         if len(log_prob.shape) == 0:
#             # Reshape 0-d tensor to avoid error
#             log_prob = log_prob.reshape(-1, 1)

#         # Reshape needed when using multiple envs with discrete observations
#         # as numpy cannot broadcast (n_discrete,) to (n_discrete, 1)
#         if isinstance(self.observation_space, spaces.Discrete):
#             obs = obs.reshape((self.n_envs, *self.obs_shape))

#         # Reshape to handle multi-dim and discrete action spaces, see GH #970 #1392
#         action = action.reshape((self.n_envs, self.action_dim))

#         self.observations[self.pos] = np.array(obs)
#         self.actions[self.pos] = np.array(action)
#         self.rewards[self.pos] = np.array(reward)
#         self.episode_starts[self.pos] = np.array(episode_start)
#         self.values[self.pos] = value.clone().cpu().numpy().flatten()
#         self.log_probs[self.pos] = log_prob.clone().cpu().numpy()
#         self.pos += 1
#         if self.pos == self.buffer_size:
#             self.full = True

#     def get(self, batch_size: int | None = None) -> Generator[RolloutBufferSamples]:
#         assert self.full, ""
#         indices = np.random.permutation(self.buffer_size * self.n_envs)
#         # Prepare the data
#         if not self.generator_ready:
#             _tensor_names = [
#                 "observations",
#                 "actions",
#                 "values",
#                 "log_probs",
#                 "advantages",
#                 "returns",
#             ]

#             for tensor in _tensor_names:
#                 self.__dict__[tensor] = self.swap_and_flatten(self.__dict__[tensor])
#             self.generator_ready = True

#         # Return everything, don't create minibatches
#         if batch_size is None:
#             batch_size = self.buffer_size * self.n_envs

#         start_idx = 0
#         while start_idx < self.buffer_size * self.n_envs:
#             yield self._get_samples(indices[start_idx : start_idx + batch_size])
#             start_idx += batch_size

#     def _get_samples(
#         self,
#         batch_inds: np.ndarray,
#     ) -> RolloutBufferSamples:
#         data = (
#             self.observations[batch_inds],
#             self.actions[batch_inds],
#             self.values[batch_inds].flatten(),
#             self.log_probs[batch_inds].flatten(),
#             self.advantages[batch_inds].flatten(),
#             self.returns[batch_inds].flatten(),
#         )
#         return RolloutBufferSamples(*tuple(map(self.to_torch, data)))
##############################################################################################

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    # wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the task"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    num_envs: int = 1
    """the number of parallel game environments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 1 # CleanRL default was 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""

    # Evaluation and checkpointing
    eval_freq: int = 100000
    """frequency of policy evaluation (in timesteps)"""
    n_eval_episodes: int = 10
    """number of episodes for policy evaluation"""

    # CompoSuite specific arguments
    use_composuite: bool = True
    """whether to use composuite environment instead of gym environment"""
    robot: str = "IIWA"
    """robot type for composuite"""
    obj: str = "Box"
    """object type for composuite"""
    obst: str = "ObjectDoor"
    """obstacle type for composuite"""
    subtask: str = "Push"
    """subtask type for composuite"""
    env_horizon: int = 500
    """environment horizon for composuite"""
    base_synthetic_data_path: str = "/home/quanpham/iterative-comp-rl-generation/results/diffusion/monolithic_seed0_train98_1"
    """base path to the synthetic dataset"""

    # Mixed buffer specific arguments
    mix_ratio_start: float = 0.5
    """initial ratio of offline data in mixed buffer"""
    mix_ratio_end: float = 0.0
    """final ratio of offline data in mixed buffer"""
    mix_ratio_decay_steps: int = 500000
    """number of steps to decay mix ratio from start to end"""
    mix_ratio_decay_start: int = 500000
    """number of steps before starting mix ratio decay"""


# Environment wrapper functions for CompoSuite
# GLOBAL_STEP_COUNTER = 0
# TIME_LIMIT = 500  # Will be updated based on args

def modified_reset(gym_env):
    original_reset = gym_env.reset

    def reset_wrapper(*args, **kwargs):
        # global GLOBAL_STEP_COUNTER
        # GLOBAL_STEP_COUNTER = 0

        # Remove seed from kwargs for Gym 0.26.2 compatibility
        if 'seed' in kwargs:
            del kwargs['seed']
        if 'options' in kwargs:
            del kwargs['options']

        result = original_reset(*args, **kwargs)
        # old Gym: reset() -> obs
        # new Gym/Gymnasium: reset() -> (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        # For Gym 0.26+, callers expect (obs, info)
        return obs, info

    gym_env.reset = reset_wrapper
    return gym_env


def modified_step(gym_env):
    original_step = gym_env.step

    def step_wrapper(*args, **kwargs):
        result = original_step(*args, **kwargs)

        # Unify old Gym (4 values) and Gymnasium/new Gym (5 values)
        if len(result) == 5:
            obs, rew, terminated, truncated, info = result
        elif len(result) == 4:
            obs, rew, done, info = result
            terminated, truncated = done, False
        else:
            raise RuntimeError(
                f"env.step returned {len(result)} items, expected 4 or 5"
            )

        # Recombine into 5-tuple (Gymnasium API)
        return obs, rew, terminated, truncated, info

    gym_env.step = step_wrapper
    return gym_env


def make_env(env_id, seed, idx, capture_video, run_name, args):
    def thunk():
        if args.use_composuite:
            # Create CompoSuite environment
            env = composuite.make(
                robot=args.robot,
                obj=args.obj,
                obstacle=args.obst,
                task=args.subtask,
                has_renderer=False,
                has_offscreen_renderer=False,
                reward_shaping=True,
                use_camera_obs=False,
                use_task_id_obs=False,
                env_horizon=args.env_horizon,
            )
 
            # Apply environment wrappers
            # Standardize reset API for old gym versions
            modified_reset(env)
            # Apply step wrapper to normalise step outputs BEFORE TimeLimit
            modified_step(env)
            # Use Gym's built-in TimeLimit instead of custom GLOBAL_STEP_COUNTER logic
            env = old_gym.wrappers.TimeLimit(env, max_episode_steps=args.env_horizon)
             
            if capture_video and idx == 0:
                env = old_gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            
            # Use old_gym RecordEpisodeStatistics for CompoSuite
            env = old_gym.wrappers.RecordEpisodeStatistics(env)
        else:
            # Use original gymnasium environment
            if capture_video and idx == 0:
                env = gym.make(env_id, render_mode="rgb_array")
                env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
            else:
                env = gym.make(env_id)
            
            # Use gymnasium RecordEpisodeStatistics for standard environments
            env = gym.wrappers.RecordEpisodeStatistics(env)
        
        env.action_space.seed(seed)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(
            np.array(env.single_observation_space.shape).prod() + np.prod(env.single_action_space.shape),
            256,
        )
        self.fc2 = nn.Linear(256, 256)
        self.ln2 = nn.LayerNorm(256) #Layer Norm, newly added July 22nd
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.ln2(x) #Layer Norm, newly added July 22nd
        x = self.fc3(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, np.prod(env.single_action_space.shape))
        self.fc_logstd = nn.Linear(256, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (env.single_action_space.high - env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (env.single_action_space.high + env.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x):
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_prob, mean


def evaluate_policy(envs, actor, device, seed=0):
    """
    Evaluate the policy using vectorized environments for parallel episode execution.
    """
    print("DEBUG: Starting evaluation...")
    actor.eval()  # Set to evaluation mode
    
    # Reset all environments
    # print("DEBUG: Resetting environments...")
    if hasattr(envs, 'reset'):
        # Vector environment
        obs, _ = envs.reset()
    else:
        # Fallback for single environment
        obs, _ = envs.reset()
        obs = np.array([obs])
    print(f"DEBUG: Reset complete. obs shape: {obs.shape}")
    
    episode_returns = np.zeros(envs.num_envs)
    episode_lengths = np.zeros(envs.num_envs)
    finished_episodes = np.zeros(envs.num_envs, dtype=bool)
    final_returns = []
    final_successes = []
    
    # Use a reasonable safety limit - environments should naturally terminate
    # via their own time limits (CompoSuite: env_horizon, Gym: built-in limits)
    # This is just a safety net to prevent infinite loops in pathological cases
    max_steps = 1000  # Safety limit - should rarely be reached
    
    step_count = 0
    while len(final_returns) < envs.num_envs and np.any(episode_lengths < max_steps):
        step_count += 1
        # if step_count % 100 == 0:
            # print(f"DEBUG: Step {step_count}, finished: {len(final_returns)}/{envs.num_envs}, lengths: {episode_lengths}")
        
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).to(device)
            # Use mean action for evaluation (deterministic)
            _, _, actions = actor.get_action(obs_tensor)
            actions = actions.cpu().numpy()
        
        # Step all environments
        # print(f"DEBUG: Stepping environments, step {step_count}...")
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)
        # print(f"DEBUG: Step complete. rewards: {rewards}, terminations: {terminations}, truncations: {truncations}")
        
        # Update episode data for environments that haven't finished
        active_mask = ~finished_episodes
        episode_returns[active_mask] += rewards[active_mask]
        episode_lengths[active_mask] += 1
        
        # Check for newly finished episodes
        dones = terminations | truncations
        newly_finished = dones & ~finished_episodes
        
        if np.any(newly_finished):
            for idx in np.where(newly_finished)[0]:
                final_returns.append(episode_returns[idx])
                # Extract success from info
                if "final_info" in infos and infos["final_info"][idx] is not None:
                    final_success = infos["final_info"][idx].get('Success', 0)
                else:
                    final_success = infos.get('Success', np.zeros(envs.num_envs))[idx] if isinstance(infos, dict) else 0
                final_successes.append(final_success)
                finished_episodes[idx] = True
            # print(f"DEBUG: Newly finished episodes: {np.where(newly_finished)[0]}, total finished: {len(final_returns)}")
        
        obs = next_obs
        
        # Safety check
        if np.all(episode_lengths >= max_steps):
            print("DEBUG: All episodes hit max_steps, breaking...")
            break
    
    print(f"DEBUG: Evaluation complete. Final returns: {final_returns}")
    actor.train()  # Set back to training mode
    return np.array(final_returns), np.array(final_successes)


if __name__ == "__main__":

    args = tyro.cli(Args)
    
    # Generate appropriate run name based on environment type
    if args.use_composuite:
        # Extract train number from the data path
        import re
        train_match = re.search(r'train(\d+)_', args.base_synthetic_data_path)
        train_number = train_match.group(1) if train_match else "unknown"
        run_name = f"composuite_{args.robot}_{args.obj}_{args.obst}_{args.subtask}__{args.exp_name}__{args.seed}__train{train_number}_{args.base_synthetic_data_path}"
    else:
        run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            # entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    if args.use_composuite:
        # Use old_gym vector environment for CompoSuite
        envs = old_gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, old_gym.spaces.Box), "only continuous action space is supported"
    else:
        # Use gymnasium vector environment for standard environments
        envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name, args) for i in range(args.num_envs)]
        )
        assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # Create checkpoints directory
    checkpoints_path = f"runs/{run_name}/checkpoints"
    os.makedirs(checkpoints_path, exist_ok=True)
    
    # Initialize evaluation tracking
    best_eval_score = float('-inf')
    best_success_rate = float('-inf')
    evaluations = []

    # Create separate evaluation environment
    if args.use_composuite:
        eval_envs = old_gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + 1000 + i, i, False, run_name, args) for i in range(args.n_eval_episodes)]
        )
    else:
        eval_envs = gym.vector.SyncVectorEnv(
            [make_env(args.env_id, args.seed + 1000 + i, i, False, run_name, args) for i in range(args.n_eval_episodes)]
        )
    
    eval_envs.action_space.seed(args.seed)

    actor = Actor(envs).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    # Load synthetic dataset
    if args.use_composuite:
        synthetic_dataset = load_single_synthetic_dataset(
            base_path=args.base_synthetic_data_path,
            robot=args.robot,
            obj=args.obj,
            obst=args.obst,
            task=args.subtask
        )
        print("Loaded synthetic dataset with", len(synthetic_dataset["observations"]), "transitions")

    # Initialize buffers
    online_buffer = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )

    if args.use_composuite:
        offline_buffer = OfflineReplayBuffer(
            synthetic_dataset,
            envs.single_observation_space,
            envs.single_action_space,
            device,
            n_envs=args.num_envs,
        )
        rb = MixedReplayBuffer(
            online_buffer=online_buffer,
            offline_buffer=offline_buffer,
            device=device,
            mix_ratio=args.mix_ratio_start,
        )
    else:
        rb = online_buffer

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    if args.use_composuite:
        # For CompoSuite (Gym 0.26.2), don't pass seed to reset
        obs, _ = envs.reset()
    else:
        # For standard environments (Gymnasium), pass seed
        obs, _ = envs.reset(seed=args.seed)

    for global_step in range(args.total_timesteps):
        # Update mix ratio if using mixed buffer
        if args.use_composuite:
            if global_step == args.mix_ratio_decay_start:
                print(f"===> Global step {global_step}: Starting mix ratio decay...")

            decay_end_step = args.mix_ratio_decay_start + args.mix_ratio_decay_steps
            if global_step == decay_end_step:
                print(f"===> Global step {global_step}: Finished mix ratio decay.")

            if global_step >= args.mix_ratio_decay_start:
                # Calculate current mix ratio
                progress = min(1.0, (global_step - args.mix_ratio_decay_start) / args.mix_ratio_decay_steps)
                current_mix_ratio = args.mix_ratio_start + progress * (args.mix_ratio_end - args.mix_ratio_start)
                rb.set_mix_ratio(current_mix_ratio)

                if global_step % 1000 == 0:  # Log mix ratio periodically
                    writer.add_scalar("charts/mix_ratio", current_mix_ratio, global_step)

        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is not None:
                    print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                    writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                    writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                    break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            data = rb.sample(args.batch_size)
            with torch.no_grad():
                next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)
                qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
                next_q_value = data.rewards.flatten() + (1 - data.dones.flatten()) * args.gamma * (min_qf_next_target).view(-1)

            qf1_a_values = qf1(data.observations, data.actions).view(-1)
            qf2_a_values = qf2(data.observations, data.actions).view(-1)
            qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
            qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
            qf_loss = qf1_loss + qf2_loss

            # optimize the model
            q_optimizer.zero_grad()
            qf_loss.backward()
            q_optimizer.step()

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.get_action(data.observations)
                    qf1_pi = qf1(data.observations, pi)
                    qf2_pi = qf2(data.observations, pi)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.get_action(data.observations)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(args.tau * param.data + (1 - args.tau) * target_param.data)

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

        # EVALUATION: Evaluate policy periodically
        if global_step > args.learning_starts and (global_step % args.eval_freq == 0 or global_step == args.total_timesteps - 1):
            print(f"Time steps: {global_step + 1}")
            eval_scores, eval_successes = evaluate_policy(
                eval_envs,
                actor,
                device=device,
                seed=args.seed,
            )
            eval_score = eval_scores.mean()
            success_rate = eval_successes.mean()
            evaluations.append(eval_score)
            print(f"Evaluation over {args.n_eval_episodes} episodes: {eval_score:.3f}, Success rate: {success_rate:.2f}")
            
            # Log evaluation results to tensorboard/wandb
            writer.add_scalar("eval/episodic_return", eval_score, global_step)
            writer.add_scalar("eval/best_return", max(best_eval_score, eval_score), global_step)
            writer.add_scalar("eval/success_rate", success_rate, global_step)
            writer.add_scalar("eval/best_success_rate", max(best_success_rate, success_rate), global_step)
            
            # Save best model checkpoint
            if eval_score > best_eval_score:
                best_eval_score = eval_score
                checkpoint_data = {
                    'actor_state_dict': actor.state_dict(),
                    'qf1_state_dict': qf1.state_dict(),
                    'qf2_state_dict': qf2.state_dict(),
                    'qf1_target_state_dict': qf1_target.state_dict(),
                    'qf2_target_state_dict': qf2_target.state_dict(),
                    'actor_optimizer_state_dict': actor_optimizer.state_dict(),
                    'q_optimizer_state_dict': q_optimizer.state_dict(),
                    'global_step': global_step,
                    'eval_score': eval_score,
                    'args': args,
                }
                if args.autotune:
                    checkpoint_data['log_alpha'] = log_alpha
                    checkpoint_data['a_optimizer_state_dict'] = a_optimizer.state_dict()
                    checkpoint_data['alpha'] = alpha
                
                torch.save(
                    checkpoint_data,
                    os.path.join(checkpoints_path, "sac_policy_model.pt"),
                )
                print(f"New best score: {best_eval_score:.3f}. Model saved.")
            
            if success_rate > best_success_rate:
                best_success_rate = success_rate

    # Close evaluation environment
    if hasattr(eval_envs, 'close'):
        eval_envs.close()

    # Add final summary to wandb
    if args.track:
        wandb.run.summary["best_eval_return"] = best_eval_score
        wandb.run.summary["best_success_rate"] = best_success_rate
        print(f"Training completed. Best evaluation score: {best_eval_score:.3f}, Best success rate: {best_success_rate:.2f}")

    envs.close()
    writer.close()
