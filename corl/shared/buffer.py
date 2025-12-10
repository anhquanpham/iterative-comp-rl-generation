# Shared functions for CORL algorithms.

from typing import Dict, List, Optional
import numpy as np
import torch

TensorBatch = List[torch.Tensor]

def return_reward_range(dataset, max_episode_steps):
    returns, lengths = [], []
    ep_ret, ep_len = 0.0, 0
    for r, d in zip(dataset["rewards"], dataset["terminals"]):
        ep_ret += float(r)
        ep_len += 1
        if d or ep_len == max_episode_steps:
            returns.append(ep_ret)
            lengths.append(ep_len)
            ep_ret, ep_len = 0.0, 0
    lengths.append(ep_len)  # but still keep track of number of steps
    assert sum(lengths) == len(dataset["rewards"])
    return min(returns), max(returns)


class RewardNormalizer:
    def __init__(self, dataset, env_name, max_episode_steps=1000):
        self.env_name = env_name
        self.scale = 1.
        self.shift = 0.
        if any(s in env_name for s in ("halfcheetah", "hopper", "walker2d")):
            min_ret, max_ret = return_reward_range(dataset, max_episode_steps)
            self.scale = max_episode_steps / (max_ret - min_ret)
        elif "antmaze" in env_name:
            self.shift = -1.

    def __call__(self, reward):
        return (reward + self.shift) * self.scale


class StateNormalizer:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def to_torch(self, device: str):
        self.mean = torch.tensor(self.mean, device=device)
        self.std = torch.tensor(self.std, device=device)

    def __call__(self, state):
        return (state - self.mean) / self.std


class ReplayBufferBase:
    def __init__(
            self,
            device: str = "cpu",
            reward_normalizer: Optional[RewardNormalizer] = None,
            state_normalizer: Optional[StateNormalizer] = None,
    ):
        self.reward_normalizer = reward_normalizer
        self.state_normalizer = state_normalizer
        if self.state_normalizer is not None:
            self.state_normalizer.to_torch(device)
        self._device = device

    # Un-normalized samples.
    def _sample(self, batch_size: int, **kwargs) -> TensorBatch:
        raise NotImplementedError

    def sample(self, batch_size: int, **kwargs) -> TensorBatch:
        states, actions, rewards, next_states, dones = self._sample(batch_size, **kwargs)
        if self.reward_normalizer is not None:
            rewards = self.reward_normalizer(rewards)
        if self.state_normalizer is not None:
            states = self.state_normalizer(states)
            next_states = self.state_normalizer(next_states)

        return [states, actions, rewards, next_states, dones]


class ReplayBuffer(ReplayBufferBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        buffer_size: int,
        device: str = "cpu",
        reward_normalizer: Optional[RewardNormalizer] = None,
        state_normalizer: Optional[StateNormalizer] = None,
    ):
        super().__init__(device, reward_normalizer, state_normalizer)
        self._buffer_size = buffer_size
        self._pointer = 0
        self._size = 0  # Initialize size counter

        self._states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._actions = torch.zeros((buffer_size, action_dim), dtype=torch.float32, device=device)
        self._rewards = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)
        self._next_states = torch.zeros((buffer_size, state_dim), dtype=torch.float32, device=device)
        self._dones = torch.zeros((buffer_size, 1), dtype=torch.float32, device=device)

    @property
    def empty(self):
        return self._size == 0

    @property
    def full(self):
        return self._size == self._buffer_size

    def __len__(self):
        return self._size

    def _to_tensor(self, data: np.ndarray) -> torch.Tensor:
        return torch.tensor(data, dtype=torch.float32, device=self._device)

    def load_transitions(self, data: Dict[str, np.ndarray]):
        if not self.empty:
            raise ValueError("Trying to load data into non-empty replay buffer.")
        n_transitions = data["observations"].shape[0]
        if n_transitions > self._buffer_size:
            raise ValueError("Replay buffer is smaller than the dataset you are trying to load!")
        self._states[:n_transitions] = self._to_tensor(data["observations"])
        self._actions[:n_transitions] = self._to_tensor(data["actions"])
        self._rewards[:n_transitions] = self._to_tensor(data["rewards"][..., None])
        self._next_states[:n_transitions] = self._to_tensor(data["next_observations"])
        self._dones[:n_transitions] = self._to_tensor(data["terminals"][..., None])
        self._pointer = n_transitions % self._buffer_size
        self._size = n_transitions
        print(f"Dataset size: {n_transitions}")

    def _sample(self, batch_size: int, **kwargs) -> TensorBatch:
        indices = np.random.randint(0, self._size, size=batch_size)
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]
        return [states, actions, rewards, next_states, dones]

    def add_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ):
        self._states[self._pointer] = self._to_tensor(state)
        self._actions[self._pointer] = self._to_tensor(action)
        self._rewards[self._pointer] = self._to_tensor(np.array([reward]))
        self._next_states[self._pointer] = self._to_tensor(next_state)
        self._dones[self._pointer] = self._to_tensor(np.array([done], dtype=np.float32))

        self._pointer = (self._pointer + 1) % self._buffer_size
        self._size = min(self._size + 1, self._buffer_size)


def prepare_replay_buffer(
        state_dim: int,
        action_dim: int,
        dataset: dict,
        num_samples: int = -1,
        device: str = "cpu",
        reward_normalizer: Optional[RewardNormalizer] = None,
        state_normalizer: Optional[StateNormalizer] = None,
):
    buffer_args = {
        'reward_normalizer': reward_normalizer,
        'state_normalizer': state_normalizer,
        'device': device,
    }

    if num_samples != -1:
        for key in dataset.keys():
            dataset[key] = dataset[key][:num_samples]
        print('Limiting size of the data to {} samples.'.format(num_samples))

    replay_buffer = ReplayBuffer(
        state_dim=state_dim,
        action_dim=action_dim,
        buffer_size=dataset['rewards'].shape[0],
        **buffer_args,
    )
    replay_buffer.load_transitions(dataset)

    return replay_buffer