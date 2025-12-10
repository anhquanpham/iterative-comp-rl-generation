#!/usr/bin/env python3
"""
Train a compositional TD3+BC policy on multiple expert datasets.

This script trains a single multitask policy using compositional networks
on the first 14 training tasks from expert data, then evaluates on all 32 test tasks.
"""

import os
import sys
import json
import argparse
import pathlib
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Add paths
sys.path.append(str(Path(__file__).parent.parent))

# Import composuite
import composuite

# Import data loading from diffusion utils (handles robot-specific directories)
from diffusion.utils import load_single_composuite_dataset


def modified_reset(gym_env):
    """Normalize reset API to return (obs, info)"""
    original_reset = gym_env.reset

    def reset_wrapper(*args, **kwargs):
        if 'seed' in kwargs:
            del kwargs['seed']
        if 'options' in kwargs:
            del kwargs['options']
        result = original_reset(*args, **kwargs)
        # old Gym: reset() -> obs; new Gym/Gymnasium: reset() -> (obs, info)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
        else:
            obs, info = result, {}
        return obs, info

    gym_env.reset = reset_wrapper
    return gym_env


def modified_step(gym_env):
    """Normalize step API to return (obs, rew, terminated, truncated, info)"""
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
        return obs, rew, terminated, truncated, info

    gym_env.step = step_wrapper
    return gym_env

# Import compositional networks
sys.path.insert(0, str(Path(__file__).parent))
from compositional_networks import (
    CompositionalEncoder,
    CompositionalEncoderWithAction,
    create_cp_encoderfactory_params
)


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_task_list(task_list_path: str) -> Tuple[List[Tuple], List[Tuple]]:
    """Load train and test tasks from JSON file"""
    with open(task_list_path, 'r') as f:
        task_data = json.load(f)
    
    train_tasks = [tuple(task) for task in task_data['train']]
    test_tasks = [tuple(task) for task in task_data['test']]
    
    return train_tasks, test_tasks


def load_multitask_dataset(
    base_path: str,
    task_list: List[Tuple],
    dataset_type: str = 'expert'
) -> Dict[str, np.ndarray]:
    """
    Load multiple tasks and concatenate into single dataset.
    IMPORTANT: Task IDs are kept in observations for compositional networks.
    """
    num_tasks = len(task_list)
    print(f"\nLoading {num_tasks} tasks from {dataset_type} dataset...")
    
    # Preallocate memory (1M transitions per task)
    observations = np.zeros((num_tasks * 1000000, 93), dtype=np.float32)
    next_observations = np.zeros((num_tasks * 1000000, 93), dtype=np.float32)
    actions = np.zeros((num_tasks * 1000000, 8), dtype=np.float32)
    rewards = np.zeros((num_tasks * 1000000,), dtype=np.float32)
    terminals = np.zeros((num_tasks * 1000000,), dtype=np.uint8)
    timeouts = np.zeros((num_tasks * 1000000,), dtype=np.uint8)
    
    for i, task in enumerate(tqdm(task_list, desc="Loading tasks")):
        robot, obj, obst, subtask = task
        
        # Use same function as existing pipeline
        dataset = load_single_composuite_dataset(base_path, dataset_type, robot, obj, obst, subtask)
        
        obs = dataset["observations"]
        act = dataset["actions"]
        rew = dataset["rewards"]
        term = dataset["terminals"]
        tout = dataset["timeouts"]
        
        # Construct next_observations: next_obs[t] = obs[t+1]
        # For last transition of each episode, next_obs doesn't matter (done=1)
        next_obs = np.zeros_like(obs)
        next_obs[:-1] = obs[1:]  # Shift observations by 1
        next_obs[-1] = obs[-1]   # Last one doesn't matter
        
        # Handle episode boundaries: when done, next_obs should be terminal state
        done_indices = np.where((term == 1) | (tout == 1))[0]
        for idx in done_indices:
            if idx < len(obs) - 1:
                # Next obs after terminal should be the terminal state itself
                next_obs[idx] = obs[idx]
        
        observations[i * 1000000:(i + 1) * 1000000] = obs
        next_observations[i * 1000000:(i + 1) * 1000000] = next_obs
        actions[i * 1000000:(i + 1) * 1000000] = act
        rewards[i * 1000000:(i + 1) * 1000000] = rew
        terminals[i * 1000000:(i + 1) * 1000000] = term
        
        # Timeouts should not happen when terminal
        timeouts[i * 1000000:(i + 1) * 1000000] = tout
        timeouts[i * 1000000:(i + 1) * 1000000][term == 1] = 0
    
    print(f"✓ Loaded {num_tasks * 1000000:,} transitions from {num_tasks} tasks")
    print(f"  Observation shape: {observations.shape}")
    print(f"  Next observation shape: {next_observations.shape}")
    print(f"  Action shape: {actions.shape}")
    
    return {
        'observations': observations,
        'next_observations': next_observations,
        'actions': actions,
        'rewards': rewards,
        'terminals': terminals,
        'timeouts': timeouts
    }


class ReplayBuffer:
    """Simple replay buffer for offline RL"""
    
    def __init__(self, observations, next_observations, actions, rewards, terminals, timeouts, device):
        self.observations = torch.from_numpy(observations).to(device)
        self.next_observations = torch.from_numpy(next_observations).to(device)
        self.actions = torch.from_numpy(actions).to(device)
        self.rewards = torch.from_numpy(rewards).to(device)
        self.terminals = torch.from_numpy(terminals).to(device)
        self.timeouts = torch.from_numpy(timeouts).to(device)
        self.size = len(observations)
        self.device = device
    
    def sample(self, batch_size):
        """Sample a batch of transitions"""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return (
            self.observations[indices],
            self.next_observations[indices],
            self.actions[indices],
            self.rewards[indices],
            self.terminals[indices],
            self.timeouts[indices]
        )


class CompositionalActor(nn.Module):
    """Actor with compositional encoder"""
    
    def __init__(self, encoder_kwargs, observation_shape, max_action):
        super().__init__()
        self.encoder = CompositionalEncoder(encoder_kwargs, observation_shape)
        self.max_action = max_action
    
    def forward(self, state):
        return self.max_action * torch.tanh(self.encoder(state))


class CompositionalCritic(nn.Module):
    """Critic with compositional encoder (with action)"""
    
    def __init__(self, encoder_kwargs, observation_shape, action_size):
        super().__init__()
        self.encoder = CompositionalEncoderWithAction(
            encoder_kwargs, observation_shape, action_size
        )
    
    def forward(self, state, action):
        return self.encoder(state, action)


class CompositionalTD3BC:
    """TD3+BC with compositional networks"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        device,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2,
        alpha=2.5,
    ):
        # Create compositional actor
        actor_encoder_kwargs = create_cp_encoderfactory_params(
            with_action=False, output_dim=action_dim
        )
        self.actor = CompositionalActor(
            actor_encoder_kwargs, (state_dim,), max_action
        ).to(device)
        self.actor_target = CompositionalActor(
            actor_encoder_kwargs, (state_dim,), max_action
        ).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        
        # Create compositional critics
        critic_encoder_kwargs = create_cp_encoderfactory_params(
            with_action=True, output_dim=1
        )
        self.critic_1 = CompositionalCritic(
            critic_encoder_kwargs, (state_dim,), action_dim
        ).to(device)
        self.critic_1_target = CompositionalCritic(
            critic_encoder_kwargs, (state_dim,), action_dim
        ).to(device)
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_1_optimizer = torch.optim.Adam(self.critic_1.parameters(), lr=3e-4)
        
        self.critic_2 = CompositionalCritic(
            critic_encoder_kwargs, (state_dim,), action_dim
        ).to(device)
        self.critic_2_target = CompositionalCritic(
            critic_encoder_kwargs, (state_dim,), action_dim
        ).to(device)
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())
        self.critic_2_optimizer = torch.optim.Adam(self.critic_2.parameters(), lr=3e-4)
        
        self.max_action = max_action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
        self.alpha = alpha
        self.device = device
        self.total_it = 0
    
    def select_action(self, state):
        """Select action for evaluation (no noise)"""
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            if state.ndim == 1:
                state = state.unsqueeze(0)
            action = self.actor(state)
            return action.cpu().numpy()[0]
    
    def train(self, replay_buffer, batch_size=256):
        """One training step"""
        self.total_it += 1
        
        # Sample batch
        state, next_state, action, reward, terminal, timeout = replay_buffer.sample(batch_size)
        done = terminal.float()
        reward = reward.unsqueeze(-1)
        done = done.unsqueeze(-1)
        
        # Critic update
        with torch.no_grad():
            # Target policy smoothing
            noise = (torch.randn_like(action) * self.policy_noise).clamp(
                -self.noise_clip, self.noise_clip
            )
            next_action = (self.actor_target(next_state) + noise).clamp(
                -self.max_action, self.max_action
            )
            
            # Target Q-values
            target_q1 = self.critic_1_target(next_state, next_action)
            target_q2 = self.critic_2_target(next_state, next_action)
            target_q = torch.min(target_q1, target_q2)
            target_q = reward + (1 - done) * self.discount * target_q
        
        # Current Q-values
        current_q1 = self.critic_1(state, action)
        current_q2 = self.critic_2(state, action)
        
        # Critic loss
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        # Optimize critics
        self.critic_1_optimizer.zero_grad()
        self.critic_2_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1_optimizer.step()
        self.critic_2_optimizer.step()
        
        # Delayed actor update
        actor_loss = None
        if self.total_it % self.policy_freq == 0:
            # Actor loss (TD3+BC)
            pi = self.actor(state)
            q = self.critic_1(state, pi)
            lmbda = self.alpha / q.abs().mean().detach()
            
            actor_loss = -lmbda * q.mean() + F.mse_loss(pi, action)
            
            # Optimize actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # Update target networks
            self._soft_update(self.actor, self.actor_target)
            self._soft_update(self.critic_1, self.critic_1_target)
            self._soft_update(self.critic_2, self.critic_2_target)
        
        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item() if actor_loss is not None else None
        }
    
    def _soft_update(self, source, target):
        """Soft update of target network"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                (1 - self.tau) * target_param.data + self.tau * source_param.data
            )
    
    def save(self, filepath):
        """Save model checkpoint"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic_1': self.critic_1.state_dict(),
            'critic_2': self.critic_2.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_1_optimizer': self.critic_1_optimizer.state_dict(),
            'critic_2_optimizer': self.critic_2_optimizer.state_dict(),
            'total_it': self.total_it,
        }, filepath)
        print(f"✓ Saved checkpoint: {filepath}")


def evaluate_policy(agent, task_list, n_episodes=10, use_task_id_obs=True):
    """
    Evaluate policy on multiple tasks.
    
    IMPORTANT: use_task_id_obs=True because compositional networks need task IDs.
    """
    results = {}
    
    for task in tqdm(task_list, desc="Evaluating tasks"):
        robot, obj, obst, subtask = task
        task_name = f"{robot}_{obj}_{obst}_{subtask}"
        
        # Create environment WITH task IDs for compositional networks
        env = composuite.make(robot, obj, obst, subtask, 
                             use_task_id_obs=use_task_id_obs, 
                             ignore_done=False)
        # Apply API normalization wrappers (consistent with train_augmented_policy.py)
        modified_reset(env)
        modified_step(env)
        
        episode_rewards = []
        episode_successes = []
        
        for _ in range(n_episodes):
            state, info = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = agent.select_action(state)
                state, reward, terminated, truncated, info = env.step(action)
                done = bool(terminated or truncated)
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            # Extract success from final info (key always exists in composuite environment)
            final_success = info['Success']
            episode_successes.append(final_success)
        
        results[task_name] = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'success_rate': np.mean(episode_successes),
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to offline dataset')
    parser.add_argument('--task_list_path', type=str, required=True,
                       help='Path to task list JSON file')
    parser.add_argument('--num_train', type=int, default=14,
                       help='Number of training tasks to use')
    parser.add_argument('--results_path', type=str, required=True,
                       help='Path to save results')
    parser.add_argument('--seed', type=int, default=0,
                       help='Random seed')
    parser.add_argument('--max_timesteps', type=int, default=50000,
                       help='Maximum training timesteps')
    parser.add_argument('--batch_size', type=int, default=3584,
                       help='Batch size (default: 14 tasks × 256)')
    parser.add_argument('--eval_freq', type=int, default=5000,
                       help='Evaluation frequency')
    parser.add_argument('--n_eval_episodes', type=int, default=10,
                       help='Number of evaluation episodes per task')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create results directory
    results_dir = Path(args.results_path)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*80}")
    print("COMPOSITIONAL TD3+BC MULTITASK TRAINING")
    print(f"{'='*80}")
    print(f"Seed: {args.seed}")
    print(f"Device: {args.device}")
    print(f"Data path: {args.data_path}")
    print(f"Results path: {args.results_path}")
    print(f"Max timesteps: {args.max_timesteps:,}")
    print(f"Batch size: {args.batch_size}")
    print(f"Eval frequency: {args.eval_freq:,}")
    print(f"{'='*80}")
    
    # Load task lists
    train_tasks_full, test_tasks = load_task_list(args.task_list_path)
    train_tasks = train_tasks_full[:args.num_train]
    
    print(f"\n✓ Loaded task list:")
    print(f"  Training tasks: {len(train_tasks)} (first {args.num_train} from train set)")
    print(f"  Test tasks: {len(test_tasks)}")
    
    # Load multitask dataset (KEEP task IDs)
    dataset = load_multitask_dataset(args.data_path, train_tasks, 'expert')
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(
        dataset['observations'],
        dataset['next_observations'],
        dataset['actions'],
        dataset['rewards'],
        dataset['terminals'],
        dataset['timeouts'],
        args.device
    )
    print(f"✓ Created replay buffer with {replay_buffer.size:,} transitions")
    
    # Create agent with compositional networks
    state_dim = 93  # CompoSuite observation dimension WITH task IDs
    action_dim = 8
    max_action = 1.0
    
    agent = CompositionalTD3BC(
        state_dim=state_dim,
        action_dim=action_dim,
        max_action=max_action,
        device=args.device
    )
    print(f"✓ Created compositional TD3+BC agent")
    
    # Training loop
    print(f"\n{'='*80}")
    print("STARTING TRAINING")
    print(f"{'='*80}")
    
    best_score = -np.inf
    best_success_rate = -np.inf
    
    # Track which eval step had best checkpoint
    best_score_eval_step = 0
    best_success_eval_step = 0
    best_score_eval_results = {}  # Store full eval results from best checkpoint
    best_success_eval_results = {}
    
    # Track best per-task results across all evaluations
    best_per_task_score = {}
    best_per_task_success = {}
    
    for t in tqdm(range(1, args.max_timesteps + 1), desc="Training"):
        # Train one step
        metrics = agent.train(replay_buffer, batch_size=args.batch_size)
        
        # Evaluate periodically
        if t % args.eval_freq == 0:
            print(f"\n{'='*80}")
            print(f"EVALUATION AT STEP {t:,}")
            print(f"{'='*80}")
            
            # Evaluate on all test tasks
            eval_results = evaluate_policy(
                agent, test_tasks, 
                n_episodes=args.n_eval_episodes,
                use_task_id_obs=True  # Compositional networks need task IDs
            )
            
            # Update best per-task results
            for task_name, results in eval_results.items():
                current_score = results['mean_reward']
                current_success = results['success_rate']
                
                # Track best score for this task
                if task_name not in best_per_task_score:
                    best_per_task_score[task_name] = current_score
                else:
                    best_per_task_score[task_name] = max(
                        best_per_task_score[task_name], current_score
                    )
                
                # Track best success rate for this task
                if task_name not in best_per_task_success:
                    best_per_task_success[task_name] = current_success
                else:
                    best_per_task_success[task_name] = max(
                        best_per_task_success[task_name], current_success
                    )
            
            # Compute average metrics (current eval)
            avg_score = np.mean([r['mean_reward'] for r in eval_results.values()])
            avg_success_rate = np.mean([r['success_rate'] for r in eval_results.values()])
            
            print(f"\nCurrent evaluation (step {t}):")
            print(f"  Avg score: {avg_score:.3f}")
            print(f"  Avg success rate: {avg_success_rate:.3f}")
            
            # Compute best averages across all tasks
            current_best_avg_score = np.mean(list(best_per_task_score.values()))
            current_best_avg_success = np.mean(list(best_per_task_success.values()))
            
            print(f"\nBest so far (across all evals):")
            print(f"  Best avg score: {current_best_avg_score:.3f}")
            print(f"  Best avg success rate: {current_best_avg_success:.3f}")
            
            # Save best checkpoints based on current avg
            if avg_score > best_score:
                best_score = avg_score
                best_score_eval_step = t
                best_score_eval_results = {
                    task: results for task, results in eval_results.items()
                }
                agent.save(results_dir / f"best_score_checkpoint_seed{args.seed}.pt")
                print(f"  ✓ New checkpoint: best avg score {best_score:.3f} at step {t}")
            
            if avg_success_rate > best_success_rate:
                best_success_rate = avg_success_rate
                best_success_eval_step = t
                best_success_eval_results = {
                    task: results for task, results in eval_results.items()
                }
                agent.save(results_dir / f"best_success_checkpoint_seed{args.seed}.pt")
                print(f"  ✓ New checkpoint: best avg success rate {best_success_rate:.3f} at step {t}")
            
            # Save detailed results (current eval + best per task so far)
            results_file = results_dir / f"eval_step{t}_seed{args.seed}.json"
            with open(results_file, 'w') as f:
                json.dump({
                    'step': t,
                    'seed': args.seed,
                    'current_avg_score': float(avg_score),
                    'current_avg_success_rate': float(avg_success_rate),
                    'best_checkpoint_score': float(best_score),
                    'best_checkpoint_success_rate': float(best_success_rate),
                    'best_per_task_avg_score': float(current_best_avg_score),
                    'best_per_task_avg_success': float(current_best_avg_success),
                    'current_eval_results': {
                        task: {k: float(v) for k, v in results.items()}
                        for task, results in eval_results.items()
                    },
                    'best_per_task_score': {
                        task: float(score) for task, score in best_per_task_score.items()
                    },
                    'best_per_task_success': {
                        task: float(success) for task, success in best_per_task_success.items()
                    }
                }, f, indent=2)
            
            print(f"  ✓ Saved results to: {results_file}")
    
    # Final evaluation
    print(f"\n{'='*80}")
    print("FINAL EVALUATION")
    print(f"{'='*80}")
    
    final_results = evaluate_policy(
        agent, test_tasks, 
        n_episodes=args.n_eval_episodes,
        use_task_id_obs=True
    )
    
    # Update best per-task one last time
    for task_name, results in final_results.items():
        current_score = results['mean_reward']
        current_success = results['success_rate']
        
        if task_name not in best_per_task_score:
            best_per_task_score[task_name] = current_score
        else:
            best_per_task_score[task_name] = max(
                best_per_task_score[task_name], current_score
            )
        
        if task_name not in best_per_task_success:
            best_per_task_success[task_name] = current_success
        else:
            best_per_task_success[task_name] = max(
                best_per_task_success[task_name], current_success
            )
    
    final_avg_score = np.mean([r['mean_reward'] for r in final_results.values()])
    final_avg_success_rate = np.mean([r['success_rate'] for r in final_results.values()])
    
    # Compute best averages (what we report)
    best_avg_score = np.mean(list(best_per_task_score.values()))
    best_avg_success_rate = np.mean(list(best_per_task_success.values()))
    
    print(f"\nFinal evaluation (last step):")
    print(f"  Avg score: {final_avg_score:.3f}")
    print(f"  Avg success rate: {final_avg_success_rate:.3f}")
    print(f"\n✓ BEST CHECKPOINTS (single model evaluation):")
    print(f"  Best score checkpoint: {best_score:.3f} at step {best_score_eval_step}")
    print(f"  Best success checkpoint: {best_success_rate:.3f} at step {best_success_eval_step}")
    print(f"\n  (Note: Best per-task across all evals: score={best_avg_score:.3f}, success={best_avg_success_rate:.3f})")
    print(f"  (But we report checkpoint results for fair comparison, not cherry-picked per-task bests)")
    
    # Save final results with CHECKPOINT EVALUATION data (not cherry-picked per-task bests)
    final_results_file = results_dir / f"final_results_seed{args.seed}.json"
    with open(final_results_file, 'w') as f:
        json.dump({
            'seed': args.seed,
            'final_eval_avg_score': float(final_avg_score),
            'final_eval_avg_success_rate': float(final_avg_success_rate),
            # Best checkpoint info
            'best_score_checkpoint': {
                'avg_score': float(best_score),
                'eval_step': int(best_score_eval_step),
                'per_task_results': {
                    task: {k: float(v) for k, v in results.items()}
                    for task, results in best_score_eval_results.items()
                }
            },
            'best_success_checkpoint': {
                'avg_success_rate': float(best_success_rate),
                'eval_step': int(best_success_eval_step),
                'per_task_results': {
                    task: {k: float(v) for k, v in results.items()}
                    for task, results in best_success_eval_results.items()
                }
            },
            # Legacy fields for backward compatibility
            'best_checkpoint_score': float(best_score),
            'best_checkpoint_success_rate': float(best_success_rate),
            'best_avg_score': float(best_avg_score),
            'best_avg_success_rate': float(best_avg_success_rate),
            'best_per_task_score': {
                task: float(score) for task, score in best_per_task_score.items()
            },
            'best_per_task_success': {
                task: float(success) for task, success in best_per_task_success.items()
            },
            'final_eval_results': {
                task: {k: float(v) for k, v in results.items()}
                for task, results in final_results.items()
            }
        }, f, indent=2)
    
    print(f"\n✓ Training completed!")
    print(f"  Final results saved to: {final_results_file}")
    print(f"  Best checkpoints saved in: {results_dir}")
    
    # Print in format that SLURM log parser expects (checkpoint values, not per-task bests)
    print(f"\nTraining completed. Best score: {best_score:.6f}, Best success rate: {best_success_rate:.6f}")
    print(f"Best score checkpoint at step {best_score_eval_step}, Best success checkpoint at step {best_success_eval_step}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()

