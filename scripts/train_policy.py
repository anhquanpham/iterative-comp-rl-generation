import os
import argparse
import pathlib
import torch
import wandb
import numpy as np
from diffusion.utils import *
from accelerate import Accelerator
import composuite
from corl.algorithms import td3_bc, iql
from corl.shared.buffer import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # Specify transitions data base paths and results folder base path
    parser.add_argument('--base_agent_data_path', type=str, required=True, help='Base path to agent datasets.')
    parser.add_argument('--base_synthetic_data_path', type=str, required=True, help='Base path to synthetic datasets.')
    parser.add_argument('--base_results_folder', type=str, required=True, help='Base path to results.')
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (expert agent, synthetic, etc.')
    # Task parameters
    parser.add_argument('--robot', type=str, required=True, help='Robot type.')
    parser.add_argument('--obj', type=str, required=True, help='Object type.')
    parser.add_argument('--obst', type=str, required=True, help='Obstacle type.')
    parser.add_argument('--subtask', type=str, required=True, help='Subtask type.')
    # Offline RL training hyperparameters
    parser.add_argument('--algorithm', type=str, required=True, help='Offline learning algorithm.')
    parser.add_argument('--max_timesteps', type=int, default=50000, help='Number of training steps.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--n_episodes', type=int, default=10, help='Number of evaluation episodes.')
    parser.add_argument('--seed', type=int, default=0, help='Seed for policy training.')
    # Underlying diffusion model type and its training data parameters; needed for building write paths and logging
    parser.add_argument('--denoiser', type=str, default='monolithic', help='Type of denoiser network.')
    parser.add_argument('--task_list_seed', type=int, default=0, help='Seed for selecting task lists.')
    parser.add_argument('--num_train', type=int, required=True, help='Number of training tasks.')
    parser.add_argument('--experiment_type', type=str, help='CompoSuite experiment type.', default='default')
    parser.add_argument('--diffusion_training_run', type=int, default=0, help='Diffusion training run.')
    # Training device
    parser.add_argument('--use_gpu', default=True, action='store_true', help='Use GPU if available.')
    # W&B project
    parser.add_argument('--wandb_project', type=str, default="policy_training")

    args = parser.parse_args()

    base_results_path = pathlib.Path(args.base_results_folder)

    task_suffix = f"{args.robot}_{args.obj}_{args.obst}_{args.subtask}_{args.seed}"
    if args.dataset_type == 'synthetic':
        synthetic_data_prefix = f"{args.denoiser}_seed{args.task_list_seed}_train{args.num_train}_{args.diffusion_training_run}"
        task_results_folder = base_results_path / f"{synthetic_data_prefix}" / f"{task_suffix}"
        task_results_folder.mkdir(parents=True, exist_ok=True)
    else:
        task_results_folder = base_results_path / args.dataset_type / f"seed_{args.task_list_seed}" / task_suffix
        task_results_folder.mkdir(parents=True, exist_ok=True)
        
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    env = composuite.make(args.robot, args.obj, args.obst, args.subtask, 
                          use_task_id_obs=False, ignore_done=False)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    representative_task_env = composuite.make(args.robot, args.obj, args.obst, args.subtask, 
                          use_task_id_obs=True, ignore_done=False)
    agent_dataset = load_single_composuite_dataset(base_path=args.base_agent_data_path, 
                                                   dataset_type='expert', 
                                                   robot=args.robot, obj=args.obj, 
                                                   obst=args.obst, task=args.subtask)
    agent_dataset, _ = remove_indicator_vectors(representative_task_env.modality_dims, transitions_dataset(agent_dataset))

    if args.dataset_type == 'synthetic':
        synthetic_data_path = os.path.join(args.base_synthetic_data_path, synthetic_data_prefix)
        synthetic_dataset = load_single_synthetic_dataset(base_path=synthetic_data_path, 
                                                          robot=args.robot, obj=args.obj, 
                                                          obst=args.obst, task=args.subtask)
        integer_dims, constant_dims = identify_special_dimensions(agent_dataset['observations'])
        synthetic_dataset = process_special_dimensions(synthetic_dataset, integer_dims, constant_dims)
        dataset = synthetic_dataset
    else:
        dataset = agent_dataset

    if args.dataset_type == 'synthetic':
        wandb.init(
            mode="online",
            project=args.wandb_project,
            group=f"{args.denoiser}_seed{args.seed}",  # group by denoiser type and seed
            name=f"{synthetic_data_prefix}_{args.algorithm}_seed{args.seed}_{task_results_folder.name}",
            tags=[
                args.dataset_type, 
                f"robot_{args.robot}",
                f"obj_{args.obj}",
                f"obst_{args.obst}",
                f"subtask_{args.subtask}",
                f"algorithm_{args.algorithm}",
                f"seed_{args.seed}", 
                args.denoiser, 
                f"task_list_seed_{args.task_list_seed}",
                f"train_{args.num_train}", 
                f"experiment_type_{args.experiment_type}",
            ],
            config={
                "dataset_type": args.dataset_type,
                "robot": args.robot,
                "obj": args.obj,
                "obst": args.obst,
                "subtask": args.subtask,
                "algorithm": args.algorithm,
                "seed": args.seed,
                "max_timesteps": args.max_timesteps,
                "batch_size": args.batch_size,
                "denoiser": args.denoiser,
                "task_list_seed": args.task_list_seed,
                "num_train": args.num_train,
            }
        )
    else:
        wandb.init(
            mode="online",
            project=args.wandb_project,
            group=f"seed{args.seed}",  # group by seed
            name=f"{args.algorithm}_seed{args.seed}_{task_results_folder.name}",
            tags=[f"seed_{args.seed}", args.dataset_type],
            config={
                "dataset_type": args.dataset_type,
                "robot": args.robot,
                "obj": args.obj,
                "obst": args.obst,
                "subtask": args.subtask,
                "algorithm": args.algorithm,
                "seed": args.seed,
                "max_timesteps": args.max_timesteps,
                "batch_size": args.batch_size,
                "task_list_seed": args.task_list_seed,
            }
        )

    if args.algorithm == 'td3_bc':
        
        config = td3_bc.TrainConfig()
        config.max_timesteps = args.max_timesteps
        config.batch_size = args.batch_size
        config.n_episodes = args.n_episodes
        config.checkpoints_path = task_results_folder
        config.device = device

        state_mean, state_std = td3_bc.compute_mean_std(dataset["observations"], eps=1e-3)
        env = td3_bc.wrap_env(env, state_mean=state_mean, state_std=state_std)
        td3_bc.set_seed(args.seed, env)
        replay_buffer = prepare_replay_buffer(
            state_dim=state_dim,
            action_dim=action_dim,
            dataset=dataset,
            num_samples=int(dataset['observations'].shape[0]),
            device=device,
            reward_normalizer=None,
            state_normalizer=StateNormalizer(state_mean, state_std),
            )
    
        actor = td3_bc.Actor(state_dim, action_dim, max_action, hidden_dim=config.network_width, n_hidden=config.network_depth)
        critic_1 = td3_bc.Critic(state_dim, action_dim, hidden_dim=config.network_width, n_hidden=config.network_depth)
        critic_2 = td3_bc.Critic(state_dim, action_dim, hidden_dim=config.network_width, n_hidden=config.network_depth)
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        critic_1_optimizer = torch.optim.Adam(critic_1.parameters(), lr=3e-4)
        critic_2_optimizer = torch.optim.Adam(critic_2.parameters(), lr=3e-4)
        actor = accelerator.prepare(actor)
        actor_optimizer = accelerator.prepare(actor_optimizer)
        critic_1 = accelerator.prepare(critic_1)
        critic_1_optimizer = accelerator.prepare(critic_1_optimizer)
        critic_2 = accelerator.prepare(critic_2)
        critic_2_optimizer = accelerator.prepare(critic_2_optimizer)

        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "critic_1": critic_1,
            "critic_1_optimizer": critic_1_optimizer,
            "critic_2": critic_2,
            "critic_2_optimizer": critic_2_optimizer,
            "discount": config.discount,
            "tau": config.tau,
            "policy_noise": config.policy_noise * max_action,
            "noise_clip": config.noise_clip * max_action,
            "policy_freq": config.policy_freq,
            "alpha": config.alpha,
        }

        trainer = td3_bc.TD3_BC(**kwargs)
        evaluations = []
        best_eval_score = float('-inf')
        best_checkpoint_path = None

        for t in range(int(config.max_timesteps)):
            batch = replay_buffer.sample(config.batch_size)
            log_dict = trainer.train(batch)
            if t % config.log_every == 0:
                wandb.log(log_dict, step=trainer.total_it)
            # Evaluate actor
            if t % config.eval_freq == 0 or t == config.max_timesteps - 1:
                print(f"Time steps: {t + 1}")
                eval_scores = td3_bc.eval_actor(
                    env,
                    actor,
                    device=config.device,
                    n_episodes=config.n_episodes,
                    seed=config.seed,
                )
                eval_score = eval_scores.mean()
                evaluations.append(eval_score)
                print(
                    f"Evaluation over {config.n_episodes} episodes: "
                    f"{eval_score:.3f}"
                )
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_checkpoint_path = config.checkpoints_path
                    # Save best model separately
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(best_checkpoint_path, "td3_bc_policy_model.pt"),
                    )
                    print(f"Current best score: {best_eval_score:.3f}.")
                log_dict = {
                    "Score": eval_score,
                    "Best Score": best_eval_score
                }
                wandb.log(log_dict, step=trainer.total_it)

        print(f"Training completed. Best score: {best_eval_score:.3f}.")
        wandb.run.summary["best_score"] = best_eval_score

    elif args.algorithm == 'iql':
        config = iql.TrainConfig()
        config.max_timesteps = args.max_timesteps
        config.batch_size = args.batch_size
        config.n_episodes = args.n_episodes
        config.checkpoints_path = task_results_folder
        config.device = device

        state_mean, state_std = iql.compute_mean_std(dataset["observations"], eps=1e-3)
        env = iql.wrap_env(env, state_mean=state_mean, state_std=state_std)
        iql.set_seed(args.seed, env)
        replay_buffer = prepare_replay_buffer(
            state_dim=state_dim,
            action_dim=action_dim,
            dataset=dataset,
            num_samples=int(dataset['observations'].shape[0]),
            device=device,
            reward_normalizer=None,
            state_normalizer=StateNormalizer(state_mean, state_std),
            )
        
        actor = (
            iql.DeterministicPolicy(state_dim, action_dim, max_action, hidden_dim=config.network_width, n_hidden=config.network_depth)
            if config.iql_deterministic else
            iql.GaussianPolicy(state_dim, action_dim, max_action, hidden_dim=config.network_width, n_hidden=config.network_depth)
        )
        actor_optimizer = torch.optim.Adam(actor.parameters(), lr=3e-4)
        q_network = iql.TwinQ(state_dim, action_dim, hidden_dim=config.network_width, n_hidden=config.network_depth)
        q_optimizer = torch.optim.Adam(q_network.parameters(), lr=3e-4)
        v_network = iql.ValueFunction(state_dim, hidden_dim=config.network_width, n_hidden=config.network_depth)
        v_optimizer = torch.optim.Adam(v_network.parameters(), lr=3e-4)
        actor = accelerator.prepare(actor)
        actor_optimizer = accelerator.prepare(actor_optimizer)
        q_network = accelerator.prepare(q_network)
        q_optimizer = accelerator.prepare(q_optimizer)
        v_network = accelerator.prepare(v_network)
        v_optimizer = accelerator.prepare(v_optimizer)

        kwargs = {
            "max_action": max_action,
            "actor": actor,
            "actor_optimizer": actor_optimizer,
            "q_network": q_network,
            "q_optimizer": q_optimizer,
            "v_network": v_network,
            "v_optimizer": v_optimizer,
            "discount": config.discount,
            "tau": config.tau,
            "device": config.device,
            "beta": config.beta,
            "iql_tau": config.iql_tau,
            "max_steps": config.max_timesteps
        }

        trainer = iql.ImplicitQLearning(**kwargs)
        evaluations = []
        best_eval_score = float('-inf')
        best_checkpoint_path = None

        for t in range(int(config.max_timesteps)):
            batch = replay_buffer.sample(config.batch_size)
            log_dict = trainer.train(batch)
            if t % config.log_every == 0:
                wandb.log(log_dict, step=trainer.total_it)
            # Evaluate actor
            if t % config.eval_freq == 0 or t == config.max_timesteps - 1:
                print(f"Time steps: {t + 1}")
                eval_scores = iql.eval_actor(
                    env,
                    actor,
                    device=config.device,
                    n_episodes=config.n_episodes,
                    seed=config.seed,
                )
                eval_score = eval_scores.mean()
                evaluations.append(eval_score)
                print(
                    f"Evaluation over {config.n_episodes} episodes: "
                    f"{eval_score:.3f}"
                )
                if eval_score > best_eval_score:
                    best_eval_score = eval_score
                    best_checkpoint_path = config.checkpoints_path
                    # Save best model separately
                    torch.save(
                        trainer.state_dict(),
                        os.path.join(best_checkpoint_path, "iql_policy_model.pt"),
                    )
                    print(f"Current best score: {best_eval_score:.3f}.")
                log_dict = {
                    "Score": eval_score,
                    "Best Score": best_eval_score
                }
                wandb.log(log_dict, step=trainer.total_it)

        print(f"Training completed. Best score: {best_eval_score:.3f}.")
        wandb.run.summary["best_score"] = best_eval_score

    else:
        raise ValueError(f"Unknown algorithm: {args.algorithm}.")

