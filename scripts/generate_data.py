import argparse
import pathlib
import torch
import time
import numpy as np
import gin
from diffusion.utils import *
from accelerate import Accelerator
import composuite
from offline_compositional_rl_datasets.utils.data_utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--base_data_path', type=str, required=True, help='Base path to datasets.')
    parser.add_argument('--base_results_folder', type=str, required=True, help='Base path to results.')
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/diffusion.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[], help='Additional gin parameters.')
    parser.add_argument('--denoiser', type=str, default='monolithic', help='Type of denoiser network.')
    parser.add_argument('--task_list_path', type=str, required=True, help='Path to task splits.')
    parser.add_argument('--num_train', type=int, required=True, help='Number of training tasks.')
    parser.add_argument('--dataset_type', type=str, required=True, help='Dataset type (e.g., expert data).')
    parser.add_argument('--experiment_type', type=str, required=True, help='CompoSuite experiment type.', default='default')
    parser.add_argument('--use_gpu', action='store_true', default=True, help='Use GPU if available.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed.')
    parser.add_argument('--run', type=int, default=0)
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    base_results_path = pathlib.Path(args.base_results_folder)
    results_folder = base_results_path / f"{args.denoiser}_seed{args.seed}_train{args.num_train}_{args.run}"
    results_folder.mkdir(parents=True, exist_ok=True)
    assert results_folder.exists(), f"The results folder associated with run {args.run} does not exist."

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)
    accelerator = Accelerator()
    device = accelerator.device

    exp_name, train_task_list, _, test_task_list = get_task_list(
        args.task_list_path,
        args.dataset_type,
        args.experiment_type,
        None,  # holdout element
        args.seed,
    )
    train_task_list = [tuple(task) for task in train_task_list]
    # test_task_list = [tuple(task) for task in test_task_list]

    # Create a representative dataset to get input and indicator dimensions.
    representative_task = train_task_list[0]
    robot, obj, obst, subtask = representative_task
    representative_env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=False, ignore_done=False)
    representative_indicators_env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=True, ignore_done=False)
    modality_dims = representative_indicators_env.modality_dims
    dataset = load_single_composuite_dataset(args.base_data_path, args.dataset_type, robot, obj, obst, subtask)
    dataset = transitions_dataset(dataset)
    dataset, indicators = remove_indicator_vectors(modality_dims, dataset)
    inputs = make_inputs(dataset)
    inputs = torch.from_numpy(inputs).float()
    indicators = torch.from_numpy(indicators).float()

    # Initialize denoiser network.
    if args.denoiser == 'compositional':
        model = CompositionalResidualMLPDenoiser(d_in=inputs.shape[1], cond_dim=indicators.shape[1])
    else:
        model = ResidualMLPDenoiser(d_in=inputs.shape[1], cond_dim=indicators.shape[1])
    model = accelerator.prepare(model)

    # Load checkpoint.
    checkpoint_path = os.path.join(results_folder, 'model-100000.pt')
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    ema_dict = checkpoint['ema']
    ema_dict = {k: v for k, v in ema_dict.items() if k.startswith('ema_model')}
    ema_dict = {k.replace('ema_model.', ''): v for k, v in ema_dict.items()}

    # Create normalizer.
    terminal_dim = inputs.shape[1] - 1
    skip_dims = []
    if terminal_dim not in skip_dims:
        skip_dims.append(terminal_dim)
    print(f"Skipping normalization for dimensions {skip_dims}.")        
    dummy_tensor = torch.zeros((1, inputs.shape[1]))
    normalizer = normalizer_factory('standard', dummy_tensor, skip_dims=skip_dims)
    normalizer.mean = checkpoint['model']['normalizer.mean']  # overwrite initialized mean
    normalizer.std = checkpoint['model']['normalizer.std']  # overwrite initialized std
    print('Means:', normalizer.mean)
    print('Stds:', normalizer.std)

    # Create diffusion model.
    diffusion = ElucidatedDiffusion(net=model, normalizer=normalizer, event_shape=[inputs.shape[1]])
    diffusion.load_state_dict(ema_dict)
    diffusion = accelerator.prepare(diffusion)
    diffusion.eval()

    # Generate synthetic data for train-validation tasks.
    for robot, obj, obst, subtask in train_task_list:
        print('Generating synthetic data for train task:', robot, obj, obst, subtask)
        subtask_folder = results_folder / f"{robot}_{obj}_{obst}_{subtask}"
        retry_count = 0
        while not subtask_folder.exists():
            try:
                subtask_folder.mkdir(parents=True, exist_ok=True)
            except Exception as exception:
                retry_count += 1
                if retry_count >= 5:
                    raise RuntimeError(f"Failed to create directory {subtask_folder}.") from exception
                time.sleep(1)  # wait before retrying

        subtask_indicator = get_task_indicator(robot, obj, obst, subtask)
        generator = SimpleDiffusionGenerator(env=representative_env, ema_model=diffusion)
        obs, actions, rewards, next_obs, terminals = generator.sample(num_samples=1000000, cond=subtask_indicator)

        if not subtask_folder.exists():
            print(f'Folder missing unexpectedly: {subtask_folder}')
            subtask_folder.mkdir(parents=True, exist_ok=True)
        
        idx = 0
        while (subtask_folder / f'samples_{idx}.npz').exists():
            idx += 1

        np.savez_compressed(
            subtask_folder / f'samples_{idx}.npz',
            observations=obs,
            actions=actions,
            rewards=rewards,
            next_observations=next_obs,
            terminals=terminals
        )

    # # Generate synthetic data for test tasks.
    # for robot, obj, obst, subtask in test_task_list:
    #     print('Generating synthetic data for test task:', robot, obj, obst, subtask)
    #     subtask_folder = results_folder / f"{robot}_{obj}_{obst}_{subtask}"
    #     retry_count = 0
    #     while not subtask_folder.exists():
    #         try:
    #             subtask_folder.mkdir(parents=True, exist_ok=True)
    #         except Exception as exception:
    #             retry_count += 1
    #             if retry_count >= 5:
    #                 raise RuntimeError(f"Failed to create directory {subtask_folder}.") from exception
    #             time.sleep(1)  # wait before retrying

    #     subtask_indicator = get_task_indicator(robot, obj, obst, subtask)
    #     generator = SimpleDiffusionGenerator(env=representative_env, ema_model=diffusion)
    #     obs, actions, rewards, next_obs, terminals = generator.sample(num_samples=1000000, cond=subtask_indicator)

    #     if not subtask_folder.exists():
    #         print(f'Folder missing unexpectedly: {subtask_folder}')
    #         subtask_folder.mkdir(parents=True, exist_ok=True)
        
    #     idx = 0
    #     while (subtask_folder / f'samples_{idx}.npz').exists():
    #         idx += 1

    #     np.savez_compressed(
    #         subtask_folder / f'samples_{idx}.npz',
    #         observations=obs,
    #         actions=actions,
    #         rewards=rewards,
    #         next_observations=next_obs,
    #         terminals=terminals
    #     )

