import argparse
import pathlib
import torch
import wandb
import numpy as np
import gin
from diffusion.utils import *
from diffusion.elucidated_diffusion import Trainer
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
    parser.add_argument('--wandb_project', type=str, default="diffusion_training")
    args = parser.parse_args()
    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    base_results_path = pathlib.Path(args.base_results_folder)
    idx = 1
    folder_prefix = f"{args.denoiser}_seed{args.seed}_train{args.num_train}"
    while (base_results_path / f"{folder_prefix}_{idx}").exists():
        idx += 1
    results_folder = base_results_path / f"{folder_prefix}_{idx}"
    results_folder.mkdir(parents=True, exist_ok=True)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)
    
    exp_name, train_task_list, _, _ = get_task_list(
        args.task_list_path,
        args.dataset_type,
        args.experiment_type,
        None,  # holdout element
        args.seed,
    )
    train_task_list = [tuple(task) for task in train_task_list]
    train_task_list = train_task_list[:args.num_train]
    representative_task = train_task_list[0]
    robot, obj, obst, subtask = representative_task
    representative_env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=False, ignore_done=False)
    representative_indicators_env = composuite.make(robot, obj, obst, subtask, use_task_id_obs=True, ignore_done=False)
    modality_dims = representative_indicators_env.modality_dims

    task_indicators_dict = {}
    num_samples = 0
    all_inputs = []
    all_indicators = []
    for task in tqdm(train_task_list, desc="Loading data"):
        robot, obj, obst, subtask = task
        dataset = load_single_composuite_dataset(args.base_data_path, args.dataset_type, robot, obj, obst, subtask)
        dataset = transitions_dataset(dataset)
        dataset, indicators = remove_indicator_vectors(modality_dims, dataset)
        task_indicators_dict[task] = indicators[0, :]
        num_samples += dataset['observations'].shape[0]
        inputs = make_inputs(dataset)
        all_inputs.append(inputs)
        all_indicators.append(indicators)
    num_features = all_inputs[0].shape[1]
    num_indicators = all_indicators[0].shape[1]
    all_inputs_matrix = np.empty((num_samples, num_features), dtype=all_inputs[0].dtype)
    all_indicators_matrix = np.empty((num_samples, num_indicators), dtype=all_indicators[0].dtype)

    current_index = 0
    for inputs, indicators in tqdm(zip(all_inputs, all_indicators), desc="Filling processed inputs and indicators matrices"):
        all_inputs_matrix[current_index:current_index + inputs.shape[0]] = inputs
        all_indicators_matrix[current_index:current_index + indicators.shape[0]] = indicators
        current_index += inputs.shape[0]
    inputs = torch.from_numpy(all_inputs_matrix).float()
    indicators = torch.from_numpy(all_indicators_matrix).float()
    dataset = torch.utils.data.TensorDataset(inputs, indicators)

    wandb.init(
        project=args.wandb_project,
        group=f"{args.denoiser}_seed{args.seed}",  # group by denoiser type and seed
        name=results_folder.name,
        tags=[args.denoiser, f"seed_{args.seed}", f"train_{args.num_train}", args.dataset_type],
        config={
            "denoiser": args.denoiser,
            "seed": args.seed,
            "num_train": args.num_train,
            "dataset_type": args.dataset_type,
            "experiment_type": args.experiment_type,
        }
    )

    diffusion = construct_diffusion_model(inputs=inputs, denoiser=args.denoiser, cond_dim=indicators.shape[1])
    trainer = Trainer(diffusion, dataset, results_folder=str(results_folder))
    trainer.train()

