#!/usr/bin/env python3
"""
Fully Automated Iterative Diffusion Pipeline

This script automates the entire 3-stage iterative diffusion process:
1. Train diffusion model (expert + good synthetic from previous iterations)
2. Generate synthetic data for target tasks  
3. Train policies on synthetic data and evaluate performance
4. Automatically analyze results and filter good/bad tasks for next iteration

Usage:
    python automated_iterative_diffusion.py --max_iterations 5 --base_config_file config.json

The script will run the specified number of iterations automatically, with each iteration
building on the successful tasks from previous iterations.
"""

import os
import sys
import json
import time
import argparse
import subprocess
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set
import pandas as pd
import numpy as np
from dataclasses import dataclass, asdict

@dataclass
class IterationConfig:
    """Configuration for a single iteration"""
    # Base paths
    base_path: str = '/home/quanpham/iterative-comp-rl-generation'
    data_path: str = '/home/quanpham/iterative-comp-rl-generation/data'
    results_path: str = '/home/quanpham/iterative-comp-rl-generation/results'
    tasks_path: str = '/home/quanpham/iterative-comp-rl-generation/offline_compositional_rl_datasets/_train_test_splits'
    
    # Diffusion training config
    denoiser: str = 'monolithic'
    num_train: int = 56
    task_list_seed: int = 0
    diffusion_seed: int = 0
    
    # Curriculum config
    curriculum_seed: int = 0
    
    # Job resource config
    diffusion_memory: int = 400
    diffusion_time: int = 24
    generation_memory: int = 50
    generation_time: int = 48
    policy_memory: int = 16
    policy_time: int = 24
    
    # RL training config
    rl_seeds: List[int] = None
    middle_seeds: List[int] = None
    algorithm: str = 'td3_bc'
    
    # Analysis config
    success_threshold: float = 0.8
    threshold_reduction_amount: float = 0.1  # Amount to reduce threshold by
    threshold_reduction_cycle: int = 1  # Number of consecutive iterations with no good tasks before reducing threshold
    min_threshold: float = 0.5  # Minimum threshold value
    
    # Retry configuration
    max_retries: int = 100
    retry_delay: int = 300  # 5 minutes between retries
    retry_memory_multiplier: float = 1.5  # Increase memory on retry
    retry_time_multiplier: float = 1.5    # Increase time on retry
    
    # Resource limits (to prevent requesting impossible resources)
    max_memory_gb: int = 400  # Maximum memory available on machines
    max_time_hours: int = 72  # Maximum time limit
    
    def __post_init__(self):
        if self.rl_seeds is None:
            self.rl_seeds = [0, 1, 2, 3, 4]
        if self.middle_seeds is None:
            self.middle_seeds = [0]


class IterativeDiscoveryPipeline:
    """Main pipeline orchestrator"""
    
    def __init__(self, config: IterationConfig, max_iterations: int):
        self.config = config
        self.max_iterations = max_iterations
        self.base_path = Path(config.base_path)
        self.results_path = Path(config.results_path)
        
        # Track tasks across iterations
        self.good_tasks_history = {}  # iteration -> list of good tasks
        self.bad_tasks_history = {}   # iteration -> list of bad tasks
        self.source_tasks_history = {} # iteration -> dict of source run -> tasks
        self.test_tasks_history = {}  # task -> {iteration -> success_rate}
        
        # Track threshold reduction mechanism
        self.consecutive_no_good_tasks = 0  # Count of consecutive iterations with no good tasks
        self.current_threshold = config.success_threshold  # Current threshold (may be reduced)
        self.threshold_reduction_history = []  # Track when threshold was reduced
        
        # Initialize dynamic curriculum system
        self.curriculum_schedule = self._generate_curriculum_schedule()
        self.curriculum_cycle_length = len(self.curriculum_schedule)
        
        # Print curriculum schedule for debugging
        self._print_curriculum_schedule()
        
        # Create necessary directories
        self.setup_directories()
        
    def setup_directories(self):
        """Create necessary directories for the pipeline"""
        for i in range(self.max_iterations + 1):
            dirs_to_create = [
                self.base_path / "scripts" / "slurm",
                self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_test_{i}",
                self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_middle_{i}",
                self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_test_{i}" / "logs",
                self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_middle_{i}" / "logs",
                self.results_path / f"augmented_{i}" / "diffusion",
                self.results_path / f"augmented_{i}" / "policies",
            ]
            for dir_path in dirs_to_create:
                dir_path.mkdir(parents=True, exist_ok=True)
    
    def _generate_curriculum_schedule(self) -> List[Dict[str, str]]:
        """Generate dynamic curriculum schedule with randomized component and value orders"""
        import random
        
        # Set seed for reproducible curriculum
        random.seed(self.config.curriculum_seed)
        
        # Define component structure - this can be easily modified for different tasks
        components = {
            'Robot': ['IIWA', 'Panda', 'Kinova3', 'Jaco'],
            'Objective': ['Trashcan', 'PickPlace', 'Shelf', 'Push'],
            'Object': ['Plate', 'Box', 'Hollowbox', 'Dumbbell'],
            'Obstacle': ['ObjectDoor', 'GoalWall', 'ObjectWall', 'None']
        }
        
        # Randomize component order
        component_names = list(components.keys())
        random.shuffle(component_names)
        
        # Generate curriculum schedule
        schedule = []
        
        for component_name in component_names:
            component_values = components[component_name].copy()
            # Randomize the order of values within this component
            random.shuffle(component_values)
            
            # Add each individual value
            for value in component_values:
                schedule.append({
                    'component_type': component_name,
                    'target_value': value,
                    'phase': 'individual'
                })
            
            # Add the "ALL" phase for this component
            schedule.append({
                'component_type': component_name,
                'target_value': 'ALL',
                'phase': 'cumulative'
            })
        
        return schedule
    
    def _print_curriculum_schedule(self):
        """Print the generated curriculum schedule for debugging"""
        print(f"\n{'='*80}")
        print(f"GENERATED CURRICULUM SCHEDULE (Seed: {self.config.curriculum_seed})")
        print(f"{'='*80}")
        print(f"Cycle Length: {self.curriculum_cycle_length} iterations")
        print(f"Curriculum starts at iteration 5")
        print(f"\nSchedule:")
        
        for i, entry in enumerate(self.curriculum_schedule):
            iteration = i + 5  # Curriculum starts at iteration 5
            component = entry['component_type']
            value = entry['target_value']
            phase = entry['phase']
            
            print(f"  Iteration {iteration:2d}: {component:<9} - {value:<12} ({phase})")
        
        # Show cycle repetition
        if self.max_iterations > 5 + self.curriculum_cycle_length:
            next_cycle_start = 5 + self.curriculum_cycle_length
            print(f"\nCycle repeats starting at iteration {next_cycle_start}")
        
        print(f"{'='*80}")
    
    def _get_curriculum_info(self, iteration: int) -> Dict[str, str]:
        """Get curriculum information for a given iteration using dynamic curriculum"""
        
        # Handle iterations before curriculum starts (0-4)
        if iteration < 5:
            return {
                'component_type': "PreCurriculum",
                'target_value': "ALL",
                'description': f"No curriculum filtering for iteration {iteration} (pre-curriculum)",
                'cycle_iteration': -1,
                'cycle_position': 0,
                'phase': 'pre_curriculum'
            }
        
        # Calculate position in curriculum cycle starting from iteration 5
        curriculum_iteration = iteration - 5
        cycle_position = curriculum_iteration % self.curriculum_cycle_length
        
        # Get curriculum info from the generated schedule
        if cycle_position < len(self.curriculum_schedule):
            curriculum_entry = self.curriculum_schedule[cycle_position]
            component_type = curriculum_entry['component_type']
            target_value = curriculum_entry['target_value']
            phase = curriculum_entry['phase']
            
            # Generate description based on phase
            if phase == 'individual':
                description = f"Using only {target_value} {component_type.lower()} tasks for iteration {iteration}"
            else:  # cumulative
                description = f"Using ALL {component_type.lower()} tasks for iteration {iteration} (full cumulative)"
            
            return {
                'component_type': component_type,
                'target_value': target_value,
                'description': description,
                'cycle_iteration': curriculum_iteration,
                'cycle_position': cycle_position,
                'phase': phase
            }
        else:
            # This shouldn't happen, but fallback to no filtering
            return {
                'component_type': "Unknown",
                'target_value': "ALL",
                'description': f"Fallback - no curriculum filtering for iteration {iteration}",
                'cycle_iteration': curriculum_iteration,
                'cycle_position': cycle_position,
                'phase': 'fallback'
            }
    
    def _filter_tasks_by_curriculum(self, source_tasks_dict: Dict, curriculum_info: Dict[str, str]) -> Dict:
        """Filter tasks based on curriculum information"""
        if curriculum_info['target_value'] == "ALL":
            # Return all tasks (for "ALL" phases or pre-curriculum iterations)
            return source_tasks_dict
        
        filtered_dict = {}
        component_type = curriculum_info['component_type']
        target_value = curriculum_info['target_value']
        
        for source_run, tasks in source_tasks_dict.items():
            if not tasks:
                filtered_dict[source_run] = []
                continue
                
            filtered_tasks = []
            for task in tasks:
                # Task format: Robot_Object_Obstacle_Objective
                parts = task.split('_')
                if len(parts) != 4:
                    continue
                    
                robot, obj, obstacle, objective = parts
                
                # Check if task matches the curriculum filter
                if component_type == "Robot" and robot == target_value:
                    filtered_tasks.append(task)
                elif component_type == "Objective" and objective == target_value:
                    filtered_tasks.append(task)
                elif component_type == "Object" and obj == target_value:
                    filtered_tasks.append(task)
                elif component_type == "Obstacle" and obstacle == target_value:
                    filtered_tasks.append(task)
                # Note: "PreCurriculum" component_type is handled by the "ALL" case above
            
            # Store the filtered tasks for this run
            filtered_dict[source_run] = filtered_tasks
        
        return filtered_dict
    
    def get_test_tasks_from_file(self) -> List[str]:
        """Load test tasks from task list file"""
        task_file = Path(self.config.tasks_path) / "default" / f"split_{self.config.task_list_seed}.json"
        with open(task_file, 'r') as f:
            task_data = json.load(f)
            test_tasks = task_data.get('test', [])
            # Convert to string format: Robot_Obj_Obst_Subtask
            return ['_'.join(task) for task in test_tasks]
    
    def get_train_and_middle_tasks_from_file(self) -> Tuple[List[str], List[str]]:
        """Load train and middle tasks from task list file"""
        task_file = Path(self.config.tasks_path) / "default" / f"split_{self.config.task_list_seed}.json"
        with open(task_file, 'r') as f:
            task_data = json.load(f)
            full_train_tasks = task_data.get('train', [])
            # Convert to string format: Robot_Obj_Obst_Subtask
            full_train_tasks = ['_'.join(task) for task in full_train_tasks]
            
            # Split based on num_train parameter
            actual_train_tasks = full_train_tasks[:self.config.num_train]  # First 56 tasks
            middle_tasks = full_train_tasks[self.config.num_train:]        # Tasks 57+
            
            return actual_train_tasks, middle_tasks
    
    def run_diffusion_training(self, iteration: int) -> bool:
        """Stage 1: Train diffusion model"""
        print(f"\n=== ITERATION {iteration}: STAGE 1 - DIFFUSION TRAINING ===")
        
        # Print source tasks information if available
        if iteration > 0 and iteration - 1 in self.source_tasks_history:
            # Component-specific curriculum for iterations 5 onwards 
            # 20-iteration cycle: Robot (5-9), Objective (10-14), Object (15-19), Obstacle (20-24), then repeat
            if iteration >= 5:
                curriculum_info = self._get_curriculum_info(iteration)
                filtered_tasks = self._filter_tasks_by_curriculum(self.source_tasks_history[iteration - 1], curriculum_info)
                
                print(f"🎯 {curriculum_info['component_type']}-specific curriculum: {curriculum_info['description']}")
                
                total_source_tasks = 0
                for source_run, tasks in filtered_tasks.items():
                    if tasks:
                        print(f"  Run {source_run}: {len(tasks)} {curriculum_info['component_type'].lower()} tasks")
                        for task in sorted(tasks):
                            print(f"    - {task}")
                        total_source_tasks += len(tasks)
                    else:
                        print(f"  Run {source_run}: No {curriculum_info['component_type'].lower()} tasks found")
                
                print(f"  Total {curriculum_info['component_type'].lower()} source tasks: {total_source_tasks}")
            else:
                print(f"📊 Using good tasks from previous iterations as additional training data:")
                for source_run, tasks in self.source_tasks_history[iteration - 1].items():
                    print(f"  Run {source_run}: {len(tasks)} good tasks")
                    for task in sorted(tasks):
                        print(f"    - {task}")
                total_source_tasks = sum(len(tasks) for tasks in self.source_tasks_history[iteration - 1].values())
                print(f"  Total source tasks: {total_source_tasks}")
        else:
            print(f"📊 No source tasks available (iteration {iteration} - using only expert data)")
        
        job_name = f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_{iteration}"
        
        # Create script generator function for retries
        def script_generator(retry_context):
            return self._build_diffusion_script(iteration, job_name, retry_context)
        
        # Build initial script content
        script_content = script_generator({})
        script_path = self.base_path / f"job_diffusion_{iteration}.sh"
        
        # Write and submit job
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        print(f"Submitting diffusion training job for iteration {iteration}")
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        if result.returncode != 0:
            print(f"Error submitting diffusion job: {result.stderr}")
            return False
        
        # Parse job ID and wait for completion with retry
        job_id = self._parse_job_id(result.stdout)
        print(f"Diffusion training job submitted: {job_id}")
        
        # Clean up script
        script_path.unlink()
        
        # Wait for job completion with retry support
        retry_context = {'attempt': 1}
        success = self._wait_for_job_completion_with_retry(job_id, "diffusion training", script_generator, retry_context)
        
        # Second layer: Check for trained model and resubmit if needed
        if success:
            success = self._ensure_diffusion_model_exists(iteration)
        
        return success
    
    def run_data_generation(self, iteration: int) -> bool:
        """Stage 2: Generate synthetic data"""
        print(f"\n=== ITERATION {iteration}: STAGE 2 - DATA GENERATION ===")
        
        # Determine which tasks to generate for
        if iteration == 0:
            # First iteration: generate for both test tasks AND middle tasks
            test_tasks = self.get_test_tasks_from_file()
            _, middle_tasks = self.get_train_and_middle_tasks_from_file()
            target_tasks = test_tasks + middle_tasks
            print(f"Initial iteration: generating data for {len(test_tasks)} test tasks + {len(middle_tasks)} middle tasks = {len(target_tasks)} total tasks")
        else:
            # Later iterations: only generate for bad tasks from previous iteration
            target_tasks = self.bad_tasks_history.get(iteration - 1, [])
            good_tasks = self.good_tasks_history.get(iteration - 1, [])
            print(f"Iteration {iteration}: generating data for {len(target_tasks)} bad tasks from previous iteration")
            if good_tasks:
                print(f"🔄 Skipping {len(good_tasks)} good tasks (already solved):")
                for task in sorted(good_tasks):
                    print(f"  - {task}")
        
        if not target_tasks:
            print("No target tasks for data generation. Skipping.")
            return True
        
        print(f"Generating data for {len(target_tasks)} tasks")
        
        # Submit jobs for each target task with retry context
        job_contexts = []
        for task in target_tasks:
            # Create script generator function for this task
            def script_generator(retry_context, task=task):
                return self._build_generation_script(iteration, task, retry_context)
            
            script_content = script_generator({})
            script_path = self.base_path / f"job_gen_{iteration}_{task.replace('_', '')[:20]}.sh"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
            if result.returncode == 0:
                job_id = self._parse_job_id(result.stdout)
                job_contexts.append({
                    'job_id': job_id,
                    'task_id': task,
                    'script_generator': script_generator,
                    'attempt': 1
                })
            else:
                print(f"Error submitting generation job for {task}: {result.stderr}")
            
            script_path.unlink()
        
        print(f"Submitted {len(job_contexts)} data generation jobs")
        
        # Wait for all generation jobs to complete with retry support
        success, _ = self._wait_for_jobs_completion_with_retry(job_contexts, "data generation")
        
        # Second layer: Check for missing synthetic data and resubmit if needed
        if success:
            success = self._ensure_all_synthetic_data_exists(iteration, target_tasks)
        
        return success
    
    def run_policy_training(self, iteration: int) -> Tuple[bool, Set[str]]:
        """Stage 3: Train policies on synthetic data"""
        print(f"\n=== ITERATION {iteration}: STAGE 3 - POLICY TRAINING ===")
        
        # Determine tasks to train on
        if iteration == 0:
            # First iteration: use test tasks for "test" and middle tasks for "middle"
            test_tasks = self.get_test_tasks_from_file()
            actual_train_tasks, middle_tasks = self.get_train_and_middle_tasks_from_file()
            
            print(f"Initial iteration task allocation:")
            print(f"  Test tasks: {len(test_tasks)} (from test set)")
            print(f"  Train tasks: {len(actual_train_tasks)} (first {self.config.num_train} from train set)")
            print(f"  Middle tasks: {len(middle_tasks)} (remaining train tasks: {self.config.num_train}+)")
        else:
            # Later iterations: filter bad tasks from previous iteration
            all_bad_tasks = self.bad_tasks_history.get(iteration - 1, [])
            
            # For test tasks: use bad test tasks from previous iteration
            test_tasks = [task for task in all_bad_tasks if task in self.get_test_tasks_from_file()]
            
            # For middle tasks: use bad middle tasks from previous iteration  
            _, original_middle_tasks = self.get_train_and_middle_tasks_from_file()
            middle_tasks = [task for task in all_bad_tasks if task in original_middle_tasks]
            
            print(f"Iterative task filtering:")
            print(f"  Test tasks: {len(test_tasks)} (bad test tasks from previous iteration)")
            print(f"  Middle tasks: {len(middle_tasks)} (bad middle tasks from previous iteration)")
            print(f"  Total bad tasks from previous iteration: {len(all_bad_tasks)}")
        
        # Validate that synthetic data exists for all tasks before training
        print(f"\nValidating synthetic data availability...")
        valid_test_tasks = self._validate_synthetic_data(test_tasks, iteration)
        valid_middle_tasks = self._validate_synthetic_data(middle_tasks, iteration)
        
        print(f"Data validation results:")
        print(f"  Test tasks: {len(valid_test_tasks)}/{len(test_tasks)} have synthetic data")
        print(f"  Middle tasks: {len(valid_middle_tasks)}/{len(middle_tasks)} have synthetic data")
        
        if not valid_test_tasks and not valid_middle_tasks:
            print("❌ No valid synthetic data found for any tasks. Policy training cannot proceed.")
            return False, set()
        
        job_contexts = []
        
        # Submit test tasks (5 seeds)
        for task in valid_test_tasks:
            for seed in self.config.rl_seeds:
                # Create script generator for this specific task+seed
                def script_generator(retry_context, task=task, seed=seed):
                    return self._build_policy_script(iteration, task, seed, 'test', retry_context)
                
                script_content = script_generator({})
                script_path = self.base_path / f"job_policy_test_{iteration}_{task.replace('_', '')[:15]}_{seed}.sh"
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = self._parse_job_id(result.stdout)
                    job_contexts.append({
                        'job_id': job_id,
                        'task_id': f"{task}_seed{seed}_test",
                        'script_generator': script_generator,
                        'attempt': 1
                    })
                
                script_path.unlink()
        
        # Submit middle tasks (1 seed)
        for task in valid_middle_tasks:
            for seed in self.config.middle_seeds:
                # Create script generator for this specific task+seed
                def script_generator(retry_context, task=task, seed=seed):
                    return self._build_policy_script(iteration, task, seed, 'middle', retry_context)
                
                script_content = script_generator({})
                script_path = self.base_path / f"job_policy_middle_{iteration}_{task.replace('_', '')[:15]}_{seed}.sh"
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = self._parse_job_id(result.stdout)
                    job_contexts.append({
                        'job_id': job_id,
                        'task_id': f"{task}_seed{seed}_middle",
                        'script_generator': script_generator,
                        'attempt': 1
                    })
                
                script_path.unlink()
        
        print(f"Submitted {len(job_contexts)} policy training jobs")
        
        # Wait for all policy jobs to complete with retry support
        success, completed_job_ids = self._wait_for_jobs_completion_with_retry(job_contexts, "policy training")
        
        # Second layer: Check for missing policy training results and resubmit if needed
        if success:
            success, resubmitted_job_ids = self._ensure_all_policy_training_completed(iteration, valid_test_tasks, valid_middle_tasks)
            # Combine job IDs from both original and resubmitted jobs
            all_successful_job_ids = completed_job_ids.union(resubmitted_job_ids)
        else:
            all_successful_job_ids = completed_job_ids
        
        return success, all_successful_job_ids
    
    def analyze_results(self, iteration: int, successful_job_ids: Set[str]) -> Tuple[List[str], List[str]]:
        """Analyze policy training results and determine good/bad tasks"""
        print(f"\n=== ITERATION {iteration}: ANALYZING RESULTS ===")
        
        # Parse logs from both test and middle training
        test_results = self._parse_policy_logs(iteration, 'test', successful_job_ids)
        middle_results = self._parse_policy_logs(iteration, 'middle', successful_job_ids)
        
        print(f"DEBUG: Using {len(successful_job_ids)} successful job IDs for parsing")
        print(f"DEBUG: test_results keys: {list(test_results.keys())}")
        print(f"DEBUG: middle_results keys: {list(middle_results.keys())}")
        
        # Analyze historical performance of test tasks
        current_best_rates, avg_best_success = self._analyze_test_tasks_history(iteration, test_results)
        
        print(f"\nTest Tasks Historical Analysis:")
        print(f"  Current threshold: {self.current_threshold:.3f}")
        print(f"  Average best success rate so far: {avg_best_success:.3f}")
        print(f"  Individual best success rates:")
        sorted_tasks = sorted(current_best_rates.items(), key=lambda x: x[1], reverse=True)
        for task, rate in sorted_tasks:
            print(f"    {task}: {rate:.3f}")
        
        # Print test task success rate table
        self._print_test_task_table(iteration)
        
        # Analyze test tasks (average across 5 seeds) using current threshold
        test_good_tasks = []
        test_bad_tasks = []
        for task, success_rate in test_results.items():
            if success_rate is not None and success_rate > self.current_threshold:
                test_good_tasks.append(task)
            else:
                test_bad_tasks.append(task)
        
        # Analyze middle tasks (single seed) using current threshold
        middle_good_tasks = []
        middle_bad_tasks = []
        for task, success_rate in middle_results.items():
            if success_rate is not None and success_rate > self.current_threshold:
                middle_good_tasks.append(task)
            else:
                middle_bad_tasks.append(task)
        
        # Combine results
        good_tasks = test_good_tasks + middle_good_tasks
        bad_tasks = test_bad_tasks + middle_bad_tasks
        
        # Threshold reduction mechanism
        if len(good_tasks) == 0:
            self.consecutive_no_good_tasks += 1
            print(f"\n⚠️  No good tasks found in iteration {iteration} (consecutive count: {self.consecutive_no_good_tasks})")
            
            # Check if we've reached the threshold reduction cycle (5 iterations = 1 robot masking cycle)
            if self.consecutive_no_good_tasks >= self.config.threshold_reduction_cycle:
                old_threshold = self.current_threshold
                new_threshold = max(
                    self.current_threshold - self.config.threshold_reduction_amount,
                    self.config.min_threshold
                )
                
                if new_threshold < old_threshold:
                    self.current_threshold = new_threshold
                    self.threshold_reduction_history.append({
                        'iteration': iteration,
                        'old_threshold': old_threshold,
                        'new_threshold': new_threshold,
                        'reason': f'No good tasks for {self.config.threshold_reduction_cycle} consecutive iterations'
                    })
                    
                    print(f"\n🔧 THRESHOLD REDUCED:")
                    print(f"   Old threshold: {old_threshold:.3f}")
                    print(f"   New threshold: {new_threshold:.3f}")
                    print(f"   Reason: No good tasks found for {self.config.threshold_reduction_cycle} consecutive iterations")
                    
                    # Re-analyze tasks with new threshold
                    print(f"\n🔄 Re-analyzing tasks with new threshold {new_threshold:.3f}...")
                    
                    # Re-analyze test tasks
                    test_good_tasks = []
                    test_bad_tasks = []
                    for task, success_rate in test_results.items():
                        if success_rate is not None and success_rate > self.current_threshold:
                            test_good_tasks.append(task)
                        else:
                            test_bad_tasks.append(task)
                    
                    # Re-analyze middle tasks
                    middle_good_tasks = []
                    middle_bad_tasks = []
                    for task, success_rate in middle_results.items():
                        if success_rate is not None and success_rate > self.current_threshold:
                            middle_good_tasks.append(task)
                        else:
                            middle_bad_tasks.append(task)
                    
                    # Update combined results
                    good_tasks = test_good_tasks + middle_good_tasks
                    bad_tasks = test_bad_tasks + middle_bad_tasks
                    
                    print(f"   Re-analysis results: {len(good_tasks)} good tasks, {len(bad_tasks)} bad tasks")
                    
                    # Reset counter only if re-analysis found good tasks
                    if len(good_tasks) > 0:
                        self.consecutive_no_good_tasks = 0
                        print(f"   Counter reset to 0 (found good tasks)")
                    else:
                        self.consecutive_no_good_tasks = 1
                        print(f"   Counter set to 1 (still no good tasks after threshold reduction)")
                else:
                    print(f"   Threshold already at minimum ({self.config.min_threshold:.3f}), cannot reduce further")
        else:
            # Reset counter when good tasks are found
            if self.consecutive_no_good_tasks > 0:
                print(f"\n✅ Good tasks found! Resetting consecutive counter from {self.consecutive_no_good_tasks} to 0")
                self.consecutive_no_good_tasks = 0
        
        # Store results
        self.good_tasks_history[iteration] = good_tasks
        self.bad_tasks_history[iteration] = bad_tasks
        
        # Update source tasks for next iteration - CUMULATIVE approach
        # Always carry forward prior cumulative history even if there are 0 new good tasks
        if iteration > 0:
            self.source_tasks_history[iteration] = self.source_tasks_history.get(iteration - 1, {}).copy()
        else:
            self.source_tasks_history[iteration] = {}

        # Add current iteration's good tasks (if any)
        if good_tasks:
            self.source_tasks_history[iteration][iteration] = good_tasks
        
        print(f"Analysis complete:")
        print(f"  Test tasks: {len(test_good_tasks)} good, {len(test_bad_tasks)} bad")
        print(f"  Middle tasks: {len(middle_good_tasks)} good, {len(middle_bad_tasks)} bad")
        print(f"  Total: {len(good_tasks)} good tasks, {len(bad_tasks)} bad tasks")
        
        # Print detailed good tasks information
        if test_good_tasks:
            print(f"\n✅ Good test tasks found (success rate > {self.current_threshold:.3f}):")
            for task in sorted(test_good_tasks):
                success_rate = test_results[task]
                print(f"  - {task} (success rate: {success_rate:.3f})")
        else:
            print(f"\n❌ No good test tasks found (all below {self.current_threshold:.3f} threshold)")
        
        if middle_good_tasks:
            print(f"\n✅ Good middle tasks found (success rate > {self.current_threshold:.3f}):")
            for task in sorted(middle_good_tasks):
                success_rate = middle_results[task]
                print(f"  - {task} (success rate: {success_rate:.3f})")
        else:
            print(f"\n❌ No good middle tasks found (all below {self.current_threshold:.3f} threshold)")
        
        # Show progression
        if iteration > 0:
            prev_bad = len(self.bad_tasks_history.get(iteration - 1, []))
            improvement = prev_bad - len(bad_tasks)
            print(f"  Progress: {improvement} tasks solved since iteration {iteration - 1}")
        
        # Show cumulative good tasks for next iteration  
        if iteration in self.source_tasks_history:
            total_cumulative = sum(len(tasks) for tasks in self.source_tasks_history[iteration].values())
            print(f"  Cumulative good tasks for next iteration: {total_cumulative}")
            
            # Show curriculum info for next iteration
            if iteration + 1 >= 5:
                next_curriculum_info = self._get_curriculum_info(iteration + 1)
                next_filtered_tasks = self._filter_tasks_by_curriculum(self.source_tasks_history[iteration], next_curriculum_info)
                
                if next_curriculum_info['target_value'] == "ALL":
                    total_tasks = sum(len(tasks) for tasks in next_filtered_tasks.values())
                    print(f"  🎯 Next iteration {iteration + 1} will use ALL {next_curriculum_info['component_type'].lower()} tasks: {total_tasks} available (full cumulative)")
                else:
                    component_specific_count = sum(len(tasks) for tasks in next_filtered_tasks.values())
                    print(f"  🎯 Next iteration {iteration + 1} will use only {next_curriculum_info['target_value']} {next_curriculum_info['component_type'].lower()} tasks: {component_specific_count} available")
        
        return good_tasks, bad_tasks
    
    def run_iteration(self, iteration: int) -> bool:
        """Run a complete iteration of the pipeline"""
        print(f"\n{'='*60}")
        print(f"STARTING ITERATION {iteration}")
        print(f"{'='*60}")
        
        # Stage 1: Train diffusion model
        if not self.run_diffusion_training(iteration):
            print(f"Diffusion training failed for iteration {iteration}")
            return False
        
        # Stage 2: Generate synthetic data
        if not self.run_data_generation(iteration):
            print(f"Data generation failed for iteration {iteration}")
            return False
        
        # Stage 3: Train policies
        success, successful_job_ids = self.run_policy_training(iteration)
        if not success:
            print(f"Policy training failed for iteration {iteration}")
            return False
        
        # Stage 4: Analyze results
        good_tasks, bad_tasks = self.analyze_results(iteration, successful_job_ids)
        
        # Check termination condition
        if not bad_tasks:
            print(f"\nAll tasks solved! Stopping at iteration {iteration}")
            return False  # Signal to stop iterations
        
        # Generate CSV analysis files
        self._generate_csv_analysis(iteration)
        
        # Save historical analysis
        self._save_historical_analysis(iteration)
        
        print(f"\nIteration {iteration} complete. {len(bad_tasks)} tasks remaining.")
        return True
    
    def run_pipeline(self):
        """Run the complete iterative pipeline"""
        print("Starting Automated Iterative Diffusion Pipeline")
        print(f"Configuration: {asdict(self.config)}")
        print(f"Max iterations: {self.max_iterations}")
        
        # Wandb tracking removed - using simple table printing instead
        
        for iteration in range(self.max_iterations):
            start_time = time.time()
            
            if not self.run_iteration(iteration):
                break  # Either failed or all tasks solved
            
            iteration_time = time.time() - start_time
            print(f"\nIteration {iteration} completed in {iteration_time/3600:.1f} hours")
        
        # Final summary
        self._print_final_summary()
        
        # Save final test task table as CSV
        if self.bad_tasks_history:
            final_iteration = max(self.bad_tasks_history.keys())
            self._save_final_test_task_table(final_iteration)
        
        # Create best test task dataset
        self._create_best_testtask_dataset()
        
        # Pipeline completed
    
    def _build_diffusion_script(self, iteration: int, job_name: str, retry_context: dict = None) -> str:
        """Build SLURM script for diffusion training"""
        memory = self.config.diffusion_memory
        time_hours = self.config.diffusion_time
        
        if retry_context:
            requested_memory = int(memory * retry_context.get('memory_multiplier', 1.0))
            requested_time = int(time_hours * retry_context.get('time_multiplier', 1.0))
            
            # Enforce resource limits with warnings
            memory = min(requested_memory, self.config.max_memory_gb)
            time_hours = min(requested_time, self.config.max_time_hours)
            
            if requested_memory > self.config.max_memory_gb:
                print(f"⚠️  Diffusion: Requested {requested_memory}GB capped at {self.config.max_memory_gb}GB")
            if requested_time > self.config.max_time_hours:
                print(f"⚠️  Diffusion: Requested {requested_time}h capped at {self.config.max_time_hours}h")
        
        script = f"""#!/bin/bash
#SBATCH --job-name=diffusion_training
#SBATCH --output=scripts/slurm/%j_{job_name}.out
#SBATCH --mem={memory}G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time={time_hours}:00:00
#SBATCH --exclude=al-l40s-0.grasp.maas
# SBATCH --partition=eaton-compute
# SBATCH --qos=ee-med

source /home/quanpham/first_3.9.6/bin/activate
export WANDB__SERVICE_WAIT=1000

# CUDA error prevention environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python {self.config.base_path}/scripts/train_augmented_diffusion.py \\
    --base_data_path {self.config.data_path} \\
    --base_results_folder {self.config.results_path} \\
    --gin_config_files {self.config.base_path}/config/diffusion.gin \\
    --denoiser {self.config.denoiser} \\
    --task_list_path {self.config.tasks_path} \\
    --task_list_seed {self.config.task_list_seed} \\
    --num_train {self.config.num_train} \\
    --dataset_type expert \\
    --experiment_type default \\
    --seed {self.config.diffusion_seed} \\
    --diffusion_training_run {iteration}"""
        
        # Add source tasks for iterations > 0 - use CUMULATIVE history
        if iteration > 0 and iteration - 1 in self.source_tasks_history:
            source_tasks_list = []
            
            # Component-specific curriculum for iterations 5 onwards
            if iteration >= 5:
                # First, show the FULL cumulative list available
                print(f"\n📊 FULL CUMULATIVE GOOD TASKS AVAILABLE (before filtering):")
                total_available = 0
                all_available_tasks = []
                for source_run, tasks in self.source_tasks_history[iteration - 1].items():
                    if tasks:
                        print(f"  Run {source_run}: {len(tasks)} tasks")
                        for task in sorted(tasks):
                            print(f"    - {task}")
                            all_available_tasks.append(task)
                        total_available += len(tasks)
                print(f"  Total available tasks: {total_available}")
                
                # Get curriculum info and filter tasks
                curriculum_info = self._get_curriculum_info(iteration)
                filtered_tasks = self._filter_tasks_by_curriculum(self.source_tasks_history[iteration - 1], curriculum_info)
                
                if curriculum_info['target_value'] == "ALL":
                    print(f"\n✅ NO FILTERING - USING ALL AVAILABLE TASKS:")
                    # Use all tasks from all runs
                    for source_run, tasks in filtered_tasks.items():
                        if tasks:
                            source_tasks_list.append(f'{source_run}:{",".join(tasks)}')
                            print(f"  Run {source_run}: {len(tasks)} tasks (all {curriculum_info['component_type'].lower()})")
                    print(f"  Total tasks used: {total_available}/{total_available}")
                else:
                    print(f"\n🔍 FILTERING FOR {curriculum_info['target_value']} {curriculum_info['component_type'].upper()} TASKS:")
                    total_filtered = 0
                    for source_run, tasks in filtered_tasks.items():
                        if tasks:
                            source_tasks_list.append(f'{source_run}:{",".join(tasks)}')
                            original_count = len(self.source_tasks_history[iteration - 1].get(source_run, []))
                            print(f"  Run {source_run}: {len(tasks)}/{original_count} {curriculum_info['target_value']} tasks selected")
                            for task in sorted(tasks):
                                print(f"    ✓ {task}")
                            total_filtered += len(tasks)
                        else:
                            original_count = len(self.source_tasks_history[iteration - 1].get(source_run, []))
                            print(f"  Run {source_run}: 0/{original_count} {curriculum_info['target_value']} tasks found")
                    print(f"  Total {curriculum_info['target_value']} tasks selected: {total_filtered}/{total_available}")
            else:
                # Normal cumulative approach for iterations 0-4
                print(f"📊 Using cumulative good tasks from all components for iteration {iteration}")
                for source_run, tasks in self.source_tasks_history[iteration - 1].items():
                    if tasks:
                        source_tasks_list.append(f'{source_run}:{",".join(tasks)}')
                        print(f"  Run {source_run}: {len(tasks)} tasks")
            
            if source_tasks_list:
                script += f' \\\n    --source_tasks {" ".join(source_tasks_list)}'
                print(f"\n📋 SOURCE TASKS LIST PASSED TO DIFFUSION MODEL:")
                print(f"   Command line argument: --source_tasks {' '.join(source_tasks_list)}")
                print(f"   Number of source runs: {len(source_tasks_list)}")
                for i, source_entry in enumerate(source_tasks_list, 1):
                    run_id, tasks_str = source_entry.split(':', 1)
                    task_list = tasks_str.split(',')
                    print(f"   {i}. Run {run_id}: {len(task_list)} tasks")
                    for task in task_list:
                        print(f"      - {task}")
            else:
                print(f"⚠️  No source tasks found for iteration {iteration}")
        else:
            print(f"📊 No source tasks available (iteration {iteration} - using only expert data)")
        
        script += '\n'
        return script
    
    def _build_generation_script(self, iteration: int, target_task: str, retry_context: dict = None) -> str:
        """Build SLURM script for data generation"""
        memory = self.config.generation_memory
        time_hours = self.config.generation_time
        
        if retry_context:
            requested_memory = int(memory * retry_context.get('memory_multiplier', 1.0))
            requested_time = int(time_hours * retry_context.get('time_multiplier', 1.0))
            
            # Enforce resource limits with warnings
            memory = min(requested_memory, self.config.max_memory_gb)
            time_hours = min(requested_time, self.config.max_time_hours)
            
            if requested_memory > self.config.max_memory_gb:
                print(f"⚠️  Generation: Requested {requested_memory}GB capped at {self.config.max_memory_gb}GB")
            if requested_time > self.config.max_time_hours:
                print(f"⚠️  Generation: Requested {requested_time}h capped at {self.config.max_time_hours}h")
        
        return f"""#!/bin/bash
#SBATCH --job-name=generate_data
#SBATCH --output=scripts/slurm/%j_generate_{self.config.denoiser}_train{self.config.num_train}_seed{self.config.diffusion_seed}_run{iteration}_{target_task}.out
#SBATCH --mem={memory}G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8
#SBATCH --time={time_hours}:00:00
#SBATCH --exclude=al-l40s-0.grasp.maas
# SBATCH --partition=eaton-compute
# SBATCH --qos=ee-med

source /home/quanpham/first_3.9.6/bin/activate

# CUDA error prevention environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python {self.config.base_path}/scripts/generate_augmented_data.py \\
    --base_data_path {self.config.data_path} \\
    --base_results_folder {self.config.results_path} \\
    --gin_config_files {self.config.base_path}/config/diffusion.gin \\
    --denoiser {self.config.denoiser} \\
    --task_list_path {self.config.tasks_path} \\
    --task_list_seed {self.config.task_list_seed} \\
    --num_train {self.config.num_train} \\
    --dataset_type expert \\
    --experiment_type default \\
    --seed {self.config.diffusion_seed} \\
    --diffusion_training_run {iteration} \\
    --target_task {target_task}
"""
    
    def _build_policy_script(self, iteration: int, task: str, seed: int, task_type: str, retry_context: dict = None) -> str:
        """Build SLURM script for policy training"""
        robot, obj, obst, subtask = task.split('_')
        job_name = f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_{self.config.algorithm}_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}"
        
        memory = self.config.policy_memory
        time_hours = self.config.policy_time
        
        if retry_context:
            requested_memory = int(memory * retry_context.get('memory_multiplier', 1.0))
            requested_time = int(time_hours * retry_context.get('time_multiplier', 1.0))
            
            # Enforce resource limits with warnings
            memory = min(requested_memory, self.config.max_memory_gb)
            time_hours = min(requested_time, self.config.max_time_hours)
            
            if requested_memory > self.config.max_memory_gb:
                print(f"⚠️  Policy: Requested {requested_memory}GB capped at {self.config.max_memory_gb}GB")
            if requested_time > self.config.max_time_hours:
                print(f"⚠️  Policy: Requested {requested_time}h capped at {self.config.max_time_hours}h")
        
        output_dir = f"policies_slurm_logs/policies_slurm_diffusionseed{self.config.diffusion_seed}/policies_slurm_{task_type}_{iteration}"
        
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=scripts/{output_dir}/%j_{job_name}.out
#SBATCH --mem={memory}G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --time={time_hours}:00:00
#SBATCH --exclude=al-l40s-0.grasp.maas
# SBATCH --partition=eaton-compute
# SBATCH --qos=ee-med

source /home/quanpham/first_3.9.6/bin/activate
export WANDB__SERVICE_WAIT=1000

# CUDA error prevention environment variables
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python {self.config.base_path}/scripts/train_augmented_policy.py \\
    --base_agent_data_path {self.config.data_path} \\
    --base_results_folder {self.config.results_path} \\
    --dataset_type synthetic \\
    --robot {robot} \\
    --obj {obj} \\
    --obst {obst} \\
    --subtask {subtask} \\
    --algorithm {self.config.algorithm} \\
    --seed {seed} \\
    --denoiser {self.config.denoiser} \\
    --task_list_seed {self.config.task_list_seed} \\
    --num_train {self.config.num_train} \\
    --diffusion_seed {self.config.diffusion_seed} \\
    --diffusion_training_run {iteration}
"""
    
    def _parse_job_id(self, sbatch_output: str) -> str:
        """Parse job ID from sbatch output"""
        match = re.search(r'Submitted batch job (\d+)', sbatch_output)
        return match.group(1) if match else None
    
    def _get_job_failure_reason(self, job_id: str) -> str:
        """Get detailed failure reason for a job"""
        try:
            result = subprocess.run(['sacct', '-j', job_id, '--format=State,ExitCode,Reason', '--noheader'], 
                                  capture_output=True, text=True)
            slurm_reason = result.stdout.strip()
            
            # Also check the actual log file for CUDA errors
            # Check both policy logs and slurm logs (for data generation)
            log_patterns = [
                f"scripts/policies_slurm_logs/policies_slurm_diffusionseed{self.config.diffusion_seed}/policies_slurm_*/{job_id}_*.out",
                f"scripts/slurm/{job_id}_*.out"
            ]
            
            log_files = []
            for pattern in log_patterns:
                log_files.extend(list(self.base_path.glob(pattern)))
            
            if log_files:
                try:
                    with open(log_files[0], 'r') as f:
                        log_content = f.read()
                    
                    # Check for CUDA errors in log content
                    cuda_error_patterns = [
                        'CUDA error: uncorrectable ECC error',
                        'RuntimeError: CUDA',
                        'CUDA kernel errors',
                        'uncorrectable ECC error encountered'
                    ]
                    
                    # Check for file system errors in log content
                    file_error_patterns = [
                        'FileNotFoundError',
                        'No such file or directory',
                        'Permission denied',
                        'ENOENT',
                        'EACCES'
                    ]
                    
                    # Check for all error types
                    all_error_patterns = cuda_error_patterns + file_error_patterns
                    
                    for pattern in all_error_patterns:
                        if pattern.lower() in log_content.lower():
                            if any(cuda in pattern.lower() for cuda in ['cuda', 'ecc']):
                                return f"CUDA ERROR: {pattern}"
                            elif any(file in pattern.lower() for file in ['file', 'directory', 'permission']):
                                return f"FILE ERROR: {pattern}"
                            else:
                                return f"RUNTIME ERROR: {pattern}"
                    
                except Exception as e:
                    print(f"Warning: Could not read log file for job {job_id}: {e}")
            
            return slurm_reason
        except:
            return "Unknown failure"
    
    def _should_retry_job(self, job_id: str, attempt: int) -> Tuple[bool, str, int, int]:
        """Determine if job should be retried and with what resources"""
        if attempt >= self.config.max_retries:
            return False, "Max retries exceeded", 0, 0
        
        failure_reason = self._get_job_failure_reason(job_id)
        
        # Determine if failure is retryable
        retryable_patterns = [
            # SLURM system errors
            'OUT_OF_MEMORY', 'TIMEOUT', 'NODE_FAIL', 'CANCELLED',
            'PREEMPTED', 'BOOT_FAIL', 'DEADLINE',
            # CUDA hardware errors - these are often transient and should be retried
            'CUDA ERROR', 'ECC ERROR', 'UNCORRECTABLE ECC', 'CUDA KERNEL ERROR',
            'RUNTIMEERROR: CUDA', 'DEVICE-SIDE ASSERTION', 'CUDA_LAUNCH_BLOCKING',
            # File system errors - often transient
            'FILENOTFOUNDERROR', 'NO SUCH FILE', 'PERMISSION DENIED', 'DISK FULL',
            'IOERROR', 'OSERROR', 'ENOENT', 'EACCES', 'ENOSPC',
            # Network/connection errors
            'CONNECTION ERROR', 'NETWORK ERROR', 'TIMEOUT ERROR', 'CONNECTION REFUSED',
            # Wandb/logging errors
            'COMMERROR', 'RUN INITIALIZATION HAS TIMED OUT', 'WANDB.ERRORS.ERRORS.COMMERROR',
            # Python/import errors - might be environment issues
            'IMPORTERROR', 'MODULENOTFOUNDERROR', 'ATTRIBUTEERROR',
            # General system errors
            'SEGMENTATION FAULT', 'BUS ERROR', 'KILLED', 'SIGNAL'
        ]
        
        #is_retryable = any(pattern in failure_reason.upper() for pattern in retryable_patterns)
        is_retryable = True #Changing this to True to always retry jobs
        
        if not is_retryable and 'FAILED' in failure_reason and '0:0' not in failure_reason:
            # Non-zero exit code - might be a code issue, but still worth retrying
            is_retryable = True
        
        # Calculate new resources (increase for OOM/timeout)
        memory_multiplier = 1.0
        time_multiplier = 1.0
        
        if 'OUT_OF_MEMORY' in failure_reason.upper():
            memory_multiplier = self.config.retry_memory_multiplier ** attempt
        if 'TIMEOUT' in failure_reason.upper():
            time_multiplier = self.config.retry_time_multiplier ** attempt
        if any(cuda_error in failure_reason.upper() for cuda_error in ['CUDA ERROR', 'ECC ERROR', 'UNCORRECTABLE ECC']):
            # For CUDA errors, try with different GPU settings and slightly more memory
            memory_multiplier = 1.2  # Slight memory increase
            time_multiplier = 1.1    # Slight time increase
        if any(file_error in failure_reason.upper() for file_error in ['FILENOTFOUNDERROR', 'NO SUCH FILE', 'ENOENT']):
            # For file errors, try with more time (might be slow filesystem)
            time_multiplier = 1.5    # More time for file operations
        if any(perm_error in failure_reason.upper() for perm_error in ['PERMISSION DENIED', 'EACCES']):
            # For permission errors, try with different settings
            memory_multiplier = 1.1  # Slight memory increase
            time_multiplier = 1.2    # Slight time increase
        
        return is_retryable, failure_reason, memory_multiplier, time_multiplier
    
    def _validate_synthetic_data(self, tasks: List[str], iteration: int) -> List[str]:
        """Validate that synthetic data exists for given tasks"""
        valid_tasks = []
        
        for task in tasks:
            # Check if synthetic data file exists
            robot, obj, obst, subtask = task.split('_')
            data_path = self.results_path / f"augmented_{iteration}" / "diffusion" / f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_{iteration}" / f"{robot}_{obj}_{obst}_{subtask}" / "samples_0.npz"
            
            if data_path.exists():
                valid_tasks.append(task)
            else:
                print(f"  ⚠️  Missing synthetic data for task: {task}")
                print(f"     Expected: {data_path}")
        
        return valid_tasks
    
    def _ensure_all_synthetic_data_exists(self, iteration: int, target_tasks: List[str]) -> bool:
        """Second layer: Check for missing synthetic data and resubmit failed jobs"""
        print(f"\n=== SECOND LAYER: VERIFYING SYNTHETIC DATA COMPLETENESS ===")
        
        max_resubmit_attempts = 100  # Maximum number of resubmit attempts per task
        resubmit_attempts = {}  # Track attempts per task
        
        for attempt in range(max_resubmit_attempts):
            missing_tasks = []
            
            # Check which tasks are missing synthetic data
            for task in target_tasks:
                robot, obj, obst, subtask = task.split('_')
                data_path = self.results_path / f"augmented_{iteration}" / "diffusion" / f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_{iteration}" / f"{robot}_{obj}_{obst}_{subtask}" / "samples_0.npz"
                
                if not data_path.exists():
                    missing_tasks.append(task)
            
            if not missing_tasks:
                print(f"✅ All {len(target_tasks)} tasks have synthetic data!")
                return True
            
            print(f"⚠️  Attempt {attempt + 1}: {len(missing_tasks)} tasks missing synthetic data")
            for task in missing_tasks:
                print(f"  ⚠️  Missing: {task}")
            
            # Resubmit missing tasks
            resubmitted_jobs = []
            for task in missing_tasks:
                if resubmit_attempts.get(task, 0) >= max_resubmit_attempts:
                    print(f"❌ Task {task} exceeded max resubmit attempts ({max_resubmit_attempts})")
                    continue
                
                # Create script generator for this task
                def script_generator(retry_context, task=task):
                    return self._build_generation_script(iteration, task, retry_context)
                
                script_content = script_generator({})
                script_path = self.base_path / f"resubmit_gen_{iteration}_{task.replace('_', '')[:20]}_{attempt + 1}.sh"
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = self._parse_job_id(result.stdout)
                    resubmitted_jobs.append({
                        'job_id': job_id,
                        'task_id': task,
                        'script_generator': script_generator,
                        'attempt': 1
                    })
                    resubmit_attempts[task] = resubmit_attempts.get(task, 0) + 1
                    print(f"🔄 Resubmitted {task} (attempt {resubmit_attempts[task]})")
                else:
                    print(f"❌ Failed to resubmit {task}: {result.stderr}")
                
                script_path.unlink()
            
            if not resubmitted_jobs:
                print("❌ No jobs were successfully resubmitted")
                return False
            
            print(f"🔄 Waiting for {len(resubmitted_jobs)} resubmitted jobs to complete...")
            
            # Wait for resubmitted jobs to complete
            success, _ = self._wait_for_jobs_completion_with_retry(resubmitted_jobs, "data generation resubmit")
            if not success:
                print("❌ Some resubmitted jobs failed")
                return False
            
            # Wait a bit before checking again
            if attempt < max_resubmit_attempts - 1:
                print("⏳ Waiting 5 minutes before next verification...")
                time.sleep(300)
        
        # Final check
        final_missing = []
        for task in target_tasks:
            robot, obj, obst, subtask = task.split('_')
            data_path = self.results_path / f"augmented_{iteration}" / "diffusion" / f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_{iteration}" / f"{robot}_{obj}_{obst}_{subtask}" / "samples_0.npz"
            
            if not data_path.exists():
                final_missing.append(task)
        
        if final_missing:
            print(f"❌ After {max_resubmit_attempts} attempts, {len(final_missing)} tasks still missing:")
            for task in final_missing:
                print(f"  ❌ {task}")
            return False
        else:
            print(f"✅ All tasks successfully generated after resubmissions!")
            return True

    def _ensure_diffusion_model_exists(self, iteration: int) -> bool:
        """Second layer: Check if diffusion model exists and resubmit if not."""
        print(f"\n=== SECOND LAYER: VERIFYING DIFFUSION MODEL COMPLETENESS ===")
        
        model_dir = self.results_path / f"augmented_{iteration}" / "diffusion" / f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_{iteration}"
        
        # Check for model checkpoint file (should be model-100000.pt based on config)
        checkpoint_file = model_dir / "model-100000.pt" #for fast testing, we reduced from 100000 to 10
        
        if checkpoint_file.exists():
            print(f"✅ Diffusion model checkpoint found at {checkpoint_file}")
            return True
        
        # Also check for config.gin file to see if training started
        config_file = model_dir / "config.gin"
        if config_file.exists():
            print(f"⚠️  Diffusion training started but checkpoint not found. Expected: {checkpoint_file}")
        else:
            print(f"❌ Diffusion model directory not found at {model_dir}")
        
        print(f"❌ Diffusion model checkpoint not found. Resubmitting diffusion training job...")
        
        # Create script generator for resubmission
        def script_generator(retry_context):
            return self._build_diffusion_script(iteration, f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_{iteration}", retry_context)
        
        # Generate a unique job name for the resubmission
        job_name = f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_resubmit_{iteration}"
        
        script_content = script_generator({})
        script_path = self.base_path / f"job_diffusion_resubmit_{iteration}.sh"
        
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
        script_path.unlink()
        
        if result.returncode == 0:
            job_id = self._parse_job_id(result.stdout)
            print(f"Resubmitted diffusion training job: {job_id}")
            
            # Wait for the resubmitted job to complete
            retry_context = {'attempt': 1}
            success = self._wait_for_job_completion_with_retry(job_id, "diffusion training", script_generator, retry_context)
            
            if success:
                # Check again for the checkpoint
                if checkpoint_file.exists():
                    print(f"✅ Resubmitted diffusion training completed successfully. Checkpoint found at {checkpoint_file}")
                    return True
                else:
                    print(f"⚠️  Resubmitted diffusion training completed but checkpoint still not found at {checkpoint_file}")
                    return False
            else:
                print(f"❌ Resubmitted diffusion training job failed after retries.")
                return False
        else:
            print(f"Failed to submit resubmitted diffusion training job: {result.stderr}")
            return False

    def _ensure_all_policy_training_completed(self, iteration: int, test_tasks: List[str], middle_tasks: List[str]) -> Tuple[bool, Set[str]]:
        """Second layer: Check for missing policy training results and resubmit failed jobs"""
        print(f"\n=== SECOND LAYER: VERIFYING POLICY TRAINING COMPLETENESS ===")
        
        max_resubmit_attempts = 100  # Maximum number of resubmit attempts per task+seed
        resubmit_attempts = {}  # Track attempts per task+seed
        successful_job_ids = set()  # Track successful job IDs
        
        for attempt in range(max_resubmit_attempts):
            missing_jobs = []
            
            # Add initial wait to allow filesystem to sync
            if attempt == 0:
                print("⏳ Waiting 30 seconds for filesystem to sync before verification...")
                time.sleep(30)
            
            # Check which test tasks are missing "Training completed" line
            for task in test_tasks:
                for seed in self.config.rl_seeds:
                    job_key = f"{task}_seed{seed}_test"
                    if resubmit_attempts.get(job_key, 0) >= max_resubmit_attempts:
                        continue
                    
                    # Check if log file exists and contains "Training completed"
                    log_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_test_{iteration}"
                    # Fix: Use the correct pattern that matches actual filenames
                    # Pattern: *_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}.out
                    robot, obj, obst, subtask = task.split('_')
                    search_pattern = f"*_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}.out"
                    log_files = list(log_dir.glob(search_pattern))
                    
                    if not log_files:
                        missing_jobs.append((task, seed, 'test'))
                        continue
                    
                    # Check if any log file contains "Training completed"
                    training_completed = False
                    max_read_attempts = 10  # Try reading up to 10 times
                    for log_file in log_files:
                        for read_attempt in range(max_read_attempts):
                            try:
                                with open(log_file, 'r') as f:
                                    content = f.read()
                                    if "Training completed" in content:
                                        training_completed = True
                                        break
                                if training_completed:
                                    break
                                # Wait before retry if not found
                                time.sleep(5)
                            except Exception as e:
                                if read_attempt == max_read_attempts - 1:
                                    print(f"Error reading log file {log_file}: {e}")
                                time.sleep(5)
                        if training_completed:
                            break
                    
                    if not training_completed:
                        missing_jobs.append((task, seed, 'test'))
            
            # Check which middle tasks are missing "Training completed" line
            for task in middle_tasks:
                for seed in self.config.middle_seeds:
                    job_key = f"{task}_seed{seed}_middle"
                    if resubmit_attempts.get(job_key, 0) >= max_resubmit_attempts:
                        continue
                    
                    # Check if log file exists and contains "Training completed"
                    log_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_middle_{iteration}"
                    # Fix: Use the correct pattern that matches actual filenames
                    # Pattern: *_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}.out
                    robot, obj, obst, subtask = task.split('_')
                    search_pattern = f"*_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}.out"
                    log_files = list(log_dir.glob(search_pattern))
                    
                    if not log_files:
                        missing_jobs.append((task, seed, 'middle'))
                        continue
                    
                    # Check if any log file contains "Training completed"
                    training_completed = False
                    max_read_attempts = 10  # Try reading up to 10 times
                    for log_file in log_files:
                        for read_attempt in range(max_read_attempts):
                            try:
                                with open(log_file, 'r') as f:
                                    content = f.read()
                                    if "Training completed" in content:
                                        training_completed = True
                                        break
                                if training_completed:
                                    break
                                # Wait before retry if not found
                                time.sleep(5)
                            except Exception as e:
                                if read_attempt == max_read_attempts - 1:
                                    print(f"Error reading log file {log_file}: {e}")
                                time.sleep(5)
                        if training_completed:
                            break
                    
                    if not training_completed:
                        missing_jobs.append((task, seed, 'middle'))
            
            if not missing_jobs:
                print(f"✅ All policy training jobs completed successfully!")
                print(f"  Test tasks: {len(test_tasks)} × {len(self.config.rl_seeds)} seeds = {len(test_tasks) * len(self.config.rl_seeds)} jobs")
                print(f"  Middle tasks: {len(middle_tasks)} × {len(self.config.middle_seeds)} seeds = {len(middle_tasks) * len(self.config.middle_seeds)} jobs")
                print(f"  Total: {len(test_tasks) * len(self.config.rl_seeds) + len(middle_tasks) * len(self.config.middle_seeds)} policy training jobs")
                return True, successful_job_ids
            
            print(f"⚠️  Attempt {attempt + 1}: {len(missing_jobs)} policy training jobs missing 'Training completed'")
            for task, seed, task_type in missing_jobs:
                print(f"  ⚠️  Missing: {task} (seed {seed}, {task_type})")
            
            # Resubmit missing jobs
            resubmitted_jobs = []
            for task, seed, task_type in missing_jobs:
                job_key = f"{task}_seed{seed}_{task_type}"
                if resubmit_attempts.get(job_key, 0) >= max_resubmit_attempts:
                    print(f"❌ Job {job_key} exceeded max resubmit attempts ({max_resubmit_attempts})")
                    continue
                
                # Create script generator for this specific task+seed+type
                def script_generator(retry_context, task=task, seed=seed, task_type=task_type):
                    return self._build_policy_script(iteration, task, seed, task_type, retry_context)
                
                script_content = script_generator({})
                script_path = self.base_path / f"resubmit_policy_{iteration}_{task.replace('_', '')[:15]}_{seed}_{task_type}_{attempt + 1}.sh"
                
                with open(script_path, 'w') as f:
                    f.write(script_content)
                
                result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
                if result.returncode == 0:
                    job_id = self._parse_job_id(result.stdout)
                    resubmitted_jobs.append({
                        'job_id': job_id,
                        'task_id': job_key,
                        'script_generator': script_generator,
                        'attempt': 1
                    })
                    resubmit_attempts[job_key] = resubmit_attempts.get(job_key, 0) + 1
                    print(f"🔄 Resubmitted {job_key} (attempt {resubmit_attempts[job_key]})")
                else:
                    print(f"❌ Failed to resubmit {job_key}: {result.stderr}")
                
                script_path.unlink()
            
            if not resubmitted_jobs:
                print("❌ No jobs were successfully resubmitted")
                return False, successful_job_ids
            
            print(f"🔄 Waiting for {len(resubmitted_jobs)} resubmitted policy jobs to complete...")
            
            # Wait for resubmitted jobs to complete
            success, completed_job_ids = self._wait_for_jobs_completion_with_retry(resubmitted_jobs, "policy training resubmit")
            if not success:
                print("❌ Some resubmitted policy jobs failed - will continue checking in next iteration")
                # Don't exit - continue the loop to check logs again
            else:
                # Track successful job IDs
                successful_job_ids.update(completed_job_ids)
            
            # Wait a bit before checking again
            if attempt < max_resubmit_attempts - 1:
                print("⏳ Waiting 5 minutes before next verification...")
                time.sleep(300)
        
        # Final check
        final_missing = []
        for task in test_tasks:
            for seed in self.config.rl_seeds:
                log_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_test_{iteration}"
                # Fix: Use the correct pattern that matches actual filenames
                # Pattern: *_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}.out
                robot, obj, obst, subtask = task.split('_')
                search_pattern = f"*_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}.out"
                log_files = list(log_dir.glob(search_pattern))
                
                training_completed = False
                for log_file in log_files:
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                            if "Training completed" in content:
                                training_completed = True
                                break
                    except Exception:
                        pass
                
                if not training_completed:
                    final_missing.append((task, seed, 'test'))
        
        for task in middle_tasks:
            for seed in self.config.middle_seeds:
                log_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_middle_{iteration}"
                # Fix: Use the correct pattern that matches actual filenames
                # Pattern: *_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}.out
                robot, obj, obst, subtask = task.split('_')
                search_pattern = f"*_{robot}_{obj}_{obst}_{subtask}_RLseed{seed}_{iteration}.out"
                log_files = list(log_dir.glob(search_pattern))
                
                training_completed = False
                for log_file in log_files:
                    try:
                        with open(log_file, 'r') as f:
                            content = f.read()
                            if "Training completed" in content:
                                training_completed = True
                                break
                    except Exception:
                        pass
                
                if not training_completed:
                    final_missing.append((task, seed, 'middle'))
        
        if final_missing:
            print(f"❌ After {max_resubmit_attempts} attempts, {len(final_missing)} policy jobs still missing:")
            for task, seed, task_type in final_missing:
                print(f"  ❌ {task} (seed {seed}, {task_type})")
            
            # Show completion summary
            total_expected = len(test_tasks) * len(self.config.rl_seeds) + len(middle_tasks) * len(self.config.middle_seeds)
            completed = total_expected - len(final_missing)
            print(f"\n📊 Policy Training Completion Summary:")
            print(f"  Expected: {total_expected} jobs")
            print(f"  Completed: {completed} jobs")
            print(f"  Missing: {len(final_missing)} jobs")
            print(f"  Success rate: {completed/total_expected*100:.1f}%")
            
            return False, successful_job_ids
        else:
            print(f"✅ All policy training jobs successfully completed after resubmissions!")
            return True, successful_job_ids
    
    def _print_test_task_table(self, iteration: int):
        """Print a table showing test task success rates across all iterations"""
        if not self.test_tasks_history:
            print("\nNo test task history available yet.")
            return
        
        print(f"\n{'='*80}")
        print(f"TEST TASK SUCCESS RATES - ITERATION {iteration}")
        print(f"{'='*80}")
        
        # Get all tasks and iterations
        all_tasks = list(self.test_tasks_history.keys())
        all_iterations = set()
        for task_history in self.test_tasks_history.values():
            all_iterations.update(task_history.keys())
        all_iterations = sorted(all_iterations)
        
        if not all_tasks:
            print("No test tasks found.")
            return
        
        # Create header
        header = f"{'Task':<30}"
        for iter_num in all_iterations:
            header += f"{'Iter ' + str(iter_num):<12}"
        header += f"{'Best':<12}"
        print(header)
        print('-' * len(header))
        
        # Print each task row
        all_best_rates = []
        for task in sorted(all_tasks):
            task_history = self.test_tasks_history[task]
            row = f"{task:<30}"
            
            # Success rates for each iteration
            task_rates = []
            for iter_num in all_iterations:
                rate = task_history.get(iter_num)
                if rate is not None:
                    row += f"{rate:<12.3f}"
                    task_rates.append(rate)
                else:
                    row += f"{'--':<12}"
            
            # Best rate for this task
            best_rate = max(task_rates) if task_rates else 0.0
            row += f"{best_rate:<12.3f}"
            all_best_rates.append(best_rate)
            
            print(row)
        
        # Bottom row with average best success rate up to each iteration
        print('-' * len(header))
        footer = f"{'AVERAGE BEST RATE':<30}"
        
        for iter_num in all_iterations:
            # Calculate average best rate up to this iteration
            best_rates_up_to_iter = []
            for task in all_tasks:
                task_history = self.test_tasks_history[task]
                # Get all rates for this task up to current iteration
                rates_so_far = [rate for i, rate in task_history.items() 
                               if i <= iter_num and rate is not None]
                if rates_so_far:
                    best_rates_up_to_iter.append(max(rates_so_far))
            
            if best_rates_up_to_iter:
                avg_best_up_to_iter = sum(best_rates_up_to_iter) / len(best_rates_up_to_iter)
                footer += f"{avg_best_up_to_iter:<12.3f}"
            else:
                footer += f"{'--':<12}"
        
        # Final column shows overall best rate so far
        final_avg_best = sum(all_best_rates) / len(all_best_rates) if all_best_rates else 0.0
        footer += f"{final_avg_best:<12.3f}"
        print(footer)
        print(f"{'='*80}")

    def _save_final_test_task_table(self, final_iteration: int):
        """Save the final test task table as CSV"""
        if not self.test_tasks_history:
            print("\nNo test task history available for CSV export.")
            return
        
        # Get all tasks and iterations
        all_tasks = list(self.test_tasks_history.keys())
        all_iterations = set()
        for task_history in self.test_tasks_history.values():
            all_iterations.update(task_history.keys())
        all_iterations = sorted(all_iterations)
        
        if not all_tasks:
            print("No test tasks found for CSV export.")
            return
        
        # Create DataFrame
        rows = []
        for task in sorted(all_tasks):
            task_history = self.test_tasks_history[task]
            row = {'task': task}
            
            # Success rates for each iteration
            task_rates = []
            for iter_num in all_iterations:
                rate = task_history.get(iter_num)
                if rate is not None:
                    row[f'iter_{iter_num}'] = rate
                    task_rates.append(rate)
                else:
                    row[f'iter_{iter_num}'] = None
            
            # Best rate for this task
            best_rate = max(task_rates) if task_rates else 0.0
            row['best_rate'] = best_rate
            rows.append(row)
        
        # Add average best rate row
        avg_row = {'task': 'AVERAGE_BEST_RATE'}
        for iter_num in all_iterations:
            # Calculate average best rate up to this iteration
            best_rates_up_to_iter = []
            for task in all_tasks:
                task_history = self.test_tasks_history[task]
                rates_so_far = [rate for i, rate in task_history.items() 
                               if i <= iter_num and rate is not None]
                if rates_so_far:
                    best_rates_up_to_iter.append(max(rates_so_far))
            
            if best_rates_up_to_iter:
                avg_best_up_to_iter = sum(best_rates_up_to_iter) / len(best_rates_up_to_iter)
                avg_row[f'iter_{iter_num}'] = avg_best_up_to_iter
            else:
                avg_row[f'iter_{iter_num}'] = None
        
        # Final average best rate
        all_best_rates = [row['best_rate'] for row in rows]
        final_avg_best = sum(all_best_rates) / len(all_best_rates) if all_best_rates else 0.0
        avg_row['best_rate'] = final_avg_best
        rows.append(avg_row)
        
        # Create DataFrame and save
        import pandas as pd
        df = pd.DataFrame(rows)
        
        # Save to CSV in the main policies_slurm_diffusionseed folder
        output_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"final_test_task_table_iter{final_iteration}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved final test task table to: {output_path}")

    def _create_best_testtask_dataset(self):
        """Create best test task dataset by copying the best synthetic data for each test task"""
        print(f"\n{'='*60}")
        print("CREATING BEST TEST TASK DATASET")
        print(f"{'='*60}")
        
        if not self.test_tasks_history:
            print("No test task history available. Cannot create best dataset.")
            return
        
        # Get all test tasks
        test_tasks = self.get_test_tasks_from_file()
        print(f"Found {len(test_tasks)} test tasks to process")
        
        # Find best iteration for each test task
        best_iterations = {}
        for task in test_tasks:
            if task in self.test_tasks_history:
                task_history = self.test_tasks_history[task]
                # Find iteration with highest success rate
                best_iter = max(task_history.items(), key=lambda x: x[1] if x[1] is not None else 0)
                best_iterations[task] = best_iter[0]
                print(f"  {task}: best iteration {best_iter[0]} (success rate: {best_iter[1]:.3f})")
            else:
                print(f"  {task}: no history found")
        
        # Create best dataset directory
        # Find the maximum iteration that was actually run
        max_iteration = 0
        for task_history in self.test_tasks_history.values():
            if task_history:
                max_iteration = max(max_iteration, max(task_history.keys()))
        
        best_dataset_name = f"best_testtask_dataset/monolithic_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_iters{max_iteration}"
        best_dataset_path = self.results_path / best_dataset_name
        best_dataset_path.mkdir(parents=True, exist_ok=True)
        
        print(f"\nCreating best dataset at: {best_dataset_path}")
        
        # Copy best synthetic data for each test task
        copied_tasks = 0
        failed_tasks = []
        
        for task in test_tasks:
            if task not in best_iterations:
                failed_tasks.append(f"{task} (no history)")
                continue
                
            best_iter = best_iterations[task]
            robot, obj, obst, subtask = task.split('_')
            
            # Source path (best iteration's synthetic data)
            source_path = self.results_path / f"augmented_{best_iter}" / "diffusion" / f"{self.config.denoiser}_tasklist{self.config.task_list_seed}_train{self.config.num_train}_diffusionseed{self.config.diffusion_seed}_{best_iter}" / f"{robot}_{obj}_{obst}_{subtask}"
            
            # Destination path (best dataset)
            dest_path = best_dataset_path / f"{robot}_{obj}_{obst}_{subtask}"
            
            if source_path.exists():
                try:
                    # Copy the entire task directory
                    import shutil
                    if dest_path.exists():
                        shutil.rmtree(dest_path)
                    shutil.copytree(source_path, dest_path)
                    copied_tasks += 1
                    print(f"  ✅ {task}: copied from iteration {best_iter}")
                except Exception as e:
                    failed_tasks.append(f"{task} (copy error: {e})")
                    print(f"  ❌ {task}: failed to copy from iteration {best_iter} - {e}")
            else:
                failed_tasks.append(f"{task} (source not found)")
                print(f"  ❌ {task}: source data not found at {source_path}")
        
        # Create summary file
        summary_data = {
            'dataset_name': best_dataset_name,
            'total_test_tasks': len(test_tasks),
            'successfully_copied': copied_tasks,
            'failed_tasks': failed_tasks,
            'best_iterations': best_iterations,
            'config': {
                'denoiser': self.config.denoiser,
                'task_list_seed': self.config.task_list_seed,
                'num_train': self.config.num_train,
                'diffusion_seed': self.config.diffusion_seed,
                'success_threshold': self.config.success_threshold,
                'final_threshold': self.current_threshold
            }
        }
        
        summary_path = best_dataset_path / "best_dataset_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary_data, f, indent=2)
        
        print(f"\n📊 BEST DATASET SUMMARY:")
        print(f"   Dataset path: {best_dataset_path}")
        print(f"   Total test tasks: {len(test_tasks)}")
        print(f"   Successfully copied: {copied_tasks}")
        print(f"   Failed tasks: {len(failed_tasks)}")
        print(f"   Success rate: {copied_tasks/len(test_tasks)*100:.1f}%")
        
        if failed_tasks:
            print(f"\n❌ Failed tasks:")
            for task in failed_tasks:
                print(f"   - {task}")
        
        print(f"\n📁 Summary saved to: {summary_path}")
        print(f"✅ Best test task dataset creation completed!")

    def _analyze_test_tasks_history(self, iteration: int, test_results: Dict[str, float]) -> Tuple[Dict[str, float], float]:
        """Analyze historical performance of test tasks and compute best rates so far"""
        # Update history
        for task, success_rate in test_results.items():
            if task not in self.test_tasks_history:
                self.test_tasks_history[task] = {}
            self.test_tasks_history[task][iteration] = success_rate
        
        # Compute best success rates so far for each task
        current_best_rates = {}
        for task in test_results.keys():
            task_history = self.test_tasks_history[task]
            # Get all success rates up to current iteration
            rates_so_far = [rate for iter_num, rate in task_history.items() 
                           if iter_num <= iteration and rate is not None]
            if rates_so_far:
                current_best_rates[task] = max(rates_so_far)
        
        # Compute average best success rate
        if current_best_rates:
            avg_best_success = sum(current_best_rates.values()) / len(current_best_rates)
        else:
            avg_best_success = 0.0
        
        return current_best_rates, avg_best_success
    
    def _save_historical_analysis(self, iteration: int):
        """Save historical analysis to CSV"""
        if not self.test_tasks_history:
            return
            
        # Create DataFrame with all iterations
        rows = []
        for task in self.test_tasks_history:
            row = {'task': task}
            # Add success rate for each iteration
            for i in range(iteration + 1):
                row[f'success_rate_iter{i}'] = self.test_tasks_history[task].get(i)
            # Add best success rate so far
            rates = [rate for rate in self.test_tasks_history[task].values() if rate is not None]
            row['best_success_rate'] = max(rates) if rates else None
            rows.append(row)
        
        # Create DataFrame and save
        import pandas as pd
        df = pd.DataFrame(rows)
        
        # Sort by best success rate
        df = df.sort_values('best_success_rate', ascending=False, na_position='last')
        
        # Save to CSV
        output_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_test_{iteration}" / "logs"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"test_tasks_history_iter{iteration}.csv"
        df.to_csv(output_path, index=False)
        print(f"\nSaved historical analysis to: {output_path}")
    
    def _wait_for_job_completion_with_retry(self, job_id: str, job_type: str, job_script_generator, retry_context: dict) -> bool:
        """Wait for job completion with automatic retry on failure"""
        attempt = retry_context.get('attempt', 1)
        
        if not job_id:
            return False
        
        print(f"Waiting for {job_type} job {job_id} to complete (attempt {attempt})...")
        
        while True:
            # Check job status
            result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
            if job_id not in result.stdout:
                # Job finished, check if successful
                result = subprocess.run(['sacct', '-j', job_id, '--format=JobID,State', '--noheader'], 
                                      capture_output=True, text=True)
                # Parse the output to find the main job (without .batch or .extern suffix)
                lines = result.stdout.strip().split('\n')
                main_job_status = None
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id_part, status = parts[0], parts[1]
                        # Check if this is the main job (no suffix)
                        if job_id_part == job_id:
                            main_job_status = status
                            break
                
                if main_job_status == 'COMPLETED':
                    print(f"{job_type} job {job_id} completed successfully")
                    return True
                else:
                    print(f"{job_type} job {job_id} failed. Status: {main_job_status}")
                    
                    # Check if we should retry
                    should_retry, failure_reason, mem_mult, time_mult = self._should_retry_job(job_id, attempt)
                    
                    if should_retry:
                        print(f"Failure reason: {failure_reason}")
                        print(f"Retrying with {mem_mult:.1f}x memory, {time_mult:.1f}x time (attempt {attempt + 1}/{self.config.max_retries})")
                        
                        # Wait before retry
                        time.sleep(self.config.retry_delay)
                        
                        # Update retry context
                        retry_context['attempt'] = attempt + 1
                        retry_context['memory_multiplier'] = mem_mult
                        retry_context['time_multiplier'] = time_mult
                        
                        # Generate new job script with updated resources
                        new_script = job_script_generator(retry_context)
                        script_path = self.base_path / f"retry_{job_type}_{attempt + 1}_{int(time.time())}.sh"
                        
                        with open(script_path, 'w') as f:
                            f.write(new_script)
                        
                        # Submit retry job
                        result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
                        script_path.unlink()
                        
                        if result.returncode == 0:
                            new_job_id = self._parse_job_id(result.stdout)
                            print(f"Retry job submitted: {new_job_id}")
                            
                            # Recursively wait for retry job
                            return self._wait_for_job_completion_with_retry(new_job_id, job_type, job_script_generator, retry_context)
                        else:
                            print(f"Failed to submit retry job: {result.stderr}")
                            return False
                    else:
                        print(f"Job not retryable or max retries exceeded. Failure: {failure_reason}")
                        return False
            
            time.sleep(60)  # Check every minute
    
    def _wait_for_job_completion(self, job_id: str, job_type: str) -> bool:
        """Wait for a single job to complete (legacy method for backward compatibility)"""
        if not job_id:
            return False
        
        print(f"Waiting for {job_type} job {job_id} to complete...")
        
        while True:
            # Check job status
            result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
            if job_id not in result.stdout:
                # Job finished, check if successful
                result = subprocess.run(['sacct', '-j', job_id, '--format=JobID,State', '--noheader'], 
                                      capture_output=True, text=True)
                # Parse the output to find the main job (without .batch or .extern suffix)
                lines = result.stdout.strip().split('\n')
                main_job_status = None
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) >= 2:
                        job_id_part, status = parts[0], parts[1]
                        # Check if this is the main job (no suffix)
                        if job_id_part == job_id:
                            main_job_status = status
                            break
                
                if main_job_status == 'COMPLETED':
                    print(f"{job_type} job {job_id} completed successfully")
                    return True
                else:
                    print(f"{job_type} job {job_id} failed. Status: {main_job_status}")
                    return False
            
            time.sleep(60)  # Check every minute
    
    def _wait_for_jobs_completion_with_retry(self, job_contexts: List[dict], job_type: str) -> Tuple[bool, Set[str]]:
        """Wait for multiple jobs to complete with retry support"""
        if not job_contexts:
            return True, set()
        
        print(f"Waiting for {len(job_contexts)} {job_type} jobs to complete...")
        
        completed_jobs = set()
        failed_jobs = set()
        # Filter out None job IDs
        valid_contexts = [ctx for ctx in job_contexts if ctx.get('job_id') is not None]
        active_contexts = {ctx['job_id']: ctx for ctx in valid_contexts}
        
        while len(completed_jobs) + len(failed_jobs) < len(valid_contexts):
            jobs_to_check = list(active_contexts.keys())
            
            for job_id in jobs_to_check:
                if job_id in completed_jobs or job_id in failed_jobs:
                    continue
                
                # Check job status
                result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
                if job_id not in result.stdout:
                    # Job finished, check if successful
                    result = subprocess.run(['sacct', '-j', job_id, '--format=JobID,State', '--noheader'], 
                                          capture_output=True, text=True)
                    # Parse the output to find the main job (without .batch or .extern suffix)
                    lines = result.stdout.strip().split('\n')
                    main_job_status = None
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2:
                            job_id_part, status = parts[0], parts[1]
                            # Check if this is the main job (no suffix)
                            if job_id_part == job_id:
                                main_job_status = status
                                break
                    
                    if main_job_status == 'COMPLETED':
                        completed_jobs.add(job_id)
                        del active_contexts[job_id]
                        print(f"✅ {job_type} job {job_id} completed successfully")
                    else:
                        print(f"❌ {job_type} job {job_id} failed. Status: {main_job_status}")
                        ctx = active_contexts[job_id]
                        attempt = ctx.get('attempt', 1)
                        
                        # Check if we should retry
                        should_retry, failure_reason, mem_mult, time_mult = self._should_retry_job(job_id, attempt)
                        
                        if should_retry:
                            print(f"🔄 Retrying {job_type} job (attempt {attempt + 1}/{self.config.max_retries})")
                            print(f"   Reason: {failure_reason}")
                            print(f"   Resources: {mem_mult:.1f}x memory, {time_mult:.1f}x time")
                            
                            # Wait before retry
                            time.sleep(self.config.retry_delay)
                            
                            # Update context
                            ctx['attempt'] = attempt + 1
                            ctx['memory_multiplier'] = mem_mult
                            ctx['time_multiplier'] = time_mult
                            
                            # Generate and submit retry job
                            new_script = ctx['script_generator'](ctx)
                            script_path = self.base_path / f"retry_{job_type}_{ctx['task_id']}_{attempt + 1}_{int(time.time())}.sh"
                            
                            with open(script_path, 'w') as f:
                                f.write(new_script)
                            
                            result = subprocess.run(["sbatch", str(script_path)], capture_output=True, text=True)
                            script_path.unlink()
                            
                            if result.returncode == 0:
                                new_job_id = self._parse_job_id(result.stdout)
                                print(f"   Retry job submitted: {new_job_id}")
                                
                                # Update tracking
                                del active_contexts[job_id]
                                ctx['job_id'] = new_job_id
                                active_contexts[new_job_id] = ctx
                            else:
                                print(f"   Failed to submit retry job: {result.stderr}")
                                failed_jobs.add(job_id)
                                del active_contexts[job_id]
                        else:
                            print(f"   Not retrying: {failure_reason}")
                            failed_jobs.add(job_id)
                            del active_contexts[job_id]
            
            if len(completed_jobs) + len(failed_jobs) < len(job_contexts):
                time.sleep(60)  # Check every minute
        
        print(f"{job_type} jobs final status: {len(completed_jobs)} successful, {len(failed_jobs)} permanently failed")
        return len(failed_jobs) == 0, completed_jobs
    
    def _wait_for_jobs_completion(self, job_ids: List[str], job_type: str) -> bool:
        """Wait for multiple jobs to complete (legacy method - no retry)"""
        if not job_ids:
            return True
        
        print(f"Waiting for {len(job_ids)} {job_type} jobs to complete...")
        
        completed_jobs = set()
        failed_jobs = set()
        
        while len(completed_jobs) + len(failed_jobs) < len(job_ids):
            for job_id in job_ids:
                if job_id in completed_jobs or job_id in failed_jobs:
                    continue
                
                # Check job status
                result = subprocess.run(['squeue', '-j', job_id], capture_output=True, text=True)
                if job_id not in result.stdout:
                    # Job finished, check if successful
                    result = subprocess.run(['sacct', '-j', job_id, '--format=State', '--noheader'], 
                                          capture_output=True, text=True)
                    if 'COMPLETED' in result.stdout:
                        completed_jobs.add(job_id)
                    else:
                        failed_jobs.add(job_id)
            
            if len(completed_jobs) + len(failed_jobs) < len(job_ids):
                time.sleep(60)  # Check every minute
        
        print(f"{job_type} jobs completed: {len(completed_jobs)} successful, {len(failed_jobs)} failed")
        return len(failed_jobs) == 0, completed_jobs
    
    def _parse_policy_logs(self, iteration: int, task_type: str, successful_job_ids: Optional[Set[str]] = None) -> Dict[str, float]:
        """Parse policy training logs to extract success rates"""
        log_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_{task_type}_{iteration}"
        results = {}
        
        if not log_dir.exists():
            return results
        
        # Pattern to match log files and extract task info
        for log_file in log_dir.glob("*.out"):
            # Parse filename to extract task - use the correct regex that matches actual filenames
            match = re.search(r'(\d+)_monolithic_tasklist(\d+)_train(\d+)_diffusionseed(\d+)_td3_bc_(\w+)_(\w+)_(\w+)_(\w+)_RLseed(\d+)_(\d+)\.out', log_file.name)
            if not match:
                continue
            
            jobid, tasklist, num_train, diffusion_seed, robot, obj, obst, subtask, rlseed, difftrainrun = match.groups()
            
            # Filter by successful job IDs if provided
            if successful_job_ids is not None and jobid not in successful_job_ids:
                continue
                
            task = f"{robot}_{obj}_{obst}_{subtask}"
            
            # Parse success rate from log content
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Look for final summary line
                final_match = re.search(r'Training completed\.\s*Best score:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*Best success rate:\s*([0-9]+(?:\.[0-9]+)?)', content, re.IGNORECASE)
                if final_match:
                    success_rate = float(final_match.group(2))
                    
                    # Aggregate across seeds for the same task
                    if task not in results:
                        results[task] = []
                    results[task].append(success_rate)
            
            except Exception as e:
                print(f"Error parsing log {log_file}: {e}")
        
        # Average success rates across seeds for each task
        for task, rates in results.items():
            results[task] = np.mean(rates) if rates else 0.0
        
        return results
    
    def _generate_csv_analysis(self, iteration: int):
        """Generate detailed CSV analysis files like the manual plots_from_logs script"""
        import pandas as pd
        import numpy as np
        
        # Generate CSV for test tasks
        self._create_comparison_csv(iteration, "test", seeds=(0, 1, 2, 3, 4))
        
        # Generate CSV for middle tasks  
        self._create_comparison_csv(iteration, "middle", seeds=(0,))
    
    def _create_comparison_csv(self, iteration: int, task_type: str, seeds: tuple):
        """Create comparison CSV for given task type and seeds"""
        import pandas as pd
        import numpy as np
        
        # Get log directory
        if task_type == "test":
            log_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_test_{iteration}"
            output_filename = f"td3bc_diffusionseed{self.config.diffusion_seed}_diffusiontrainingrun{iteration}.csv"
        else:  # middle
            log_dir = self.base_path / "scripts" / "policies_slurm_logs" / f"policies_slurm_diffusionseed{self.config.diffusion_seed}" / f"policies_slurm_middle_{iteration}"
            output_filename = f"td3bc_diffusionseed{self.config.diffusion_seed}_diffusiontraining{iteration}_middle.csv"
        
        if not log_dir.exists():
            print(f"Warning: Log directory {log_dir} does not exist")
            return
        
        # Parse all log files and build dataframe
        rows = []
        for log_file in log_dir.glob("*.out"):
            # Extract metadata from filename using regex like the original script
            import re
            match = re.match(
                r'(\d+)_monolithic_tasklist(\d+)_train(\d+)_diffusionseed(\d+)_td3_bc_(\w+)_(\w+)_(\w+)_(\w+)_RLseed(\d+)_(\d+)\.out',
                log_file.name
            )
            if not match:
                continue
                
            jobid, tasklist, num_train, diffusion_seed, robot, obj, obst, subtask, rlseed, difftrainrun = match.groups()
            
            # Parse log content for score and success rate
            try:
                with open(log_file, 'r') as f:
                    content = f.read()
                
                # Look for final summary line
                final_match = re.search(
                    r'Training completed\.\s*Best score:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*Best success rate:\s*([0-9]+(?:\.[0-9]+)?)',
                    content, re.IGNORECASE
                )
                
                if final_match:
                    score = float(final_match.group(1))
                    success = float(final_match.group(2))
                else:
                    # Fallback parsing (like original script)
                    score_matches = re.findall(r'Current best score:\s*([0-9]+(?:\.[0-9]+)?)', content, re.IGNORECASE)
                    success_matches = re.findall(r'Success rate:\s*([0-9]+(?:\.[0-9]+)?)', content, re.IGNORECASE)
                    
                    score = float(max(score_matches)) if score_matches else None
                    success = float(max(success_matches)) if success_matches else None
                
                # Add row to dataframe
                rows.append({
                    'jobid': int(jobid),
                    'denoiser': 'monolithic',
                    'tasklist_seed': int(tasklist),
                    'num_train': int(num_train),
                    'diffusion_seed': int(diffusion_seed),
                    'algorithm': 'td3_bc',
                    'robot': robot,
                    'object': obj,
                    'obstacle': obst,
                    'subtask': subtask,
                    'rlseed': int(rlseed),
                    'difftrainrun': int(difftrainrun),
                    'score': score,
                    'success': success,
                    'log_path': str(log_file)
                })
            
            except Exception as e:
                print(f"Error parsing {log_file}: {e}")
        
        if not rows:
            print(f"No valid log files found in {log_dir}")
            return
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Filter by configuration parameters
        df = df[
            (df['tasklist_seed'] == self.config.task_list_seed) &
            (df['num_train'] == self.config.num_train) &
            (df['diffusion_seed'] == self.config.diffusion_seed) &
            (df['difftrainrun'] == iteration)
        ]
        
        # Create comparison pivot table
        comparison_df = self._create_comparison_pivot(df, seeds)
        
        # Save CSV
        csv_output_dir = log_dir / "logs"
        csv_output_dir.mkdir(exist_ok=True)
        output_path = csv_output_dir / output_filename
        
        comparison_df.to_csv(output_path, index=False)
        print(f"Generated CSV analysis: {output_path}")
        
        # Print summary like original script
        print(f"CSV analysis for {task_type} tasks (iteration {iteration}):")
        print(f"  Shape: {comparison_df.shape}")
        print(f"  Seeds analyzed: {seeds}")
        
    def _create_comparison_pivot(self, df: pd.DataFrame, seeds: tuple) -> pd.DataFrame:
        """Create seed-wise comparison table like the original script"""
        import pandas as pd
        import numpy as np
        
        group_vars = ["robot", "object", "obstacle", "subtask"]
        
        # Build seed-wise row per unique task combination
        rows = []
        for keys, grp in df.groupby(group_vars):
            if not isinstance(keys, tuple):
                keys = (keys,)
            row = dict(zip(group_vars, keys))
            
            score_vals, succ_vals = [], []
            for s in seeds:
                g = grp[grp["rlseed"] == s]
                if not g.empty:
                    score = g["score"].iloc[0]
                    succ = g["success"].iloc[0]
                else:
                    score, succ = np.nan, np.nan
                
                row[f"score_seed{s}"] = score
                row[f"success_seed{s}"] = succ
                if not pd.isna(score):
                    score_vals.append(score)
                if not pd.isna(succ):
                    succ_vals.append(succ)
            
            row["avg_score"] = float(np.nanmean(score_vals)) if score_vals else np.nan
            row["avg_success"] = float(np.nanmean(succ_vals)) if succ_vals else np.nan
            rows.append(row)
        
        # Create ordered DataFrame
        out = pd.DataFrame(rows)
        if out.empty:
            return out
            
        ordered = list(group_vars)
        for s in seeds:
            ordered += [f"score_seed{s}", f"success_seed{s}"]
        ordered += ["avg_score", "avg_success"]
        
        return out.reindex(columns=ordered)
    
    def _print_final_summary(self):
        """Print final summary of the pipeline"""
        print("\n" + "="*60)
        print("PIPELINE SUMMARY")
        print("="*60)
        
        for iteration in range(len(self.good_tasks_history)):
            good_count = len(self.good_tasks_history.get(iteration, []))
            bad_count = len(self.bad_tasks_history.get(iteration, []))
            print(f"Iteration {iteration}: {good_count} good tasks, {bad_count} bad tasks")
        
        # Threshold reduction summary
        if self.threshold_reduction_history:
            print(f"\n🔧 THRESHOLD REDUCTION HISTORY:")
            print(f"   Initial threshold: {self.config.success_threshold:.3f}")
            print(f"   Final threshold: {self.current_threshold:.3f}")
            print(f"   Total reductions: {len(self.threshold_reduction_history)}")
            for reduction in self.threshold_reduction_history:
                print(f"   Iteration {reduction['iteration']}: {reduction['old_threshold']:.3f} → {reduction['new_threshold']:.3f}")
        else:
            print(f"\n🔧 THRESHOLD: No reductions needed (maintained at {self.current_threshold:.3f})")
        
        # Final status
        if not self.bad_tasks_history:
            print("\n❌ PIPELINE FAILED: No analysis completed (policy training failed)")
        else:
            final_iteration = max(self.bad_tasks_history.keys())
            remaining_tasks = len(self.bad_tasks_history.get(final_iteration, []))
            
            if remaining_tasks == 0:
                print("\n🎉 SUCCESS: All tasks solved!")
            else:
                print(f"\n📊 FINAL STATUS: {remaining_tasks} tasks remaining")


def main():
    parser = argparse.ArgumentParser(description="Automated Iterative Diffusion Pipeline")
    parser.add_argument('--max_iterations', type=int, required=True, 
                       help='Maximum number of iterations to run')
    parser.add_argument('--config_file', type=str, 
                       help='JSON config file (optional, uses defaults if not provided)')
    parser.add_argument('--denoiser', type=str, default='monolithic',
                       help='Denoiser type')
    parser.add_argument('--num_train', type=int, default=56,
                       help='Number of training tasks')
    parser.add_argument('--task_list_seed', type=int, default=0,
                       help='Task list seed')
    parser.add_argument('--diffusion_seed', type=int, default=0,
                       help='Diffusion seed')
    parser.add_argument('--curriculum_seed', type=int, default=0,
                       help='Curriculum randomization seed')
    parser.add_argument('--success_threshold', type=float, default=0.8,
                       help='Success rate threshold for good tasks')
    parser.add_argument('--threshold_reduction_amount', type=float, default=0.1,
                       help='Amount to reduce threshold by when no good tasks found')
    parser.add_argument('--threshold_reduction_cycle', type=int, default=1,
                       help='Number of consecutive iterations with no good tasks before reducing threshold')
    parser.add_argument('--min_threshold', type=float, default=0.5,
                       help='Minimum threshold value (cannot reduce below this)')
    parser.add_argument('--max_retries', type=int, default=100,
                       help='Maximum number of retries for failed jobs')
    parser.add_argument('--retry_delay', type=int, default=300,
                       help='Delay in seconds between retries')
    parser.add_argument('--retry_memory_multiplier', type=float, default=1.5,
                       help='Memory multiplier for retries')
    parser.add_argument('--retry_time_multiplier', type=float, default=1.5,
                       help='Time multiplier for retries')
    parser.add_argument('--max_memory_gb', type=int, default=400,
                       help='Maximum memory available on machines (GB)')
    parser.add_argument('--max_time_hours', type=int, default=72,
                       help='Maximum time limit for jobs (hours)')
    
    args = parser.parse_args()
    
    # Create config
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config_dict = json.load(f)
        config = IterationConfig(**config_dict)
    else:
        config = IterationConfig(
            denoiser=args.denoiser,
            num_train=args.num_train,
            task_list_seed=args.task_list_seed,
            diffusion_seed=args.diffusion_seed,
            curriculum_seed=args.curriculum_seed,
            success_threshold=args.success_threshold,
            threshold_reduction_amount=args.threshold_reduction_amount,
            threshold_reduction_cycle=args.threshold_reduction_cycle,
            min_threshold=args.min_threshold,
            max_retries=args.max_retries,
            retry_delay=args.retry_delay,
            retry_memory_multiplier=args.retry_memory_multiplier,
            retry_time_multiplier=args.retry_time_multiplier,
            max_memory_gb=args.max_memory_gb,
            max_time_hours=args.max_time_hours
        )
    
    # Create and run pipeline
    pipeline = IterativeDiscoveryPipeline(config, args.max_iterations)
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
