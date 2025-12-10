#!/usr/bin/env python3
"""
Run Compositional TD3+BC Baseline Pipeline

This script submits SLURM jobs for training compositional TD3+BC policies,
monitors their completion, and analyzes results.

Mirrors the setup from automated_iterative_diffusion_dits_iiwa.py but without
diffusion training or data generation stages.
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
from dataclasses import dataclass

@dataclass
class BaselineConfig:
    """Configuration for compositional baseline"""
    # Base paths
    base_path: str = '/home/quanpham/iterative-comp-rl-generation'
    data_path: str = '/home/quanpham/iterative-comp-rl-generation/data'
    results_path: str = '/home/quanpham/iterative-comp-rl-generation/results/compositional_baseline'
    tasks_path: str = '/home/quanpham/iterative-comp-rl-generation/offline_compositional_rl_datasets/_train_test_splits'
    
    # Task config
    num_train: int = 14  # First 14 training tasks
    task_list_seed: int = 0
    
    # Training config
    seeds: List[int] = None  # [0, 1, 2, 3, 4]
    max_timesteps: int = 50000  # Same as individual policies
    batch_size: int = 3584  # 14 tasks × 256
    eval_freq: int = 5000  # Evaluation frequency
    n_eval_episodes: int = 10  # Episodes per evaluation
    
    # Job resource config
    memory: int = 50  # GB
    time: int = 24  # hours
    
    # Retry configuration
    max_retries: int = 100
    retry_delay: int = 300  # 5 minutes
    retry_memory_multiplier: float = 1.5
    retry_time_multiplier: float = 1.5
    max_memory_gb: int = 400
    max_time_hours: int = 72
    
    def __post_init__(self):
        if self.seeds is None:
            self.seeds = [0, 1, 2, 3, 4]


class CompositionalBaselinePipeline:
    """Pipeline for running compositional TD3+BC baseline"""
    
    def __init__(self, config: BaselineConfig):
        self.config = config
        self.base_path = Path(config.base_path)
        self.results_path = Path(config.results_path)
        
        # Create necessary directories
        self.setup_directories()
    
    def setup_directories(self):
        """Create necessary directories"""
        dirs_to_create = [
            self.base_path / "scripts" / "slurm",
            self.base_path / "scripts" / "compositional_baseline_logs",
            self.results_path,
        ]
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def load_task_lists(self) -> Tuple[List[str], List[str]]:
        """Load train and test tasks from JSON file"""
        task_file = Path(self.config.tasks_path) / "component-wise" / "split_0.json"
        with open(task_file, 'r') as f:
            task_data = json.load(f)
        
        # Convert to string format: Robot_Obj_Obst_Subtask
        train_tasks = ['_'.join(task) for task in task_data['train']]
        test_tasks = ['_'.join(task) for task in task_data['test']]
        
        return train_tasks, test_tasks
    
    def submit_training_jobs(self) -> List[Dict]:
        """Submit training jobs for all seeds"""
        print(f"\n{'='*80}")
        print("SUBMITTING TRAINING JOBS")
        print(f"{'='*80}")
        
        train_tasks, test_tasks = self.load_task_lists()
        train_tasks_subset = train_tasks[:self.config.num_train]
        
        print(f"Training on: {len(train_tasks_subset)} tasks (first {self.config.num_train} from train set)")
        for i, task in enumerate(train_tasks_subset, 1):
            print(f"  {i:2d}. {task}")
        
        print(f"\nEvaluating on: {len(test_tasks)} test tasks")
        print(f"Seeds: {self.config.seeds}")
        
        job_contexts = []
        
        for seed in self.config.seeds:
            # Create script generator function for this seed
            def script_generator(retry_context, seed=seed):
                return self._build_training_script(seed, retry_context)
            
            script_content = script_generator({})
            script_path = self.base_path / f"job_compositional_baseline_seed{seed}.sh"
            
            with open(script_path, 'w') as f:
                f.write(script_content)
            
            result = subprocess.run(["sbatch", str(script_path)], 
                                   capture_output=True, text=True)
            
            if result.returncode == 0:
                job_id = self._parse_job_id(result.stdout)
                job_contexts.append({
                    'job_id': job_id,
                    'seed': seed,
                    'script_generator': script_generator,
                    'attempt': 1
                })
                print(f"✓ Submitted seed {seed}: job {job_id}")
            else:
                print(f"✗ Failed to submit seed {seed}: {result.stderr}")
            
            script_path.unlink()
        
        print(f"\n✓ Submitted {len(job_contexts)} training jobs")
        return job_contexts
    
    def _build_training_script(self, seed: int, retry_context: dict = None) -> str:
        """Build SLURM script for training"""
        memory = self.config.memory
        time_hours = self.config.time
        
        if retry_context:
            requested_memory = int(memory * retry_context.get('memory_multiplier', 1.0))
            requested_time = int(time_hours * retry_context.get('time_multiplier', 1.0))
            
            memory = min(requested_memory, self.config.max_memory_gb)
            time_hours = min(requested_time, self.config.max_time_hours)
            
            if requested_memory > self.config.max_memory_gb:
                print(f"⚠️  Requested {requested_memory}GB capped at {self.config.max_memory_gb}GB")
            if requested_time > self.config.max_time_hours:
                print(f"⚠️  Requested {requested_time}h capped at {self.config.max_time_hours}h")
        
        job_name = f"comp_baseline_seed{seed}"
        
        return f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=scripts/compositional_baseline_logs/%j_{job_name}.out
#SBATCH --mem={memory}G
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=10
#SBATCH --time={time_hours}:00:00
#SBATCH --partition=eaton-compute
#SBATCH --qos=ee-high

source /home/quanpham/first_3.9.6/bin/activate
export WANDB__SERVICE_WAIT=1000

# CUDA error prevention
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=0

python {self.config.base_path}/scripts/train_compositional_td3bc_multitask.py \\
    --data_path {self.config.data_path} \\
    --task_list_path {self.config.tasks_path}/component-wise/split_0.json \\
    --num_train {self.config.num_train} \\
    --results_path {self.config.results_path}/seed_{seed} \\
    --seed {seed} \\
    --max_timesteps {self.config.max_timesteps} \\
    --batch_size {self.config.batch_size} \\
    --eval_freq {self.config.eval_freq} \\
    --n_eval_episodes {self.config.n_eval_episodes}

echo "Training completed"
"""
    
    def _parse_job_id(self, sbatch_output: str) -> str:
        """Parse job ID from sbatch output"""
        match = re.search(r'Submitted batch job (\d+)', sbatch_output)
        return match.group(1) if match else None
    
    def _get_job_failure_reason(self, job_id: str) -> str:
        """Get detailed failure reason for a job"""
        try:
            result = subprocess.run(
                ['sacct', '-j', job_id, '--format=State,ExitCode,Reason', '--noheader'],
                capture_output=True, text=True
            )
            slurm_reason = result.stdout.strip()
            
            # Check log file for errors
            log_pattern = f"scripts/compositional_baseline_logs/{job_id}_*.out"
            log_files = list(self.base_path.glob(log_pattern))
            
            if log_files:
                try:
                    with open(log_files[0], 'r') as f:
                        log_content = f.read()
                    
                    error_patterns = [
                        'CUDA error', 'RuntimeError', 'FileNotFoundError',
                        'ImportError', 'ModuleNotFoundError', 'AssertionError'
                    ]
                    
                    for pattern in error_patterns:
                        if pattern.lower() in log_content.lower():
                            return f"ERROR: {pattern}"
                except Exception:
                    pass
            
            return slurm_reason
        except:
            return "Unknown failure"
    
    def _should_retry_job(self, job_id: str, attempt: int) -> Tuple[bool, str, float, float]:
        """Determine if job should be retried"""
        if attempt >= self.config.max_retries:
            return False, "Max retries exceeded", 0, 0
        
        failure_reason = self._get_job_failure_reason(job_id)
        
        # Always retry (as in original pipeline)
        is_retryable = True
        
        memory_multiplier = 1.0
        time_multiplier = 1.0
        
        if 'OUT_OF_MEMORY' in failure_reason.upper():
            memory_multiplier = self.config.retry_memory_multiplier ** attempt
        if 'TIMEOUT' in failure_reason.upper():
            time_multiplier = self.config.retry_time_multiplier ** attempt
        if 'CUDA' in failure_reason.upper():
            memory_multiplier = 1.2
            time_multiplier = 1.1
        
        return is_retryable, failure_reason, memory_multiplier, time_multiplier
    
    def _wait_for_jobs_completion_with_retry(self, job_contexts: List[dict]) -> Tuple[bool, Set[str]]:
        """Wait for jobs to complete with retry support"""
        if not job_contexts:
            return True, set()
        
        print(f"\n{'='*80}")
        print(f"WAITING FOR {len(job_contexts)} JOBS TO COMPLETE")
        print(f"{'='*80}")
        
        completed_jobs = set()
        failed_jobs = set()
        active_contexts = {ctx['job_id']: ctx for ctx in job_contexts}
        
        while len(completed_jobs) + len(failed_jobs) < len(job_contexts):
            jobs_to_check = list(active_contexts.keys())
            
            for job_id in jobs_to_check:
                if job_id in completed_jobs or job_id in failed_jobs:
                    continue
                
                # Check job status
                result = subprocess.run(['squeue', '-j', job_id], 
                                       capture_output=True, text=True)
                
                if job_id not in result.stdout:
                    # Job finished
                    result = subprocess.run(
                        ['sacct', '-j', job_id, '--format=JobID,State', '--noheader'],
                        capture_output=True, text=True
                    )
                    
                    # Parse main job status
                    lines = result.stdout.strip().split('\n')
                    main_job_status = None
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) >= 2 and parts[0] == job_id:
                            main_job_status = parts[1]
                            break
                    
                    if main_job_status == 'COMPLETED':
                        completed_jobs.add(job_id)
                        ctx = active_contexts[job_id]
                        print(f"✓ Seed {ctx['seed']} completed (job {job_id})")
                        del active_contexts[job_id]
                    else:
                        print(f"✗ Job {job_id} failed: {main_job_status}")
                        ctx = active_contexts[job_id]
                        attempt = ctx.get('attempt', 1)
                        
                        # Check if should retry
                        should_retry, reason, mem_mult, time_mult = self._should_retry_job(
                            job_id, attempt
                        )
                        
                        if should_retry:
                            print(f"🔄 Retrying seed {ctx['seed']} (attempt {attempt + 1})")
                            print(f"   Reason: {reason}")
                            print(f"   Resources: {mem_mult:.1f}x mem, {time_mult:.1f}x time")
                            
                            time.sleep(self.config.retry_delay)
                            
                            # Update context
                            ctx['attempt'] = attempt + 1
                            ctx['memory_multiplier'] = mem_mult
                            ctx['time_multiplier'] = time_mult
                            
                            # Submit retry
                            new_script = ctx['script_generator'](ctx)
                            script_path = self.base_path / f"retry_seed{ctx['seed']}_{attempt + 1}.sh"
                            
                            with open(script_path, 'w') as f:
                                f.write(new_script)
                            
                            result = subprocess.run(["sbatch", str(script_path)],
                                                   capture_output=True, text=True)
                            script_path.unlink()
                            
                            if result.returncode == 0:
                                new_job_id = self._parse_job_id(result.stdout)
                                print(f"   Retry submitted: {new_job_id}")
                                
                                del active_contexts[job_id]
                                ctx['job_id'] = new_job_id
                                active_contexts[new_job_id] = ctx
                            else:
                                print(f"   Failed to submit retry: {result.stderr}")
                                failed_jobs.add(job_id)
                                del active_contexts[job_id]
                        else:
                            print(f"   Not retrying: {reason}")
                            failed_jobs.add(job_id)
                            del active_contexts[job_id]
            
            if len(completed_jobs) + len(failed_jobs) < len(job_contexts):
                time.sleep(60)  # Check every minute
        
        print(f"\n{'='*80}")
        print(f"JOBS COMPLETED: {len(completed_jobs)}/{len(job_contexts)}")
        print(f"{'='*80}")
        
        return len(failed_jobs) == 0, completed_jobs
    
    def analyze_results(self) -> pd.DataFrame:
        """Analyze results and generate summary table"""
        print(f"\n{'='*80}")
        print("ANALYZING RESULTS")
        print(f"{'='*80}")
        
        train_tasks, test_tasks = self.load_task_lists()
        
        # Collect results from all seeds
        all_results = []
        
        for seed in self.config.seeds:
            seed_dir = self.results_path / f"seed_{seed}"
            final_results_file = seed_dir / f"final_results_seed{seed}.json"
            
            # Try JSON first
            if final_results_file.exists():
                try:
                    with open(final_results_file, 'r') as f:
                        results = json.load(f)
                    
                    # Use best_score_checkpoint results (single model evaluation, not cherry-picked)
                    checkpoint_data = results.get('best_score_checkpoint', {})
                    per_task_results = checkpoint_data.get('per_task_results', {})
                    
                    # Extract score and success rate from per_task_results
                    per_task_score = {
                        task: task_results['mean_reward']
                        for task, task_results in per_task_results.items()
                    }
                    per_task_success = {
                        task: task_results['success_rate']
                        for task, task_results in per_task_results.items()
                    }
                    
                    all_results.append({
                        'seed': seed,
                        'avg_score': checkpoint_data.get('avg_score', results.get('best_checkpoint_score', 0)),
                        'avg_success_rate': np.mean(list(per_task_success.values())) if per_task_success else 0,
                        'checkpoint_eval_step': checkpoint_data.get('eval_step', 0),
                        'per_task_score': per_task_score,
                        'per_task_success': per_task_success
                    })
                    print(f"✓ Loaded results for seed {seed} (from best_score_checkpoint at step {checkpoint_data.get('eval_step', '?')})")
                    continue
                except Exception as e:
                    print(f"⚠️  Error reading JSON for seed {seed}: {e}")
            
            # Fallback: Parse SLURM log
            print(f"⚠️  JSON not found for seed {seed}, trying SLURM log...")
            log_dir = self.base_path / "scripts" / "compositional_baseline_logs"
            log_pattern = f"*_comp_baseline_seed{seed}.out"
            log_files = list(log_dir.glob(log_pattern))
            
            if log_files:
                try:
                    with open(log_files[0], 'r') as f:
                        log_content = f.read()
                    
                    # Parse "Training completed. Best score: X, Best success rate: Y"
                    match = re.search(
                        r'Training completed\.\s*Best score:\s*([0-9]+(?:\.[0-9]+)?)\s*,\s*Best success rate:\s*([0-9]+(?:\.[0-9]+)?)',
                        log_content, re.IGNORECASE
                    )
                    
                    if match:
                        avg_score = float(match.group(1))
                        avg_success_rate = float(match.group(2))
                        
                        print(f"✓ Loaded results for seed {seed} (from SLURM log)")
                        print(f"  WARNING: Per-task breakdown not available from log")
                        print(f"  Only overall averages: score={avg_score:.3f}, success={avg_success_rate:.3f}")
                        
                        # Can't get per-task from log, so skip this seed
                        print(f"⚠️  Skipping seed {seed} - need JSON for per-task results")
                    else:
                        print(f"⚠️  Could not parse results from SLURM log for seed {seed}")
                except Exception as e:
                    print(f"⚠️  Error reading SLURM log for seed {seed}: {e}")
            else:
                print(f"⚠️  No SLURM log found for seed {seed}")
        
        if not all_results:
            print("❌ No results found!")
            return None
        
        # Create summary table: 32 test tasks × (5 seeds × 2 metrics + 2 averages) = 32×12
        # IMPORTANT: Uses results from best_score_checkpoint evaluation (single model, not cherry-picked)
        rows = []
        
        for task in test_tasks:
            row = {'task': task}
            
            task_scores = []
            task_success_rates = []
            
            # Add columns for each seed (both score and success rate)
            for result in all_results:
                seed = result['seed']
                # Use checkpoint evaluation results (from one model evaluation)
                if task in result['per_task_score']:
                    score = result['per_task_score'][task]
                    success_rate = result['per_task_success'][task]
                    row[f'score_seed{seed}'] = score
                    row[f'success_seed{seed}'] = success_rate
                    task_scores.append(score)
                    task_success_rates.append(success_rate)
                else:
                    row[f'score_seed{seed}'] = None
                    row[f'success_seed{seed}'] = None
            
            # Add average columns
            row['avg_score'] = np.mean(task_scores) if task_scores else None
            row['avg_success'] = np.mean(task_success_rates) if task_success_rates else None
            rows.append(row)
        
        # Add overall average row
        avg_row = {'task': 'OVERALL_AVERAGE'}
        for seed in self.config.seeds:
            seed_results = [r for r in all_results if r['seed'] == seed]
            if seed_results:
                avg_row[f'score_seed{seed}'] = seed_results[0]['avg_score']
                avg_row[f'success_seed{seed}'] = seed_results[0]['avg_success_rate']
            else:
                avg_row[f'score_seed{seed}'] = None
                avg_row[f'success_seed{seed}'] = None
        
        overall_scores = [r['avg_score'] for r in all_results]
        overall_success_rates = [r['avg_success_rate'] for r in all_results]
        avg_row['avg_score'] = np.mean(overall_scores) if overall_scores else None
        avg_row['avg_success'] = np.mean(overall_success_rates) if overall_success_rates else None
        rows.append(avg_row)
        
        df = pd.DataFrame(rows)
        
        # Reorder columns: task, score_seed0, success_seed0, score_seed1, success_seed1, ..., avg_score, avg_success
        ordered_cols = ['task']
        for seed in self.config.seeds:
            ordered_cols.append(f'score_seed{seed}')
            ordered_cols.append(f'success_seed{seed}')
        ordered_cols.extend(['avg_score', 'avg_success'])
        df = df[ordered_cols]
        
        # Save to CSV
        csv_path = self.results_path / "compositional_baseline_results.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n✓ Saved results table to: {csv_path}")
        
        # Print table
        print(f"\n{'='*80}")
        print("COMPOSITIONAL TD3+BC BASELINE RESULTS")
        print(f"{'='*80}")
        print(f"Results from best_score_checkpoint evaluation (single model per seed)")
        print(f"Table shape: {df.shape} (32 test tasks + 1 avg row) × 13 columns")
        print(f"Columns: task | score_seed0 success_seed0 ... | avg_score avg_success")
        
        # Show which eval steps were used
        eval_steps = [r['checkpoint_eval_step'] for r in all_results]
        print(f"Checkpoint eval steps: {eval_steps}")
        
        print(df.to_string(index=False, float_format=lambda x: f'{x:.3f}' if pd.notna(x) else 'N/A'))
        print(f"{'='*80}")
        
        return df
    
    def run(self):
        """Run the complete pipeline"""
        print(f"\n{'='*80}")
        print("COMPOSITIONAL TD3+BC BASELINE PIPELINE")
        print(f"{'='*80}")
        
        train_tasks, test_tasks = self.load_task_lists()
        train_tasks_subset = train_tasks[:self.config.num_train]
        
        print(f"Configuration:")
        print(f"  Training on: {self.config.num_train} tasks")
        print(f"  Evaluating on: {len(test_tasks)} test tasks")
        print(f"  Seeds: {self.config.seeds}")
        print(f"  Max timesteps: {self.config.max_timesteps:,}")
        print(f"  Batch size: {self.config.batch_size}")
        print(f"  Eval frequency: {self.config.eval_freq:,}")
        print(f"  Results path: {self.config.results_path}")
        print(f"\nTraining tasks:")
        for i, task in enumerate(train_tasks_subset, 1):
            print(f"  {i:2d}. {task}")
        print(f"{'='*80}")
        
        # Submit jobs
        job_contexts = self.submit_training_jobs()
        
        if not job_contexts:
            print("❌ No jobs were submitted")
            return
        
        # Wait for completion
        success, completed_jobs = self._wait_for_jobs_completion_with_retry(job_contexts)
        
        if not success:
            print("⚠️  Some jobs failed")
        
        # Analyze results
        results_df = self.analyze_results()
        
        if results_df is not None:
            print("\n✅ Pipeline completed successfully!")
        else:
            print("\n⚠️  Pipeline completed with missing results")


def main():
    parser = argparse.ArgumentParser(
        description="Run Compositional TD3+BC Baseline Pipeline"
    )
    parser.add_argument('--num_train', type=int, default=14,
                       help='Number of training tasks')
    parser.add_argument('--max_timesteps', type=int, default=50000,
                       help='Maximum training timesteps')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4],
                       help='Seeds to run')
    parser.add_argument('--memory', type=int, default=32,
                       help='Memory per job (GB)')
    parser.add_argument('--time', type=int, default=24,
                       help='Time per job (hours)')
    
    args = parser.parse_args()
    
    # Create config
    config = BaselineConfig(
        num_train=args.num_train,
        max_timesteps=args.max_timesteps,
        seeds=args.seeds,
        memory=args.memory,
        time=args.time
    )
    
    # Run pipeline
    pipeline = CompositionalBaselinePipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()

