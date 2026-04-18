#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --job-name=ppo_lag
#SBATCH --output=logs/ppo_lag_%j.out
#SBATCH --error=logs/ppo_lag_%j.err

# PPO-Lagrangian baseline comparison
# Bisection-tuned constant penalty (simulates oracle Lagrangian) vs AC schedule
# 50 episodes per config for tight error bars

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

cd /scratch/project_465002609/julian/ac_budget_experiment
mkdir -p logs results

pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

PYTHONUNBUFFERED=1 pixi run python scripts/ppo_lag_baseline.py \
    --budget 15 \
    --n-eval-episodes 50 \
    --output results/ppo_lag_comparison.json

echo "Job $SLURM_JOB_ID finished at $(date)"
