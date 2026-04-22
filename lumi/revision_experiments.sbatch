#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=revision
#SBATCH --output=logs/revision_%j.out
#SBATCH --error=logs/revision_%j.err

# All reviewer revision experiments in one job:
# 1. Sensitivity analyses (wall threshold, lambda_max, stochastic costs, timing)
# 2. Linear ramp baseline for Saute comparison
# 3. Constant penalty sweep (already exists, re-run for consistency)

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs

# Sensitivity analyses
echo "=== Running sensitivity analyses ==="
pixi run python scripts/sensitivity_analysis.py \
    --checkpoint checkpoints/sac_cheetah.pt \
    --n-episodes 10

echo "Job $SLURM_JOB_ID finished at $(date)"
