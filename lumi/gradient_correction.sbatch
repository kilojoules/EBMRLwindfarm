#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=grad_corr
#SBATCH --output=logs/grad_corr_%j.out
#SBATCH --error=logs/grad_corr_%j.err

# Gradient correction vs action scaling comparison
# Tests both linear velocity cost and nonlinear loading surrogate

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

cd /scratch/project_465002609/julian/ac_budget_experiment
mkdir -p logs

PYTHONUNBUFFERED=1 pixi run python scripts/gradient_correction.py \
    --checkpoint checkpoints/sac_cheetah.pt \
    --env HalfCheetah-v5 \
    --n-episodes 10 \
    --budget-frac 0.25

echo "Job $SLURM_JOB_ID finished at $(date)"
