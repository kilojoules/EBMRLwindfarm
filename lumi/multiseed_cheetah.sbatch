#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --job-name=cheetah_s%a
#SBATCH --output=logs/cheetah_seed%a_%j.out
#SBATCH --error=logs/cheetah_seed%a_%j.err
#SBATCH --array=1-5

# ===========================================================================
# Multi-seed HalfCheetah: train + ablations for each seed
# Submit with: sbatch lumi/multiseed_cheetah.sbatch
# ===========================================================================

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "Job $SLURM_JOB_ID seed=$SEED started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# Train
pixi run python scripts/safety_gym_budget.py \
    --train \
    --env HalfCheetah-v5 \
    --total-timesteps 100000 \
    --checkpoint "checkpoints/sac_cheetah_s${SEED}.pt"

# Ablations on this seed's checkpoint
pixi run python scripts/halfcheetah_ablations.py \
    --checkpoint "checkpoints/sac_cheetah_s${SEED}.pt" \
    --ablation all \
    --n-episodes 10

echo "Job $SLURM_JOB_ID seed=$SEED finished at $(date)"
