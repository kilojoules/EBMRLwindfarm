#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --job-name=saute_baseline
#SBATCH --output=logs/saute_baseline_%j.out
#SBATCH --error=logs/saute_baseline_%j.err

# Saute RL baseline: train with budget in state, then compare to post-hoc AC
set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# Train Saute RL at 25% budget
pixi run python scripts/saute_rl_baseline.py \
    --train \
    --budget-frac 0.25 \
    --total-timesteps 100000 \
    --checkpoint checkpoints/saute_cheetah_b25.pt

# Compare Saute vs post-hoc AC
pixi run python scripts/saute_rl_baseline.py \
    --compare \
    --checkpoint checkpoints/saute_cheetah_b25.pt \
    --ac-checkpoint checkpoints/sac_cheetah.pt \
    --n-episodes 10

echo "Job $SLURM_JOB_ID finished at $(date)"
