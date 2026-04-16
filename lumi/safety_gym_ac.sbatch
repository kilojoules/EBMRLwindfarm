#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=03:00:00
#SBATCH --job-name=safety_ac
#SBATCH --output=logs/safety_ac_%j.out
#SBATCH --error=logs/safety_ac_%j.err

# ===========================================================================
# Safety Gym AC Budget: train unconstrained, evaluate with budget constraint
#
# Uses Ant-v5: point robot navigating to goal with hazards.
# The "budget" is the allowed timesteps near hazards per episode.
# ===========================================================================

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# MuJoCo already installed from previous jobs
pixi run pip install mujoco 2>/dev/null || true

# Train unconstrained SAC (200k steps — Safety Gym needs more than HalfCheetah)
echo ""
echo "=== Training unconstrained SAC on Ant-v5 ==="
pixi run python scripts/safety_gym_ac_budget.py \
    --train \
    --env Ant-v5 \
    --total-timesteps 200000 \
    --checkpoint checkpoints/sac_safety_point.pt \
    --seed 1

# Full comparison: unconstrained vs constant vs AC at multiple budgets
echo ""
echo "=== Comparing methods ==="
pixi run python scripts/safety_gym_ac_budget.py \
    --compare \
    --env Ant-v5 \
    --checkpoint checkpoints/sac_safety_point.pt \
    --n-episodes 20 \
    --horizon 1000

echo "Job $SLURM_JOB_ID finished at $(date)"
