#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=safety_gym_budget
#SBATCH --output=logs/safety_gym_budget_%j.out
#SBATCH --error=logs/safety_gym_budget_%j.err

# ===========================================================================
# Domain 2: Safety-Gymnasium HalfCheetah velocity budget experiment
#
# Train unconstrained SAC, then evaluate with AC-inspired velocity budget.
# Time-varying reward multiplier creates temporal structure for allocation.
# ===========================================================================

set -euo pipefail

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# Install mujoco (HalfCheetah-v5 is built into gymnasium[mujoco])
pixi run pip install mujoco 2>/dev/null || true

# Train + evaluate in one shot
pixi run python scripts/safety_gym_budget.py \
    --train \
    --eval \
    --env HalfCheetah-v5 \
    --total-timesteps 100000 \
    --checkpoint checkpoints/sac_cheetah.pt \
    --budget-frac 0.25 \
    --n-episodes 10 \
    --horizon 1000 \
    --reward-amplitude 0.5 \
    --reward-period 200

echo "Job $SLURM_JOB_ID finished at $(date)"
