#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=grad_corr
#SBATCH --output=logs/grad_corr_%j.out
#SBATCH --error=logs/grad_corr_%j.err

# Gradient correction vs action scaling comparison
# Tests both linear velocity cost and nonlinear loading surrogate

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

cd /scratch/project_465002609/julian/ac_budget_experiment
mkdir -p logs

# Ensure deps
pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

echo "=== HalfCheetah: Gradient Correction vs Action Scaling ==="
PYTHONUNBUFFERED=1 pixi run python scripts/gradient_correction.py \
    --checkpoint checkpoints/sac_cheetah.pt \
    --env HalfCheetah-v5 \
    --n-episodes 10 \
    --budget-frac 0.25

echo ""
echo "=== Wind Farm: Gradient Correction with Nonlinear Cost ==="
# The wind farm test uses the EBT actor with gradient correction
# on a nonlinear fatigue cost function (not just binary yaw<0)
PYTHONUNBUFFERED=1 pixi run python scripts/gradient_correction_windfarm.py \
    --n-episodes 5 \
    --budget 15 \
    --horizon 200

echo "Job $SLURM_JOB_ID finished at $(date)"
