#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --job-name=log_traj
#SBATCH --output=logs/log_traj_%j.out
#SBATCH --error=logs/log_traj_%j.err

# Per-step trajectory logging for Figs 1 & 2
set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs results

# Ensure deps
pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

# Log HalfCheetah trajectories (wind farm needs checkpoint loading fix)
pixi run python scripts/log_trajectories.py \
    --domain cheetah \
    --checkpoint checkpoints/sac_cheetah.pt \
    --output results/trajectories_cheetah.npz \
    --n-episodes 10

echo "Job $SLURM_JOB_ID finished at $(date)"
