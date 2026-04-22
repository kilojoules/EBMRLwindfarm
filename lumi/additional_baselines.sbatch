#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=01:00:00
#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%j.out
#SBATCH --error=logs/baselines_%j.err

# Additional baselines: PID, linear ramp, constant sweep
# Addresses reviewer criticism #5

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

cd /scratch/project_465002609/julian/ac_budget_experiment
mkdir -p logs

pixi run python scripts/additional_baselines.py \
    --checkpoint checkpoints/sac_cheetah.pt \
    --n-episodes 10

echo "Job $SLURM_JOB_ID finished at $(date)"
