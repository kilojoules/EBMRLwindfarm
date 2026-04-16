#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=12:00:00
#SBATCH --job-name=wf_sweep
#SBATCH --output=logs/wf_sweep_%j.out
#SBATCH --error=logs/wf_sweep_%j.err

# Comprehensive wind farm budget sweep
# 9 budgets × 6 η × 3 k × 4 gs = 648 configs × 50 episodes
# Plus 1/u schedule comparison (12 configs)

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

cd /scratch/project_465002609/julian/ac_budget_experiment
mkdir -p logs results

pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

PYTHONUNBUFFERED=1 pixi run python scripts/windfarm_sweep.py \
    --n-episodes 50 \
    --horizon 200 \
    --output results/windfarm_sweep.json

echo "Job $SLURM_JOB_ID finished at $(date)"
