#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=wf_ablate
#SBATCH --output=logs/wf_ablate_%j.out
#SBATCH --error=logs/wf_ablate_%j.err

# Wind farm ablations: hard guard, schedule comparison, budget flexibility, timing
# Eval-only — uses existing trained EBT checkpoint

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

cd /scratch/project_465002609/julian/ac_budget_experiment
mkdir -p logs results

pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

PYTHONUNBUFFERED=1 pixi run python scripts/windfarm_ablations.py \
    --ablation all \
    --n-episodes 10 \
    --horizon 200 \
    --output results/windfarm_ablations.json

echo "Job $SLURM_JOB_ID finished at $(date)"
