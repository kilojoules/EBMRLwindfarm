#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --job-name=wf_qc
#SBATCH --output=logs/wf_qc_%j.out
#SBATCH --error=logs/wf_qc_%j.err

# Wind farm cost critic: collect fatigue data → train Q_c → evaluate
# Q_c composes with EBT actor via per-turbine energy addition

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

cd /scratch/project_465002609/julian/ac_budget_experiment
mkdir -p logs data checkpoints results

pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

PYTHONUNBUFFERED=1 pixi run python scripts/windfarm_cost_critic.py \
    --collect --train --compare \
    --n-collect-episodes 100 \
    --n-eval-episodes 10 \
    --horizon 200 \
    --gamma-c 0.99 \
    --epochs 200 \
    --data-path data/windfarm_cost_data.npz \
    --qc-path checkpoints/windfarm_qc.pt \
    --output-json results/windfarm_qc.json

echo "Job $SLURM_JOB_ID finished at $(date)"
