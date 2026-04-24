#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=08:00:00
#SBATCH --job-name=omnisafe
#SBATCH --output=logs/omnisafe_%A_%a.out
#SBATCH --error=logs/omnisafe_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env
pip install omnisafe 2>&1 | tail -3 || true

BUDGETS=(10 25 40)
BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}

python -u scripts/omnisafe_baseline.py \
    --algo PPOLag --env SafetyPointGoal1-v0 \
    --budget $BUDGET --steps 2000000 --seed 0 \
    --log-dir runs/omnisafe_PPOLag_B${BUDGET} \
    --out results/omnisafe_eval.json
