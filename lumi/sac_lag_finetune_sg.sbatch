#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=05:00:00
#SBATCH --job-name=sac_ft
#SBATCH --output=logs/sac_ft_%A_%a.out
#SBATCH --error=logs/sac_ft_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

BUDGETS=(10 25 40)
BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}

python -u scripts/sac_lag_finetune_sg.py \
    --pretrained-seed 1 --budget $BUDGET --horizon 1000 \
    --eval-at 0 50000 200000 500000 \
    --mu-init 0.1 --mu-lr 0.005 --lr 1e-4 \
    --out results/saclag_finetune.json
