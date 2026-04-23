#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --job-name=saclag
#SBATCH --output=logs/saclag_%A_%a.out
#SBATCH --error=logs/saclag_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

# Array index selects budget
BUDGETS=(10 25 40)
BUDGET=${BUDGETS[$SLURM_ARRAY_TASK_ID]}

python -u scripts/sac_lag_sg.py \
    --budget $BUDGET --seed 1 --total-steps 100000 \
    --horizon 1000 --mu-init 0.5 --mu-lr 0.01 \
    --out runs/sac_lag_sg_B${BUDGET}/actor.pt
