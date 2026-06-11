#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=mpc_fe
#SBATCH --output=logs/mpc_fe_%A_%a.out
#SBATCH --error=logs/mpc_fe_%A_%a.err
#SBATCH --array=1-5

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

SEED=$SLURM_ARRAY_TASK_ID

# Feasibility-filtered MPC: among candidates, filter by predicted cost ≤ budget,
# pick max reward among feasible. Falls back to min-cost if none feasible.
python scripts/mpc_true_env_sg.py \
    --seed $SEED --budget 10 --n-eps 10 \
    --horizon-mpc 10 --n-candidates 16 --cem-iters 1 \
    --feasibility \
    --out results/mpc_feasibility_sg.json
