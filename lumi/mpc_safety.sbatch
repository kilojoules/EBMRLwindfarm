#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=mpc_sg
#SBATCH --output=logs/mpc_sg_%A_%a.out
#SBATCH --error=logs/mpc_sg_%A_%a.err
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

# Offline MPC at d=10, 25, 40
for B in 10 25 40; do
    python scripts/mpc_correction_safety.py \
        --budget $B --seed $SEED --n-eps 20 \
        --mpc-horizon 5 --n-candidates 64 --cem-iters 2 \
        --dyn-eps 100 --dyn-iters 5000 \
        --out results/mpc_safety_offline.json
done

# Online Q_c refit (budget 10 only, enough to show effect)
python scripts/mpc_correction_safety.py \
    --budget 10 --seed $SEED --n-eps 20 --online-qc \
    --mpc-horizon 5 --n-candidates 64 --cem-iters 2 \
    --dyn-eps 100 --dyn-iters 5000 \
    --out results/mpc_safety_online.json
