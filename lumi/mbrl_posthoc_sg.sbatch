#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --job-name=mbrl_sg
#SBATCH --output=logs/mbrl_sg_%A_%a.out
#SBATCH --error=logs/mbrl_sg_%A_%a.err
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

# Full post-hoc MBRL: ensemble f_θ + Q_r + Q_c + CEM with β·Q_r − λ·Q_c
# Sweep β to find reward/sat balance.
for BETA in 1.0 10.0 100.0; do
    python scripts/mbrl_posthoc_sg.py \
        --seed $SEED --budget 10 --n-eps 10 \
        --horizon-mpc 10 --n-candidates 32 --cem-iters 1 \
        --dyn-eps 100 --dyn-ensemble 5 --dyn-iters 3000 --qr-iters 3000 \
        --beta $BETA \
        --out results/mbrl_posthoc_sg.json
done

# Online refit variant at best β (start with 10.0, adjust after seeing offline)
python scripts/mbrl_posthoc_sg.py \
    --seed $SEED --budget 10 --n-eps 10 \
    --horizon-mpc 10 --n-candidates 32 --cem-iters 1 \
    --dyn-eps 100 --dyn-ensemble 5 --dyn-iters 3000 --qr-iters 3000 \
    --beta 10.0 --online-refit --refit-every 3 --refit-iters 500 \
    --out results/mbrl_posthoc_sg.json
