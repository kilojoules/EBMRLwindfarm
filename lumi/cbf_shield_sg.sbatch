#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=cbf_sg
#SBATCH --output=logs/cbf_sg_%A_%a.out
#SBATCH --error=logs/cbf_sg_%A_%a.err
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

# Critic's baseline test: constant α first. Sweep α to find reward/cost balance.
for ALPHA in 0.3 0.5 0.8; do
    python -u scripts/cbf_shield_sg.py \
        --seed $SEED --budget 10 --n-eps 20 \
        --alpha $ALPHA --schedule constant --n-candidates 16 \
        --out results/cbf_shield_sg.json
done

# Urgency-scheduled α (only meaningful if constant α already works)
python -u scripts/cbf_shield_sg.py \
    --seed $SEED --budget 10 --n-eps 20 \
    --alpha 0.5 --schedule urgency --eta 3.0 --n-candidates 16 \
    --out results/cbf_shield_sg.json
