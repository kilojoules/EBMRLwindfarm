#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --job-name=cbfmk
#SBATCH --output=logs/cbfmk_%A_%a.out
#SBATCH --error=logs/cbfmk_%A_%a.err
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

# Multi-step CBF sweep: H ∈ {3, 5, 10}, α ∈ {0.3, 0.6}, K=64 w/ structured + noisy candidates
for H in 3 5 10; do
    for ALPHA in 0.3 0.6; do
        python -u scripts/cbf_shield_kinematic.py \
            --seed $SEED --budget 10 --n-eps 20 \
            --alpha $ALPHA --schedule constant \
            --horizon-cbf $H --n-candidates 64 --noise-std 1.0 \
            --out results/cbf_kinematic_multistep.json
    done
done

# Urgency-scheduled α (best H from above sweep)
python -u scripts/cbf_shield_kinematic.py \
    --seed $SEED --budget 10 --n-eps 20 \
    --alpha 0.5 --schedule urgency --eta 3.0 \
    --horizon-cbf 5 --n-candidates 64 --noise-std 1.0 \
    --out results/cbf_kinematic_multistep.json
