#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=apf_tune
#SBATCH --output=logs/apf_tune_%j.out
#SBATCH --error=logs/apf_tune_%j.err

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

# Tune APF: stronger goal-attractor, smaller repulsor radius.
# Run pure-safe (force max blending via very tight budget d=1, sharpness high).
# Evaluate reward + cost.
for ALPHA in 0.7 0.85 0.95; do
    for R in 0.3 0.5; do
        python -u scripts/apf_blend_sg.py \
            --seed 1 --budget 1 --n-eps 10 \
            --sigma-sharpness 100.0 \
            --alpha-apf $ALPHA --r-repel $R --k-rep 1.0 \
            --out results/apf_tuning_sweep.json
    done
done
