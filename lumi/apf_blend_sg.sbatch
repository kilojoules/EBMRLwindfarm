#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=apf_sg
#SBATCH --output=logs/apf_sg_%A_%a.out
#SBATCH --error=logs/apf_sg_%A_%a.err
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

# Sweep sharpness (how fast σ ramps up when budget runs out).
# sharpness=0 → no blending (pure actor; should match uncon baseline).
for SHARPNESS in 0.0 1.0 3.0 10.0; do
    python -u scripts/apf_blend_sg.py \
        --seed $SEED --budget 10 --n-eps 20 \
        --sigma-sharpness $SHARPNESS \
        --r-repel 0.6 --alpha-apf 0.5 \
        --out results/apf_blend_sg.json
done

# Also test at budget=25, 40
for B in 25 40; do
    python -u scripts/apf_blend_sg.py \
        --seed $SEED --budget $B --n-eps 20 \
        --sigma-sharpness 3.0 \
        --r-repel 0.6 --alpha-apf 0.5 \
        --out results/apf_blend_sg.json
done
