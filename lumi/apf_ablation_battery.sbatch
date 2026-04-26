#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --job-name=apf_abl
#SBATCH --output=logs/apf_abl_%A_%a.out
#SBATCH --error=logs/apf_abl_%A_%a.err
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

# Ablation battery: separate value of urgency schedule from value of π_safe
# All run at d=10 and d=25.
for B in 10 25; do
    # 1. Pure π_safe (APF only, σ=1 always) — what does the safety controller do alone?
    python -u scripts/apf_blend_sg.py --seed $SEED --budget $B --n-eps 20 \
        --mode pure_safe --r-repel 0.6 --alpha-apf 0.5 \
        --out results/apf_ablation.json

    # 2. Constant σ blends (no urgency adaptation): σ ∈ {0.3, 0.5, 0.7}
    for SIG in 0.3 0.5 0.7; do
        python -u scripts/apf_blend_sg.py --seed $SEED --budget $B --n-eps 20 \
            --mode const --const-sigma $SIG --r-repel 0.6 --alpha-apf 0.5 \
            --out results/apf_ablation.json
    done

    # 3. Threshold switch (σ=1 if u<thresh else 0)
    for TH in 0.5 0.8; do
        python -u scripts/apf_blend_sg.py --seed $SEED --budget $B --n-eps 20 \
            --mode switch --switch-thresh $TH --r-repel 0.6 --alpha-apf 0.5 \
            --out results/apf_ablation.json
    done

    # 4. Projected-budget enforcement: σ=1 if extrapolated final cost > d
    python -u scripts/apf_blend_sg.py --seed $SEED --budget $B --n-eps 20 \
        --mode projected --r-repel 0.6 --alpha-apf 0.5 \
        --out results/apf_ablation.json
done
