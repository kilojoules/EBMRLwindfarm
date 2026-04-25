#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --job-name=baselines
#SBATCH --output=logs/baselines_%A_%a.out
#SBATCH --error=logs/baselines_%A_%a.err
#SBATCH --array=0-5

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

# Battery: 3 baselines × 2 budgets ({10, 25}) at 500k steps from pretrained actor
# Array task → (budget, method) pairs:
#   0: SAUTE d=10        — state augmentation + reward masking
#   1: SAUTE d=25
#   2: SAC-Lag warmup d=10  — finetune with 50k μ-warmup, reward-scale
#   3: SAC-Lag warmup d=25
#   4: SAC-Lag dual-lr 0.05 d=10  — aggressive dual ascent
#   5: SAC-Lag dual-lr 0.05 d=25
case $SLURM_ARRAY_TASK_ID in
  0) python -u scripts/saute_sac_sg.py --budget 10 --total-steps 500000 \
        --seed 1 --out runs/saute_sg_B10 ;;
  1) python -u scripts/saute_sac_sg.py --budget 25 --total-steps 500000 \
        --seed 1 --out runs/saute_sg_B25 ;;
  2) python -u scripts/sac_lag_finetune_sg.py --pretrained-seed 1 --budget 10 \
        --eval-at 0 50000 200000 500000 --mu-warmup 50000 --reward-scale 5.0 \
        --mu-init 0.05 --mu-lr 0.005 --lr 1e-4 \
        --out results/saclag_finetune_warmup_B10.json ;;
  3) python -u scripts/sac_lag_finetune_sg.py --pretrained-seed 1 --budget 25 \
        --eval-at 0 50000 200000 500000 --mu-warmup 50000 --reward-scale 5.0 \
        --mu-init 0.05 --mu-lr 0.005 --lr 1e-4 \
        --out results/saclag_finetune_warmup_B25.json ;;
  4) python -u scripts/sac_lag_finetune_sg.py --pretrained-seed 1 --budget 10 \
        --eval-at 0 50000 200000 500000 --mu-warmup 0 --reward-scale 5.0 \
        --mu-init 0.5 --mu-lr 0.05 --lr 1e-4 \
        --out results/saclag_finetune_aggressive_B10.json ;;
  5) python -u scripts/sac_lag_finetune_sg.py --pretrained-seed 1 --budget 25 \
        --eval-at 0 50000 200000 500000 --mu-warmup 0 --reward-scale 5.0 \
        --mu-init 0.5 --mu-lr 0.05 --lr 1e-4 \
        --out results/saclag_finetune_aggressive_B25.json ;;
esac
