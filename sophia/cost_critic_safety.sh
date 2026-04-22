#!/bin/bash
#SBATCH --partition=windq
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=12:00:00
#SBATCH --array=1-5
#SBATCH --job-name=sg_qc
#SBATCH --output=/work/users/juqu/ac_budget/logs/sg_qc_%A_%a.out
#SBATCH --error=/work/users/juqu/ac_budget/logs/sg_qc_%A_%a.err

# Cost critic Q_c for Safety Gym on Sophia (CPU-only)

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "Job $SLURM_JOB_ID seed=$SEED started at $(date)"

PROJECT_DIR=/work/users/juqu/ac_budget
cd "$PROJECT_DIR"
mkdir -p logs data checkpoints results

source /home/juqu/miniconda3/etc/profile.d/conda.sh
conda activate safety_cpu

CKPT="$PROJECT_DIR/checkpoints/sac_safety_point_seed${SEED}.pt"
DATA="$PROJECT_DIR/data/cost_critic_seed${SEED}.npz"
QC="$PROJECT_DIR/checkpoints/cost_critic_seed${SEED}.pt"
RESULTS="$PROJECT_DIR/results/cost_critic_seed${SEED}.json"

# Train SAC if no checkpoint
if [ ! -f "$CKPT" ]; then
    echo "Training SAC (seed=$SEED)..."
    PYTHONUNBUFFERED=1 python scripts/safety_gym_ac_budget.py \
        --train --env SafetyPointGoal1-v0 \
        --total-timesteps 200000 --checkpoint "$CKPT" --seed "$SEED"
fi

# Collect + train Q_c + compare
PYTHONUNBUFFERED=1 python scripts/cost_critic.py \
    --domain safety_gym \
    --checkpoint "$CKPT" \
    --collect --train --compare \
    --n-collect-episodes 200 \
    --n-eval-episodes 20 \
    --horizon 1000 \
    --gamma-c 0.99 \
    --epochs 200 \
    --seed "$SEED" \
    --data-path "$DATA" \
    --qc-path "$QC" \
    --output-json "$RESULTS"

echo "Job $SLURM_JOB_ID seed=$SEED finished at $(date)"
