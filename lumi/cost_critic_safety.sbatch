#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=1-5
#SBATCH --job-name=sg_qc
#SBATCH --output=logs/sg_qc_%A_%a.out
#SBATCH --error=logs/sg_qc_%A_%a.err

# Cost critic Q_c for Safety Gym: collect → train → evaluate per seed

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "Job $SLURM_JOB_ID seed=$SEED started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"
mkdir -p logs data checkpoints results

module load rocm/6.0.3

export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

CKPT="$PROJECT_DIR/checkpoints/sac_safety_point_seed${SEED}.pt"
DATA="$PROJECT_DIR/data/cost_critic_seed${SEED}.npz"
QC="$PROJECT_DIR/checkpoints/cost_critic_seed${SEED}.pt"
RESULTS="$PROJECT_DIR/results/cost_critic_seed${SEED}.json"

if [ ! -f "$CKPT" ]; then
    echo "Training SAC first (seed=$SEED)..."
    PYTHONUNBUFFERED=1 python scripts/safety_gym_ac_budget.py \
        --train --env SafetyPointGoal1-v0 \
        --total-timesteps 200000 --checkpoint "$CKPT" --seed "$SEED"
fi

# Collect 200 episodes (200k transitions) for Q_c training
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
