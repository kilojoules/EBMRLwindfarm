#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --array=1-5
#SBATCH --job-name=sg_cpred
#SBATCH --output=logs/sg_cpred_%A_%a.out
#SBATCH --error=logs/sg_cpred_%A_%a.err

# Full pipeline per seed: collect data → train cost predictor → evaluate
# Uses isolated conda env with gymnasium 0.28.1

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
DATA="$PROJECT_DIR/data/safety_gym_seed${SEED}.npz"
PRED="$PROJECT_DIR/checkpoints/cost_predictor_seed${SEED}.pt"
RESULTS="$PROJECT_DIR/results/safety_gym_cpred_seed${SEED}.json"

if [ ! -f "$CKPT" ]; then
    echo "Checkpoint $CKPT not found. Training first..."
    PYTHONUNBUFFERED=1 python scripts/safety_gym_ac_budget.py \
        --train --env SafetyPointGoal1-v0 \
        --total-timesteps 200000 --checkpoint "$CKPT" --seed "$SEED"
fi

# Step 1: Collect transitions
echo "=== Collecting transitions (seed=$SEED) ==="
PYTHONUNBUFFERED=1 python scripts/safety_gym_cost_predictor.py \
    --collect \
    --env SafetyPointGoal1-v0 \
    --checkpoint "$CKPT" \
    --n-collect-episodes 100 \
    --horizon 1000 \
    --data-path "$DATA" \
    --seed "$SEED"

# Step 2: Train cost predictor
echo "=== Training cost predictor (seed=$SEED) ==="
PYTHONUNBUFFERED=1 python scripts/safety_gym_cost_predictor.py \
    --train \
    --data-path "$DATA" \
    --predictor-path "$PRED" \
    --seed "$SEED"

# Step 3: Compare methods
echo "=== Running comparison (seed=$SEED) ==="
PYTHONUNBUFFERED=1 python scripts/safety_gym_cost_predictor.py \
    --compare \
    --env SafetyPointGoal1-v0 \
    --checkpoint "$CKPT" \
    --predictor-path "$PRED" \
    --n-eval-episodes 20 \
    --horizon 1000 \
    --output-json "$RESULTS" \
    --seed "$SEED"

echo "Job $SLURM_JOB_ID seed=$SEED finished at $(date)"
