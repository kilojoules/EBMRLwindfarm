#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --array=1-5
#SBATCH --job-name=sg_multi
#SBATCH --output=logs/sg_multi_%A_%a.out
#SBATCH --error=logs/sg_multi_%A_%a.err

# Safety Gym multi-seed: train + compare for each seed
# Uses isolated conda env with gymnasium 0.28.1

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "Job $SLURM_JOB_ID seed=$SEED started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"
mkdir -p logs results

module load rocm/6.0.3

export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

CKPT="$PROJECT_DIR/checkpoints/sac_safety_point_seed${SEED}.pt"

# Train
PYTHONUNBUFFERED=1 python "$PROJECT_DIR/scripts/safety_gym_ac_budget.py" \
    --train \
    --env SafetyPointGoal1-v0 \
    --total-timesteps 200000 \
    --checkpoint "$CKPT" \
    --seed "$SEED"

# Compare
PYTHONUNBUFFERED=1 python "$PROJECT_DIR/scripts/safety_gym_ac_budget.py" \
    --compare \
    --env SafetyPointGoal1-v0 \
    --checkpoint "$CKPT" \
    --n-episodes 20 \
    --horizon 1000 \
    --seed "$SEED" \
    --output-json "$PROJECT_DIR/results/safety_gym_seed_${SEED}.json"

echo "Job $SLURM_JOB_ID seed=$SEED finished at $(date)"
