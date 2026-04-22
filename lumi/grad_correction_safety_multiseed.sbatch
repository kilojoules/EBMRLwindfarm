#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --array=1-5
#SBATCH --job-name=gc_safety
#SBATCH --output=logs/gc_safety_%A_%a.out
#SBATCH --error=logs/gc_safety_%A_%a.err

# Gradient correction on Safety Gym — one job per seed checkpoint
# Depends on safety_gym_multiseed.sbatch completing first

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "Job $SLURM_JOB_ID seed=$SEED started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"
mkdir -p logs

module load rocm/6.0.3

export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

CKPT="$PROJECT_DIR/checkpoints/sac_safety_point_seed${SEED}.pt"

if [ ! -f "$CKPT" ]; then
    echo "ERROR: Checkpoint $CKPT not found. Run safety_gym_multiseed first."
    exit 1
fi

PYTHONUNBUFFERED=1 python "$PROJECT_DIR/scripts/gradient_correction_safety.py" \
    --checkpoint "$CKPT" \
    --env SafetyPointGoal1-v0 \
    --n-episodes 20 \
    --horizon 1000

echo "Job $SLURM_JOB_ID seed=$SEED finished at $(date)"
