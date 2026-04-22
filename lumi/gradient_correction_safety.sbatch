#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=grad_safety
#SBATCH --output=logs/grad_safety_%j.out
#SBATCH --error=logs/grad_safety_%j.err

# Gradient correction vs action scaling on Safety Gym
# Uses the isolated conda env with gymnasium 0.28.1

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"
mkdir -p logs

module load rocm/6.0.3

export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

PYTHONUNBUFFERED=1 python "$PROJECT_DIR/scripts/gradient_correction_safety.py" \
    --checkpoint "$PROJECT_DIR/checkpoints/sac_safety_point.pt" \
    --env SafetyPointGoal1-v0 \
    --n-episodes 20 \
    --horizon 1000

echo "Job $SLURM_JOB_ID finished at $(date)"
