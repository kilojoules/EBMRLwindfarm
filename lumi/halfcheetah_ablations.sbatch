#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=cheetah_ablations
#SBATCH --output=logs/cheetah_ablations_%j.out
#SBATCH --error=logs/cheetah_ablations_%j.err

# ===========================================================================
# All HalfCheetah ablations for the paper:
# 1. Hard guard vs AC schedule vs both (core scientific question)
# 2. Constant penalty sweep (fair baseline)
# 3. 1/u vs exp schedule comparison (theory validation)
# 4. Budget flexibility (10-75% budgets, same policy)
# 5. Boltzmann response validation (theoretical assumption check)
# ===========================================================================

set -euo pipefail

echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs

pixi run python scripts/halfcheetah_ablations.py \
    --checkpoint checkpoints/sac_cheetah.pt \
    --ablation all \
    --n-episodes 10

echo "Job $SLURM_JOB_ID finished at $(date)"
