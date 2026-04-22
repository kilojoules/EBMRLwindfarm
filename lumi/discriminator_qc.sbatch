#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --job-name=disc_qc
#SBATCH --output=logs/disc_qc_%j.out
#SBATCH --error=logs/disc_qc_%j.err

set -euo pipefail
echo "Job $SLURM_JOB_ID start $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"
mkdir -p logs results

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env

PYTHONUNBUFFERED=1 python scripts/discriminator_qc.py

echo "Job $SLURM_JOB_ID done $(date)"
