#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=oracle_yaw
#SBATCH --output=logs/oracle_yaw_%A_%a.out
#SBATCH --error=logs/oracle_yaw_%A_%a.err
#SBATCH --array=1-3

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

case $SLURM_ARRAY_TASK_ID in
  1) LAYOUT="multi_modal"; WD=268; TAG="mm" ;;
  2) LAYOUT="stag4_5d";    WD=270; TAG="stag4" ;;
  3) LAYOUT="3turb";       WD=270; TAG="3turb" ;;
esac

set -e
pixi run python -u scripts/oracle_yaw_sweep.py \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout "$LAYOUT" --wd "$WD" --ws 9 --ti 0.07 \
    --n-yaw-levels 9 --yaw-max 30 \
    --out-json "results/oracle_yaw_${TAG}.json" \
    --out-fig "results/oracle_yaw_${TAG}.png"
