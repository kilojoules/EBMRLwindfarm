#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=windfarm_s%a
#SBATCH --output=logs/windfarm_seed%a_%j.out
#SBATCH --error=logs/windfarm_seed%a_%j.err
#SBATCH --array=1-3

# ===========================================================================
# Multi-seed wind farm: train + budget eval sweep for each seed
# Submit with: sbatch lumi/multiseed_windfarm.sbatch
# ===========================================================================

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "Job $SLURM_JOB_ID seed=$SEED started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs

# Ensure deps
pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

pixi run python ebt_sac_windfarm.py \
    --layouts 3turb \
    --config hard \
    --total_timesteps 100000 \
    --seed $SEED \
    --dt_sim 1 \
    --dt_env 1 \
    --save_interval 50000 \
    --eval_interval 50000 \
    --num_eval_steps 200 \
    --load_surrogate_type neg_yaw_budget \
    --load_steepness 3.0 \
    --neg_yaw_budget_hours 0.004167 \
    --neg_yaw_horizon_hours 0.055556 \
    --neg_yaw_risk_aversion 2.0 \
    --neg_yaw_threshold_deg 0.0 \
    --no-track

echo "Job $SLURM_JOB_ID seed=$SEED finished at $(date)"
