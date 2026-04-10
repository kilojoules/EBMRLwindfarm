#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=ebt_3turb_budget
#SBATCH --output=logs/ebt_3turb_budget_%j.out
#SBATCH --error=logs/ebt_3turb_budget_%j.err

# ===========================================================================
# Experiment 1: Train EBT-SAC on 3turb, then evaluate with budget surrogate
#
# Train unconstrained, evaluate with NegativeYawBudgetSurrogate post-hoc.
# The budget eval sweep (RA x guidance_scale grid) runs automatically at
# the end of training when load_surrogate_type=neg_yaw_budget.
# ===========================================================================

set -euo pipefail

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"

# --- Environment setup ---
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

cd "$PROJECT_DIR"
mkdir -p logs

# Ensure torch ROCm + WindGym are installed (idempotent, runs on compute node)
pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

# Verify environment
pixi run python -c "import torch; print(f'torch={torch.__version__}, hip={torch.version.hip}, gpu={torch.cuda.is_available()}')"

# --- Training: EBT-SAC on 3-turbine farm ---
# The neg_yaw_budget surrogate type triggers the budget eval sweep at the end
pixi run python ebt_sac_windfarm.py \
    --layouts 3turb \
    --total_timesteps 100000 \
    --seed 1 \
    --dt_sim 1 \
    --dt_env 1 \
    --save_interval 10000 \
    --eval_interval 25000 \
    --num_eval_steps 200 \
    --load_surrogate_type neg_yaw_budget \
    --neg_yaw_budget_hours 0.004167 \
    --neg_yaw_horizon_hours 0.055556 \
    --neg_yaw_risk_aversion 2.0 \
    --neg_yaw_threshold_deg 0.0 \
    --no-track

# Budget hours conversion (with dt_env=1s):
#   budget_steps = neg_yaw_budget_hours * 3600 / dt_env
#   15 budget steps: 0.004167 * 3600 / 1 = 15
#   200 horizon steps: 0.055556 * 3600 / 1 = 200

echo "Job $SLURM_JOB_ID finished at $(date)"
