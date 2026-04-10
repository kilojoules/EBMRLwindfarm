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
module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

# Activate conda environment (adjust path as needed)
source /scratch/project_465002609/julian/miniconda3/bin/activate
conda activate windfarm

# --- Working directory ---
cd /scratch/project_465002609/julian/ac_budget_experiment
mkdir -p logs

# --- Training: EBT-SAC on 3-turbine farm ---
# Train unconstrained (the budget surrogate only activates at eval time)
# The neg_yaw_budget surrogate type triggers the budget eval sweep at the end
python ebt_sac_windfarm.py \
    --layouts 3turb \
    --total_timesteps 100000 \
    --seed 1 \
    --save_interval 10000 \
    --eval_interval 25000 \
    --num_eval_steps 200 \
    --load_surrogate_type neg_yaw_budget \
    --neg_yaw_budget_hours 0.041667 \
    --neg_yaw_horizon_hours 0.555556 \
    --neg_yaw_risk_aversion 2.0 \
    --neg_yaw_threshold_deg 0.0 \
    --track False

# Budget hours conversion (with dt_env=10s):
#   budget_steps = neg_yaw_budget_hours * 3600 / dt_env
#   15 budget steps: 0.041667 * 3600 / 10 = 15
#   200 horizon steps: 0.555556 * 3600 / 10 = 200

echo "Job $SLURM_JOB_ID finished at $(date)"
