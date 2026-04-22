#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=ebt_3turb_budget
#SBATCH --output=logs/ebt_3turb_budget_%j.out
#SBATCH --error=logs/ebt_3turb_budget_%j.err

# ===========================================================================
# Experiment 1 (v2): Train EBT-SAC on 3turb with variable wind,
# then evaluate with calibrated budget surrogate sweep.
#
# Fixes from v1:
#   - Variable wind (wd 225-315, ws 10-14) so budget allocation matters
#   - Steepness sweep [2, 3, 5] instead of fixed k=10
#   - Guidance scale [0.05, 0.1, 0.5, 1.0] instead of [0.5-10]
#   - Multiple episodes (5) per config for error bars
#   - Multiple budget levels [15, 30, 50, 100]
#   - Baselines: hard-clip + constant penalty ablation
#   - Power ratio metric (constrained/unconstrained)
# ===========================================================================

set -euo pipefail

echo "Job $SLURM_JOB_ID started at $(date)"
echo "Node: $(hostname)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment

module purge
module load LUMI/24.03
module load partition/G
module load rocm/6.0.3

cd "$PROJECT_DIR"
mkdir -p logs

# Ensure deps (idempotent on compute node)
pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

# Verify
pixi run python -c "import torch; print(f'torch={torch.__version__}, gpu={torch.cuda.is_available()}')"

# --- Train EBT-SAC with variable wind conditions ---
# The 'hard' config gives variable wind within each episode:
#   wd: 225-315 deg (wake-aligned range)
#   ws: 10-14 m/s (above cut-in, moderate range)
# Each episode reset samples new conditions, providing diversity.
# The budget eval sweep runs at the end of training.

pixi run python ebt_sac_windfarm.py \
    --layouts 3turb \
    --config hard \
    --total_timesteps 100000 \
    --seed 1 \
    --dt_sim 1 \
    --dt_env 1 \
    --save_interval 10000 \
    --eval_interval 25000 \
    --num_eval_steps 200 \
    --load_surrogate_type neg_yaw_budget \
    --load_steepness 3.0 \
    --neg_yaw_budget_hours 0.004167 \
    --neg_yaw_horizon_hours 0.055556 \
    --neg_yaw_risk_aversion 2.0 \
    --neg_yaw_threshold_deg 0.0 \
    --no-track

# Budget hours conversion (with dt_env=1s):
#   15 budget steps: 0.004167 * 3600 / 1 = 15
#   200 horizon steps: 0.055556 * 3600 / 1 = 200
#
# The budget eval sweep runs automatically at end of training.
# It sweeps: 4 budgets x 3 steepness x 4 gs x 4 RA x 5 episodes
# Plus unconstrained baseline, hard-clip baseline, constant penalty ablation.

echo "Job $SLURM_JOB_ID finished at $(date)"
