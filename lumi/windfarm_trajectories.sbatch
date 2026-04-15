#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=wf_traj
#SBATCH --output=logs/wf_traj_%j.out
#SBATCH --error=logs/wf_traj_%j.err

# Train wind farm + log per-step yaw trajectories for visualization
# Produces results/windfarm_trajectories.json

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs results

pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install /scratch/project_465002609/julian/windgym 2>/dev/null || true

# Train + budget sweep + trajectory logging (all in one)
pixi run python ebt_sac_windfarm.py \
    --layouts 3turb \
    --config hard \
    --total_timesteps 100000 \
    --seed 42 \
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

echo "Job $SLURM_JOB_ID finished at $(date)"
