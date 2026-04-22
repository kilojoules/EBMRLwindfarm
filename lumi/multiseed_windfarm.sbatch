#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=wf_s%a
#SBATCH --output=logs/wf_seed%a_%j.out
#SBATCH --error=logs/wf_seed%a_%j.err
#SBATCH --array=2-5

# 5 seeds total: seed 1 already trained (step_100000.pt)
# This array adds seeds 2-5 in parallel.

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
echo "Job $SLURM_JOB_ID seed=$SEED started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs

pixi run pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>/dev/null || true
pixi run pip install --force-reinstall --no-deps /scratch/project_465002609/julian/windgym 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -rv /scratch/project_465002609/julian/windgym/WindGym/* "$SITE/WindGym/" 2>&1 | tail -3 || true
pixi run python -c "from WindGym import WindFarmEnv; print('WindGym OK:', WindFarmEnv)"

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
    --exp_name "ebt_wf_seed${SEED}" \
    --no-track

echo "Job $SLURM_JOB_ID seed=$SEED finished at $(date)"
