#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --job-name=revision_all
#SBATCH --output=logs/revision_all_%j.out
#SBATCH --error=logs/revision_all_%j.err

# All revision experiments in one job:
# A1. Constant-reward HalfCheetah (no sinusoidal multiplier)
# A2. PID gain sweep + bisection + proportional controller
# A3. Saute RL + hard clip
# A4. 5% budget test
# A6. Re-run baselines for multi-seed consistency

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

echo ""
echo "================================================================"
echo "  A1: Constant-reward HalfCheetah (amplitude=0)"
echo "================================================================"
# Train with constant reward
pixi run python scripts/safety_gym_budget.py \
    --train --env HalfCheetah-v5 \
    --total-timesteps 100000 \
    --checkpoint checkpoints/sac_cheetah_const.pt \
    --reward-amplitude 0.0

# Run ablations with constant reward
pixi run python scripts/halfcheetah_ablations.py \
    --checkpoint checkpoints/sac_cheetah_const.pt \
    --ablation hard_guard flexibility \
    --reward-amplitude 0.0 \
    --n-episodes 10

echo ""
echo "================================================================"
echo "  A2: PID sweep + bisection + proportional baselines"
echo "================================================================"
pixi run python scripts/additional_baselines.py \
    --checkpoint checkpoints/sac_cheetah.pt \
    --n-episodes 10

echo ""
echo "================================================================"
echo "  A3: Saute RL + hard clip"
echo "================================================================"
# Train Saute + clip
pixi run python scripts/saute_rl_baseline.py \
    --train --hard-clip \
    --budget-frac 0.25 \
    --total-timesteps 100000 \
    --checkpoint checkpoints/saute_clip_b25.pt

# Evaluate both Saute variants
pixi run python scripts/saute_rl_baseline.py \
    --eval --checkpoint checkpoints/saute_clip_b25.pt --n-episodes 10

pixi run python scripts/saute_rl_baseline.py \
    --eval --checkpoint checkpoints/saute_cheetah_b25.pt --n-episodes 10

echo ""
echo "================================================================"
echo "  A4: 5% budget (tight budget test)"
echo "================================================================"
pixi run python scripts/halfcheetah_ablations.py \
    --checkpoint checkpoints/sac_cheetah.pt \
    --ablation flexibility \
    --n-episodes 10

echo ""
echo "Job $SLURM_JOB_ID finished at $(date)"
