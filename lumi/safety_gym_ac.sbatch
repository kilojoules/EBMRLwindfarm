#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=safety_ac
#SBATCH --output=logs/safety_ac_%j.out
#SBATCH --error=logs/safety_ac_%j.err

# ===========================================================================
# Safety Gym AC Budget — ISOLATED environment
#
# Uses its own conda/venv with gymnasium~=0.29 + safety-gymnasium 1.0.0,
# completely separate from the pixi env used for wind farm experiments.
# This avoids the gymnasium>=1.2 conflict.
# ===========================================================================

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

# --- Create isolated Python environment ---
SAFETY_VENV="$PROJECT_DIR/.safety_venv"

if [ ! -d "$SAFETY_VENV" ]; then
    echo "Creating isolated venv for Safety Gym..."
    # Use the pixi Python but create a standalone venv
    $(pixi run which python) -m venv "$SAFETY_VENV"
fi

source "$SAFETY_VENV/bin/activate"

# Install compatible versions (only on first run, idempotent)
pip install --quiet \
    "gymnasium==0.29.1" \
    "safety-gymnasium==1.0.0" \
    "mujoco==2.3.7" \
    "torch==2.5.1" --index-url https://download.pytorch.org/whl/rocm6.2 \
    "numpy<2" \
    2>/dev/null || true

# Verify
python -c "
import gymnasium; print(f'gymnasium={gymnasium.__version__}')
import safety_gymnasium; print(f'safety_gymnasium={safety_gymnasium.__version__}')
import torch; print(f'torch={torch.__version__}, gpu={torch.cuda.is_available()}')
import mujoco; print(f'mujoco={mujoco.__version__}')
"

echo ""
echo "=== Training unconstrained SAC on SafetyPointGoal1-v0 ==="
python "$PROJECT_DIR/scripts/safety_gym_ac_budget.py" \
    --train \
    --env SafetyPointGoal1-v0 \
    --total-timesteps 200000 \
    --checkpoint "$PROJECT_DIR/checkpoints/sac_safety_point.pt" \
    --seed 1

echo ""
echo "=== Comparing methods ==="
python "$PROJECT_DIR/scripts/safety_gym_ac_budget.py" \
    --compare \
    --env SafetyPointGoal1-v0 \
    --checkpoint "$PROJECT_DIR/checkpoints/sac_safety_point.pt" \
    --n-episodes 20 \
    --horizon 1000

deactivate
echo "Job $SLURM_JOB_ID finished at $(date)"
