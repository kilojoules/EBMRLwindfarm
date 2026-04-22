#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=06:00:00
#SBATCH --job-name=safety_ac
#SBATCH --output=logs/safety_ac_%j.out
#SBATCH --error=logs/safety_ac_%j.err

# ===========================================================================
# Safety Gym AC Budget — ISOLATED conda env (Python 3.10)
# ===========================================================================

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

module load rocm/6.0.3

export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

# Create env with Python 3.10 + numpy 1.23 (if not exists)
if ! conda env list | grep -q safety_env; then
    echo "Creating Python 3.10 env..."
    conda create -y -n safety_env python=3.10 "numpy=1.23.5" --override-channels -c conda-forge
fi

conda activate safety_env

# Force numpy version (conda sometimes overrides)
pip install "numpy==1.23.5" --force-reinstall --quiet 2>&1 | tail -1

# Install all deps from requirements file
pip install -r "$PROJECT_DIR/lumi/safety_requirements.txt" --quiet 2>&1 | tail -5

# Torch ROCm separately (different index)
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 --quiet 2>&1 | tail -3

# Verify
python -c "
import numpy; print(f'numpy={numpy.__version__}')
import gymnasium; print(f'gymnasium={gymnasium.__version__}')
import safety_gymnasium; print('safety_gymnasium OK')
import torch; print(f'torch={torch.__version__}, gpu={torch.cuda.is_available()}')
env = safety_gymnasium.make('SafetyPointGoal1-v0')
obs, info = env.reset()
print(f'SafetyPointGoal1-v0: obs={obs.shape}')
env.close()
print('ALL OK')
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

echo "Job $SLURM_JOB_ID finished at $(date)"
