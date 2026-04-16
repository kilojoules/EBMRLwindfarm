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
# Safety Gym AC Budget — ISOLATED conda environment
#
# Uses Python 3.10 + gymnasium 0.29.1 + safety-gymnasium 1.0.0
# Completely separate from the pixi env (which uses Python 3.11 / gymnasium 1.2)
# Python 3.10 is required because safety-gymnasium uses mutable numpy defaults
# in dataclasses, which Python 3.11+ rejects.
# ===========================================================================

set -euo pipefail
echo "Job $SLURM_JOB_ID started at $(date)"

PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"
mkdir -p logs checkpoints

module load rocm/6.0.3

# --- Setup conda env (once) ---
if [ ! -d "$CONDA_DIR" ]; then
    echo "Installing miniconda..."
    wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/mc.sh
    bash /tmp/mc.sh -b -p "$CONDA_DIR"
    rm /tmp/mc.sh
fi

export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"

if ! conda env list | grep -q safety_env; then
    echo "Creating Python 3.10 environment..."
    conda config --add channels conda-forge
    conda config --set channel_priority strict
    conda config --remove channels defaults 2>/dev/null || true
    conda create -y -n safety_env python=3.10 numpy=1.24 -c conda-forge
fi

conda activate safety_env

# Install packages (idempotent)
pip install --quiet "gymnasium==0.29.1" "mujoco==2.3.7" "safety-gymnasium==1.0.0" 2>&1 | tail -5
pip install --quiet torch==2.5.1 --index-url https://download.pytorch.org/whl/rocm6.2 2>&1 | tail -3
pip install --quiet pyyaml xmltodict scipy matplotlib 2>&1 | tail -3

# Verify
python -c "
import gymnasium; print(f'gymnasium={gymnasium.__version__}')
import safety_gymnasium; print('safety_gymnasium OK')
import torch; print(f'torch={torch.__version__}, gpu={torch.cuda.is_available()}')
env = safety_gymnasium.make('SafetyPointGoal1-v0')
obs, info = env.reset()
print(f'SafetyPointGoal1-v0 OK, obs={obs.shape}')
env.close()
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
