#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --job-name=anim_sg_n
#SBATCH --output=logs/anim_sg_n_%j.out
#SBATCH --error=logs/anim_sg_n_%j.err

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env
pip install imageio-ffmpeg 2>/dev/null || true

# OSMesa software rendering (no DRI/GPU display needed)
conda install -y -c conda-forge mesalib 2>/dev/null || true
export MUJOCO_GL=osmesa
export PYOPENGL_PLATFORM=osmesa
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"

python scripts/animate_safety_gym_native.py --budget 40 --seed 1 \
    --out results/safety_gym_ep_native.mp4 --horizon 500
