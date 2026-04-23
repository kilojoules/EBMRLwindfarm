#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --job-name=anim_apf
#SBATCH --output=logs/anim_apf_%j.out
#SBATCH --error=logs/anim_apf_%j.err

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"

module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env
pip install imageio-ffmpeg 2>/dev/null || true

# Challenging but respected: d=40 with η=3
python -u scripts/animate_apf_blend_sg.py \
    --budget 40 --seed 1 --eta 3.0 --horizon 500 \
    --out results/sg_apf_blend_B40.mp4

# Fully safe mode: budget small (agent forced to APF early)
python -u scripts/animate_apf_blend_sg.py \
    --budget 1 --seed 1 --eta 3.0 --horizon 500 \
    --out results/sg_apf_blend_safe.mp4
