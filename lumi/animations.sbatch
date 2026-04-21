#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --job-name=anim
#SBATCH --output=logs/anim_%j.out
#SBATCH --error=logs/anim_%j.err

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
CONDA_DIR="$PROJECT_DIR/.safety_conda"
cd "$PROJECT_DIR"
mkdir -p logs results

# Safety Gym (conda env)
module load rocm/6.0.3
export PATH="$CONDA_DIR/bin:$PATH"
eval "$($CONDA_DIR/bin/conda shell.bash hook)"
conda activate safety_env
pip install imageio imageio-ffmpeg 2>/dev/null || true

echo "=== Safety Gym animation ==="
python scripts/animate_safety_gym.py --budget 40 --seed 1 \
    --out results/safety_gym_ep_B40.mp4 --horizon 500

conda deactivate

# Wind farm (pixi env)
echo "=== Wind farm animation ==="
pixi run python scripts/animate_windfarm.py \
    --budget 15 --eta 5.0 --gs 0.1 \
    --out results/windfarm_ep_B15.mp4 --horizon 200
