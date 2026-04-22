#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=00:30:00
#SBATCH --job-name=anim_wf
#SBATCH --output=logs/anim_wf_%j.out
#SBATCH --error=logs/anim_wf_%j.err

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

pixi run pip install imageio-ffmpeg 2>/dev/null || true
pixi run pip install --force-reinstall --no-deps /scratch/project_465002609/julian/windgym 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r /scratch/project_465002609/julian/windgym/WindGym/* "$SITE/WindGym/" 2>/dev/null || true

# Find latest wind farm checkpoint
CKPT=$(ls runs/ebt_sac_windfarm*/checkpoints/step_*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then
    CKPT=$(ls runs/*/checkpoints/step_*.pt 2>/dev/null | head -1)
fi
echo "Using checkpoint: $CKPT"

pixi run python scripts/animate_windfarm.py \
    --checkpoint "$CKPT" \
    --budget 5 --eta 5.0 --gs 0.1 \
    --out results/windfarm_ep_B5_tight.mp4 --horizon 200
