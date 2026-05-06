#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=fdb_anim
#SBATCH --output=logs/fdb_anim_%j.out
#SBATCH --error=logs/fdb_anim_%j.err

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

# Need ffmpeg for matplotlib FFMpegWriter
pixi run pip install --quiet imageio-ffmpeg 2>/dev/null || true

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
pixi run python -u scripts/animate_flap_del_blend.py \
    --checkpoint runs/fdb_sweep_b3/checkpoints/step_50000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --budgets 51738.4,57909.3,30484.6 \
    --horizon 200 --eta 3.0 \
    --grid-nx 56 --grid-ny 28 --frame-stride 2 --fps 12 \
    --out results/flap_del_blend_animation.mp4
