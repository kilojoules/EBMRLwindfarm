#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=blend_cmp
#SBATCH --output=logs/blend_cmp_%j.out
#SBATCH --error=logs/blend_cmp_%j.err

# Headline experiment: linear vs rejection blend at sigma=0.7, sweep pi_safe
# direction. Linear should show U-shape DEL bulge near yaw=0 (alignment
# hazard); rejection should match pi_perf when safe move worsens DEL.

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
pixi run python -u scripts/blend_modes_compare.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --horizon 200 --n-episodes 10 --sigma 0.7 \
    --safe-yaw-grid="-25,-15,-7.5,0,7.5,15,25" \
    --modes "linear,rejection,argmin" \
    --out-json results/blend_modes_compare_3way.json
