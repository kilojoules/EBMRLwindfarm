#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --job-name=sweep_blend
#SBATCH --output=logs/sweep_blend_%j.out
#SBATCH --error=logs/sweep_blend_%j.err

# Sweep blend modes + budgets -> JSON for paper Pareto figure.
# Modes: perf, safe, per_turb (B_T2 grid), uniform (B_total grid)
# 3 eps per config. ~12 configs total. ~10 min each => ~2h.

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
pixi run python -u scripts/sweep_blend_budgets.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --safe-yaw-deg "7.5,7.5,7.5" \
    --horizon 200 --n-episodes 3 --eta 0.5 \
    --t2-budgets "19500,20500,21000,21300,21500,99999" \
    --aggregate-budgets "58000,60000,62000,63000,64000,200000" \
    --out-json results/blend_sweep_b1p0.json
