#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=sigsw7p5
#SBATCH --output=logs/sigsw7p5_%j.out
#SBATCH --error=logs/sigsw7p5_%j.err

# Cleaner sigma-sweep: pi_safe = +7.5 (oracle sweet spot, DEL ~ 62708 < pi_perf
# DEL ~ 63085). Both endpoints below midpoint -> strict 'above both endpoints'
# demonstration of the hump at sigma where blend yaw crosses 0.
# Finer grid around sigma in [0.3, 0.7] where yaw crossing expected.

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
pixi run python -u scripts/sigma_sweep_test.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --horizon 200 --n-episodes 10 \
    --safe-yaw-deg "7.5,7.5,7.5" \
    --sigma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --out-json results/sigma_sweep_safe_plus7p5.json
