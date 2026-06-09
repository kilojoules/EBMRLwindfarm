#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=sigsw50
#SBATCH --output=logs/sigsw50_%j.out
#SBATCH --error=logs/sigsw50_%j.err

# (i) Larger n_episodes to drive down noise. n=50 instead of n=10.
# Tests whether the wind hump is real but noise-buried vs absent.
# Two configs: pi_safe=+7.5 (low-DEL endpoint, both endpoints below
# midpoint candidate) and pi_safe=+15 (hazardous, blend crosses 0).

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
echo "=== A: pi_safe=+7.5, n=50 ==="
pixi run python -u scripts/sigma_sweep_test.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --horizon 200 --n-episodes 50 \
    --safe-yaw-deg "7.5,7.5,7.5" \
    --sigma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --out-json results/sigma_sweep_safe_plus7p5_n50.json

echo "=== B: pi_safe=+15, n=50 ==="
pixi run python -u scripts/sigma_sweep_test.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --horizon 200 --n-episodes 50 \
    --safe-yaw-deg "15,15,15" \
    --sigma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --out-json results/sigma_sweep_safe_plus15_n50.json
