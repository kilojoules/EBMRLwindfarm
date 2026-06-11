#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:30:00
#SBATCH --job-name=referee
#SBATCH --output=logs/referee_%j.out
#SBATCH --error=logs/referee_%j.err

# Referee fixes (M1, M6):
# (1) rejection v2 (probes BLENDED action, restores guarantee) vs
#     rejection_endpoint (old, flawed) vs linear vs argmin on the MM
#     direction sweep at sigma=0.7, n=10.
# (2) beta-sensitivity: evaluate b0p5 and b2p0 MM checkpoints vs zero-yaw
#     (M6: 'beta must be chosen per layout' needs more than one beta).

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e

echo "=== 1: rejection v2 vs endpoint vs linear vs argmin, MM direction sweep ==="
pixi run python -u scripts/blend_modes_compare.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --horizon 200 --n-episodes 10 --sigma 0.7 \
    --safe-yaw-grid="-25,-15,-7.5,0,7.5,15,25" \
    --modes "linear,rejection,rejection_endpoint,argmin" \
    --out-json results/blend_modes_rejv2_mm.json

echo "=== 2a: beta=0.5 MM actor vs zero-yaw ==="
pixi run python -u scripts/eval_pi_perf_vs_safe.py \
    --checkpoint runs/del_aware_b0p5/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --horizon 200 --n-episodes 10 \
    --safe-yaw-deg "0,0,0" \
    --out-json results/beta_sens_b0p5_mm.json

echo "=== 2b: beta=2.0 MM actor vs zero-yaw ==="
pixi run python -u scripts/eval_pi_perf_vs_safe.py \
    --checkpoint runs/del_aware_b2p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --horizon 200 --n-episodes 10 \
    --safe-yaw-deg "0,0,0" \
    --out-json results/beta_sens_b2p0_mm.json
