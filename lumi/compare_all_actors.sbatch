#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=cmp_all
#SBATCH --output=logs/cmp_all_%j.out
#SBATCH --error=logs/cmp_all_%j.err

# Comprehensive comparison for pivot question:
#   (1) del_aware_b1p0 (multi_modal) vs zero-yaw vs blend
#   (2) std_mm_b1p0    (multi_modal) vs zero-yaw vs DEL-aware
#   (3) del_aware_b1p0_stag4_5d vs zero-yaw vs blend
#
# Tests:
#   A. Does DEL-aware advantage hold cross-layout? (cmp 1 vs 3)
#   B. Is DEL-aware reward redundant given tracking? (cmp 1 vs 2 on MM)
#   C. Where does blend matter? (lhs of pivot question)

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e

echo "=== multi_modal: DEL-aware vs zero ==="
pixi run python -u scripts/eval_pi_perf_vs_safe.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --horizon 200 --n-episodes 10 \
    --safe-yaw-deg "0,0,0" \
    --out-json results/cmp_mm_delaware_vs_zero.json

echo "=== multi_modal: STANDARD vs zero ==="
pixi run python -u scripts/eval_pi_perf_vs_safe.py \
    --checkpoint runs/std_mm_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --horizon 200 --n-episodes 10 \
    --safe-yaw-deg "0,0,0" \
    --out-json results/cmp_mm_std_vs_zero.json

echo "=== stag4_5d: DEL-aware vs zero ==="
pixi run python -u scripts/eval_pi_perf_vs_safe.py \
    --checkpoint runs/del_aware_b1p0_stag4_5d/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout stag4_5d --horizon 200 --n-episodes 10 \
    --safe-yaw-deg "0,0,0,0" \
    --out-json results/cmp_stag4_delaware_vs_zero.json

echo "=== blend sweep on multi_modal DEL-aware (existing) ==="
pixi run python -u scripts/blend_modes_compare.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --horizon 200 --n-episodes 10 --sigma 0.7 \
    --safe-yaw-grid="-25,-15,-7.5,0,7.5,15,25" \
    --modes "linear,rejection,argmin" \
    --out-json results/blend_modes_b1p0_mm_n10.json

echo "=== blend sweep on multi_modal STANDARD ==="
pixi run python -u scripts/blend_modes_compare.py \
    --checkpoint runs/std_mm_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --horizon 200 --n-episodes 10 --sigma 0.7 \
    --safe-yaw-grid="-25,-15,-7.5,0,7.5,15,25" \
    --modes "linear,rejection,argmin" \
    --out-json results/blend_modes_std_mm_n10.json

echo "=== blend sweep on stag4_5d DEL-aware ==="
pixi run python -u scripts/blend_modes_compare.py \
    --checkpoint runs/del_aware_b1p0_stag4_5d/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout stag4_5d --horizon 200 --n-episodes 10 --sigma 0.7 \
    --safe-yaw-grid="-25,-15,-7.5,0,7.5,15,25" \
    --modes "linear,rejection,argmin" \
    --out-json results/blend_modes_stag4_n10.json
