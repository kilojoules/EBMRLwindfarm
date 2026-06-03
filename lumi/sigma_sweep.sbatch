#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=sigsweep
#SBATCH --output=logs/sigsweep_%j.out
#SBATCH --error=logs/sigsweep_%j.err

# Cost-vs-sigma sweep: the experiment that matches the Proposition.
# Fix pi_perf (DEL-aware MM actor at ~-15) + pi_safe at +15 setpoint.
# Sweep sigma in [0,1]. If DEL non-convex on segment, expect hump near
# sigma=0.5 where blend yaw crosses 0.
# Two pi_safe targets:
#  (a) +15: blend yaw range -15 -> +15 spans 0
#  (b) -15 (same direction as perf): control; expect monotone

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e

echo "=== A: pi_safe=+15 (crosses zero) ==="
pixi run python -u scripts/sigma_sweep_test.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --horizon 200 --n-episodes 10 \
    --safe-yaw-deg "15,15,15" \
    --sigma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --out-json results/sigma_sweep_safe_plus15.json

echo "=== B: pi_safe=-25 (extends actor direction) ==="
pixi run python -u scripts/sigma_sweep_test.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --horizon 200 --n-episodes 10 \
    --safe-yaw-deg="-25,-25,-25" \
    --sigma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --out-json results/sigma_sweep_safe_minus25.json
