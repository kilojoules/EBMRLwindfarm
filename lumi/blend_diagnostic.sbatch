#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=blend_diag
#SBATCH --output=logs/blend_diag_%j.out
#SBATCH --error=logs/blend_diag_%j.err

# Diagnostic: 3 budget regimes + yaw-trajectory print at each step.
# v2 with B=21000 blew up (cum=75180, 3.6x pi_safe alone). Need to
# isolate cause. Three runs:
#   (a) B=99999 -> sigma_T2 should stay 0, blend == pi_perf, cum should
#       match eval pi_perf (~21580). Sanity check.
#   (b) B=21300 -> just below pi_perf, mild engagement expected
#   (c) B=21000 + eta=0.5 -> slow schedule, no violent transient

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
COMMON="--checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --safe-yaw-deg 7.5,7.5,7.5 \
    --horizon 200 --grid-nx 32 --grid-ny 16 \
    --frame-stride 5 --fps 10 --guidance-scale 0.0"

echo === a: loose budget (sanity) ===
pixi run python -u scripts/animate_flap_del_blend.py $COMMON \
    --budgets "1000000000,1000000000,99999" --eta 3.0 \
    --out results/blend_diag_a_loose.mp4

echo === b: B=21300 mild ===
pixi run python -u scripts/animate_flap_del_blend.py $COMMON \
    --budgets "1000000000,1000000000,21300" --eta 3.0 \
    --out results/blend_diag_b_mild.mp4

echo === c: B=21000 eta=0.5 slow ===
pixi run python -u scripts/animate_flap_del_blend.py $COMMON \
    --budgets "1000000000,1000000000,21000" --eta 0.5 \
    --out results/blend_diag_c_slow.mp4
