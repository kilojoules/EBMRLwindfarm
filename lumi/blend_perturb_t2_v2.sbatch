#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=blend_t2v2
#SBATCH --output=logs/blend_t2v2_%j.out
#SBATCH --error=logs/blend_t2v2_%j.err

# v2: pi_safe = +7.5,+7.5,+7.5 (oracle's principled choice; verified DEL_safe
# < DEL_perf on T0+T2). Smaller yaw distance from pi_perf basin (-7..-16)
# means smoother blend. B_T2 = 21000 (between pi_perf 21482 and pi_safe
# 20888) -> partial-engagement regime where AC schedule actually trades off.
#
# v1 with pi_safe=+30 + B=19500 blew up: 37deg yaw distance + tight budget
# forced violent transient when sigma_T2 fired, yielding util=3.34 (much
# worse than either policy alone). v2 fixes both.

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
pixi run python -u scripts/animate_flap_del_blend.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --safe-yaw-deg "7.5,7.5,7.5" \
    --budgets "1000000000,1000000000,21000" \
    --horizon 200 --eta 3.0 \
    --grid-nx 48 --grid-ny 24 \
    --frame-stride 2 --fps 15 \
    --guidance-scale 0.0 \
    --out results/blend_perturb_t2_v2_b1p0.mp4
