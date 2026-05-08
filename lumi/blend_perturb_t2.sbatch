#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=blend_t2
#SBATCH --output=logs/blend_t2_%j.out
#SBATCH --error=logs/blend_t2_%j.err

# Per-turbine budget pacing demo: T2 binding, T0+T1 loose.
# pi_perf T2 DEL ~= 21580 unblended; B_T2* = 19500 (~10% below) forces
# AC schedule to engage σ_T2 while σ_T0,σ_T1 stay near 0. pi_safe is the
# +30,+30,+30 setpoint (already verified DEL_safe[T2] < DEL_perf[T2] by 711).
# This is the "single experiment, both halves hold" reviewer ask.

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
    --safe-yaw-deg "30,30,30" \
    --budgets "1000000000,1000000000,19500" \
    --horizon 200 --eta 3.0 \
    --grid-nx 48 --grid-ny 24 \
    --frame-stride 2 --fps 15 \
    --guidance-scale 0.0 \
    --out results/blend_perturb_t2_b1p0.mp4
