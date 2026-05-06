#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=qrqc_feas
#SBATCH --output=logs/qrqc_feas_%j.out
#SBATCH --error=logs/qrqc_feas_%j.err

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

pixi run pip install --quiet imageio-ffmpeg 2>/dev/null || true
WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
# B = baseline DEL × T (the calibration zero-yaw budget). Lagrangian planner
# should keep util sub-1 because surrogate's argmin-DEL action (~yaw +30°)
# gives per-step DEL below baseline at TI=0.07.
pixi run python -u scripts/animate_qr_qc_lagrangian.py \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal \
    --budgets 155215.2,173727.8,91453.8 \
    --horizon 200 --eta 3.0 \
    --del-ref 648.6 --p-ref 1500.0 \
    --n-yaw-levels 5 --yaw-max 30 \
    --grid-nx 56 --grid-ny 28 --frame-stride 2 --fps 12 \
    --out results/flap_del_qrqc_feasible.mp4
