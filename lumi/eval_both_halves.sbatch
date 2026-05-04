#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --job-name=both_halves
#SBATCH --output=logs/both_halves_%A_%a.out
#SBATCH --error=logs/both_halves_%A_%a.err
#SBATCH --array=1-2

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

case $SLURM_ARRAY_TASK_ID in
  1) LAYOUT="stag4_5d";    CKPT="runs/pi_perf_stag4_5d_300k/checkpoints/step_290000.pt" ;;
  2) LAYOUT="multi_modal"; CKPT="runs/pi_perf_mm_300k/checkpoints/step_290000.pt"        ;;
esac

set -e
pixi run python -u scripts/eval_pi_perf_vs_safe.py \
    --checkpoint "$CKPT" \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout "$LAYOUT" --horizon 200 --n-episodes 5 \
    --out-json "results/pi_perf_vs_safe_${LAYOUT}.json"
