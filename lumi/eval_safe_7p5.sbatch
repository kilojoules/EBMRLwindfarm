#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --job-name=eval_7p5
#SBATCH --output=logs/eval_7p5_%j.out
#SBATCH --error=logs/eval_7p5_%j.err

# Eval pi_perf vs pi_safe[+7.5,+7.5,+7.5] (oracle's max-pwr-with-all-DEL<zero
# config: only -1% power, all 3 turbines DEL below zero baseline). Tests
# whether oracle prediction holds in WindGym dynamics for the principled-
# pi_safe lemma in the unified framing.

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
pixi run python -u scripts/eval_pi_perf_vs_safe.py \
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout multi_modal --horizon 200 --n-episodes 5 \
    --safe-yaw-deg "7.5,7.5,7.5" \
    --out-json results/pi_perf_vs_safe_del_aware_b1p0_safe7p5.json
