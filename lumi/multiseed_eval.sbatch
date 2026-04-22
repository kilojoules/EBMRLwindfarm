#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:00:00
#SBATCH --job-name=wf_eval%a
#SBATCH --output=logs/wf_eval_s%a_%j.out
#SBATCH --error=logs/wf_eval_s%a_%j.err
#SBATCH --array=2-4

set -euo pipefail
SEED=$SLURM_ARRAY_TASK_ID
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs results

pixi run pip install --force-reinstall --no-deps /scratch/project_465002609/julian/windgym 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r /scratch/project_465002609/julian/windgym/WindGym/* "$SITE/WindGym/" 2>/dev/null || true

CKPT="runs/ebt_wf_seed${SEED}/checkpoints/step_100000.pt"

PYTHONUNBUFFERED=1 pixi run python scripts/ac_eval_single.py \
    --checkpoint "$CKPT" \
    --budget 15 --eta 5.0 --gs 0.1 --k 2.0 \
    --n-episodes 50 \
    --output "results/ac_eval_seed${SEED}.json"
