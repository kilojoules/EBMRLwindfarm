#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:00:00
#SBATCH --job-name=wf_blend
#SBATCH --output=logs/wf_blend_%A_%a.out
#SBATCH --error=logs/wf_blend_%A_%a.err
#SBATCH --array=1-4

set -euo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

pixi run pip install imageio-ffmpeg 2>/dev/null || true
pixi run pip install --force-reinstall --no-deps /scratch/project_465002609/julian/windgym 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r /scratch/project_465002609/julian/windgym/WindGym/* "$SITE/WindGym/" 2>/dev/null || true

SEED=$SLURM_ARRAY_TASK_ID
if [ "$SEED" = "1" ]; then
    CKPT=runs/ebt_sac_windfarm/checkpoints/step_100000.pt
else
    CKPT=runs/ebt_wf_seed${SEED}/checkpoints/step_100000.pt
fi
echo "seed=$SEED CKPT=$CKPT"
[ ! -f "$CKPT" ] && { echo "ckpt not found: $CKPT"; exit 1; }

# Sharpness sweep
for SH in 1.0 3.0 10.0; do
    pixi run python -u scripts/yaw_safe_blend_wf.py \
        --checkpoint "$CKPT" --budget 15 --horizon 200 \
        --sharpness $SH --n-episodes 50 \
        --output results/yaw_safe_blend_seed${SEED}_sh${SH}.json
done
