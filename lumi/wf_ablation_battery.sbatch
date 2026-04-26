#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30G
#SBATCH --time=01:30:00
#SBATCH --job-name=wf_abl
#SBATCH --output=logs/wf_abl_%A_%a.out
#SBATCH --error=logs/wf_abl_%A_%a.err
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

OUT_PREFIX=results/wf_ablation_seed${SEED}

# Pure π_safe
pixi run python -u scripts/yaw_safe_blend_wf.py --checkpoint "$CKPT" \
    --budget 15 --horizon 200 --mode pure_safe --n-episodes 50 \
    --output ${OUT_PREFIX}_pure_safe.json

# Constant σ
for SIG in 0.3 0.5 0.7; do
    pixi run python -u scripts/yaw_safe_blend_wf.py --checkpoint "$CKPT" \
        --budget 15 --horizon 200 --mode const --const-sigma $SIG --n-episodes 50 \
        --output ${OUT_PREFIX}_const${SIG}.json
done

# Threshold switch
for TH in 0.5 0.8; do
    pixi run python -u scripts/yaw_safe_blend_wf.py --checkpoint "$CKPT" \
        --budget 15 --horizon 200 --mode switch --switch-thresh $TH --n-episodes 50 \
        --output ${OUT_PREFIX}_switch${TH}.json
done

# Projected
pixi run python -u scripts/yaw_safe_blend_wf.py --checkpoint "$CKPT" \
    --budget 15 --horizon 200 --mode projected --n-episodes 50 \
    --output ${OUT_PREFIX}_projected.json
