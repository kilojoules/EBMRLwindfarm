#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=sigsws4
#SBATCH --output=logs/sigsws4_%j.out
#SBATCH --error=logs/sigsws4_%j.err

# (ii) Different layout: stag4_5d with DEL-aware actor. Actor yaws POSITIVE
# (mean ~+5, individual yaws [+15, +10, +3, -5]). Hazardous pi_safe NEGATIVE
# so blend yaw crosses 0. Control pi_safe POSITIVE extended.
# n=30 to balance noise vs runtime (stag4 has 4 turbines, slower than 3-turb MM).

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
echo "=== A: stag4 hazardous pi_safe=-15 ==="
pixi run python -u scripts/sigma_sweep_test.py \
    --checkpoint runs/del_aware_b1p0_stag4_5d/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout stag4_5d \
    --horizon 200 --n-episodes 30 \
    --safe-yaw-deg="-15,-15,-15,-15" \
    --sigma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --out-json results/sigma_sweep_stag4_safe_minus15.json

echo "=== B: stag4 control pi_safe=+25 (extends actor direction) ==="
pixi run python -u scripts/sigma_sweep_test.py \
    --checkpoint runs/del_aware_b1p0_stag4_5d/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout stag4_5d \
    --horizon 200 --n-episodes 30 \
    --safe-yaw-deg "25,25,25,25" \
    --sigma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --out-json results/sigma_sweep_stag4_safe_plus25.json

echo "=== C: stag4 control pi_safe=0 ==="
pixi run python -u scripts/sigma_sweep_test.py \
    --checkpoint runs/del_aware_b1p0_stag4_5d/checkpoints/step_200000.pt \
    --bundle checkpoints/teodor_dlc12_torch.pt \
    --layout stag4_5d \
    --horizon 200 --n-episodes 30 \
    --safe-yaw-deg "0,0,0,0" \
    --sigma-grid "0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0" \
    --out-json results/sigma_sweep_stag4_safe_zero.json
