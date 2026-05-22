#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --job-name=sg_hazard
#SBATCH --output=logs/sg_hazard_%j.out
#SBATCH --error=logs/sg_hazard_%j.err

# Safety Gym 2nd-domain hazard test. Sweep APF rotation theta in {0, pi/4, pi/2, 3pi/4, pi}.
# Linear vs rejection vs argmin at sigma=0.7. n=10 episodes per config.
# At theta=0 APF normal (safety). At theta=pi APF anti-safety (toward hazards).
# Hazard claim: linear blend at non-zero theta -> averaged direction grazes hazards
# more than either endpoint.

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

set -e
module load cray-python/3.10.10
SG_PY=/scratch/project_465002609/julian/sg_venv/bin/python
$SG_PY -u scripts/sg_hazard_test.py \
    --checkpoint checkpoints/sac_safety_point_seed1.pt \
    --horizon 1000 --n-episodes 10 --sigma 0.7 \
    --theta-grid="0,0.785,1.571,2.356,3.141" \
    --modes "linear,rejection,argmin" \
    --out-json results/sg_hazard_test.json
