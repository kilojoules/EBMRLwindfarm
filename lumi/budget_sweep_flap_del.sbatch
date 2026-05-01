#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=02:30:00
#SBATCH --job-name=fdb_sweep
#SBATCH --output=logs/fdb_sweep_%A_%a.out
#SBATCH --error=logs/fdb_sweep_%A_%a.err
#SBATCH --array=1-4

# ===========================================================================
# Budget-sweep ablation for FlapDELBudget on 3-turb multi_modal.
#
# Baseline calibration (zero yaw, T=200, 5 eps):
#   B_baseline = 155215, 173728, 91454  (turb0, turb1, turb2)  [kNm-step]
#
# Array tasks vary scale of per-turbine budgets:
#   task 1: very tight  B/5
#   task 2: tight       B/3   (was 17976535 — re-run for consistency)
#   task 3: medium      B/2
#   task 4: loose       B    (== baseline; should not bind)
# ===========================================================================

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"
mkdir -p logs

module purge
module load LUMI/24.03 partition/G rocm/6.0.3

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

case $SLURM_ARRAY_TASK_ID in
  1)  LABEL="b5";  BUDGETS="31043.0,34745.6,18290.8"  ;;
  2)  LABEL="b3";  BUDGETS="51738.4,57909.3,30484.6"  ;;
  3)  LABEL="b2";  BUDGETS="77607.6,86863.9,45727.0"  ;;
  4)  LABEL="b1";  BUDGETS="155215.2,173727.8,91453.8";;
  *)  echo "unknown task $SLURM_ARRAY_TASK_ID"; exit 1 ;;
esac
echo "task=$SLURM_ARRAY_TASK_ID label=$LABEL budgets=$BUDGETS"

set -e
pixi run python -u ebt_sac_windfarm.py \
    --layouts multi_modal --config multi_modal \
    --total_timesteps 50000 --num_envs 1 --utd_ratio 2 \
    --ebt_opt_steps_train 5 --ebt_opt_steps_eval 20 --ebt_num_candidates 16 \
    --ebt_langevin_noise 0.05 --ebt_energy_reg 0.05 \
    --learning_starts 2000 --history_length 1 --max_episode_steps 200 \
    --load_surrogate_type flap_del_budget \
    --flap_del_per_turbine_budgets "$BUDGETS" \
    --flap_del_horizon_steps 200 \
    --flap_del_risk_aversion 1.0 --load_steepness 10 \
    --viz_every_n_evals 1 \
    --seed 1 --exp_name "fdb_sweep_${LABEL}" --no-track
