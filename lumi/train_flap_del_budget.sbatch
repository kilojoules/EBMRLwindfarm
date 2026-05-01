#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=04:00:00
#SBATCH --job-name=flap_del_b
#SBATCH --output=logs/flap_del_b_%j.out
#SBATCH --error=logs/flap_del_b_%j.err

# ===========================================================================
# 50k-step EBT-SAC training on 3-turbine multi_modal with FlapDELBudget cost.
#
# Per-turbine budgets = 1/3 × baseline cumulative DEL (over T=200 steps),
# so the AC schedule actually binds. Calibration: see
# results/flap_del_budgets_multi_modal.json (mean baseline B_i = 155215,
# 173728, 91454; tightened to ~1/3 for tension).
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

set -e
pixi run python -u ebt_sac_windfarm.py \
    --layouts multi_modal --config multi_modal \
    --total_timesteps 50000 --num_envs 1 --utd_ratio 2 \
    --ebt_opt_steps_train 5 --ebt_opt_steps_eval 20 --ebt_num_candidates 16 \
    --ebt_langevin_noise 0.05 --ebt_energy_reg 0.05 \
    --learning_starts 2000 --history_length 1 --max_episode_steps 200 \
    --load_surrogate_type flap_del_budget \
    --flap_del_per_turbine_budgets 51738.4,57909.3,30484.6 \
    --flap_del_horizon_steps 200 \
    --flap_del_risk_aversion 1.0 --load_steepness 10 \
    --viz_every_n_evals 1 \
    --seed 1 --exp_name ebt_flap_del_budget_mm --no-track
