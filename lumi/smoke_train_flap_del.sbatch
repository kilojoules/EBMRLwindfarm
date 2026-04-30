#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --job-name=smk_fdb
#SBATCH --output=logs/smk_fdb_%j.out
#SBATCH --error=logs/smk_fdb_%j.err

set -uo pipefail
PROJECT_DIR=/scratch/project_465002609/julian/ac_budget_experiment
cd "$PROJECT_DIR"

WG_DIR=/scratch/project_465002609/julian/windgym
pixi run pip install --force-reinstall --no-deps "$WG_DIR" 2>/dev/null || true
SITE=$(pixi run python -c "import sys; print([p for p in sys.path if 'site-packages' in p][0])")
mkdir -p "$SITE/WindGym"
cp -r "$WG_DIR/WindGym/"* "$SITE/WindGym/" 2>/dev/null || true

set -e
# Smoke test: 3-turbine multi_modal, 5k steps, flap_del_budget surrogate
# wired to per-step rotor-disk flow context. Verifies the env loop hooks
# don't crash and DEL accumulator advances each step.
pixi run python -u ebt_sac_windfarm.py \
    --layouts multi_modal --config multi_modal \
    --total_timesteps 5000 --num_envs 1 --utd_ratio 2 \
    --ebt_opt_steps_train 5 --ebt_opt_steps_eval 20 --ebt_num_candidates 16 \
    --ebt_langevin_noise 0.05 --ebt_energy_reg 0.05 \
    --learning_starts 500 --history_length 1 --max_episode_steps 200 \
    --load_surrogate_type flap_del_budget \
    --flap_del_per_turbine_budgets 155215.2,173727.8,91453.8 \
    --flap_del_horizon_steps 200 \
    --flap_del_risk_aversion 1.0 --load_steepness 10 \
    --seed 1 --exp_name smoke_flap_del_budget --no-track
