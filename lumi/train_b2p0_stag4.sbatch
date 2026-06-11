#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --job-name=b2stag4
#SBATCH --output=logs/b2stag4_%j.out
#SBATCH --error=logs/b2stag4_%j.err

# Referee M6: does beta=2 DEL-aware reward Pareto-dominate zero-yaw on
# stag4_5d (where beta=1 traded +5-7% DEL for +22% power)? If yes, the
# 'just retrain with the right beta' objection survives and the cross-
# layout motivation paragraph must be rewritten. If no, claim stands.

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
    --layouts stag4_5d --config default \
    --total_timesteps 200000 --num_envs 1 --utd_ratio 2 \
    --ebt_opt_steps_train 5 --ebt_opt_steps_eval 20 --ebt_num_candidates 16 \
    --ebt_langevin_noise 0.05 --ebt_energy_reg 0.05 \
    --learning_starts 2000 --history_length 1 --max_episode_steps 1000 \
    --load_surrogate_type t1_positive_only \
    --del_aware_reward --del_reward_beta 2.0 \
    --flap_del_bundle checkpoints/teodor_dlc12_torch.pt \
    --flap_del_ref 648.6 \
    --viz_every_n_evals 1 \
    --seed 1 --exp_name "del_aware_b2p0_stag4_5d" --no-track
