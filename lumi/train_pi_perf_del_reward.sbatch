#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --job-name=del_aware
#SBATCH --output=logs/del_aware_%A_%a.out
#SBATCH --error=logs/del_aware_%A_%a.err
#SBATCH --array=1-3

# ===========================================================================
# Both-halves experiment, take 2: train pi_perf with DEL-aware reward.
# Oracle sweep showed wake-steering has NO power headroom on multi_modal /
# stag4_5d / 3turb but ~17.8% DEL_max headroom (yaw=+30 minimises DEL).
# So the right framing is: pi_perf = load-minimising actor (wake-steers for
# DEL reduction at power cost). Then AC blend trades power for DEL slack as
# budget tightens.
#
# Three beta levels in array (0.5, 1.0, 2.0) — picks how heavily reward
# punishes DEL. Larger beta -> actor yaws more aggressively.
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
  1) BETA=0.5; TAG="b0p5" ;;
  2) BETA=1.0; TAG="b1p0" ;;
  3) BETA=2.0; TAG="b2p0" ;;
  *) echo "unknown task"; exit 1 ;;
esac
echo "task=$SLURM_ARRAY_TASK_ID beta=$BETA"

set -e
pixi run python -u ebt_sac_windfarm.py \
    --layouts multi_modal --config multi_modal \
    --total_timesteps 200000 --num_envs 1 --utd_ratio 2 \
    --ebt_opt_steps_train 5 --ebt_opt_steps_eval 20 --ebt_num_candidates 16 \
    --ebt_langevin_noise 0.05 --ebt_energy_reg 0.05 \
    --learning_starts 2000 --history_length 1 --max_episode_steps 1000 \
    --load_surrogate_type t1_positive_only \
    --del_aware_reward --del_reward_beta "$BETA" \
    --flap_del_bundle checkpoints/teodor_dlc12_torch.pt \
    --flap_del_ref 648.6 \
    --viz_every_n_evals 1 \
    --seed 1 --exp_name "del_aware_${TAG}" --no-track
