#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --job-name=pi_perf_vw
#SBATCH --output=logs/pi_perf_vw_%A_%a.out
#SBATCH --error=logs/pi_perf_vw_%A_%a.err
#SBATCH --array=1-2

# ===========================================================================
# Both-halves retry: pi_perf training with VARIABLE wind direction
# (wd in [225, 315] degrees) so wake-steering has a real optimisation target.
# Fixed wd training did not produce pi_perf > pi_safe on power.
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
  1) LAYOUT="stag4_5d";    CONFIG="varwd";              EXP="pi_perf_stag4_5d_vw_300k" ;;
  2) LAYOUT="multi_modal"; CONFIG="multi_modal_varwd";  EXP="pi_perf_mm_vw_300k"        ;;
  *) echo "unknown task"; exit 1 ;;
esac
echo "task=$SLURM_ARRAY_TASK_ID layout=$LAYOUT config=$CONFIG exp=$EXP"

set -e
pixi run python -u ebt_sac_windfarm.py \
    --layouts "$LAYOUT" --config "$CONFIG" \
    --total_timesteps 300000 --num_envs 1 --utd_ratio 2 \
    --ebt_opt_steps_train 5 --ebt_opt_steps_eval 20 --ebt_num_candidates 16 \
    --ebt_langevin_noise 0.05 --ebt_energy_reg 0.05 \
    --learning_starts 2000 --history_length 1 --max_episode_steps 1000 \
    --load_surrogate_type t1_positive_only \
    --viz_every_n_evals 1 \
    --seed 1 --exp_name "$EXP" --no-track
