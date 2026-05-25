#!/bin/bash
#SBATCH --account=project_465002609
#SBATCH --partition=small-g
#SBATCH --gpus-per-node=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=7
#SBATCH --mem=60G
#SBATCH --time=08:00:00
#SBATCH --job-name=std_mm
#SBATCH --output=logs/std_mm_%j.out
#SBATCH --error=logs/std_mm_%j.err

# Ablation: standard (non-DEL-aware) EBT actor on multi_modal, same training
# budget. Separates the dynamic-tracking advantage from the DEL-reward effect.
# Then evaluate both against zero-yaw to isolate DEL-reward contribution.

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
    --total_timesteps 200000 --num_envs 1 --utd_ratio 2 \
    --ebt_opt_steps_train 5 --ebt_opt_steps_eval 20 --ebt_num_candidates 16 \
    --ebt_langevin_noise 0.05 --ebt_energy_reg 0.05 \
    --learning_starts 2000 --history_length 1 --max_episode_steps 1000 \
    --load_surrogate_type t1_positive_only \
    --viz_every_n_evals 1 \
    --seed 1 --exp_name "std_mm_b1p0" --no-track
