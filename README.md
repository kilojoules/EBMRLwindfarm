# Adaptive Budget Scheduling for Constrained RL Deployment

Code for the paper **"Post-Hoc Budget Constraints for Pre-Trained RL Policies: A Closed-Form Schedule and Its Coupling-Strength Scope"**. The method adds cumulative-budget constraints to a pre-trained RL policy at inference time via a closed-form urgency-ratio schedule $\lambda(t) = \exp(\eta(1/u - 1))$, with zero policy retraining. A single dimensionless quantity — the action-to-cost coupling strength $\kappa = \mathbb{E}\|\nabla_a Q_c\| / \mathbb{E}\|\nabla_s Q_c\|$ — predicts scope *a priori*: wind farm yaw control ($\kappa\!\approx\!0.72$) succeeds, Safety Gymnasium ($\kappa\!\approx\!0.02$) predictably fails.

Paper PDF: `latex_paper/main.pdf`.

## Repository Layout

| Path | Purpose |
|------|---------|
| `transformer_sac_windfarm.py` | SAC training (Transformer actor+critic) |
| `ebt_sac_windfarm.py`         | Energy-Based Transformer SAC training |
| `diffusion_sac_windfarm.py`   | Diffusion-SAC training |
| `ebt.py`                      | EBT actor (explicit energy head) |
| `networks.py`                 | Actor/critic/TQC + encoding factories |
| `config.py`                   | All CLI args (tyro dataclass) |
| `evaluate.py`                 | Evaluation pipeline |
| `load_surrogates.py`          | Post-hoc constraint surrogates (neg-yaw budget, etc.) |
| `scripts/`                    | Budget/AC experiments, analyses, animations |
| `lumi/`                       | SLURM scripts for LUMI (AMD/ROCm) compute |
| `latex_paper/`                | Paper source + figures + compiled PDF |
| `checkpoints/`                | Trained models (not tracked; sync from LUMI) |
| `results/`                    | Eval outputs, figures, animations |

## Install

```bash
# Core deps (CPU/GPU)
pip install torch numpy gymnasium tyro wandb matplotlib scipy
pip install py_wake            # wake-model backend

# Wind-farm environment (required for training/eval)
pip install git+https://github.com/DTUWindEnergy/WindGym.git

# Safety Gymnasium (required for §6.2 experiments)
pip install safety-gymnasium

# Animations
pip install imageio-ffmpeg
```

Python 3.9+. PyTorch 2.0+.

## Reproducing the Paper

### 1. Wind farm: train the unconstrained EBT actor (§6.1)

```bash
python ebt_sac_windfarm.py \
    --layouts 3turb \
    --total_timesteps 100000 \
    --seed 1
```

Checkpoints land in `runs/ebt_sac_windfarm*/checkpoints/step_*.pt`. Allow ~3 GPU-hours per seed on MI250X.

For multi-seed (Table 2, last row):

```bash
sbatch lumi/multiseed_windfarm.sbatch       # 4-seed array
```

### 2. Wind farm: evaluate the AC schedule (Table 2)

```bash
python scripts/ac_eval_single.py \
    --checkpoint runs/ebt_sac_windfarm*/checkpoints/step_100000.pt \
    --budget 15 --eta 5.0 --gs 0.1 --eps 50
```

Writes `results/ac_eval_seed{N}.json`. Reproduce the oracle-constant baseline via `scripts/windfarm_sweep.py` (15-step bisection).

### 3. Wind farm: nonlinear fatigue cost (Table 3)

```bash
python scripts/nonlinear_cost_eval.py \
    --checkpoint <ckpt> --cost-type nonlinear --eta 5.0
```

### 4. Wind farm: learned $Q_c$ (Table 5, §5)

```bash
# Collect 100 frozen-policy episodes + train Q_c
python scripts/train_cost_critic_wf.py --checkpoint <ckpt> --n-eps 100

# Evaluate Q_c-guided AC
python scripts/qc_eval_wf.py --qc-checkpoint checkpoints/windfarm_qc.pt \
    --actor-checkpoint <ckpt> --eta 5 --gs 0.5
```

### 5. Safety Gymnasium: train SAC + $Q_c$, evaluate AC (Table 4)

```bash
# Unconstrained SAC, 5 seeds (1.5 GPU-h per seed)
for s in 1 2 3 4 5; do
    python scripts/sac_safety_point.py --seed $s
done

# Offline Q_c training (MC returns, γ_c = 0.99, ~100 episodes)
for s in 1 2 3 4 5; do
    python scripts/train_cost_critic_sg.py --seed $s
done

# AC evaluation (Table 4)
python scripts/safety_gym_cpred_eval.py \
    --budgets 10 25 40 --seeds 1 2 3 4 5 --n-eps 20
```

### 6. Coupling strength κ (Table 1, Figure 1)

```bash
# Wind farm: κ = 0.72
python scripts/coupling_metric.py --domain wind_farm --n-samples 2000

# Safety Gym: κ = 0.02
python scripts/coupling_metric.py --domain safety_gym --n-samples 2000
```

Results → `results/coupling_metric.json`. Figure 1 is built from two animation snapshots:

```bash
python scripts/animate_windfarm.py --checkpoint <ckpt> --budget 5 --gs 0.1 \
    --out results/windfarm_ep_B5_tight.mp4
python scripts/animate_safety_gym_compare.py --budget 10 --seed 1 \
    --out results/safety_gym_compare_B10.mp4
python scripts/stitch_kappa_figure.py
```

### 7. Pessimism null ablation (Appendix Table 7)

```bash
python scripts/uncertainty_gated_qc.py       # 5-seed × twin × {q=0,1,4} × d∈{10,25,40}
python scripts/discriminator_qc.py           # ensemble σ vs ||Δa||
```

### 8. Compile the paper

```bash
cd latex_paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## LUMI (AMD / ROCm) quickstart

All `lumi/*.sbatch` scripts expect the project under
`/scratch/project_465002609/julian/ac_budget_experiment`. Submit via:

```bash
sbatch lumi/multiseed_windfarm.sbatch     # WF training array (seeds 2-5)
sbatch lumi/multiseed_eval.sbatch         # WF AC eval
sbatch lumi/uncertainty_gated.sbatch      # SG pessimism sweep
sbatch lumi/animation_windfarm.sbatch     # WF animation + snap
sbatch lumi/animation_sg_compare.sbatch   # SG side-by-side with ∇_a arrows
```

WindGym install on LUMI needs a manual copy after pip install (the
pypi `WindGym/__init__.py` is empty); see the top of each sbatch for
the `cp -r /scratch/.../windgym/WindGym/* $SITE/WindGym/` step.

## Key Files Produced

| File | Paper reference |
|------|-----------------|
| `results/coupling_metric.json`             | Table 1, §5 |
| `results/ac_eval_seed*.json`               | Table 2 (multi-seed row) |
| `results/safety_gym_cpred_seed*.json`      | Table 4 |
| `results/uncertainty_gated_qc.json`        | Table 7 (pessimism null) |
| `results/windfarm_ep_B5_tight.mp4`         | Animation of tight-budget WF |
| `results/safety_gym_compare_B10.mp4`       | Side-by-side uncon vs AC on SG |
| `latex_paper/figures/fig_kappa_diagnostic.png` | Figure 1 |

## Architecture (wind farm)

Turbines are **tokens** in a transformer sequence. Per-turbine observations → shared encoder → per-turbine actions (EBT: gradient descent on learned energy; SAC: Gaussian head). Positional encoding variants in `positional_encodings/`; profile encoders in `profile_encodings/`. Wind-relative frame: turbine positions are rotated to a canonical frame (wind from 270°) before encoding.

## License

MIT. See `LICENSE`.

## Citation

```bibtex
@misc{quick2026adaptive,
  title  = {Post-Hoc Budget Constraints for Pre-Trained RL Policies:
            A Closed-Form Schedule and Its Coupling-Strength Scope},
  author = {Quick, Julian and Nilsen, Marcus Binder},
  year   = {2026},
  note   = {Anonymized for NeurIPS review}
}
```
