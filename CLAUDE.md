# CLAUDE.md — Project Guide for Claude Code

## General notes
- Always work in a new branch when implementing new features

## Project Overview

**Composable Energy Policies for Wind Farm Control.** Energy-Based Transformer policies that enable zero-shot constraint composition at deployment — train once for power maximization, then compose arbitrary operational constraints as additive energy terms, with no retraining. The transformer backbone treats each turbine as a token, enabling a single policy to generalize across farm sizes.

**Current status (2026-04-14):** ✅ **Proof of concept achieved.** The central claim (emergent cooperative adaptation under post-hoc constraint composition) has been validated on the 3-turbine `multi_modal` layout. Remaining work is reproducibility (seed variance), baseline comparisons, scaling experiments, and writing.

**Headline winning config:** `ebt_150k_nodroq_reg05` — see [`EXPERIMENTS.md`](EXPERIMENTS.md) for full details and reproduction command.

**Reading order for picking this work up later:**
1. [`EXPERIMENTS.md`](EXPERIMENTS.md) — Full experimental log with all findings, winning configs, gotchas
2. [`TODO.md`](TODO.md) — Live task list; what's done and what's next
3. [`planning/paper_plan.md`](planning/paper_plan.md) — Paper framing, abstract draft, baselines
4. [`CONTEXT.md`](CONTEXT.md) — Technical background on EBT and SAC↔EBM connection

**Environment dependency:** Training requires `WindGym` (open-source wind farm gym). The environment is NOT included in this repo. Evaluation and network development can be done standalone.

## Architecture

The core idea: **turbines are tokens**. A transformer processes variable-length sequences of turbine observations, enabling a single policy to control any farm size.

- **Actor/Critic** (`networks.py`): Transformer encoder → per-turbine embeddings. Actor outputs per-turbine yaw actions, critic mean-pools embeddings then outputs a single farm-level Q-value.
- **Positional encodings** (`positional_encodings/`): 14+ variants — absolute (MLP, sinusoidal, polar), relative bias (MLP, ALiBi, directional), RoPE, spatial, GAT-based. Selected via `--pos_encoding_type` string.
- **Profile encodings** (`profile_encodings/`): CNN/Fourier encoders for wake receptivity/influence profiles. Selected via `--profile_encoding_type` string.
- **Config** (`config.py`): Single `@dataclass Args` parsed by `tyro`. All hyperparameters here.

## Code Conventions

- **Config pattern:** All hyperparameters in `config.py` as a `tyro` dataclass. Never use argparse.
- **Factory pattern for encodings:** `create_positional_encoding()` and `create_profile_encoder()` in `networks.py` map string names to classes.
- **Type hints:** Used throughout (`typing.Optional`, `Tuple`, `List`, etc.)
- **Module organization:** `positional_encodings/` and `profile_encodings/` use `_prefixed.py` private modules re-exported via `__init__.py`.
- **Lazy imports:** Heavy dependencies (PyWake, scipy) are imported lazily in `helpers/__init__.py`.
- **Wind-relative coordinates:** Turbine positions are always transformed to a wind-relative frame before encoding (wind from 270°).

## Critical training gotchas (learned the hard way)

See [`EXPERIMENTS.md`](EXPERIMENTS.md) "Training gotchas" section for full details. Short version:

- **Do NOT apply SAC entropy framework to EBT naively** — energies are unnormalized, scale is 200× wrong, training diverges.
- **`ebt_energy_reg=0.05` is the minimum safe value.** `0.0` (config default) and `0.01` allow landscape collapse.
- **`max_episode_steps≥500`, ideally 1000.** Default 100 is too short for yaw convergence (0.5°/step actuator).
- **Use `--pos_encoding_type relative_mlp`.** Setting it to `None` breaks constrained cooperative behavior even with profile encoder.
- **Avoid `--use_droq`** — currently a net negative on 3-turbine layouts; causes landscape collapse unless paired with `ebt_energy_reg≥0.05` AND accept rigid constraint response.
- **Avoid `--shuffle_turbs`** on small farms — hurts convergence. Retest on 9+.
- **Watch `debug/energy_mean` in wandb.** Healthy [-20, +20]. Past -50 = collapsing.

## Key Files

| File | Purpose |
|------|---------|
| `EXPERIMENTS.md` | **Full experimental log** — all runs, findings, winning configs, gotchas |
| `TODO.md` | **Live task list** — what's done, what's left before NeurIPS submission |
| `planning/paper_plan.md` | Research plan, paper framing, abstract draft, baseline comparisons |
| `CONTEXT.md` | Technical background on EBT and SAC↔EBM connection |
| `ebt_sac_windfarm.py` | EBT-SAC training loop — headline method |
| `ebt.py` | EBT actor: explicit energy head, gradient-descent actions, per-turbine energy composition |
| `load_surrogates.py` | 7 differentiable constraint surrogates for post-hoc composition (includes Quadratic/Linear T1 variants) |
| `diffusion_sac_windfarm.py` | Diffusion-SAC training loop (alternative actor) |
| `diffusion.py` | Diffusion actor: DDPM denoiser + classifier guidance |
| `transformer_sac_windfarm.py` | Baseline Gaussian SAC training loop |
| `networks.py` | Transformer actor/critic architectures + encoding factories (~45KB). `TransformerCritic` has DroQ support in Q-head (lines 955-963). |
| `config.py` | All CLI args as tyro dataclass. `max_episode_steps=100` default. |
| `evaluate.py` | Evaluation pipeline |
| `scripts/evaluate_constraints.py` | **Comprehensive constraint sweep tool** — zero-init, steady-state averaging, xarray output. Use for all post-training evaluation. |
| `replay_buffer.py` | Experience replay with variable-size turbine sequences |
| `helpers/agent.py` | `WindFarmAgent` — wraps actor for inference |
| `helpers/constraint_viz.py` | Energy landscape visualization (yaw trajectory, local energy landscape, yaw-vs-λ, power-vs-λ) |
| `helpers/multi_layout_env.py` | Multi-layout env wrapper (trains on diverse farms). Handles `shuffle_turbs`. |
| `helpers/geometric_profiles.py` | Vectorized geometric wake profile computation (fast, ~5ms) |
| `helpers/helper_funcs.py` | Checkpoint save/load, coordinate transforms |
| `helpers/layouts.py` | Farm layout definitions (turbine x,y positions) |
| `profile_encodings/_fourier.py` | `FourierProfileEncoder` — FFT-based harmonic decomposition of wake profiles |
| `test.ipynb` | **PyWake brute-force ground truth** — cell 2 computes the true unconstrained/constrained optima |
| `scripts/demo_per_turbine_constraints.py` | Demo per-turbine constraints + travel budget |
| `scripts/find_constraint_coupling.py` | Search for layouts with coupled constraint response (produced `multi_modal` layout) |
| `scripts/fetch_wandb_results.py` | Fetch and plot wandb experiment results |
| `scripts/run_sweep.py` | Run hyperparameter sweep experiments |

## Common Commands

```bash
# EBT-SAC training — current best config (R2 from 2026-04-14)
python ebt_sac_windfarm.py \
    --layouts multi_modal --config multi_modal \
    --total_timesteps 150000 --utd_ratio 2 \
    --ebt_opt_steps_train 5 --ebt_opt_steps_eval 20 --ebt_num_candidates 16 \
    --ebt_langevin_noise 0.05 --ebt_energy_reg 0.05 \
    --learning_starts 2000 --history_length 1 --max_episode_steps 1000 \
    --load_surrogate_type t1_positive_only --viz_every_n_evals 1 \
    --seed 1 --exp_name ebt_150k_nodroq_reg05 --track

# Diffusion-SAC training (alternative actor, not headline method)
python diffusion_sac_windfarm.py --layouts 3turb --noise_schedule cosine --bc_weight_start 1.0

# Baseline Gaussian SAC training
python transformer_sac_windfarm.py --layouts square_1 --total_timesteps 100000 --seed 1

# Comprehensive constraint sweep evaluation (use this, not evaluate.py)
python scripts/evaluate_constraints.py \
    --checkpoint runs/<run>/checkpoints/step_150000.pt \
    --output-dir results/eval_<run> \
    --constraint-types t1_positive_only \
    --steepness-values 6.0 \
    --lambdas 0.0,0.5,1.0,2.0,5.0,10.0 \
    --yaw-init zeros --steady-state-steps 200 --num-steps 300 --num-episodes 3

# Quick single-checkpoint evaluation
python evaluate.py --checkpoint runs/<run>/checkpoints/step_150000.pt --eval_layouts multi_modal

# Constraint composition demo (diffusion actor)
python scripts/demo_per_turbine_constraints.py --checkpoint runs/<run>/checkpoints/step_10000.pt

# Fetch wandb results
python scripts/fetch_wandb_results.py --filter "ebt_150k"

# All config options
python ebt_sac_windfarm.py --help
```

## Reference ground truth

From `test.ipynb` (PyWake brute-force at ws=9, wd=268, TI=0.07):
- **Unconstrained optimum:** `[-16.0°, -17.3°, 0.0°]` → 9.71 MW
- **Constrained (T0 ≥ 0):** `[+22.7°, -9.3°, 0.0°]` → 9.39 MW (3.3% power loss)
- **Kill criterion:** T0 must flip positive AND T1 must measurably shift toward -9.3° under constraint

## Dependencies

Core: `torch>=2.0`, `numpy`, `gymnasium`, `tyro`, `wandb`, `matplotlib`, `scipy`
Environment: `py_wake`, `WindGym`

