# Composable Energy Policies for Wind Farm Control

**Energy-Based Transformer policies** for wind farm yaw control that enable **zero-shot constraint composition** at deployment. Train once for power maximization, then compose arbitrary operational constraints — load limits, travel budgets, per-turbine restrictions — as additive energy terms, with no retraining.

## Key Idea

Each turbine is a **token** in a transformer sequence. The actor learns an energy function E(s,a) over joint turbine configurations. At deployment, constraints compose by energy addition:

```
E_total(s, a) = E_actor(s, a) + Σ λᵢ · E_constraint_i(a)
```

Gradient-based action optimization on the composed energy naturally respects all objectives. Crucially, this doesn't just clip actions to feasible regions — it steers the policy to **genuinely new cooperative optima** where unconstrained turbines adapt to compensate.

## Architecture

1. **Per-turbine tokenization** — Local observations (wind speed, direction, yaw) become token features
2. **Wind-relative positional encoding** — Turbine positions in canonical wind frame (14+ encoding variants)
3. **Energy-based actor** — Explicit energy head E(s,a), actions via gradient descent with self-verification
4. **Variable-size generalization** — Single policy controls farms of any size (3, 9, 25+ turbines)

## Research Direction

See [`planning/paper_plan.md`](planning/paper_plan.md) for the full research plan. See [`CONTEXT.md`](CONTEXT.md) for technical background.

**Core contributions:**
1. Post-hoc constraint composition via energy addition — train for power, deploy with arbitrary constraints
2. Emergent cooperative adaptation — constraining one turbine causes others to cooperatively reorganize
3. Turbines-as-tokens generalization — single policy for any farm size
4. Formal connection to Lagrangian duality — guidance scale λ is the Lagrangian multiplier

## Project Structure

```
├── ebt_sac_windfarm.py           # EBT-SAC training loop (headline method)
├── diffusion_sac_windfarm.py     # Diffusion-SAC training loop (alternative actor)
├── transformer_sac_windfarm.py   # Baseline Gaussian SAC training loop
├── ebt.py                        # EBT actor: energy head + gradient-based action optimization
├── diffusion.py                  # Diffusion actor: DDPM denoiser + classifier guidance
├── networks.py                   # Transformer actor/critic architectures + encoding factories
├── load_surrogates.py            # Differentiable constraint surrogates (6 types)
├── config.py                     # CLI config (tyro dataclass)
├── evaluate.py                   # Evaluation pipeline
├── replay_buffer.py              # Experience replay buffer
├── helpers/
│   ├── agent.py                  # WindFarmAgent inference wrapper
│   ├── constraint_viz.py         # Energy landscape visualization
│   ├── helper_funcs.py           # Checkpoint I/O, coordinate transforms
│   ├── layouts.py                # Farm layout definitions
│   ├── multi_layout_env.py       # Multi-layout training environment
│   └── ...
├── positional_encodings/         # 14+ positional encoding variants
├── profile_encodings/            # Wake profile encoders (Fourier, CNN)
├── planning/                     # Research planning documents
│   └── paper_plan.md             # Paper plan and experiment roadmap
├── papers/                       # Literature review
│   └── PAPERS.md                 # Curated reading list
├── scripts/
│   ├── demo_per_turbine_constraints.py  # Constraint composition demo
│   ├── run_sweep.py              # Hyperparameter sweep runner
│   └── fetch_wandb_results.py    # Wandb experiment results
├── CONTEXT.md                    # Research context and technical background
└── TODO.md                       # Task tracker
```

## Quick Start

```bash
# EBT-SAC training (headline method, requires WindGym)
python ebt_sac_windfarm.py --layouts multi_modal --total_timesteps 100000 --seed 1

# Diffusion-SAC training (alternative actor)
python diffusion_sac_windfarm.py --layouts 3turb --noise_schedule cosine --bc_weight_start 1.0

# Baseline Gaussian SAC training
python transformer_sac_windfarm.py --layouts square_1 --total_timesteps 100000 --seed 1

# Evaluation
python evaluate.py --checkpoint runs/<run>/checkpoints/step_100000.pt --eval_layouts square_1

# Constraint composition demo
python scripts/demo_per_turbine_constraints.py --checkpoint runs/<run>/checkpoints/step_10000.pt

# All config options
python ebt_sac_windfarm.py --help
```

## Constraint Surrogates

Six differentiable constraint surrogates available for post-hoc composition:

| Surrogate | Description |
|-----------|-------------|
| `ReluLoadSurrogate` | Simple ReLU penalty on yaw magnitude |
| `YawThresholdLoadSurrogate` | Quadratic ramp when |yaw| exceeds threshold |
| `ExponentialYawSurrogate` | Exponential wall beyond threshold |
| `PerTurbineYawSurrogate` | Per-turbine heterogeneous yaw limits |
| `YawTravelBudgetSurrogate` | Stateful rolling-window yaw travel budget |
| `PositiveYawT1Surrogate` | Constrains turbine 1 to positive yaw only |

All implement `per_turbine_energy()` (for EBT composition) and `forward()` (for diffusion guidance).

## Dependencies

Core: `torch>=2.0`, `numpy`, `gymnasium`, `tyro`, `wandb`, `matplotlib`, `scipy`
Environment: `py_wake`, `WindGym` (required for training, not for evaluation/development)

## License

MIT License. See [LICENSE](LICENSE) for details.
