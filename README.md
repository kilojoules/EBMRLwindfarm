# EBM + RL + Transformers for Wind Farm Control

Exploring **Energy-Based Models (EBMs)** combined with **Reinforcement Learning** and **Transformer architectures** for wind farm yaw control. Built on top of a proven Transformer-SAC codebase that achieves zero-shot generalization across farm layouts.

## Background

This repository originated as a Transformer-based SAC agent for wind farm yaw control (see `CONTEXT.md` for research context). The core insight: treating turbines as tokens in a transformer sequence enables a single policy to generalize across farm sizes and layouts. We're now extending this with Energy-Based Models to explore learned energy landscapes over joint turbine configurations.

## Project Structure

```
├── transformer_sac_windfarm.py   # Main SAC training loop
├── networks.py                   # Transformer actor/critic architectures
├── config.py                     # CLI config (tyro dataclass)
├── evaluate.py                   # Evaluation pipeline
├── replay_buffer.py              # Experience replay buffer
├── helpers/
│   ├── agent.py                  # WindFarmAgent inference wrapper
│   ├── env_configs.py            # Environment configuration presets
│   ├── eval_utils.py             # Evaluation helpers
│   ├── helper_funcs.py           # Checkpoint I/O, coordinate transforms
│   ├── layouts.py                # Farm layout definitions
│   ├── multi_layout_env.py       # Multi-layout training environment
│   ├── geometric_profiles.py     # Geometry-based wake profiles
│   ├── receptivity_profiles.py   # PyWake-based turbine profiles
│   ├── training_utils.py         # Training utilities
│   └── data_loader.py            # Data loading utilities
├── positional_encodings/         # Positional encoding variants
│   ├── _absolute.py              # Absolute MLP, sinusoidal 2D
│   ├── _bias.py                  # Relative attention bias (MLP, shared)
│   ├── _gat.py                   # Graph attention-style encodings
│   ├── _rope.py                  # Rotary position embeddings
│   └── _spatial.py               # Spatial/neighborhood encodings
├── profile_encodings/            # Wake profile encoders
│   ├── _fourier.py               # Fourier-based encoding
│   ├── _cnn.py                   # CNN-based encoding
│   └── _blocks.py                # Shared building blocks
├── CONTEXT.md                    # Research context and goals
├── TODO.md                       # Research task tracker
└── requirements.txt              # Python dependencies
```

## Key Architecture

Each turbine is a **token** in a transformer sequence:

1. **Per-turbine tokenization** — Local observations (wind speed, direction, yaw) become token features
2. **Wind-relative positional encoding** — Turbine positions rotated to canonical wind frame, then encoded (MLP, sinusoidal, RoPE, ALiBi, relative bias, etc.)
3. **Wake profile conditioning** — Optional Fourier/CNN-encoded receptivity/influence profiles for layout-aware context
4. **Permutation-equivariant output** — Shared actor/critic heads produce actions for all turbines simultaneously

The transformer handles variable-length sequences, so a single policy controls farms of any size.

## Quick Start

```bash
# Training
python transformer_sac_windfarm.py \
    --layouts square_1 \
    --total_timesteps 100000 \
    --pos_encoding_type absolute_mlp \
    --seed 1

# Evaluation
python evaluate.py \
    --checkpoint runs/<run_name>/checkpoints/step_100000.pt \
    --eval_layouts square_1
```

## License

MIT License. See [LICENSE](LICENSE) for details.
