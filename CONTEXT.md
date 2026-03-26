# Research Context: EBM + RL + Transformers for Wind Farm Control

## What are Energy-Based Models (EBMs)?

Energy-Based Models assign a scalar energy to each configuration of variables. Low energy = high compatibility/probability. Unlike explicit density models, EBMs don't need to normalize over the full space — they just need to assign *relative* energies. This makes them flexible for modeling complex, multimodal distributions over high-dimensional spaces.

Key properties:
- **Unnormalized**: E(x) is a scalar; no partition function needed for many uses
- **Multimodal**: Can represent multiple modes without mode collapse
- **Composable**: Energies add — E_total(x) = E_1(x) + E_2(x) enables compositional reasoning
- **Implicit**: Define "what's good" rather than "how to generate it"

## Why Combine EBMs with RL?

Standard RL learns a policy π(a|s) that maps states to action distributions. EBMs offer an alternative: learn an energy function E(s, a) over state-action pairs, then derive actions by minimizing energy.

Potential advantages for wind farm control:
1. **Multimodal actions** — Multiple good yaw configurations may exist for a given wind condition. EBMs naturally represent this without mode collapse (unlike Gaussian policies).
2. **Compositional objectives** — Combine separate energy terms (power, fatigue, wake, turbulence) by addition. Add/remove objectives without retraining from scratch.
3. **Implicit planning** — Energy minimization over action sequences enables implicit lookahead without explicit model-based planning.
4. **Uncertainty** — Energy landscape curvature provides natural uncertainty estimates.

## How the Existing Infrastructure Fits

The current codebase already solves several hard problems that transfer directly:

| Component | EBM Role |
|-----------|----------|
| Transformer encoder | Backbone for E(s,a) — processes variable-length turbine sequences |
| Positional encodings | Encode spatial structure of turbine layouts |
| Profile encodings | Encode wake interaction patterns |
| Wind-relative frame | Canonical coordinate system for invariant energy functions |
| Replay buffer | Off-policy data for contrastive energy training |
| Multi-layout env | Diverse training data for generalization |

The key change: instead of Actor → actions, train an energy network E_θ(s, a) and derive actions via gradient-based optimization (Langevin dynamics, MCMC, etc.).

## Research Goals

1. **Implement EBM critic** — Replace or augment the Q-function with an energy-based formulation
2. **Action generation via energy minimization** — Langevin dynamics or amortized optimization to extract actions from E(s,a)
3. **Compositional objectives** — Separate energy terms for power, loads, wake penalties
4. **Evaluate vs. standard SAC** — Compare sample efficiency, multimodality, generalization

## Relevant Approaches

- **Implicit Behavioral Cloning (IBC)** — Florence et al. (2022). Uses EBMs for policy learning. Actions via Langevin MCMC on E(s,a).
- **Contrastive RL** — Eysenbach et al. (2022). Learns representations via contrastive objectives with RL structure.
- **Diffusion Policy** — Chi et al. (2023). Uses score-based diffusion (closely related to EBMs) for policy learning.
- **Energy-Based Models for Continuous RL** — Various works on using EBMs as Q-functions or policies.

## Open Questions

- How to handle the partition function during training? Contrastive divergence? Noise contrastive estimation?
- Langevin dynamics for action optimization — how many steps at inference time? Amortized sampler?
- Should the energy function replace the critic, the actor, or both?
- How does composability of energy terms interact with the multi-objective wind farm problem?
- Can energy landscapes transfer across farm layouts better than explicit policies?
