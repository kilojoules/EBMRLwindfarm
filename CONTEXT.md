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

## The SAC ↔ EBM Connection

A critical insight: **SAC is already an EBM-adjacent method.** The maximum entropy RL framework defines the optimal policy as:

```
π*(a|s) ∝ exp(Q*(s,a) / α)
```

This is a Boltzmann distribution — the canonical EBM policy. SAC's predecessor, **Soft Q-Learning** (Haarnoja et al. 2017), made this explicit: it trained an energy function E(s,a) = -Q(s,a) and sampled actions via amortized Stein Variational Gradient Descent (SVGD). SAC (Haarnoja et al. 2018) replaced SVGD with a reparameterized Gaussian actor for stability and speed.

What SAC gained: stable training, fast inference, practical scalability.
What SAC lost: multimodal action distributions, explicit energy landscape, compositionality.

**Our research isn't "applying EBMs to a new domain" — it's completing the circle.** We're making the implicit EBM structure in SAC explicit again, and leveraging EBM-specific properties (compositionality, energy landscapes, multimodality) that SAC's Gaussian approximation discards.

## Research Directions

### Direction A: Diffusion Policy with Transformer Backbone

Replace the Gaussian actor with diffusion-based action generation. The existing transformer encoder becomes the denoising backbone, conditioning on turbine observations to iteratively refine noisy yaw actions into optimal configurations.

- **Approach**: Conditional denoising diffusion process (DDPM/DDIM) over per-turbine yaw actions, conditioned on transformer-encoded turbine states.
- **Why it fits**: Diffusion handles multimodal action distributions naturally — multiple valid yaw configurations for a given wind condition. The iterative refinement process is analogous to Langevin dynamics on an energy landscape.
- **Prior art**: Chi et al. (2023) Diffusion Policy; Wang et al. (2023) Diffusion-QL; Chen et al. (2024) Consistency Policy for real-time inference.
- **Novelty**: No prior work applies diffusion policies to variable-size turbine sequences with layout generalization. The "turbines as tokens" architecture makes diffusion conditioning particularly natural — each token's action is denoised in context of all other turbines.
- **Risk**: Inference speed. Diffusion requires multiple denoising steps. Consistency models (Chen et al. 2024) offer 45x speedup via single-step inference — a critical mitigation for real-time control.

### Direction B: Compositional Energy Objectives

Learn separate energy terms for different objectives: E_power(s,a), E_fatigue(s,a), E_wake(s,a). Compose them by addition:

```
E_total(s,a) = E_power(s,a) + λ₁·E_fatigue(s,a) + λ₂·E_wake(s,a)
```

- **Approach**: Train independent energy networks for each objective using contrastive divergence or noise contrastive estimation. Actions are derived by minimizing the composed energy via Langevin dynamics or an amortized sampler.
- **Why it fits**: Wind farms have natural compositional structure — power maximization, structural fatigue, wake interactions, and turbulence are genuinely separate physical phenomena. Changing λ weights at deployment lets operators rebalance objectives (e.g., prioritize load reduction during storms) *without retraining*.
- **Prior art**: Du et al. (2020) compose energy functions for multi-concept image generation; Liu et al. (2022) extend this to diffusion models.
- **Novelty**: This is a domain where EBM compositionality genuinely matters, not just a toy demo. Wind farm operators need to adjust objective tradeoffs based on real-time conditions (wind speed, grid demand, maintenance schedules). No current RL approach offers test-time objective rebalancing.
- **Risk**: Training stability. Multiple energy networks must be calibrated to similar scales. May need careful normalization or joint training with shared representations.

### Direction C: Energy Landscape for Layout Transfer

Learn an energy function E(s,a) that captures the landscape of good yaw configurations. Hypothesis: energy landscapes transfer better than explicit policies across farm layouts, because they encode "what's good" (low energy = high power) rather than "what to do" (specific yaw angles).

- **Approach**: Train E_θ(s,a) on diverse layouts. At test time on an unseen layout, derive actions via energy minimization. The energy landscape should capture wind-physics invariants (wake deflection angles, turbine spacing effects) that generalize.
- **Why it fits**: Layout generalization is the existing codebase's core strength (transformer + positional encodings enable zero-shot transfer). The question is whether EBM formulation *improves* transfer, since energy functions define a landscape of relative quality rather than committing to specific action mappings.
- **Prior art**: Cao et al. (2023) show energy-based regularization improves OOD generalization in offline RL.
- **Novelty**: Testing whether energy landscapes are more layout-invariant than explicit policies. If true, this would be a significant result for the "turbines as tokens" approach.
- **Risk**: May not provide measurable improvement over standard SAC if the transformer already captures sufficient layout invariance. Need careful experimental design to isolate the EBM contribution.

### Recommended Priority

**Direction B (Compositional Energy) is the strongest angle.** "Diffusion policy for wind farms" is solid but incremental. "Compositional energy objectives with test-time rebalancing" is a genuinely new capability that EBMs uniquely enable and wind farms uniquely need. Direction A could serve as a stepping stone (diffusion is better understood than raw EBM training), and Direction C provides the evaluation story.

A practical path: start with Direction A (diffusion actor, proves the denoising architecture works), then extend to Direction B (compositional energy terms within the diffusion framework via classifier guidance).

## Open Questions

- **Training**: How to handle the partition function? Contrastive divergence vs. noise contrastive estimation vs. score matching?
- **Inference speed**: How many Langevin/denoising steps at inference time? Can consistency models achieve single-step inference?
- **Architecture**: Should the energy function replace the critic, the actor, or both? (Direction A replaces actor; B replaces both.)
- **Composability**: How do separate energy terms interact? Do they need shared representations or can they be fully independent?
- **Evaluation**: How to measure multimodality and composability beyond standard RL metrics?
- **Scaling**: Does the EBM formulation change the scaling properties of the transformer backbone?
