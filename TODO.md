# TODO — EBM + RL + Transformers

## Setup
- [x] Spring cleaning: remove old notebooks, archive, edge scripts
- [x] Update README for new direction
- [x] Create CLAUDE.md, CONTEXT.md, TODO.md
- [x] Update .gitignore
- [ ] Update requirements.txt for EBM dependencies (if needed)

## Near-Term Research
- [ ] Implement basic energy network E(s, a) using existing transformer backbone
- [ ] Implement Langevin dynamics action sampler
- [ ] Add contrastive training loss (noise contrastive estimation or InfoNCE)
- [ ] Baseline comparison: standard SAC vs. EBM-based policy on single layout
- [ ] Evaluate multimodality of learned energy landscape

## Architecture
- [ ] Design EBM head architecture (how to merge state and action for scalar energy output)
- [ ] Decide: replace critic, replace actor, or add alongside existing SAC
- [ ] Explore amortized action generation (learned sampler network)

## Experiments
- [ ] Single layout: EBM policy vs. SAC policy (power, convergence, stability)
- [ ] Multi-layout generalization: does energy landscape transfer better?
- [ ] Compositional objectives: separate power/load/wake energy terms
- [ ] Visualize energy landscapes for interpretability

## Ideas to Explore
- [ ] Diffusion policy as alternative to Langevin sampling
- [ ] Score matching vs. contrastive divergence for training
- [ ] EBM for world model (predict next state energy)
- [ ] Combine EBM uncertainty with exploration bonus
