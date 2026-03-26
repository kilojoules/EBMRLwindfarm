# TODO — EBM + RL + Transformers

## Literature & Exploration (Start Here)
- [ ] Read Soft Q-Learning paper (Haarnoja 2017) — understand the SAC→EBM connection
- [ ] Read Diffusion Policy paper (Chi et al. 2023) — understand state-of-the-art diffusion policy
- [ ] Read IBC paper (Florence et al. 2021) — understand EBM policy training challenges
- [ ] Read Compositional EBM paper (Du et al. 2020) — understand energy compositionality
- [ ] Run existing SAC baseline on 2-3 layouts, record performance for comparison
- [ ] Build a tiny synthetic "wind farm" (3 turbines, simple wake model) for rapid EBM prototyping
- [ ] Prototype: minimal EBM energy head on existing transformer (even if broken, builds intuition)
- [ ] Decide primary research direction after reading phase

## Setup
- [x] Spring cleaning: remove old notebooks, archive, edge scripts
- [x] Update README for new direction
- [x] Create CLAUDE.md, CONTEXT.md, TODO.md
- [x] Update .gitignore
- [x] Create papers/PAPERS.md — curated literature review
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
- [ ] Test-time objective rebalancing: change λ weights without retraining
- [ ] Visualize energy landscapes for interpretability

## Ideas to Explore
- [ ] Diffusion policy as alternative to Langevin sampling
- [ ] Consistency models for single-step inference (real-time control)
- [ ] Score matching vs. contrastive divergence for training
- [ ] EBM for world model (predict next state energy)
- [ ] Combine EBM uncertainty with exploration bonus
- [ ] Classifier guidance on diffusion policy for compositional objectives
