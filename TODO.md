# TODO — Composable Energy Policies for Constrained Multi-Agent Control

> Organized by priority for NeurIPS 2026 submission. See `planning/paper_plan.md` for full research plan.

## Priority 1: Hero Experiment — Kill Test (by Apr 17)

- [ ] Train EBT-SAC on `multi_modal` 3-turbine layout to convergence
- [ ] Evaluate unconstrained → expect yaws near [-16°, -17.3°, 0°]
- [ ] Evaluate with `t1_positive_only` at λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- [ ] **Kill criterion:** T1 must flip positive AND T2 must measurably change
- [ ] If fail → diagnose energy landscape shape, consider ICLR 2027

## Priority 2: Energy Landscape Visualization

- [ ] Visualize 2D energy landscape (T1 yaw vs T2 yaw, T3 fixed at 0°)
- [ ] Side-by-side: E_actor alone vs. E_actor + λ·E_constraint
- [ ] Confirm two distinct basins in composed landscape
- [ ] Polish as hero figure for the paper

## Priority 3: Pareto Front

- [ ] Sweep λ from 0 to large for each constraint type
- [ ] Plot (power output, constraint satisfaction) Pareto curve
- [ ] Baseline: naive post-hoc action clipping
- [ ] Baseline: Lagrangian SAC retrained per constraint level (if time permits)
- [ ] Compute cooperative adaptation metric (how much unconstrained turbines change)

## Priority 4: Scale to Larger Farms

- [ ] Train on small layouts (3-turbine)
- [ ] Evaluate on 9-turbine layouts WITH constraints (zero-shot)
- [ ] Evaluate on 16+ turbine layouts WITH constraints (zero-shot)
- [ ] Show "locality of cooperation" — nearby turbines adapt, distant ones don't

## Priority 5: Multi-Constraint Composition

- [ ] Compose 2-3 constraints simultaneously
  - [ ] E_total = E_actor + λ₁·E_yaw_limit + λ₂·E_travel_budget + λ₃·E_t1_positive
- [ ] Show each constraint independently controllable via its λ
- [ ] Show removing one (λ=0) recovers unconstrained solution for that dimension
- [ ] Show joint solution differs from satisfying constraints sequentially

## Priority 6: Attention Visualization

- [ ] Extract attention weights from transformer during evaluation
- [ ] Visualize how attention shifts when constraints are applied
- [ ] Test hypothesis: constrained turbines receive more attention from others

## Writing

- [ ] Method section (architecture + composition mechanism + Lagrangian connection)
- [ ] Experiments section
- [ ] Introduction
- [ ] Related work
- [ ] Conclusion
- [ ] Figures and tables
- [ ] Abstract (final version)

## Backlog (post-submission or for ICLR 2027)

- [ ] Automatic λ-tuning via dual ascent
- [ ] Learned constraint surrogates (from data, not hand-crafted)
- [ ] Second domain beyond wind farms (cooperative navigation or similar)
- [ ] Consistency distillation for single-step inference
- [ ] Formal convergence analysis of composed energy optimization

---

## Completed

### Infrastructure
- [x] EBT actor implementation (`ebt.py`)
- [x] EBT-SAC training pipeline (`ebt_sac_windfarm.py`)
- [x] Diffusion actor implementation (`diffusion.py`)
- [x] Diffusion-SAC training pipeline (`diffusion_sac_windfarm.py`)
- [x] 6 load surrogates (`load_surrogates.py`)
- [x] Per-turbine energy composition in EBT actor
- [x] Classifier guidance in diffusion actor
- [x] Demo script for constraint scenarios
- [x] Multi-layout training environment
- [x] 14+ positional encoding variants
- [x] Evaluation pipeline
- [x] Wandb integration and sweep scripts
- [x] PyWake brute-force validation of multi-modal optima
- [x] Research plan and paper framing (`planning/paper_plan.md`)

### Literature
- [x] Curated reading list (`papers/PAPERS.md`)
- [x] Competitive landscape analysis (see `planning/paper_plan.md`)
