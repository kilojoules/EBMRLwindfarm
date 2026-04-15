# TODO — NeurIPS 2026 Sprint (2026-04-15 → 2026-05-06)

> Organized by priority. See [`EXPERIMENTS.md`](EXPERIMENTS.md) for the full experimental log and [`planning/paper_plan.md`](planning/paper_plan.md) for the research plan.
>
> **Status (2026-04-14):** ✅ Proof of concept achieved. Hero experiment works on `ebt_150k_nodroq_reg05` (see EXPERIMENTS.md). Remaining work is about reproducibility, scaling, and baselines for publication.

---

## Priority 1: Hero Experiment — Kill Test ✅ COMPLETED

- [x] Train EBT-SAC on `multi_modal` 3-turbine layout to convergence
- [x] Evaluate unconstrained → close to [-16°, -17.3°, 0°] (R2: [-14.2, -21.5, -8.2]; F1: [-12.6, -18.8, -0.7])
- [x] Evaluate with `t1_positive_only` at λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- [x] **Kill criterion satisfied:** T0 flips positive AND T1 measurably changes. R2 at λ=0.5 gives [+18.9, -9.0, -4.2] vs ground truth [+22.7, -9.3, 0.0] — T1 within 0.3°.
- [x] Constraint saturation at +30° (Round 0 failure mode) is FIXED at 150K steps with energy_reg=0.05

## Priority 2: Publication-critical experiments (MUST DO for NeurIPS)

### 2a. Seed variance on the winning config
- [ ] Run 3-5 additional seeds of `ebt_150k_nodroq_reg05` (R2) at 150K or 200K steps
- [ ] Compute mean±std of constrained yaw angles at λ=0.5 across seeds
- [ ] Establish that the cooperative adaptation is seed-robust (not cherry-picked)
- **Why:** Every current result uses seed=1. A reviewer will immediately ask "is this cherry-picked?"
- **Cost:** ~15-20h compute in parallel (4 seeds × 4h each, or 5 seeds on 4 GPUs)

### 2b. Baseline: naive action clipping
- [ ] Take the trained R2 checkpoint
- [ ] Eval-time modification: replace constraint composition with `torch.clamp(actions[..., 0], 0, 1)` on T0
- [ ] Run the same constraint sweep as the main experiment
- [ ] Show that clipping gives T0 at 0° (boundary) and NO cooperative adaptation in T1/T2
- **Why:** This is the "null hypothesis" — shows that energy composition produces qualitatively different behavior than trivial clipping.
- **Cost:** ~1h (eval only, no training)

### 2c. Baseline: Lagrangian SAC (retrained per constraint)
- [ ] Train a Gaussian SAC (or EBT) from scratch with the constraint baked into the reward: `r - λ * constraint_violation`
- [ ] Use the same 150K training budget
- [ ] Evaluate at the same λ values
- [ ] Compare cooperative adaptation quality vs our zero-shot method
- **Why:** This is the "strong" baseline — shows that retraining gives similar or worse results despite costing far more.
- **Cost:** ~12h training (1-2 runs needed for the comparison)

### 2d. Pareto front
- [ ] Use R2 checkpoint, sweep λ ∈ {0, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0}
- [ ] For each λ, compute: (mean power, constraint violation magnitude)
- [ ] Plot Pareto curve
- [ ] Overlay baseline curves from 2b and 2c
- **Why:** Standard visualization for constraint trade-offs.
- **Cost:** ~2h (eval only)

### 2e. Pick ONE main result config
- [ ] Decide between R2 (hero constraint adaptation) and F1 (best unconstrained)
- [ ] Document the tradeoff explicitly in the paper
- [ ] Alternative: run Round 3 experiments to find a config that's good on both (R2 base + profile encoder with stronger energy_reg)
- **Cost:** 0h if we just pick; ~16h if we run Round 3

## Priority 3: Publication-strengthening experiments (SHOULD DO)

### 3a. Zero-shot layout generalization
- [ ] Evaluate the R2 checkpoint on a 9-turbine layout (existing in `helpers/layouts.py`)
- [ ] Evaluate with and without the `t1_positive_only` constraint
- [ ] Show that the transformer handles the larger farm zero-shot
- [ ] Ideally show "locality of cooperation" — nearby turbines adapt to T0's flip, distant ones don't
- **Why:** Core paper claim is "turbines-as-tokens enables zero-shot scale generalization." Without this, it's just a 3-turbine paper.
- **Cost:** ~2h eval (no training)

### 3b. Multi-constraint composition
- [ ] Compose 2-3 constraints: `E_total = E_actor + λ₁·E_t1_positive + λ₂·E_yaw_limit`
- [ ] Show each constraint is independently controllable via its λ
- [ ] Show removing one (λ=0) recovers single-constraint solution
- **Why:** The word "composition" in the paper title requires showing actual composition of multiple constraints.
- **Cost:** ~2h eval

### 3c. Hero figure: 2D energy landscape
- [ ] Use `helpers/constraint_viz.py:plot_local_energy_landscape` on R2 checkpoint
- [ ] Render side-by-side E_actor vs E_actor + λ·E_constraint
- [ ] Polish for paper (axis labels, colorbars, minimum markers, two-basin annotation)
- [ ] Generate PNG and PDF at publication resolution
- **Why:** This is the figure reviewers will remember.
- **Cost:** ~30 min rendering + ~1h polishing

### 3d. Power ratio clarification
- [ ] Document the baseline that power ratios are computed against (likely greedy zero-yaw)
- [ ] Alternative: report absolute power in MW and normalize against PyWake optimum (9.71 MW unconstrained)
- [ ] Ensure "power cost of constraint" is never negative in the final plots
- **Cost:** ~1h analysis

## Priority 4: Nice-to-have (TIME PERMITTING)

### 4a. Cooperative adaptation metric
- [ ] Define: `adapt_score = mean(|y_constrained - y_unconstrained|)` over unconstrained turbines
- [ ] Compute for ours vs clipping baseline (should be zero for clipping by definition)
- [ ] Compute for Lagrangian baseline (should be nonzero but expensive to produce)
- **Why:** Gives a single number that quantifies the "emergence" claim.
- **Cost:** ~1h analysis

### 4b. Attention visualization
- [ ] Extract attention weights from the transformer during eval
- [ ] Visualize how attention shifts when constraint is applied
- [ ] Test hypothesis: constrained turbines receive more attention from others
- **Why:** Interpretability — reviewers like this.
- **Cost:** ~3h

### 4c. Inference-time compute scaling
- [ ] Sweep `ebt_opt_steps_eval` ∈ {5, 10, 20, 50, 100}
- [ ] Show that more optimization steps → better constraint satisfaction (or better unconstrained power)
- [ ] Frame as "test-time compute scaling for control"
- **Why:** Connects to the broader trend of compute-efficient inference scaling.
- **Cost:** ~1h eval

## Priority 5: Speculative / research questions (POST-SUBMISSION)

- [ ] SAC entropy with proper log_Z estimation via candidate logsumexp (requires modifying `ebt.py:optimize_actions` to return all candidate energies)
- [ ] Automatic λ-tuning via dual ascent
- [ ] Learned constraint surrogates from SCADA data (instead of hand-crafted analytical ones)
- [ ] Second domain beyond wind farms (cooperative navigation, traffic signal control, etc.)
- [ ] Consistency distillation for single-step EBT inference
- [ ] Formal convergence analysis of composed energy optimization

## Writing tasks

- [ ] Method section — architecture, composition mechanism, Lagrangian connection
- [ ] Experiments section (blocked by P2 and P3)
- [ ] Introduction
- [ ] Related work (mostly done, see `planning/paper_plan.md`)
- [ ] Conclusion
- [ ] Final figures and tables (blocked by P3c)
- [ ] Abstract — final version with real numbers

## Deferred to CoRL 2026 / future work

- [ ] Real SCADA + strain-gauge validation (Lillgrund, Horns Rev)
- [ ] Second domain beyond wind farms (cooperative navigation, multi-drone)
- [ ] End-to-end joint training of DEL surrogates with the EBT actor

---

## Completed infrastructure (historical)

### Code
- [x] EBT actor implementation (`ebt.py`)
- [x] EBT-SAC training pipeline (`ebt_sac_windfarm.py`)
- [x] Diffusion actor implementation (`diffusion.py`) — alternative, not headline
- [x] 7 constraint surrogates (`load_surrogates.py`) — original 6 + `QuadraticPositiveYawT1Surrogate` + `LinearPositiveYawT1Surrogate`
- [x] Per-turbine energy composition in EBT actor
- [x] Multi-layout training environment (`helpers/multi_layout_env.py`)
- [x] 14+ positional encoding variants (`positional_encodings/`)
- [x] 6 profile encoder variants including `FourierProfileEncoder` (`profile_encodings/`)
- [x] Comprehensive constraint evaluation script (`scripts/evaluate_constraints.py`)
- [x] Wandb integration with periodic constraint-viz figures
- [x] Existing energy landscape visualization (`helpers/constraint_viz.py`)
- [x] Turbine shuffle support (`shuffle_turbs` flag)
- [x] Geometric profile computation (`helpers/geometric_profiles.py`)

### Validation
- [x] PyWake brute-force ground truth in `test.ipynb`
- [x] `scripts/find_constraint_coupling.py` to identify layouts with coupled constraint response
- [x] `multi_modal` 3-turbine layout identified as the kill-test scenario

### Experiments
- [x] Round 0: Initial 30K runs (2026-04-11) — baseline that failed to converge
- [x] Round 1: 60K DroQ × ep_length sweep (2026-04-12) — characterized the DroQ collapse issue
- [x] Round 1': 150K DroQ × energy_reg sweep (2026-04-14) — **R2 winner for hero experiment**
- [x] Round 2: 150K Fourier profile × shuffle_turbs sweep (2026-04-14) — F1 winner for unconstrained
- [x] All Round 1' and Round 2 checkpoints evaluated with zero-init + steady-state averaging
- [x] Identified and fixed hardcoded `min(50, num_eval_steps)` viz cap

### Literature
- [x] Curated reading list (`papers/PAPERS.md`)
- [x] NeurIPS 2026 LaTeX template (`paper/main.tex`, `paper/figs/`)
