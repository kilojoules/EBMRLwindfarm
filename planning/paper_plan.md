# Paper Plan — Pointer

> **Status:** superseded by the 2026-04-15 re-plan. The execution-ready plan lives at
> `~/.claude/plans/gentle-kindling-papert.md`. The original 2026-04-10 plan (dramatic-flip framing)
> is preserved verbatim at `planning/paper_plan_v1.md` for the record.

## What changed on 2026-04-15

After re-reading the empirical results (`scripts/constraint_coupling_results.md`,
`results/constraint_eval_v2_histlen2/summary.csv`, `results/eval_droq_60k/summary.csv`) and
checking the literature with fresh eyes, the original "dramatic bimodal flip"
framing was replaced. Two hard facts drove the pivot:

1. **Physics ceiling.** PyWake brute-force over 1500+ layouts shows genuine strategy
   shifts under yaw constraints top out at ~10° per free turbine at ≥5D spacing; larger
   apparent flips are `cos²(+θ) = cos²(-θ)` inline-symmetry artifacts. The original
   `[-16°, -17°, 0°] → [+22.7°, -9.3°, 0°]` story does not robustly exist in realistic
   farm geometries.
2. **Training fragility.** The EBT-SAC model on `multi_modal` does show emergent
   cooperative adaptation, but only in a narrow `(λ, steepness)` window, and the
   effect is sensitive to training recipe across seeds.

And one hard fact from the literature:

3. **EBT-Policy** (Gladstone et al., arXiv 2510.27545, Oct 2025) already claims
   "energy-based transformers as visuomotor policies with inference-time compute
   scaling and emergent behaviors." This closes the "EBT for control" framing and
   forces the headline claim to be **multi-agent-specific**.

## The new headline

> In multi-agent energy-based control, composing per-agent × per-channel constraint
> energies at deployment produces *cooperative reorganization across agent tokens* —
> a structural property that single-agent composable-energy methods (CEP, CoDiG,
> EBT-Policy) cannot express. A single EBT-SAC policy trained for power
> maximization admits arbitrary per-turbine × per-damage-channel constraint
> profiles at deployment, tracing Pareto frontiers across multiple fatigue
> channels with zero retraining. We characterize the regime in which cooperative
> emergence occurs and validate against Lagrangian-retrained and action-clipping
> baselines.

## Key operational changes

- **Multi-channel DEL surrogates** (real neural surrogates fit to high-fidelity
  aeroelastic simulations, all basic channels: tower base, blade root, main bearing,
  etc.) replace the toy load surrogates in `load_surrogates.py` as the hero constraint.
- **Phase diagram** `(coupling strength × constraint strength × channel)` turns the
  fragility into a scientific characterization.
- **CEP-style ablation** (strip cross-agent attention from the energy head) is the
  crucial scientific control isolating the multi-agent joint-optimization claim.
- **Second hero layout**: `stag4_5d` added to `helpers/layouts.py` — validated by
  PyWake as the layout with strongest constraint coupling.

See `~/.claude/plans/gentle-kindling-papert.md` for the full experiment plan, timeline,
decision gates, and positioning table.
