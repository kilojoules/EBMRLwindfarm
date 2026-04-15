# TODO — NeurIPS 2026 Sprint (2026-04-15 → 2026-05-06)

> Execution checklist for the re-planned NeurIPS submission. Full plan:
> `~/.claude/plans/gentle-kindling-papert.md`. Archived original plan:
> `planning/paper_plan_v1.md`.

**Deadline:** abstract 2026-05-04, full paper 2026-05-06. Fallback: CoRL 2026.

---

## Decision gates

- [ ] **G0 (Apr 17)** — `stag4_5d` layout trains end-to-end on EBT-SAC without errors
- [ ] **G1 (Apr 22)** — E1 hero: cooperative reorganization reproduces on ≥1 real DEL
      constraint on BOTH hero layouts across ≥5 seeds (variance < half effect size)
- [ ] **G2 (Apr 22)** — E4 scale-up: cooperative adaptation on 7/9-turbine is ≥ 0.5×
      the 3-turbine value across ≥3 seeds
- [ ] **G3 (Apr 27)** — E3 phase diagram readable AND E5 CEP-ablation decisive
- [ ] **G4 (Apr 28)** — commit main-track-only vs. main+D&B after verifying the
      NeurIPS 2026 D&B deadline

---

## Window 1: Apr 15 – 17 (foundation)

- [x] Add `stag4_5d` layout to `helpers/layouts.py`
- [x] Archive `planning/paper_plan.md` → `paper_plan_v1.md`; write new pointer
- [x] Update this TODO.md
- [ ] **User:** plumb real DEL surrogates (tower-base fore-aft, blade-root flapwise,
      main-bearing moment, …) into `load_surrogates.py` as `nn.Module` classes
      exposing `per_turbine_energy(action, mask) → (batch, n_turbines, 1)`
- [ ] Kick off EBT-SAC training on `multi_modal` (≥5 seeds)
- [ ] Kick off EBT-SAC training on `stag4_5d` (≥5 seeds)
- [ ] Begin drafting `paper/main.tex` — submission track fix + abstract + methodology
- [ ] **G0 check** — confirm `stag4_5d` training runs to completion on ≥1 seed

## Window 2: Apr 17 – 22 (hero experiments)

- [ ] **E1**: evaluate trained hero models with `t1_positive_only` (sanity baseline)
      and real tower-base DEL constraint across the existing sweep harness
- [ ] **E4**: evaluate zero-shot on `r1` (6-turb) and `square_3x3` (9-turb) WITH
      constraints; compute per-turbine shift vs. graph-distance-from-constrained
- [ ] **E5 baselines — launch early**: fork `transformer_sac_windfarm.py` as
      `transformer_sac_lagrangian.py` with multi-channel DEL summed into reward;
      kick off ~8 Lagrangian-retrained runs across the λ-vector lattice
- [ ] **E5 scientific control — wire in**: add `no_cross_attention_energy` flag in
      `networks.py` (per-turbine energy MLP only, no cross-token information flow)
- [ ] **E2**: start multi-channel Pareto sweeps as soon as ≥2 DEL channels are
      plumbed in
- [ ] **G1 check** — hero result reproduces across ≥5 seeds on real DEL constraint
- [ ] **G2 check** — scale-up result holds on 9-turbine layout

## Window 3: Apr 22 – 27 (characterization + ablations)

- [ ] **E2**: complete multi-channel Pareto sweep (~40 λ-weightings × multiple
      layouts); generate the 3D Pareto-surface hero figure; run the per-turbine ×
      per-channel heterogeneity demo
- [ ] **E3**: phase-diagram sweep across `(constraint strength × coupling strength
      × channel)`; classify cells as {clipping, composition-matches, violating,
      inconclusive}; generate heatmap figure
- [ ] **E5**: train CEP-style ablation (no cross-agent attention); compare
      cooperative adaptation metric against full EBT
- [ ] **E5**: run diffusion-actor + classifier guidance as "approximate composition"
      baseline
- [ ] **E6**: stacked multi-constraint demo (multi-channel DEL + travel budget +
      per-turbine yaw limit); verify orthogonal controllability
- [ ] Draft Experiments section in `paper/main.tex` as results land
- [ ] **G3 check** — phase diagram readable AND CEP-ablation result decisive

## Window 4: Apr 27 – May 4 (writing)

- [ ] **G4 (Apr 28)**: check NeurIPS 2026 D&B deadline, commit main-only vs. main+D&B
- [ ] Draft Introduction (1–1.5 pages; 3–4 numbered contributions)
- [ ] Draft Related Work — cite EBT-Policy (arXiv 2510.27545), WFCRL (NeurIPS 2024
      D&B), CoDiG (CoRL 2025, arXiv 2505.13131), CEP (Urain IJRR 2023), ALGD
      (arXiv 2602.02924), Kadoche 2025 PhD, CCPO (NeurIPS 2023), UPDeT (ICLR 2021),
      Haarnoja Soft Q-Learning (2017)
- [ ] Draft Background (SAC ↔ EBM connection, tight)
- [ ] Polish Methodology section
- [ ] Polish all figures — hero (Pareto surface), phase diagram, scale-up locality,
      energy landscape, CEP-ablation comparison
- [ ] Write Analysis/Discussion (limitations, phase-diagram interpretation, compute)
- [ ] Write Conclusion (from the draft in the plan)
- [ ] Final Abstract
- [ ] Internal read-through for flow
- [ ] **May 4**: abstract submission

## Window 5: May 4 – 6 (polish + submit)

- [ ] Final figure polish and captions
- [ ] Double-check all citations and bib entries
- [ ] Final anonymization check (no `final` option, no author-revealing text)
- [ ] Fill `checklist.tex`
- [ ] Upload appendix/supplementary
- [ ] **May 6**: full paper submission

## Post-May-6 (conditional)

- [ ] If G4 committed to D&B: repackage `scripts/constraint_coupling_results.md`
      + phase-diagram + baseline table + WindGym wrapper as WindCoupleBench
      (~5 days extra writing)

---

## Backlog (nice-to-haves if time permits)

- [ ] Attention-shift visualization (which turbines attend to which when
      constraints are applied)
- [ ] Automatic λ-tuning via dual ascent
- [ ] Second mid-size layout for cross-layout generalization of E2
- [ ] Consistency distillation for single-step inference
- [ ] Formal convergence analysis of composed energy optimization

## Deferred to CoRL 2026 / future work

- [ ] Real SCADA + strain-gauge validation (Lillgrund, Horns Rev)
- [ ] Second domain beyond wind farms (cooperative navigation, multi-drone)
- [ ] End-to-end joint training of DEL surrogates with the EBT actor

---

## Completed infrastructure (carry-over from the 2026-04-10 plan)

- [x] EBT actor (`ebt.py`) + EBT-SAC training (`ebt_sac_windfarm.py`)
- [x] Diffusion actor (`diffusion.py`) + Diffusion-SAC (`diffusion_sac_windfarm.py`)
- [x] 6 toy load surrogates (`load_surrogates.py`) with per-turbine interface
- [x] Per-turbine energy composition in EBT actor
- [x] Classifier guidance in diffusion actor
- [x] Multi-layout training environment (`helpers/multi_layout_env.py`)
- [x] 14+ positional encoding variants
- [x] Evaluation + sweep pipeline (`scripts/evaluate_constraints.py` — 1068 lines)
- [x] Energy landscape viz (`scripts/visualize_energy_landscape.py`)
- [x] PyWake brute-force constraint coupling scan (`scripts/find_constraint_coupling.py`,
      `scripts/constraint_coupling_results.md`)
- [x] Curated reading list (`papers/PAPERS.md`)
- [x] NeurIPS 2026 LaTeX template (`paper/main.tex`, `paper/figs/`)
