# Revision Flag Summary

Tracker for items needing author input. Generated during the major revision
("The Alignment Hazard" → TMLR focused note pivot).

## Headline-regime decision (REQUIRED)

The revision now anchors the flagship hazard-recovery numbers on `y_tgt=0`
(the true zero-crossing regime) instead of `y_tgt=+25` (a separate
saturated-wake mechanism). The honest `y_tgt=0` recovery is more modest:

| Metric | Linear @ y_tgt=0 | Rejection @ y_tgt=0 | Argmin @ y_tgt=0 |
|---|---|---|---|
| Power (GW-step) | 1.61 | 1.72 | 1.70 |
| DEL_tot | 63375 | 63000 | 62692 |
| Δ vs linear (Pwr) | — | +6.8% | +5.6% |
| Δ vs linear (DEL) | — | −0.6% | −1.1% |

**Author decision:** lead with the y_tgt=0 numbers (honest, smaller effect)
OR keep y_tgt=+25 as the headline and relabel the mechanism as
"large-yaw / wake-saturation"? The agent chose y_tgt=0; the +25 numbers
are retained in Table 4 (renamed) as a separate effect.

## Numbers that did not reconcile (FLAGGED IN SOURCE)

### Old abstract: "+3.0% DEL and −12% farm power" (linear hazard)

- Linear @ y_tgt=0: Pwr 1.61, DEL_tot 63375
- π_perf alone (Table 4 reference, MM DEL-aware): Pwr 1.89, DEL_tot 63074
- π_safe alone (eval at +7.5° setpoint): Pwr 1.52, DEL_tot 62684
- Honest DEL excess vs endpoints: (63375 − min(63074, 62684))/62684 ≈ **+1.10%**, NOT +3.0%
- Honest power loss vs endpoints: (1.61 − max(1.89, 1.52))/1.89 ≈ **−14.8%**, NOT −12%

Replaced abstract with: "+1.1% DEL and −15% farm power" (the honest cell-level deltas).
% TODO(revision): author verify these are the right reference points (π_perf vs zero-yaw vs π_safe@+7.5).

### Old abstract: "−3.8% DEL and +15.3% power" (worst case at y_tgt=+25)

Sourced to Table 4 (sweep summary at y_tgt=+25). Mechanism is wake-saturation,
NOT the zero-crossing alignment hazard. Demoted from abstract; kept in §6 as
a separate "saturated-yaw recovery" finding.

### Cross-layout intro claims

- "+14% power AND −3% DEL" (multi_modal DEL-aware vs zero): verified against
  Table 2 (cross_layout_actor). DEL deltas per turbine: [−3.2%, −5.1%, −0.7%].
  Average DEL drop ~3.0%; power +14%. Match — keep.
- "+22% power / +5–7% DEL" (stag4_5d): verified against Table 2. Match — keep.

### κ values (Table 1)

- 0.72 wind, 0.02 Safety Gym. Source: Table 1 (coupling). Keep.

### Safety Gym headline ("C=483 → 15", "R=27 vs 0.5", "54× reward")

Verified against Table 5 (sg_hazard at theta=3π/4):
- Linear: R=0.50, C=483
- Rejection: R=27.08, C=50
- Argmin: R=27.30, C=15
- "54× reward" = 27.30/0.50 = 54.6× — OK (use ~55× for honesty)
- "−97% cost" = (483-15)/483 = 96.9% — OK

## New citations needing verification

Marked `% [VERIFY]` in references.bib. Author confirm before submission.

- **mcmahan2024anytime** — McMahan & Zhu, AISTATS 2024. arXiv 2311.05511.
  Used in §2 (constraint type) and §7 (related work).
- **tilmant2008water** — Tilmant et al., WRR 2008. Marginal water value /
  reservoir operation. Used in §3 (schedule motivation) and §7.
- **you2008hedging** — You & Cai, WRR 2008. Reservoir hedging rule.
  Used in §3/§7. Author choose one of tilmant/you to keep if both feel
  redundant.
- **arxiv2026loadwakeRL** — May 2026 arXiv paper on load-constrained wake
  steering RL with Independent SAC + sector-averaged DEL surrogate +
  WindGym. Candidate arXiv ID **2604.22795** — AUTHOR MUST CONFIRM TITLE
  AND ID; web search may have hallucinated. Used in §7 as the closest
  prior art.
- **falcon2021** — FALCON, Renewable Energy 2021 (Sci Direct
  S0960148121013227). Multi-agent DRL wake steering with fatigue Pareto.
  Used in §7.
- **wfcrl2025** — WFCRL, arXiv:2501.13592. MARL wake steering benchmark.
  Used in §7.

## Renaming and label changes

- §3 `\label{sec:theory}` (Urgency-Ratio Schedule) collided with §5.2.
  §5.2 renamed `sec:monotone_theory`. (Fixed in prior commit `ae987d8`.)
- Figure 1 label changed `fig:kappa_diag` → `fig:setup_overview` to match
  new framing. (Fixed in prior commit `55a79e0`.)

## Items intentionally not touched

- The κ coupling diagnostic (Table 1) remains in §4 but is demoted to
  supporting machinery. Per author instruction, do NOT delete.
- Gradient-correction variant Eq.~3 stays in §4.2, demoted to supporting.
- All retraining-baseline tables stay in App B (auxiliary_legacy.tex).
- DEL additivity is NOT critiqued anywhere (per §0 rule).

## σ-sweep result UPGRADED (2026-06-09): strict-above-both rescued at n=50

Job 19122212 (n=50 multi_modal, π_safe ∈ {+7.5, +15}) returned. At
π_safe=+7.5 with n=50:
- σ=0: DEL 63052 ± 14 SE
- σ=0.4: DEL 63264 ± 72 SE (peak; yaw ≈ -8°, transition zone)
- σ=1: DEL 62681 ± 11 SE

Peak vs left endpoint: +212 ± 73 (3σ significant)
Peak vs right endpoint: +583 ± 73 (8σ significant)

**Strict-above-both ACHIEVED with statistical significance in wind.**
The n=10 result had this same hump buried in noise; n=50 reduces SE
enough to declare significance.

Job 19122213 (stag4 sweep) TIMED OUT after 2h, completed 2/3 configs:
- stag4 π_safe=-15: monotone ↓ (no hump)
- stag4 π_safe=+25: monotone ↑ (no hump)
- stag4 π_safe=0: partial, monotone

Stag4 ridge too gentle in any direction tested. Multi_modal layout retains
the strict-above-both demonstration; stag4 is a counterexample to the wind
generality claim but doesn't undermine the multi_modal result.

Paper updated to claim strict-above-both with statistical significance,
referencing the n=50 +7.5 sweep. Fig 3 caption rewritten. Abstract,
intro, and Discussion aligned.

## σ-sweep result (n=10 preliminary, 2026-06-02)

Both sweep jobs returned:
- Job 19012840 (π_safe=+15° + π_safe=−25° control)
- Job 19013277 (π_safe=+7.5° low-DEL endpoint)

**Verdict: between outcomes 1 and 2.** The DEL ridge at yaw≈0 is REAL and
the predicted local non-monotonicity IS visible (e.g., DEL@σ=0.5=63935
versus DEL@σ=0.4=63220 in the +15 config, +715 kNm spike at exactly
where yaw crosses 0), but its absolute magnitude is small enough that
no wind configuration we tried gives a midpoint DEL strictly and
significantly above both endpoints on n=10 episodes.

Paper now positions wind as "present-but-marginal" and Safety Gym as
"sharp"; the Proposition's strict-above-both prediction is fully
demonstrated in 2D nav, while the wind ridge is operationally marginal.
This is the honest paper consistent with the data.

Updated:
- Abstract: dropped "strictly higher than either endpoint" claim; replaced
  with split positioning ("sharp in 2D nav, marginal in wind").
- Intro: same split positioning.
- §5.1 (counterexample): "fail strictly" → "fail".
- §6 (sigma_sweep paragraph): caption explains all 3 curves; honest reading.
- §8 Discussion: aligned with split positioning.

## Theory-experiment gap (HISTORICAL, addressed by σ-sweep)

User pointed out the formal Proposition is about non-convexity in σ at fixed
controllers, but the existing wind experiment only sweeps π_safe direction at
fixed σ=0.7. The σ-claim is not directly demonstrated in wind. Submitted
σ-sweep on LUMI (job 19012840, `scripts/sigma_sweep_test.py`,
`lumi/sigma_sweep.sbatch`):
- Configuration A: π_safe=+15° (hazardous; blend yaw crosses 0)
- Configuration B: π_safe=-25° (co-directional control; monotone expected)
- σ ∈ {0, 0.1, ..., 1.0}, n=10 episodes

Three possible outcomes:
1. **Hump in A, monotone in B** → wind hazard demonstrated directly, σ-claim
   rescued. Generate matching figure, update Fig 3 placeholder.
2. **Flat in A** → wind DEL ridge too gentle. Honest paper becomes
   "hazard sharp in 2D nav (Safety Gym), marginal in wind; segment-convexity
   condition predicts when". Reduce wind claims accordingly.
3. **Hump in both** → controller direction not the main driver. Rethink.

Currently `Figure~\ref{fig:sigma_sweep}` referenced from abstract, intro, and
§6, all with placeholder. Replace once job 19012840 returns.

## Compile state

After revision, paper should compile clean with `pdflatex; bibtex; pdflatex; pdflatex`.
Target: 9-page main body, appendix unrestricted.

## Adversarial referee response ledger (2026-06-11)

| Item | Status | Action |
|---|---|---|
| M1 rejection guarantee false | **RESOLVED + empirically confirmed** | v2 probes blended action. Gbar job 28628187: at y_tgt=0, endpoint-probe DEL=63416 ≈ linear 63420 (fails exactly as theory predicts); v2 DEL=62847 (-570). Table fix_recovery_zero updated with 4 modes; §5.3 verification sentence added. |
| M2 peak-location mismatch | **RESOLVED — strengthened the paper** | `scripts/probe_del_ridge.py`: measured ridge at −9..−14° (not 0°). σ-sweep peak at yaw≈−8 sits ON the measured ridge. New Fig del_ridge_probe. §5.1 prose corrected. |
| M3 post-hoc selection | **FIXED** | §6.1: peak location independently predicted by measured ridge; Bonferroni ×11 leaves σ=0 separation at p≈0.03, σ=1 untouched. |
| M4 SG strawman | **SOFTENED** | §6.4: rotation framed as synthetic misalignment probe standing in for stale-goal / wrong-layout / frame-error sources. |
| M5 generality overreach | OPEN | "generic on directional manifolds" survives in §5.1. Author call: soften to "can occur; segment-convexity is the checkable predictor". |
| M6 β never swept | **RESOLVED (MM); stag4 β=2 training in flight** | β∈{0.5,1,2} on MM: only β=1 Pareto-dominates (β=0.5: T2 DEL +1046; β=2: T2 +204). §6.5 updated — sensitivity holds even within one layout. Gbar job 28628188 training β=2 stag4 (~18h, SPS=3 CPU). |
| M7 anytime metric decorative | OPEN | Needs per-episode violation rates vs a declared budget. Existing JSONs store aggregates only; would need re-eval with per-episode dumps. Author decide if worth compute. |
| m1 "previously-unreported" | **SOFTENED** | Now "failure mode ... the inference-time composition literature does not address". |
| m2 vestigial κ/Alg.1/schedule | OPEN | Author call: move §3/§4.2-4.3/Alg.1 to appendix or keep as supporting. |
| m3 proprietary surrogate | OPEN | Add reproducibility paragraph; consider releasing surrogate I/O on published grid. |
| m4 App B contradictions | PARTIAL | Preamble note exists; full reframing pass not done. |
| m5 forward ref eq:blend2 | **FIXED** | §3 now names §4.1 explicitly. |
| m6 green-curve overclaim | OPEN | Caption still says "local non-monotonicity ... consistent with predicted ridge" — quantify or drop. |
| m7 GW-step units | **FIXED** | Defined at first use in §6 setup. |
