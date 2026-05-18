# Paper Restructure: "The Alignment Hazard"

## New title
**"The Alignment Hazard: Linear Blending of Pre-Trained Policies Can Increase Cumulative Cost, and a Cost-Monotone Fix"**

Alt: *"When Linear Policy Blending Hurts: A Cost-Monotone Composition for Budget-Aware Safe RL"*

## New abstract (draft)

> Composing pre-trained controllers at inference is an attractive alternative to retraining under new safety constraints: blend a performance policy $\pi_{\mathrm{perf}}$ with a safety controller $\pi_{\mathrm{safe}}$ via $a = (1-\sigma)\pi_{\mathrm{perf}} + \sigma\pi_{\mathrm{safe}}$, where $\sigma$ is paced against a budget. We identify a **fundamental failure mode** of this scheme on directional action manifolds: when $\pi_{\mathrm{perf}}$ and $\pi_{\mathrm{safe}}$ act in opposite directions on a cost ridge, the linear blend's midpoint produces cost **strictly higher than either endpoint**. We name this the **alignment hazard** and prove its existence via a wind-farm yaw control experiment: at fixed $\sigma=0.7$ across seven $\pi_{\mathrm{safe}}$ yaw targets, blade-flap DEL peaks where the blend forces yaw through zero ($+3.0\%$ over both endpoints, $-12\%$ farm power). We propose two cost-monotone fixes: **rejection blending** (engage $\sigma$ only when a 1-step cost probe confirms $\pi_{\mathrm{safe}}$ helps) and **argmin selection** (hard-pick the lower-cost candidate). On the same sweep, rejection recovers the Pareto frontier: $-3.8\%$ DEL and $+8.9\%$ power at the worst-case $\pi_{\mathrm{safe}}$. The 1-step probe costs $\sim$one surrogate evaluation per step. This generalises beyond yaw: any setting where action lives on a directional manifold and cost is non-convex along the segment $\pi_{\mathrm{perf}}\to\pi_{\mathrm{safe}}$ is exposed.

## Three claimed contributions

1. **Phenomenon (alignment hazard).** Identify and demonstrate that linear policy blending under budget pacing is not cost-monotone in $\sigma$ when controllers straddle a cost ridge. Existence proof via yaw control with blade-flap DEL.

2. **Theory.** Sufficient condition for cost-monotonicity: cost convex along segment $\pi_{\mathrm{perf}}\to\pi_{\mathrm{safe}}$. Counterexample family: directional actions with cost maximised near neutral.

3. **Cost-monotone fixes.** Rejection blending and argmin selection, both implemented with 1-step lookahead. Provably no-worse-than-$\pi_{\mathrm{perf}}$. Empirical Pareto recovery.

## Restructured sections

| Old | New |
|---|---|
| §3 Urgency-Ratio Schedule | §3 Setup: Blending + Pacing |
| §4 Method (linear blend) | §4 The Alignment Hazard (NEW + counterexample) |
| §5 Wind Farm (piecewise validation) | §5 Cost-Monotone Composition (NEW: rejection + argmin) |
| §5.2 Physics-grounded DEL | §6 Experiments: yaw sweep showing hazard + recovery |
| §5.3 Both-halves negative result | DEMOTED to appendix |
| §6 Safety Gym | §7 Safety Gym (does hazard appear there too?) |

## Headline figure (new)

Pareto plot: x-axis aggregate DEL, y-axis farm power. Three curves across $\pi_{\mathrm{safe}} \in \{-25, -15, -7.5, 0, +7.5, +15, +25\}$:
- Linear blend (U-shape; anti-Pareto bulge centred at $\pi_{\mathrm{safe}}=0$ and inflating at $\pi_{\mathrm{safe}}=+25$)
- Rejection blend (flat / Pareto-monotone)
- Argmin selection (TBD, expected similar to rejection)

Plus baseline points: $\pi_{\mathrm{perf}}$ alone, $\pi_{\mathrm{safe}}$ alone for each target.

## Data status (2026-05-18)

| Run | Result | Status |
|---|---|---|
| linear σ=0.7 sweep | U-shape confirmed | ✓ in `results/alignment_hazard_sigma07.json` |
| rejection σ=0.7 sweep | Pareto recovered | ✓ in `results/blend_modes_compare.json` |
| argmin σ=0.7 sweep | TBD | submitted 18708520 |

## Reviewer defenses

**"You adversarially chose opposite-direction π_safe":** false. Oracle's principled choice (max-power-with-all-DEL-below-zero) gives $\pi_{\mathrm{safe}}=[+7.5,+7.5,+7.5]$, while DEL-aware reward training drove $\pi_{\mathrm{perf}}$ to negative yaws. Opposite directions emerged from independent objectives. We sweep the full direction grid to show the hazard is **robust to choice**.

**"Just use a different blend formula":** rejection blend IS that fix. We propose it.

**"Doesn't generalise":** add Safety Gym experiment as second domain. Sweep $\pi_{\mathrm{safe}}$ direction (action sign) and check if cost-bulge appears.

## Open questions before submission

1. Does Safety Gym show the same hazard? (Run sweep there as second domain proof.)
2. Theoretical statement: precise sufficient condition + minimal counterexample.
3. Where does the hazard NOT appear? (Convex-cost cases — show as positive result for linear blend.)
