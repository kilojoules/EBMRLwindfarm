# Paper Plan: Composable Energy Policies for Constrained Multi-Agent Control

> Research planning document — synthesized from structured ideation session (2026-04-10).
> Target: NeurIPS 2026 (abstract deadline ~May 4, full paper ~May 6). Backup: ICLR 2027 (~Oct 2026).

---

## The Nugget

**Energy-based transformer policies enable zero-shot constraint composition in multi-agent control: constraining one agent at deployment causes the entire team to cooperatively reorganize to a new joint optimum — without retraining.**

## Framing Strategy

Lead with **one headline contribution**: post-hoc constraint composition via energy addition that produces emergent cooperative adaptation. Everything else is supporting evidence.

| Element | Role in paper |
|---------|--------------|
| Post-hoc constraint composition + emergent cooperation | **Headline** — the "wow" result |
| Lagrangian duality connection | **Theory** — grounds the mechanism formally |
| Turbines-as-tokens zero-shot generalization | **Supporting** — Section 4.x, not the headline |
| Inference-time compute scaling (more opt steps → better) | **Appendix** — nice-to-have |
| EBT vs. diffusion comparison | **Not included** — doesn't excite, dilutes focus |

### What this paper is NOT

- "We applied EBMs to wind farms" — too narrow, niche framing
- "We combined transformer + EBM + RL" — no clear thesis, the "kitchen sink" trap
- A wind energy paper — the domain is the demonstration, not the contribution

### What this paper IS

A **general principle for deployable constrained multi-agent RL**, demonstrated on a compelling physical domain. The pitch:

> "Operational constraints in multi-agent systems often affect only a subset of agents, yet standard constrained MARL methods require retraining the entire team for each new constraint configuration. We show that energy-based transformer policies enable exact post-hoc composition of per-agent constraints as additive energy terms. Constraining a single agent causes the team to cooperatively reorganize to a genuinely new joint optimum — not merely clipping the constrained agent's actions — with zero retraining."

---

## Title Candidates

1. **Composable Energy Policies: Zero-Shot Constraint Adaptation in Multi-Agent Control**
2. Train Once, Constrain Anywhere: Energy-Based Policies for Deployable Constrained Control
3. Composable Energy Policies for Generalizable Wind Farm Control

Preference: (1) — clean, specific, signals both composition and multi-agent.

---

## Draft Abstract

> **(Topic)** Deploying reinforcement learning policies in multi-agent physical systems requires adapting to operational constraints that change over the system's lifetime — load limits on individual agents, travel budgets, safety exclusion zones — yet standard constrained RL methods require costly retraining for each new constraint configuration.
>
> **(Problem)** This retraining bottleneck is particularly severe when constraints affect only a subset of agents, because optimal cooperative strategies may change qualitatively: the best joint response to a single-agent constraint is not simply clipping that agent's actions, but reorganizing the entire team.
>
> **(Method)** We introduce Composable Energy Policies for multi-agent control, where a transformer backbone treats each agent as a token and an energy-based actor learns the task objective as an energy landscape E(s,a). At deployment, arbitrary differentiable constraints are composed as additive energy terms — E_total = E_task + Σ λ_i · E_constraint_i — and gradient-based action optimization naturally respects all objectives without retraining.
>
> **(Results)** On wind farm yaw control, a single policy trained on small layouts generalizes zero-shot to larger farms and, when per-turbine constraints are composed at deployment, discovers qualitatively different cooperative strategies: constraining one turbine to positive-only yaw causes it to flip from −16° to +23° while unconstrained neighbors cooperatively adjust their angles, recovering [X]% of optimal power. We show this composition is formally equivalent to Lagrangian relaxation and that sweeping λ traces the Pareto frontier between performance and constraint satisfaction.
>
> **(Why it matters)** Our results establish energy-based policies as a principled framework for deploying adaptive multi-agent control under changing operational constraints, eliminating the retraining cycle that currently limits real-world RL deployment.

## Draft Conclusion

> We have shown that energy-based transformer policies fundamentally change how constraints interact with multi-agent RL: rather than retraining for each constraint configuration, operators compose constraint energies at deployment and the policy cooperatively adapts. The key finding is not merely that constraints can be enforced post-hoc — it is that the energy landscape's joint optimization produces emergent cooperative behavior, where constraining one agent causes others to reorganize to a qualitatively different optimum. This is impossible with post-hoc action clipping and prohibitively expensive with constraint-conditioned retraining. Combined with the turbines-as-tokens architecture that enables zero-shot generalization across system sizes, this yields a single trained model deployable to any farm with any constraint profile — a practical paradigm shift for operational RL.

---

## Experiment Plan

### Priority 1: Hero Experiment (Kill Test) — by April 17

**Goal:** Validate that the EBT actor discovers emergent cooperative behavior under constraints.

- Train EBT-SAC on `multi_modal` 3-turbine layout
- Evaluate with `t1_positive_only` constraint at guidance scales λ ∈ {0.1, 0.5, 1.0, 2.0, 5.0}
- **Success criterion:** T1 flips from negative to positive yaw (ideally near +22.7°) AND at least one other turbine measurably changes its yaw
- **Kill criterion:** If T1 stays negative across all guidance scales, the central claim fails → target ICLR 2027

**Known optimal solutions (PyWake brute-force):**
- Unconstrained: `[-16°, -17.3°, 0°]`
- With `t1_positive_only`: `[+22.7°, -9.3°, 0°]`

### Priority 2: Energy Landscape Visualization — 1 day

The figure reviewers will remember. Side-by-side:
- Left: Actor energy E_actor(s, a) landscape (T1 vs T2 yaw, T3 fixed)
- Right: Composed energy E_actor + λ·E_constraint landscape
- Show the minimum shifting to a different basin

### Priority 3: Pareto Front — 1-2 days

Sweep λ from 0 to large. Plot (power output, constraint satisfaction) as a Pareto curve.
Compare against:
- Naive post-hoc action clipping
- Lagrangian SAC retrained at each constraint level (if time permits)

### Priority 4: Scale to 9-16 Turbines — 3-5 days

Train on small layouts, evaluate on 9+ turbine layouts WITH constraints.
Key result: show "locality of cooperation" — nearby wake-affected turbines adapt, distant ones don't.
Addresses the "3 turbines is too small" criticism.

### Priority 5: Multi-Constraint Composition — 2-3 days

Compose 2-3 constraints simultaneously:
`E_total = E_actor + λ₁·E_yaw_limit + λ₂·E_travel_budget + λ₃·E_t1_positive`

Show:
- Each constraint independently controllable via its λ
- Removing one (λ=0) recovers the unconstrained solution for that dimension
- Joint solution differs from satisfying constraints sequentially

### Priority 6: Cooperative Adaptation Metric — 1 day

Define and compute: how much do unconstrained turbines change their actions when a constraint is added to another turbine?
- Clipping: exactly zero (by definition)
- Energy composition: nonzero (the point of the paper)
- Lagrangian retraining: nonzero but requires expensive retraining

### Priority 7: Attention Visualization — 1-2 days

Show which turbines attend to which others, and how attention shifts when constraints are applied.
Hypothesis: when T1 is constrained, other turbines attend more to T1.

---

## Baseline Comparisons

| Method | Retraining needed? | Can find new optima? | Handles variable agents? |
|--------|-------------------|---------------------|-------------------------|
| Post-hoc action clipping | No | No (clips to boundary) | N/A |
| Lagrangian SAC (retrained) | Yes (per constraint) | Yes (but expensive) | No (fixed architecture) |
| CCPO (constraint-conditioned) | Yes (trains on constraint range) | Partially | No |
| **Ours (composable energy)** | **No** | **Yes** | **Yes** |

---

## Theoretical Contribution: Lagrangian Duality

Adding λ·E_constraint to E_actor is exactly Lagrangian relaxation:

```
min_a  E_actor(s, a)    subject to    E_constraint(a) ≤ 0
```

- λ is the Lagrangian multiplier
- The guidance scale has a formal interpretation
- Sweeping λ traces the Pareto frontier (by Lagrangian duality)
- Can implement automatic λ-tuning via dual ascent (~20 lines of code)

**Note:** Do NOT overclaim this as a contribution — ALGD (Feb 2026) already frames Lagrangian-as-energy for diffusion. Present as supporting theory/background.

---

## Key Prior Work and Positioning

### Must cite and clearly differentiate from:

| Paper | What they did | What's new in our work |
|-------|--------------|----------------------|
| **Urain et al. (RSS 2021, IJRR 2023)** — Composable Energy Policies | Energy composition for single-robot reactive motion | Multi-agent, RL-trained energies, emergent cross-agent cooperation |
| **Du & Mordatch (NeurIPS 2020)** — Compositional EBMs | Additive energy composition for image generation | Sequential control (not one-shot generation), multi-agent |
| **Gladstone et al. (2025)** — Energy-Based Transformers | EBT architecture for supervised learning | RL training, control applications, constraint composition |
| **UPDeT (ICLR 2021)** — Agents-as-tokens | Transformer for multi-agent RL | Energy-based actor (not just transformer backbone), constraint composition |
| **Haarnoja et al. (ICML 2017)** — Soft Q-Learning | Original EBM + RL connection | Explicit energy with post-hoc composition, transformer backbone |
| **CCPO (NeurIPS 2023)** | Zero-shot constraint adaptation | No constraint-conditioned training needed; energy composition is post-hoc |
| **CoDiG (CoRL 2025)** | Constraint-aware diffusion guidance | EBM (not diffusion), multi-agent cooperation (not single-robot) |
| **ALGD (Feb 2026)** | Lagrangian as energy for diffusion | Multi-agent, cooperative emergence, post-hoc (not retrained) |

### Cross-field connections to mention:

- **Protein design:** Composable energy landscapes for multi-property optimization (binding + stability + solubility)
- **Test-time compute scaling:** EBT's variable optimization steps as inference-time scaling for control (brief mention, not headline)

---

## Competitive Landscape

**Scooping risk: LOW.** No group is pursuing the exact combination of EBM + multi-agent + transformer + post-hoc constraints.

**Groups to watch:**
- Julen Urain (META FAIR) — composable energy policies inventor, moved to flow matching
- Alexi Gladstone (MIT) — EBT paper, focused on supervised learning
- Hao Ma / Melanie Zeilinger (ETH Zurich) — CoDiG, constraint-aware diffusion

**Timing: Well-timed.** Within a 12-18 month optimal window. EBMs resurging, composable policies trending, safe RL hot. The window is open but will narrow as diffusion-based constraint methods converge toward energy-based formulations.

---

## Positioning Dos and Don'ts

**Do:**
- Lead with emergent cooperative adaptation — it's the surprise
- Frame as general multi-agent principle, wind farms as compelling domain
- Acknowledge Urain et al. explicitly, state clearly what's new
- Compare against clipping AND Lagrangian retraining baselines
- Explain why EBM composition > diffusion guidance (exact vs. approximate)

**Don't:**
- Lead with "we applied EBMs to wind farms"
- Position as a wind energy paper
- Overclaim Lagrangian duality as novel (ALGD exists)
- Compare only against naive clipping
- Try to sell all contributions equally — one headline, rest supporting

---

## Paper Structure (Suggested)

1. **Introduction** — The deployment problem: constraints change, retraining is expensive. Our solution: composable energy policies.
2. **Background** — EBMs, SAC, the SAC↔EBM connection, transformer for variable-size agents
3. **Method**
   - 3.1 Energy-Based Transformer Actor (architecture)
   - 3.2 Post-Hoc Constraint Composition (the mechanism)
   - 3.3 Connection to Lagrangian Duality (theoretical grounding)
4. **Experiments**
   - 4.1 Hero: Emergent cooperative adaptation (multi_modal + t1_positive_only)
   - 4.2 Pareto fronts and constraint trade-offs
   - 4.3 Multi-constraint composition
   - 4.4 Zero-shot layout generalization with constraints
   - 4.5 Energy landscape visualization
5. **Related Work** — Composable EBMs, constrained RL, transformer MARL, wind farm RL
6. **Conclusion**

---

## Timeline

| Date | Milestone |
|------|-----------|
| Apr 10-17 | **Kill test:** Train EBT-SAC on multi_modal, validate hero experiment |
| Apr 17-18 | Energy landscape visualization |
| Apr 18-20 | Pareto fronts + baseline comparisons |
| Apr 20-25 | Scale-up experiments (9-16 turbines) + multi-constraint |
| Apr 25-May 1 | Writing (method + experiments sections) |
| May 1-4 | Writing (intro, related work, conclusion), figures |
| May 4 | Abstract submission |
| May 6 | Full paper submission |

**If hero experiment fails by Apr 17:** Stop. Diagnose. Target ICLR 2027 with more time to fix training (multi-modal energy regularization, contrastive samples, etc.).

---

## Key Diagnostic: Energy Landscape Shape

Before running the full hero experiment, visualize the 2D energy landscape (T1 yaw vs T2 yaw, T3 fixed at 0°):
- **With composition:** Should show two distinct basins (one near [-16, -17], one near [+23, -9])
- **Without composition:** Should show single basin near [-16, -17]

If the composed landscape is unimodal with a repulsive wall (constraint pushes to boundary, no second basin), the cooperation claim will fail and training may need adjustment.
