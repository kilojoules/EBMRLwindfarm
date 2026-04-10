# Experiment Plan: AC Budget Surrogate for ICML

## Goal

Validate that the Almgren-Chriss-inspired NegativeYawBudgetSurrogate works
as a post-hoc constraint on a **trained** EBT-SAC agent in WindGym. This is
the critical experiment needed before submission.

## Experiments (Priority Order)

### Experiment 1: Post-Hoc Constraint on Trained EBT Agent (MUST-HAVE)

**Hypothesis:** A pre-trained unconstrained EBT-SAC agent, when evaluated
with NegativeYawBudgetSurrogate composed at inference, respects the negative-yaw
budget while retaining >90% of unconstrained power.

**Protocol:**
1. Train EBT-SAC on `3turb` layout, 100k steps, no budget constraint
2. Evaluate the trained checkpoint with budget surrogate at inference:
   - Risk aversion: [0.0, 0.5, 1.0, 2.0, 5.0]
   - Guidance scale: [0.5, 1.0, 2.0, 5.0, 10.0]
   - Budget: 15 steps (out of 200-step episode)
3. Compare to unconstrained evaluation (guidance_scale=0)

**Measurements:**
- Farm power (mean episode return)
- Negative yaw count per turbine per episode
- Power retention: constrained_power / unconstrained_power
- Budget violation rate

**LUMI job:** `lumi/train_ebt_3turb.sbatch`

### Experiment 2: Post-Hoc vs. Retrained CMDP (MUST-HAVE)

**Hypothesis:** Post-hoc composition matches retrained CMDP in power but
wins on flexibility (zero retraining for new budget levels).

**Protocol:**
1. Use checkpoint from Exp 1 + budget surrogate (post-hoc)
2. Train a NEW agent with negative-yaw penalty baked into reward
3. Compare both at budget=15 and then at budget=30 (no retraining for post-hoc)

### Experiment 3: Guidance Scale Sensitivity (MUST-HAVE)

**Hypothesis:** A robust guidance_scale range exists where budget is
satisfied across wind conditions.

**Protocol:**
- Log sweep: guidance_scale in [0.01, 0.1, 0.5, 1, 2, 5, 10, 50, 100]
- Fixed budget=15, RA=2.0
- Multiple evaluation episodes

### Experiment 4: Budget Adaptation Mid-Episode (NICE-TO-HAVE)

**Hypothesis:** AC schedule adapts within one step when budget changes.

### Experiment 5: PID Lagrangian Comparison (NICE-TO-HAVE)

### Experiment 6: PyWake/FLORIS Validation (NICE-TO-HAVE)

## Current Status

- [x] NegativeYawBudgetSurrogate implemented and tested
- [x] Factory + config integration
- [x] Threshold-policy demo with 10 years of wind data
- [x] Closed-loop gradient-based demo
- [x] Theoretical properties verified
- [ ] **Experiment 1: trained EBT agent** ← NEXT
- [ ] Experiment 2: retrained CMDP baseline
- [ ] Experiment 3: guidance scale sweep
