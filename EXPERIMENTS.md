# Experiment Plan: Post-Hoc Cumulative Budget Constraints via Optimal Execution Theory

## Paper Structure: Three Domains

The ICML paper demonstrates generality across three application areas:

1. **Engineering domain:** Wind farm yaw control (primary, WindGym)
2. **Standard ML benchmark:** Safety-Gymnasium locomotion velocity tasks
3. **OR/Economics domain:** Ad auction budget pacing

Each domain shares the same structure:
- An RL agent optimizing reward with continuous actions
- A "risky action" that boosts short-term reward but incurs cumulative cost
- A lifetime budget on that cumulative cost
- Time-varying conditions where the risky action's value changes

The same `NegativeYawBudgetSurrogate` (generalized as `CumulativeBudgetSurrogate`)
applies to all three, with only the cost indicator function swapped.

---

## Domain 1: Wind Farm Yaw Control (WindGym)

**Risky action:** Negative yaw (steers rotor against conventional direction)
**Budget:** Total timesteps at negative yaw per turbine per year
**Temporal variation:** Wind speed and direction follow a real 10-year time
series from Energy Island (Denmark). Negative yaw is most valuable during
wake-aligned wind events at moderate speeds.

### Experiment 1.1: Post-Hoc Constraint on Trained EBT Agent

**Status:** Running on LUMI (job 17432996) with Energy Island wind time series.

**Protocol:**
1. Train EBT-SAC on `3turb` layout, 100k steps, unconstrained, with
   Energy Island CSV driving wind conditions
2. Evaluate with budget surrogate post-hoc:
   - Steepness: [2, 3, 5] (calibrated to not overwhelm learned energy head)
   - Guidance scale: [0.05, 0.1, 0.5, 1.0]
   - Risk aversion: [0, 1, 2, 5]
   - Budget levels: [15, 30, 50, 100] steps out of 200
   - 5 episodes per config
3. Baselines: unconstrained, hard-clip (no neg yaw ever), constant penalty (RA=0)

**Measurements:**
- Power ratio: constrained_power / unconstrained_power
- Budget utilization: neg_yaw_steps / budget
- Budget violation rate

### Experiment 1.2: Post-Hoc vs. Retrained CMDP

Train a separate agent with neg-yaw penalty in reward. Compare:
- Performance at budget=15, then change to budget=30 (no retraining for post-hoc)
- Training compute (wall-clock time)

### Experiment 1.3: Multiple Farm Layouts

Test on 3turb, square_1, and a larger layout to demonstrate zero-shot generalization
of the budget constraint (the turbines-as-tokens architecture handles variable farm sizes).

---

## Domain 2: Safety-Gymnasium Locomotion (Standard ML Benchmark)

**Environment:** `SafetyHalfCheetahVelocity-v1` from Safety-Gymnasium (NeurIPS 2023)

**Risky action:** Running fast (velocity above threshold)
**Budget:** Total timesteps allowed above velocity threshold per episode
**Cost indicator:** c(s,a) = 1[velocity > v_threshold]

**Temporal variation (added via wrapper):** A sinusoidal reward multiplier
simulates varying terrain difficulty or task urgency:
```
reward_multiplier(t) = 1 + A * sin(2*pi*t / T_period)
```
During high-multiplier phases, sprinting is worth more. The AC schedule
should concentrate speeding budget into these phases.

### Experiment 2.1: AC Budget vs. Constant Lagrangian

**Protocol:**
1. Train unconstrained SAC on HalfCheetah (standard, ~30 min)
2. Add time-varying reward wrapper
3. Evaluate with budget surrogate post-hoc:
   - Cost indicator: velocity > threshold
   - Budget: [10%, 25%, 50%] of episode length
   - RA: [0 (constant Lagrangian), 1, 2, 5]
   - Guidance scale sweep
4. Baselines: constant Lagrangian (RA=0), hard velocity cap, PPO-Lagrangian (retrained)

**Key hypothesis:** AC(RA>0) concentrates speeding into high-multiplier phases,
achieving higher cumulative reward than AC(RA=0) for the same budget.

### Experiment 2.2: Comparison to OmniSafe Baselines

OmniSafe provides PPO-Lagrangian, CPPO-PID, and other constrained RL
baselines. Compare post-hoc AC to retrained constrained agents.

### Implementation Plan

```
scripts/safety_gym_budget.py:
  - TimeVaryingRewardWrapper: multiplies reward by sin(t) schedule
  - VelocityBudgetSurrogate: adapts NegativeYawBudgetSurrogate
    with cost = 1[velocity > threshold]
  - Training: standard SAC (stable-baselines3 or clean-rl)
  - Evaluation: same sweep structure as wind domain
```

**Dependencies:** safety-gymnasium, mujoco (runs on LUMI with ROCm)

---

## Domain 3: Ad Auction Budget Pacing (OR/Economics)

**Environment:** Lightweight custom Gymnasium env (~150 lines)

**Setup:** An advertiser participates in sequential ad auctions over a "day"
(episode). At each timestep, an ad impression arrives with features
(user demographics, time of day). The agent bids a continuous amount.
Winning costs money from the daily budget.

**Risky action:** Bidding high (wins auctions, depletes budget)
**Budget:** Total daily ad spend in dollars
**Cost indicator:** c(s,a) = payment_if_won (continuous, or simplified to binary: won/lost)

**Temporal variation (natural):** Impression values follow a time-of-day
pattern — morning commute has high-value users, late night has low-value.
The agent should bid aggressively during high-value periods and conserve
budget during low-value periods. This IS the Almgren-Chriss problem.

### Experiment 3.1: AC Pacing vs. Uniform Pacing

**Protocol:**
1. Train unconstrained bidding agent (no budget constraint)
2. Evaluate with budget surrogate post-hoc:
   - Budget: [50%, 30%, 10%] of "unlimited" daily spend
   - RA: [0, 1, 2, 5]
3. Baselines: uniform pacing (bid same amount every step), greedy (bid high until broke),
   throttled (random 50% of auctions)

**Key hypothesis:** AC pacing concentrates spend into high-value impression
periods, achieving higher total conversions than uniform pacing.

### Implementation Plan

```
scripts/ad_bidding_budget.py:
  - AdAuctionEnv(gym.Env): simple second-price auction MDP
    - State: (remaining_budget, time_remaining, impression_features)
    - Action: bid amount (continuous, [0, max_bid])
    - Reward: value of won impression (0 if lost)
    - Time-varying: impression value ~ sin(time_of_day) pattern
  - BiddingBudgetSurrogate: adapts the AC schedule for spend budgets
```

**Dependencies:** None beyond numpy, torch, gymnasium

---

## Implementation Priority

| Priority | Task | Domain | Time Est |
|---|---|---|---|
| 1 | Complete wind farm experiments on LUMI | Wind | Running |
| 2 | Build Safety-Gym velocity budget experiment | Locomotion | 2-3 days |
| 3 | Build ad auction budget pacing experiment | Economics | 3-5 days |
| 4 | Retrained CMDP baselines for all domains | All | 1 week |
| 5 | Multi-layout wind farm experiments | Wind | 2-3 days |
| 6 | Paper writing | - | 2-3 weeks |

## Generalized Surrogate Interface

For the paper, rename `NegativeYawBudgetSurrogate` to show it's domain-agnostic:

```python
class CumulativeBudgetSurrogate(nn.Module):
    """
    AC-inspired time-varying penalty for cumulative action budgets.

    Args:
        budget_steps: Total allowed "risky" timesteps
        horizon_steps: Planning horizon
        risk_aversion: AC concentration parameter (0=TWAP)
        cost_fn: Callable(action) -> per-element cost indicator
    """
```

The domain-specific part is only the `cost_fn`:
- Wind: `cost_fn = lambda a: (a < -threshold).float()`
- Locomotion: `cost_fn = lambda v: (v > v_threshold).float()`
- Bidding: `cost_fn = lambda bid: (bid > reserve_price).float()`

---

## Current Status

- [x] NegativeYawBudgetSurrogate implemented and tested
- [x] Threshold-policy demo with 10 years of wind data (85% of oracle)
- [x] Closed-loop gradient-based demo (99% power retention)
- [x] Theoretical properties verified (TWAP recovery, monotonicity, ~O(√T) regret)
- [x] LUMI environment set up (torch ROCm + WindGym)
- [x] Experiment 1.1 v1 complete (budget respected, penalty calibration needs fix)
- [x] Experiment 1.1 v2 submitted with calibrated steepness + time series wind
- [x] WindGym modified with TimeSeriesWindManager for CSV-driven wind
- [ ] Experiment 1.1 v2 results pending (LUMI job 17432996)
- [ ] Safety-Gymnasium velocity budget experiment
- [ ] Ad auction budget pacing experiment
- [ ] Retrained CMDP baselines
- [ ] Paper writing
