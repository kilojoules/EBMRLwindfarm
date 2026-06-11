# Budgeted Long-Horizon Risk Control

This folder starts the replacement paper direction:

> Safety constraints and turbine loads are state-mediated, delayed, and budgeted. The core object is not an instantaneous penalty. It is a long-horizon cost-to-go critic.

The current AC schedule paper becomes a baseline/initializer. The new method is test-time constrained policy improvement using learned reward and cost critics:

```text
maximize_a    Q_r(s, h, a)
subject to    C_used + Q_c(s, h, a) <= B
```

or the soft version:

```text
maximize_a    Q_r(s, h, a) - lambda_t Q_c(s, h, a) - beta uncertainty(Q_c)
```

## Layout

| Path | Purpose |
|---|---|
| `paper/main.tex` | Initial LaTeX draft for the new paper |
| `paper/references.bib` | Minimal bibliography for the draft |
| `scripts/budgeted_qc_core.py` | Shared critics, replay utilities, budget scoring |
| `scripts/wake_proxy_qc_experiment.py` | Self-contained wake-convection proxy testbed |
| `scripts/safety_gym_budgeted_qc.py` | Safety Gym frozen-policy `Q_r/Q_c` testing script |
| `scripts/summarize_results.py` | Tiny JSON result summarizer |

## Quick smoke test

The wake proxy requires PyTorch but does not require WindGym, PyWake, or Safety Gym:

```bash
python budgeted_long_horizon_control/scripts/wake_proxy_qc_experiment.py \
  --quick \
  --out results/budgeted_lhc/wake_proxy_quick.json
```

The Safety Gym script expects an existing unconstrained checkpoint in the format used by the existing repo scripts:

```bash
python budgeted_long_horizon_control/scripts/safety_gym_budgeted_qc.py \
  --checkpoint checkpoints/sac_safety_point_seed1.pt \
  --quick \
  --out results/budgeted_lhc/safety_gym_quick.json
```

## Initial experiment claims to test

1. Direct action penalties fail when cost is delayed through dynamics.
2. A learned `Q_c(s,h,a)` produces useful action ranking before it produces perfect cost calibration.
3. Budget feasibility should be measured per episode, not as mean cost alone.
4. Wind-farm wake-convection proxies and Safety Gym hazards should fail in similar ways under one-step penalties.
5. The final wind-farm loading surrogate can replace the proxy if it implements the same transition-level cost API.
