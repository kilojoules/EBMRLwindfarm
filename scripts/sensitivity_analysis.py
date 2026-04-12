#!/usr/bin/env python3
"""
Sensitivity analyses for paper revision (Reviewer Criticism #7).

Sweeps:
1. Wall threshold: {1%, 2%, 5%, 10%, 20%}
2. Lambda_max: {1e3, 1e4, 1e6, 1e8}
3. Stochastic costs: add noise sigma={0, 0.05, 0.1, 0.2} to velocity
4. Timing comparison: measure overhead per step

Usage:
    python scripts/sensitivity_analysis.py --checkpoint checkpoints/sac_cheetah.pt
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import gymnasium as gym

from scripts.safety_gym_budget import (
    GaussianActor, TimeVaryingRewardWrapper, VelocityBudgetSurrogate,
)


def load_agent(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt["act_limit"]


def run_ep(env, actor, act_limit, surr, gs, horizon, v_threshold,
           noise_sigma=0.0):
    if surr is not None and hasattr(surr, 'reset'):
        surr.reset()
    obs, _ = env.reset()
    ep_rew, violations = 0.0, 0
    budget = surr.budget_steps if surr else horizon

    for _ in range(horizon):
        with torch.no_grad():
            a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
        action = a.squeeze(0).numpy() * act_limit

        if surr is not None and gs > 0 and violations < budget:
            penalty = surr.penalize_action(action / act_limit)
            action = action / (1.0 + 0.1 * penalty)

        obs, rew, t, tr, info = env.step(action)
        velocity = abs(obs[8]) if len(obs) > 8 else abs(obs[0])
        # Add noise for stochastic cost analysis
        noisy_velocity = velocity + np.random.normal(0, noise_sigma)
        if noisy_velocity > v_threshold:
            violations += 1
        if surr is not None:
            surr.update(noisy_velocity, v_threshold)
        ep_rew += rew
        if t or tr:
            break
    return ep_rew, violations


def sweep_wall_threshold(env, actor, act_limit, v_threshold, n_ep=10):
    """Sweep the hard wall activation threshold."""
    print("\n" + "="*60)
    print("  SENSITIVITY 1: Wall Threshold")
    print("="*60)
    horizon, budget_frac, ra, gs = 1000, 0.25, 2.0, 1.0
    budget = int(horizon * budget_frac)

    print(f"{'Threshold':>10s} | {'Reward':>10s} | {'Violations':>12s} | {'%Used':>6s}")
    print("-" * 50)

    for wall_pct in [0.01, 0.02, 0.05, 0.10, 0.20]:
        rewards, viols = [], []
        for _ in range(n_ep):
            surr = VelocityBudgetSurrogate(
                budget_steps=budget, horizon_steps=horizon,
                risk_aversion=ra, steepness=3.0,
            )
            # Monkey-patch wall threshold
            original = surr._compute_lambda
            def make_patched(s, pct):
                def patched():
                    eps = 1e-6
                    b_rem = max(s.budget_steps - s.cumulative_violations, 0)
                    t_rem = max(s.horizon_steps - s.current_step, 1)
                    bf = b_rem / max(s.budget_steps, 1)
                    tf = t_rem / max(s.horizon_steps, 1)
                    u = max(bf / max(tf, eps), eps)
                    ac = np.exp(s.risk_aversion * (1.0 / u - 1.0))
                    dep = max(1.0 - bf / pct, 0)
                    wall = np.exp(s.steepness * dep)
                    return min(ac * wall, 1e6)
                return patched
            surr._compute_lambda = make_patched(surr, wall_pct)
            r, v = run_ep(env, actor, act_limit, surr, gs, horizon, v_threshold)
            rewards.append(r)
            viols.append(v)
        pct_used = 100 * np.mean(viols) / budget
        print(f"  {wall_pct*100:7.0f}%  | {np.mean(rewards):10.1f} | "
              f"{np.mean(viols):8.0f}/{budget} | {pct_used:5.1f}%")


def sweep_lambda_max(env, actor, act_limit, v_threshold, n_ep=10):
    """Sweep lambda_max clamp value."""
    print("\n" + "="*60)
    print("  SENSITIVITY 2: Lambda Max")
    print("="*60)
    horizon, budget_frac, ra, gs = 1000, 0.25, 2.0, 1.0
    budget = int(horizon * budget_frac)

    print(f"{'LambdaMax':>10s} | {'Reward':>10s} | {'Violations':>12s} | {'%Used':>6s}")
    print("-" * 50)

    for lam_max in [1e2, 1e3, 1e4, 1e6, 1e8]:
        rewards, viols = [], []
        for _ in range(n_ep):
            surr = VelocityBudgetSurrogate(
                budget_steps=budget, horizon_steps=horizon,
                risk_aversion=ra, steepness=3.0,
            )
            original = surr._compute_lambda
            def make_clamped(s, mx):
                def clamped():
                    eps = 1e-6
                    b_rem = max(s.budget_steps - s.cumulative_violations, 0)
                    t_rem = max(s.horizon_steps - s.current_step, 1)
                    bf = b_rem / max(s.budget_steps, 1)
                    tf = t_rem / max(s.horizon_steps, 1)
                    u = max(bf / max(tf, eps), eps)
                    ac = np.exp(s.risk_aversion * (1.0 / u - 1.0))
                    dep = max(1.0 - bf / 0.05, 0)
                    wall = np.exp(s.steepness * dep)
                    return min(ac * wall, mx)
                return clamped
            surr._compute_lambda = make_clamped(surr, lam_max)
            r, v = run_ep(env, actor, act_limit, surr, gs, horizon, v_threshold)
            rewards.append(r)
            viols.append(v)
        pct_used = 100 * np.mean(viols) / budget
        print(f"  {lam_max:8.0e}  | {np.mean(rewards):10.1f} | "
              f"{np.mean(viols):8.0f}/{budget} | {pct_used:5.1f}%")


def sweep_stochastic_costs(env, actor, act_limit, v_threshold, n_ep=10):
    """Test robustness to noisy cost observations."""
    print("\n" + "="*60)
    print("  SENSITIVITY 3: Stochastic Costs (noisy velocity)")
    print("="*60)
    horizon, budget_frac, ra, gs = 1000, 0.25, 2.0, 1.0
    budget = int(horizon * budget_frac)

    print(f"{'NoiseSigma':>11s} | {'Reward':>10s} | {'Violations':>12s} | {'%Used':>6s}")
    print("-" * 50)

    for sigma in [0.0, 0.05, 0.1, 0.2, 0.5]:
        rewards, viols = [], []
        for _ in range(n_ep):
            surr = VelocityBudgetSurrogate(
                budget_steps=budget, horizon_steps=horizon,
                risk_aversion=ra, steepness=3.0,
            )
            r, v = run_ep(env, actor, act_limit, surr, gs, horizon,
                          v_threshold, noise_sigma=sigma)
            rewards.append(r)
            viols.append(v)
        pct_used = 100 * np.mean(viols) / budget
        print(f"  {sigma:10.2f}  | {np.mean(rewards):10.1f} | "
              f"{np.mean(viols):8.0f}/{budget} | {pct_used:5.1f}%")


def timing_comparison(env, actor, act_limit, v_threshold):
    """Measure computational overhead of the AC schedule."""
    print("\n" + "="*60)
    print("  SENSITIVITY 4: Computational Overhead")
    print("="*60)
    horizon = 1000
    budget = 250

    # Baseline: no constraint
    obs, _ = env.reset()
    start = time.perf_counter()
    for _ in range(horizon):
        with torch.no_grad():
            a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
        action = a.squeeze(0).numpy() * act_limit
        obs, _, t, tr, _ = env.step(action)
        if t or tr:
            break
    baseline_time = (time.perf_counter() - start) / horizon * 1000

    # With AC schedule
    surr = VelocityBudgetSurrogate(budget_steps=budget, horizon_steps=horizon,
                                    risk_aversion=2.0, steepness=3.0)
    surr.reset()
    obs, _ = env.reset()
    start = time.perf_counter()
    for _ in range(horizon):
        lam = surr._compute_lambda()
        with torch.no_grad():
            a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
        action = a.squeeze(0).numpy() * act_limit
        penalty = surr.penalize_action(action / act_limit)
        action = action / (1.0 + 0.1 * penalty)
        obs, _, t, tr, _ = env.step(action)
        velocity = abs(obs[8]) if len(obs) > 8 else abs(obs[0])
        surr.update(velocity, v_threshold)
        if t or tr:
            break
    ac_time = (time.perf_counter() - start) / horizon * 1000

    print(f"  Baseline (no constraint): {baseline_time:.3f} ms/step")
    print(f"  With AC schedule:         {ac_time:.3f} ms/step")
    print(f"  Overhead:                 {ac_time - baseline_time:.3f} ms/step "
          f"({(ac_time/baseline_time - 1)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sac_cheetah.pt")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--v-threshold", type=float, default=None)
    cli = parser.parse_args()

    actor, act_limit = load_agent(cli.checkpoint)
    env = gym.make("HalfCheetah-v5")
    env = TimeVaryingRewardWrapper(env, amplitude=0.5, period=200)

    if cli.v_threshold is None:
        velocities = []
        for _ in range(3):
            obs, _ = env.reset()
            for _ in range(1000):
                with torch.no_grad():
                    a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
                obs, _, t, tr, _ = env.step(a.squeeze(0).numpy() * act_limit)
                velocities.append(abs(obs[8]) if len(obs) > 8 else abs(obs[0]))
                if t or tr:
                    break
        v_threshold = np.percentile(velocities, 50)
    else:
        v_threshold = cli.v_threshold

    print(f"Velocity threshold: {v_threshold:.2f}")

    sweep_wall_threshold(env, actor, act_limit, v_threshold, cli.n_episodes)
    sweep_lambda_max(env, actor, act_limit, v_threshold, cli.n_episodes)
    sweep_stochastic_costs(env, actor, act_limit, v_threshold, cli.n_episodes)
    timing_comparison(env, actor, act_limit, v_threshold)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
