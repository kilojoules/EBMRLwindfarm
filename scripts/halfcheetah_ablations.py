#!/usr/bin/env python3
"""
HalfCheetah ablation experiments for the budget constraint paper.

Runs all critical ablations on a trained SAC checkpoint:
1. Hard-guard ablation: AC only vs guard only vs both
2. Constant-penalty sweep matching AC's budget usage
3. 1/u vs exp(eta*(1/u-1)) schedule comparison
4. Budget flexibility: 10%, 25%, 50% budgets with same policy
5. Boltzmann response validation: spending rate vs lambda

Usage:
    python scripts/halfcheetah_ablations.py --checkpoint checkpoints/sac_cheetah.pt
    python scripts/halfcheetah_ablations.py --checkpoint checkpoints/sac_cheetah.pt --ablation all
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import gymnasium as gym

from scripts.safety_gym_budget import (
    GaussianActor, TimeVaryingRewardWrapper, VelocityBudgetSurrogate,
)


def load_agent(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt["act_limit"], ckpt["obs_dim"]


def make_env(env_name="HalfCheetah-v5", amplitude=0.5, period=200):
    env = gym.make(env_name)
    env = TimeVaryingRewardWrapper(env, amplitude=amplitude, period=period)
    return env


def run_episode(env, actor, act_limit, surr, gs, horizon=1000,
                v_threshold=5.66, use_hard_guard=True):
    """Run one episode with optional budget surrogate and hard guard."""
    if surr is not None:
        surr.reset()
    obs, _ = env.reset()
    ep_rew = 0.0
    violations = 0
    budget = surr.budget_steps if surr is not None else horizon

    for t in range(horizon):
        with torch.no_grad():
            a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
        action = a.squeeze(0).numpy() * act_limit

        # Apply budget penalty to action
        if surr is not None and gs > 0:
            can_spend = (not use_hard_guard) or (violations < budget)
            if can_spend:
                penalty = surr.penalize_action(action / act_limit)
                scale = 1.0 / (1.0 + 0.1 * penalty)
                action = action * scale

        obs, reward, term, trunc, info = env.step(action)
        velocity = abs(obs[8]) if len(obs) > 8 else abs(obs[0])

        if velocity > v_threshold:
            violations += 1

        if surr is not None:
            surr.update(velocity, v_threshold)

        ep_rew += reward
        if term or trunc:
            break

    return ep_rew, violations


def run_config(env, actor, act_limit, budget_frac, ra, steepness, gs,
               horizon, v_threshold, n_episodes, use_hard_guard=True,
               schedule_type="exp"):
    """Run multiple episodes for one configuration."""
    budget = int(horizon * budget_frac)
    rewards, viols = [], []
    for _ in range(n_episodes):
        surr = VelocityBudgetSurrogate(
            budget_steps=budget, horizon_steps=horizon,
            risk_aversion=ra, steepness=steepness,
        )
        # Monkey-patch schedule type for inverse test
        if schedule_type == "inverse" and ra > 0:
            original_compute = surr._compute_lambda
            def make_inverse_lambda(s, eta):
                def inverse_lambda():
                    eps = 1e-6
                    b_rem = max(s.budget_steps - s.cumulative_violations, 0)
                    t_rem = max(s.horizon_steps - s.current_step, 1)
                    bf = b_rem / max(s.budget_steps, 1)
                    tf = t_rem / max(s.horizon_steps, 1)
                    u = bf / max(tf, eps)
                    u = max(u, eps)
                    ac = u ** (-eta)
                    dep = max(1.0 - bf / 0.05, 0)
                    wall = np.exp(s.steepness * dep)
                    return min(ac * wall, 1e6)
                return inverse_lambda
            surr._compute_lambda = make_inverse_lambda(surr, ra)

        r, v = run_episode(env, actor, act_limit, surr, gs, horizon,
                           v_threshold, use_hard_guard)
        rewards.append(r)
        viols.append(v)
    return np.mean(rewards), np.std(rewards), np.mean(viols), int(budget)


# =============================================================================
# ABLATIONS
# =============================================================================

def ablation_hard_guard(env, actor, act_limit, v_threshold, n_ep=10):
    """Ablation 1: Hard guard vs AC schedule vs both."""
    print("\n" + "="*70)
    print("  ABLATION 1: Hard Guard vs AC Schedule vs Both")
    print("="*70)
    horizon, budget_frac, ra, k, gs = 1000, 0.25, 2.0, 3.0, 1.0
    budget = int(horizon * budget_frac)

    configs = [
        ("Both (AC + guard)", ra, True, "exp"),
        ("AC only (no guard)", ra, False, "exp"),
        ("Guard only (no AC, RA=0)", 0.0, True, "exp"),
        ("Neither (unconstrained)", 0.0, False, "exp"),
    ]

    print(f"Budget: {budget}/{horizon} steps, RA={ra}, k={k}, gs={gs}")
    print(f"{'Config':<30s} | {'Reward':>10s} | {'Violations':>12s} | {'%Budget':>8s}")
    print("-" * 70)

    for name, ra_val, guard, sched in configs:
        mr, sr, mv, b = run_config(env, actor, act_limit, budget_frac,
                                    ra_val, k, gs, horizon, v_threshold,
                                    n_ep, guard, sched)
        pct = 100 * mv / b if b > 0 else 0
        print(f"  {name:<28s} | {mr:10.1f} | {mv:8.0f}/{b:<3d} | {pct:7.1f}%")


def ablation_constant_sweep(env, actor, act_limit, v_threshold, n_ep=10):
    """Ablation 2: Sweep constant penalty to match AC's budget usage."""
    print("\n" + "="*70)
    print("  ABLATION 2: Constant Penalty Sweep (matching budget usage)")
    print("="*70)
    horizon, budget_frac, k = 1000, 0.25, 3.0

    # First run AC to get target budget usage
    mr_ac, _, mv_ac, budget = run_config(env, actor, act_limit, budget_frac,
                                          2.0, k, 1.0, horizon, v_threshold, n_ep)
    print(f"AC (RA=2): Reward={mr_ac:.1f}, Violations={mv_ac:.0f}/{budget}")

    print(f"\nConstant penalty sweep (RA=0):")
    print(f"{'gs':>6s} | {'Reward':>10s} | {'Violations':>12s} | {'%Budget':>8s}")
    print("-" * 50)
    for gs in [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0]:
        mr, sr, mv, b = run_config(env, actor, act_limit, budget_frac,
                                    0.0, k, gs, horizon, v_threshold, n_ep)
        pct = 100 * mv / b if b > 0 else 0
        marker = " <-- matches AC usage" if abs(mv - mv_ac) < 20 else ""
        print(f"  {gs:5.2f} | {mr:10.1f} | {mv:8.0f}/{b:<3d} | {pct:7.1f}%{marker}")


def ablation_schedule_comparison(env, actor, act_limit, v_threshold, n_ep=10):
    """Ablation 3: 1/u (optimal) vs exp(eta*(1/u-1)) (practical)."""
    print("\n" + "="*70)
    print("  ABLATION 3: Schedule Comparison (1/u vs exp)")
    print("="*70)
    horizon, budget_frac, k, gs = 1000, 0.25, 3.0, 1.0

    print(f"{'Schedule':<15s} {'RA':>4s} | {'Reward':>10s} | {'Violations':>12s} | {'%Budget':>8s}")
    print("-" * 65)

    for sched in ["exp", "inverse"]:
        for ra in [0.0, 0.5, 1.0, 2.0, 5.0]:
            mr, sr, mv, b = run_config(env, actor, act_limit, budget_frac,
                                        ra, k, gs, horizon, v_threshold,
                                        n_ep, True, sched)
            pct = 100 * mv / b if b > 0 else 0
            label = f"{'exp':>6s}" if sched == "exp" else f"{'1/u':>6s}"
            print(f"  {label:<13s} {ra:4.1f} | {mr:10.1f} | {mv:8.0f}/{b:<3d} | {pct:7.1f}%")


def ablation_budget_flexibility(env, actor, act_limit, v_threshold, n_ep=10):
    """Ablation 4: Same policy, different budget levels."""
    print("\n" + "="*70)
    print("  ABLATION 4: Budget Flexibility (same policy, varying budgets)")
    print("="*70)
    horizon, ra, k, gs = 1000, 2.0, 3.0, 1.0

    # Unconstrained baseline
    env2 = make_env()
    uncon_rewards = []
    for _ in range(n_ep):
        obs, _ = env2.reset()
        ep_rew = 0.0
        for _ in range(horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            obs, rew, t, tr, _ = env2.step(a.squeeze(0).numpy() * act_limit)
            ep_rew += rew
            if t or tr:
                break
        uncon_rewards.append(ep_rew)
    uncon_mean = np.mean(uncon_rewards)
    print(f"Unconstrained: {uncon_mean:.1f}")

    print(f"\n{'Budget%':>8s} | {'Reward':>10s} | {'%Uncon':>7s} | {'Violations':>12s} | {'%Used':>6s}")
    print("-" * 60)
    for frac in [0.10, 0.15, 0.25, 0.50, 0.75]:
        mr, sr, mv, b = run_config(env, actor, act_limit, frac,
                                    ra, k, gs, horizon, v_threshold, n_ep)
        pct_uncon = 100 * mr / uncon_mean if uncon_mean > 0 else 0
        pct_used = 100 * mv / b if b > 0 else 0
        print(f"  {frac*100:5.0f}%  | {mr:10.1f} | {pct_uncon:6.1f}% | {mv:8.0f}/{b:<4d} | {pct_used:5.1f}%")


def ablation_boltzmann_response(env, actor, act_limit, v_threshold, n_ep=5):
    """Ablation 5: Validate Boltzmann response assumption."""
    print("\n" + "="*70)
    print("  ABLATION 5: Boltzmann Response Validation")
    print("  (Is spending rate ~ exp(-alpha * lambda)?)")
    print("="*70)
    horizon, budget_frac, k = 1000, 0.50, 3.0  # generous budget

    print(f"{'gs':>8s} | {'SpendRate':>10s} | {'ln(rate)':>10s}")
    print("-" * 35)
    for gs in [0.0, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]:
        _, _, mv, b = run_config(env, actor, act_limit, budget_frac,
                                  0.0, k, gs, horizon, v_threshold, n_ep)
        rate = mv / horizon
        ln_rate = np.log(rate) if rate > 0 else float('-inf')
        print(f"  {gs:7.2f} | {rate:10.4f} | {ln_rate:10.4f}")

    print("\n  If Boltzmann: ln(rate) should be approximately linear in gs.")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="HalfCheetah budget ablations")
    parser.add_argument("--checkpoint", default="checkpoints/sac_cheetah.pt")
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--ablation", nargs="+",
                        default=["all"],
                        choices=["all", "hard_guard", "constant_sweep",
                                 "schedule", "flexibility", "boltzmann"])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--v-threshold", type=float, default=None)
    cli = parser.parse_args()

    actor, act_limit, obs_dim = load_agent(cli.checkpoint)
    env = make_env(cli.env)

    # Auto-detect velocity threshold if not provided
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
        print(f"Auto velocity threshold (50th pct): {v_threshold:.2f}")
    else:
        v_threshold = cli.v_threshold

    ablations = set(cli.ablation)
    if "all" in ablations:
        ablations = {"hard_guard", "constant_sweep", "schedule",
                     "flexibility", "boltzmann"}

    n_ep = cli.n_episodes

    if "hard_guard" in ablations:
        ablation_hard_guard(env, actor, act_limit, v_threshold, n_ep)

    if "constant_sweep" in ablations:
        ablation_constant_sweep(env, actor, act_limit, v_threshold, n_ep)

    if "schedule" in ablations:
        ablation_schedule_comparison(env, actor, act_limit, v_threshold, n_ep)

    if "flexibility" in ablations:
        ablation_budget_flexibility(env, actor, act_limit, v_threshold, n_ep)

    if "boltzmann" in ablations:
        ablation_boltzmann_response(env, actor, act_limit, v_threshold, n_ep)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
