#!/usr/bin/env python3
"""
Gradient correction for Safety Gym SafetyPointGoal1-v0.

Action scaling fails on Safety Gym because scaling action magnitude
makes the agent slower but doesn't redirect it away from hazards.
Gradient correction uses the cost function's gradient to steer the
agent's velocity DIRECTION away from hazards.

The key: Safety Gym's cost depends on the agent's position relative
to hazards. The action controls velocity. The gradient of hazard cost
w.r.t. action points in the direction that INCREASES hazard proximity.
Subtracting this gradient steers the agent AWAY from hazards.

Usage:
    python scripts/gradient_correction_safety.py \
        --checkpoint checkpoints/sac_safety_point.pt
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

# Import the actor and surrogate from the safety gym script
from scripts.safety_gym_ac_budget import (
    GaussianActor, HazardBudgetSurrogate,
)


def hazard_cost_from_obs(obs_tensor, action_tensor):
    """
    Differentiable directional hazard cost from observation.

    The hazard lidar (last 16 dims) encodes hazard proximity in angular
    bins. We compute the dot product between the action and the hazard
    direction — penalizing actions that move the agent *toward* hazards.

    The gradient w.r.t. action points away from hazard directions,
    steering the agent around obstacles rather than just slowing it.
    """
    n_lidar = 16
    hazard_lidar = obs_tensor[-n_lidar:].clamp(min=0)

    angles = torch.linspace(0, 2 * np.pi * (1 - 1 / n_lidar), n_lidar)

    # Hazard direction in ego frame
    hx = (hazard_lidar * torch.cos(angles)).sum()
    hy = (hazard_lidar * torch.sin(angles)).sum()

    # Alignment: how much does the action move toward hazards?
    # action[0] = forward, action[1] = rotation
    alignment = action_tensor[0] * hx + action_tensor[1] * hy

    hazard_intensity = hazard_lidar.max()

    # Softplus for smooth gradient (unlike ReLU which has zero gradient when alignment < 0)
    return F.softplus(alignment, beta=2.0) * (1.0 + hazard_intensity)


def run_episode_gradient(env, actor, act_limit, budget_surr, horizon,
                          obs_tensor_device="cpu", correction_lr=0.05,
                          correction_steps=5):
    """Run one episode with gradient-based action correction."""
    budget_surr.reset()
    obs, _ = env.reset()
    ep_rew, ep_cost = 0.0, 0.0

    for t in range(horizon):
        lam = budget_surr.compute_lambda()

        with torch.no_grad():
            a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
        action = a.squeeze(0) * act_limit  # tensor

        # Gradient correction: steer away from hazard directions
        obs_t = torch.FloatTensor(obs)
        a_corr = action.clone().detach()

        # Use high lambda even after budget exhaustion (maximum avoidance)
        effective_lam = lam if budget_surr.cumulative_cost < budget_surr.budget_steps else 100.0

        for _ in range(correction_steps):
            a_corr.requires_grad_(True)
            cost = hazard_cost_from_obs(obs_t, a_corr)
            grad = torch.autograd.grad(cost, a_corr)[0]
            a_corr = a_corr.detach() - effective_lam * correction_lr * grad
            a_corr = a_corr.clamp(-act_limit, act_limit)

        action_np = a_corr.detach().numpy()

        step_result = env.step(action_np)
        if len(step_result) == 6:
            obs, reward, cost, term, trunc, info = step_result
        else:
            obs, reward, term, trunc, info = step_result
            cost = info.get("cost", 0.0)

        budget_surr.update(cost)
        ep_rew += reward
        ep_cost += cost

        if term or trunc:
            break

    return ep_rew, budget_surr.cumulative_cost


def run_episode_scaling(env, actor, act_limit, budget_surr, horizon):
    """Run one episode with action scaling (for comparison)."""
    budget_surr.reset()
    obs, _ = env.reset()
    ep_rew, ep_cost = 0.0, 0.0

    for t in range(horizon):
        lam = budget_surr.compute_lambda()

        with torch.no_grad():
            a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
        action = a.squeeze(0).numpy() * act_limit

        # Action scaling (always active — stronger when budget is depleted)
        effective_lam = lam if budget_surr.cumulative_cost < budget_surr.budget_steps else 100.0
        action_mag = np.linalg.norm(action)
        penalty = max(0, action_mag - 0.3)
        scale = 1.0 / (1.0 + 0.1 * effective_lam * penalty)
        action = action * scale

        step_result = env.step(action)
        if len(step_result) == 6:
            obs, reward, cost, term, trunc, info = step_result
        else:
            obs, reward, term, trunc, info = step_result
            cost = info.get("cost", 0.0)

        budget_surr.update(cost)
        ep_rew += reward
        ep_cost += cost

        if term or trunc:
            break

    return ep_rew, budget_surr.cumulative_cost


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sac_safety_point.pt")
    parser.add_argument("--env", default="SafetyPointGoal1-v0")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    cli = parser.parse_args()

    import safety_gymnasium

    ckpt = torch.load(cli.checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    env = safety_gymnasium.make(cli.env)

    # Get unconstrained cost baseline
    print(f"Running unconstrained baseline...")
    uncon_rewards, uncon_costs = [], []
    for _ in range(cli.n_episodes):
        obs, _ = env.reset()
        ep_rew, ep_cost = 0.0, 0.0
        for _ in range(cli.horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            step_result = env.step(a.squeeze(0).numpy() * act_limit)
            if len(step_result) == 6:
                obs, reward, cost, term, trunc, info = step_result
            else:
                obs, reward, term, trunc, info = step_result
                cost = info.get("cost", 0.0)
            ep_rew += reward
            ep_cost += cost
            if term or trunc:
                break
        uncon_rewards.append(ep_rew)
        uncon_costs.append(ep_cost)

    uncon_cost = np.mean(uncon_costs)
    print(f"Unconstrained: Reward={np.mean(uncon_rewards):.1f}±{np.std(uncon_rewards):.1f}, "
          f"Cost={uncon_cost:.0f}±{np.std(uncon_costs):.0f}")

    # Test budget levels relative to unconstrained cost
    cost_budgets = [
        int(uncon_cost * 0.25),  # 25% of unconstrained
        int(uncon_cost * 0.50),  # 50%
        int(uncon_cost * 0.75),  # 75%
        int(uncon_cost),         # 100%
    ]

    print(f"\n{'='*75}")
    print(f"  Gradient Correction vs Action Scaling on {cli.env}")
    print(f"  Unconstrained cost = {uncon_cost:.0f}")
    print(f"{'='*75}")
    print(f"  {'Method':<35s} {'Budget':>6s} {'Reward':>8s} {'Cost':>6s} "
          f"{'%Uncon':>7s} {'CostOK':>7s}")
    print("-" * 75)

    for cost_budget in cost_budgets:
        for ra in [2.0, 5.0]:
            for method_name, run_fn in [("Action Scaling", "scaling"),
                                         ("Gradient Correction", "gradient")]:
                rewards, costs = [], []
                for _ in range(cli.n_episodes):
                    surr = HazardBudgetSurrogate(
                        budget_steps=cost_budget, horizon_steps=cli.horizon,
                        risk_aversion=ra, steepness=3.0,
                    )
                    if run_fn == "gradient":
                        r, c = run_episode_gradient(env, actor, act_limit,
                                                     surr, cli.horizon)
                    else:
                        r, c = run_episode_scaling(env, actor, act_limit,
                                                    surr, cli.horizon)
                    rewards.append(r)
                    costs.append(c)

                mr = np.mean(rewards)
                mc = np.mean(costs)
                pct = 100 * mr / np.mean(uncon_rewards)
                ok = "✓" if mc <= cost_budget * 1.1 else "✗"
                label = f"{method_name} η={ra}"
                print(f"  {label:<35s} {cost_budget:>4d} {mr:>8.1f} {mc:>6.0f} "
                      f"{pct:>6.1f}% {ok:>7s}")

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
