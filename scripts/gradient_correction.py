#!/usr/bin/env python3
"""
Gradient-based action correction for post-hoc budget constraints.

Replaces action scaling with gradient descent on the cost function,
enabling nonlinear and spatial cost constraints on MLP actors.

The correction:
    a_corrected = a_actor - lambda(t) * lr * nabla_a c(s, a)

This is mathematically equivalent to EBT energy composition but applied
to the output of a frozen Gaussian actor. The AC lambda(t) schedule
from the budget surrogate weights the correction strength.

Comparison: action_scaling vs gradient_correction on HalfCheetah and
optionally Safety Gym.

Usage:
    python scripts/gradient_correction.py \
        --checkpoint checkpoints/sac_cheetah.pt \
        --env HalfCheetah-v5
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F
import gymnasium as gym

from scripts.safety_gym_budget import (
    GaussianActor, TimeVaryingRewardWrapper,
)


class GradientCorrectionBudget:
    """
    AC budget surrogate with gradient-based action correction.

    Instead of scaling action magnitude, runs K gradient descent steps
    on the action to minimize the cost function, weighted by lambda(t).

    For velocity cost: c(a) = relu(|a| - threshold)^2
    The gradient pushes actions below the threshold.

    For nonlinear costs: c(s, a) can be any differentiable function.
    The gradient naturally captures nonlinear structure.
    """

    def __init__(self, budget_steps, horizon_steps, risk_aversion=2.0,
                 steepness=3.0, correction_lr=0.1, correction_steps=5):
        self.budget_steps = budget_steps
        self.horizon_steps = horizon_steps
        self.risk_aversion = risk_aversion
        self.steepness = steepness
        self.correction_lr = correction_lr
        self.correction_steps = correction_steps

        self.current_step = 0
        self.cumulative_violations = 0

    def reset(self):
        self.current_step = 0
        self.cumulative_violations = 0

    def update(self, cost):
        if cost > 0:
            self.cumulative_violations += 1
        self.current_step += 1

    def compute_lambda(self):
        eps = 1e-6
        budget_remaining = max(self.budget_steps - self.cumulative_violations, 0)
        time_remaining = max(self.horizon_steps - self.current_step, 1)
        budget_fraction = budget_remaining / max(self.budget_steps, 1)
        time_fraction = time_remaining / max(self.horizon_steps, 1)
        urgency = budget_fraction / max(time_fraction, eps)
        safe_urgency = max(urgency, eps)
        ac_weight = np.exp(self.risk_aversion * (1.0 / safe_urgency - 1.0))
        depletion = max(1.0 - budget_fraction / 0.05, 0)
        hard_wall = np.exp(self.steepness * depletion)
        return min(ac_weight * hard_wall, 1e6)

    def correct_action(self, action_tensor, cost_fn, act_limit):
        """
        Gradient-based action correction.

        Args:
            action_tensor: (act_dim,) tensor, the actor's proposed action (scaled)
            cost_fn: callable(action_tensor) -> scalar cost (differentiable)
            act_limit: float, action space bound

        Returns:
            corrected action as numpy array
        """
        lam = self.compute_lambda()
        a = action_tensor.clone().detach().float()

        for _ in range(self.correction_steps):
            a.requires_grad_(True)
            cost = cost_fn(a)
            if cost.item() == 0:
                a = a.detach()
                break
            grad = torch.autograd.grad(cost, a)[0]
            a = a.detach() - lam * self.correction_lr * grad
            a = a.clamp(-act_limit, act_limit)

        return a.detach().numpy()


def velocity_cost_fn(action, threshold=0.5):
    """Differentiable cost: penalizes large action magnitudes."""
    excess = F.relu(action.abs().mean() - threshold)
    return excess ** 2


def nonlinear_load_cost_fn(action, wind_speed=12.0):
    """
    Nonlinear loading surrogate: fatigue cost depends on yaw AND wind speed.
    Higher wind speed + negative yaw = exponentially more damage.
    This can't be captured by simple action scaling.
    """
    yaw_magnitude = action.abs().mean()
    # Nonlinear interaction: cost scales as yaw^2 * (ws/10)^3
    ws_factor = (wind_speed / 10.0) ** 3
    return yaw_magnitude ** 2 * ws_factor


def run_episode_with_method(env, actor, act_limit, budget, horizon,
                             v_threshold, method="gradient", ra=2.0,
                             cost_fn=None):
    """Run one episode with either gradient correction or action scaling."""
    if method == "gradient":
        surr = GradientCorrectionBudget(
            budget_steps=budget, horizon_steps=horizon,
            risk_aversion=ra, steepness=3.0,
            correction_lr=0.05, correction_steps=5,
        )
    else:
        # Import action-scaling surrogate for comparison
        from scripts.safety_gym_budget import VelocityBudgetSurrogate
        surr = VelocityBudgetSurrogate(
            budget_steps=budget, horizon_steps=horizon,
            risk_aversion=ra, steepness=3.0,
        )

    surr.reset()
    obs, _ = env.reset()
    ep_rew, violations = 0.0, 0

    for t in range(horizon):
        with torch.no_grad():
            a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
        action = a.squeeze(0) * act_limit  # tensor, scaled

        if method == "gradient" and violations < budget:
            if cost_fn is None:
                cfn = lambda a: velocity_cost_fn(a, threshold=0.5)
            else:
                cfn = cost_fn
            action_np = surr.correct_action(action, cfn, act_limit)
        elif method == "scaling" and violations < budget:
            action_np = action.numpy()
            penalty = surr.penalize_action(action_np / act_limit)
            scale = 1.0 / (1.0 + 0.1 * surr._compute_lambda() * penalty)
            action_np = action_np * scale
        else:
            action_np = action.numpy()

        obs, rew, term, trunc, info = env.step(action_np)
        velocity = abs(obs[8]) if len(obs) > 8 else abs(obs[0])

        cost = 1.0 if velocity > v_threshold else 0.0
        if method == "gradient":
            surr.update(cost)
        else:
            surr.update(velocity, v_threshold)

        if velocity > v_threshold:
            violations += 1

        ep_rew += rew
        if term or trunc:
            break

    return ep_rew, violations


def compare_methods(checkpoint, env_name="HalfCheetah-v5", n_episodes=10,
                    horizon=1000, budget_frac=0.25):
    """Compare gradient correction vs action scaling."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    env = gym.make(env_name)
    env = TimeVaryingRewardWrapper(env, amplitude=0.5, period=200)

    budget = int(horizon * budget_frac)

    # Auto velocity threshold
    velocities = []
    for _ in range(3):
        obs, _ = env.reset()
        for _ in range(horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            obs, _, t, tr, _ = env.step(a.squeeze(0).numpy() * act_limit)
            velocities.append(abs(obs[8]) if len(obs) > 8 else abs(obs[0]))
            if t or tr:
                break
    v_threshold = np.percentile(velocities, 50)
    print(f"Velocity threshold: {v_threshold:.2f}")
    print(f"Budget: {budget}/{horizon} steps ({budget_frac*100:.0f}%)")

    print(f"\n{'='*70}")
    print(f"  Gradient Correction vs Action Scaling")
    print(f"{'='*70}")
    print(f"  {'Method':<30s} {'RA':>4s} {'Reward':>10s} {'Violations':>12s} {'%Used':>6s}")
    print("-" * 70)

    # Unconstrained
    rewards_u = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_rew = 0.0
        for _ in range(horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            obs, rew, t, tr, _ = env.step(a.squeeze(0).numpy() * act_limit)
            ep_rew += rew
            if t or tr:
                break
        rewards_u.append(ep_rew)
    print(f"  {'Unconstrained':<30s} {'—':>4s} {np.mean(rewards_u):10.1f} {'—':>12s} {'—':>6s}")

    for ra in [0.0, 2.0, 5.0]:
        for method, label in [("scaling", "Action Scaling"),
                               ("gradient", "Gradient Correction")]:
            rewards, viols = [], []
            for _ in range(n_episodes):
                r, v = run_episode_with_method(
                    env, actor, act_limit, budget, horizon,
                    v_threshold, method=method, ra=ra,
                )
                rewards.append(r)
                viols.append(v)
            mr = np.mean(rewards)
            mv = np.mean(viols)
            util = 100 * mv / budget
            pct = 100 * mr / np.mean(rewards_u)
            print(f"  {label+f' (η={ra})':<30s} {ra:4.1f} {mr:10.1f} "
                  f"{mv:5.0f}/{budget} {util:5.1f}%")

    # Also test with nonlinear cost function
    print(f"\n--- Nonlinear loading surrogate (cost ~ yaw^2 * ws^3) ---")
    for ra in [2.0, 5.0]:
        for method, label in [("scaling", "Scaling"),
                               ("gradient", "Gradient")]:
            rewards, viols = [], []
            for _ in range(n_episodes):
                r, v = run_episode_with_method(
                    env, actor, act_limit, budget, horizon,
                    v_threshold, method=method, ra=ra,
                    cost_fn=lambda a: nonlinear_load_cost_fn(a, wind_speed=14.0),
                )
                rewards.append(r)
                viols.append(v)
            mr = np.mean(rewards)
            mv = np.mean(viols)
            util = 100 * mv / budget
            print(f"  {label+f' (η={ra}, nonlinear)':<30s} {ra:4.1f} {mr:10.1f} "
                  f"{mv:5.0f}/{budget} {util:5.1f}%")

    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sac_cheetah.pt")
    parser.add_argument("--env", default="HalfCheetah-v5")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--budget-frac", type=float, default=0.25)
    cli = parser.parse_args()

    compare_methods(cli.checkpoint, cli.env, cli.n_episodes,
                    budget_frac=cli.budget_frac)


if __name__ == "__main__":
    main()
