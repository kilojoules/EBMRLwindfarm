#!/usr/bin/env python3
"""
Additional baselines for the Saute comparison (Reviewer Criticism #4/#5).

Implements:
1. Linear ramp: lambda increases linearly with time (simple heuristic)
2. PID controller on penalty weight (adapts based on violation rate)
3. Constant penalty sweep (find weight matching AC's budget usage)

Usage:
    python scripts/additional_baselines.py --checkpoint checkpoints/sac_cheetah.pt
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


def load_agent(checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt["act_limit"]


def run_ep_with_lambda_fn(env, actor, act_limit, lambda_fn, horizon, v_threshold,
                          budget, steepness=3.0, action_threshold=0.5):
    """Run one episode with a custom lambda(t, budget_remaining) function."""
    obs, _ = env.reset()
    ep_rew, violations = 0.0, 0

    for t in range(horizon):
        lam = lambda_fn(t, budget - violations, budget, horizon - t, horizon)

        with torch.no_grad():
            a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
        action = a.squeeze(0).numpy() * act_limit

        if violations < budget:
            action_mag = np.abs(action / act_limit).mean()
            excess = max(action_mag - action_threshold, 0)
            penalty = np.exp(steepness * excess) - 1.0
            scale = 1.0 / (1.0 + 0.1 * lam * penalty)
            action = action * scale

        obs, rew, term, trunc, info = env.step(action)
        velocity = abs(obs[8]) if len(obs) > 8 else abs(obs[0])
        if velocity > v_threshold:
            violations += 1
        ep_rew += rew
        if term or trunc:
            break

    return ep_rew, violations


def run_baseline(env, actor, act_limit, name, lambda_fn, horizon, v_threshold,
                 budget, n_episodes):
    """Run multiple episodes and report stats."""
    rewards, viols = [], []
    for _ in range(n_episodes):
        r, v = run_ep_with_lambda_fn(env, actor, act_limit, lambda_fn,
                                      horizon, v_threshold, budget)
        rewards.append(r)
        viols.append(v)
    mr, sr = np.mean(rewards), np.std(rewards)
    mv = np.mean(viols)
    util = 100 * mv / budget
    print(f"  {name:<30s} | {mr:8.1f} ± {sr:5.1f} | {mv:5.0f}/{budget} ({util:5.1f}%)")
    return mr, mv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="checkpoints/sac_cheetah.pt")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--v-threshold", type=float, default=None)
    cli = parser.parse_args()

    actor, act_limit = load_agent(cli.checkpoint)
    env = gym.make("HalfCheetah-v5")
    env = TimeVaryingRewardWrapper(env, amplitude=0.5, period=200)

    horizon = 1000
    budget_frac = 0.25
    budget = int(horizon * budget_frac)
    n_ep = cli.n_episodes

    if cli.v_threshold is None:
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
    else:
        v_threshold = cli.v_threshold

    print(f"Velocity threshold: {v_threshold:.2f}")
    print(f"Budget: {budget}/{horizon} steps ({budget_frac*100:.0f}%)")
    print()
    print("=" * 70)
    print("  Additional Baselines Comparison")
    print("=" * 70)
    print(f"  {'Method':<30s} | {'Reward':>14s} | {'Violations':>15s}")
    print("-" * 70)

    # 1. Unconstrained
    run_baseline(env, actor, act_limit, "Unconstrained",
                 lambda t, br, B, tr, T: 0.0,
                 horizon, v_threshold, budget, n_ep)

    # 2. Constant penalty (sweep to find best)
    best_const_reward = -1e9
    for lam_const in [0.1, 0.5, 1.0, 2.0, 5.0]:
        mr, mv = run_baseline(
            env, actor, act_limit, f"Constant (λ={lam_const})",
            lambda t, br, B, tr, T, lc=lam_const: lc,
            horizon, v_threshold, budget, n_ep)

    # 3. Linear ramp: lambda increases linearly from 0 to 2 over the episode
    run_baseline(env, actor, act_limit, "Linear ramp (0→2)",
                 lambda t, br, B, tr, T: 2.0 * t / T,
                 horizon, v_threshold, budget, n_ep)

    # 4. Linear ramp based on budget depletion
    run_baseline(env, actor, act_limit, "Budget-linear (br/B → 0)",
                 lambda t, br, B, tr, T: max(1.0 - br / max(B, 1), 0) * 5.0,
                 horizon, v_threshold, budget, n_ep)

    # 5. PID controller on violation rate
    # Tracks error = (actual_rate - target_rate) and adjusts lambda
    class PIDLambda:
        def __init__(self, kp=2.0, ki=0.5, kd=0.1, target_rate=None):
            self.kp, self.ki, self.kd = kp, ki, kd
            self.integral = 0.0
            self.prev_error = 0.0
            self.lam = 1.0

        def __call__(self, t, br, B, tr, T):
            if t == 0:
                self.integral = 0.0
                self.prev_error = 0.0
                self.lam = 1.0
                return 1.0

            # Target: spend budget uniformly (TWAP rate)
            target_spent = B * t / T
            actual_spent = B - br
            error = actual_spent - target_spent  # positive = overspending

            self.integral += error
            derivative = error - self.prev_error
            self.prev_error = error

            self.lam = max(0.01, 1.0 + self.kp * error + self.ki * self.integral + self.kd * derivative)
            return min(self.lam, 1e6)

    run_baseline(env, actor, act_limit, "PID (Kp=2, Ki=0.5, Kd=0.1)",
                 PIDLambda(kp=2.0, ki=0.5, kd=0.1),
                 horizon, v_threshold, budget, n_ep)

    run_baseline(env, actor, act_limit, "PID (Kp=5, Ki=1, Kd=0.2)",
                 PIDLambda(kp=5.0, ki=1.0, kd=0.2),
                 horizon, v_threshold, budget, n_ep)

    # 6. AC schedule (our method)
    class ACLambda:
        def __init__(self, eta=2.0):
            self.eta = eta
        def __call__(self, t, br, B, tr, T):
            eps = 1e-6
            bf = max(br, 0) / max(B, 1)
            tf = max(tr, 1) / max(T, 1)
            u = max(bf / max(tf, eps), eps)
            ac = np.exp(self.eta * (1.0 / u - 1.0))
            dep = max(1.0 - bf / 0.05, 0)
            wall = np.exp(3.0 * dep)
            return min(ac * wall, 1e6)

    run_baseline(env, actor, act_limit, "AC schedule (η=2)",
                 ACLambda(eta=2.0),
                 horizon, v_threshold, budget, n_ep)

    run_baseline(env, actor, act_limit, "AC schedule (η=5)",
                 ACLambda(eta=5.0),
                 horizon, v_threshold, budget, n_ep)

    env.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
