#!/usr/bin/env python3
"""
Saute RL baseline for HalfCheetah velocity budget.

Saute RL (Sootla et al., ICML 2022) augments the state with remaining
safety budget and reshapes the reward to penalize constraint violations.
This requires retraining the policy for each budget level.

We compare:
1. Saute RL (retrained with budget in state) at budget=25%
2. Our post-hoc AC schedule (zero retraining) at budget=25%
3. Our post-hoc AC schedule at budget=10%, 50% (no retraining needed)

The key comparison: Saute RL trains a separate policy per budget level.
Our method uses ONE policy for ALL budget levels.

Usage:
    python scripts/saute_rl_baseline.py --train --budget-frac 0.25
    python scripts/saute_rl_baseline.py --eval --checkpoint checkpoints/saute_cheetah_b25.pt
    python scripts/saute_rl_baseline.py --compare
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

from scripts.safety_gym_budget import (
    GaussianActor, QNetwork, ReplayBuffer, TimeVaryingRewardWrapper,
)


class SauteWrapper(gym.Wrapper):
    """
    Saute RL wrapper: augments observation with remaining budget fraction
    and reshapes reward to penalize budget violations.

    obs_augmented = [obs, budget_remaining / budget_total]
    reward_saute = reward - penalty * max(0, violations - budget)
    """

    def __init__(self, env, budget_frac=0.25, v_threshold=5.8,
                 penalty_coeff=10.0, horizon=1000):
        super().__init__(env)
        self.budget = int(horizon * budget_frac)
        self.v_threshold = v_threshold
        self.penalty_coeff = penalty_coeff
        self.horizon = horizon
        self.violations = 0
        self.step_count = 0

        # Augment observation space with budget fraction
        low = np.concatenate([env.observation_space.low, [0.0]])
        high = np.concatenate([env.observation_space.high, [1.0]])
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.violations = 0
        self.step_count = 0
        return self._augment(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        self.step_count += 1

        velocity = abs(obs[8]) if len(obs) > 8 else abs(obs[0])
        if velocity > self.v_threshold:
            self.violations += 1

        # Saute penalty: penalize when cumulative violations exceed budget
        if self.violations > self.budget:
            reward -= self.penalty_coeff

        info["saute_violations"] = self.violations
        info["saute_budget_remaining"] = max(0, self.budget - self.violations)

        return self._augment(obs), reward, term, trunc, info

    def _augment(self, obs):
        budget_frac = max(0, self.budget - self.violations) / max(self.budget, 1)
        return np.concatenate([obs, [budget_frac]])


def train_saute(budget_frac=0.25, total_timesteps=100000, v_threshold=5.8,
                save_path="checkpoints/saute_cheetah.pt",
                reward_amplitude=0.5, reward_period=200, seed=1):
    """Train SAC with Saute RL wrapper (budget-aware)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = gym.make("HalfCheetah-v5")
    env = TimeVaryingRewardWrapper(env, amplitude=reward_amplitude, period=reward_period)
    env = SauteWrapper(env, budget_frac=budget_frac, v_threshold=v_threshold)

    obs_dim = env.observation_space.shape[0]  # original + 1 (budget frac)
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = GaussianActor(obs_dim, act_dim)
    qf1 = QNetwork(obs_dim, act_dim)
    qf2 = QNetwork(obs_dim, act_dim)
    qf1_target = QNetwork(obs_dim, act_dim)
    qf2_target = QNetwork(obs_dim, act_dim)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    q_opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=3e-4)

    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -act_dim

    buf = ReplayBuffer(obs_dim, act_dim)
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    learning_starts = 5000

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_len, ep_count = 0.0, 0, 0

    for step in range(1, total_timesteps + 1):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = a.squeeze(0).numpy() * act_limit

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        buf.add(obs, action / act_limit, reward, next_obs, float(term))
        obs = next_obs
        ep_ret += reward
        ep_len += 1

        if done:
            ep_count += 1
            if ep_count % 10 == 0:
                viol = info.get("saute_violations", 0)
                print(f"Step {step}: ep_return={ep_ret:.1f}, violations={viol}, ep_len={ep_len}")
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if step >= learning_starts and buf.size >= batch_size:
            o, a, r, no, d = buf.sample(batch_size)
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                na, nlp, _ = actor.sample(no)
                qt = torch.min(qf1_target(no, na), qf2_target(no, na)) - alpha * nlp
                target = r + gamma * (1 - d) * qt

            q_loss = F.mse_loss(qf1(o, a), target) + F.mse_loss(qf2(o, a), target)
            q_opt.zero_grad()
            q_loss.backward()
            q_opt.step()

            sa, slp, _ = actor.sample(o)
            actor_loss = (alpha * slp - torch.min(qf1(o, sa), qf2(o, sa))).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            alpha_loss = -(log_alpha * (slp.detach() + target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            for p, pt in zip(qf1.parameters(), qf1_target.parameters()):
                pt.data.lerp_(p.data, tau)
            for p, pt in zip(qf2.parameters(), qf2_target.parameters()):
                pt.data.lerp_(p.data, tau)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_limit": act_limit,
        "budget_frac": budget_frac,
        "v_threshold": v_threshold,
    }, save_path)
    print(f"Saved Saute checkpoint to {save_path}")
    env.close()


def eval_saute(checkpoint, n_episodes=10, horizon=1000,
               reward_amplitude=0.5, reward_period=200):
    """Evaluate a Saute-trained agent."""
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]
    budget_frac = ckpt["budget_frac"]
    v_threshold = ckpt["v_threshold"]

    env = gym.make("HalfCheetah-v5")
    env = TimeVaryingRewardWrapper(env, amplitude=reward_amplitude, period=reward_period)
    env = SauteWrapper(env, budget_frac=budget_frac, v_threshold=v_threshold)

    budget = int(horizon * budget_frac)
    rewards, violations_list = [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_rew = 0.0
        for _ in range(horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = a.squeeze(0).numpy() * act_limit
            obs, rew, term, trunc, info = env.step(action)
            # Use raw reward (without Saute penalty) for fair comparison
            ep_rew += info.get("raw_reward", rew)
            if term or trunc:
                break
        rewards.append(ep_rew)
        violations_list.append(info.get("saute_violations", 0))

    mean_rew = np.mean(rewards)
    mean_viol = np.mean(violations_list)
    util = 100 * mean_viol / budget

    print(f"Saute RL (budget={budget_frac*100:.0f}%): "
          f"Reward={mean_rew:.1f} +/- {np.std(rewards):.1f}, "
          f"Violations={mean_viol:.0f}/{budget} ({util:.0f}%)")

    env.close()
    return {"reward": mean_rew, "violations": mean_viol, "utilization": util}


def compare_all(saute_ckpt, ac_ckpt, n_episodes=10):
    """Compare Saute RL vs post-hoc AC at same and different budgets."""
    from scripts.safety_gym_budget import VelocityBudgetSurrogate

    print("=" * 70)
    print("  Saute RL vs Post-Hoc AC Budget Scheduling")
    print("=" * 70)

    # Saute at its trained budget
    saute_data = torch.load(saute_ckpt, map_location="cpu", weights_only=False)
    budget_frac = saute_data["budget_frac"]
    v_threshold = saute_data["v_threshold"]

    print(f"\n--- Budget = {budget_frac*100:.0f}% (Saute was trained at this level) ---")
    saute_result = eval_saute(saute_ckpt, n_episodes)

    # AC at same budget (post-hoc, no retraining)
    from scripts.safety_gym_budget import eval_with_budget
    print(f"\nAC post-hoc at same budget ({budget_frac*100:.0f}%):")
    eval_with_budget("HalfCheetah-v5", ac_ckpt, budget_frac,
                     risk_aversion=2.0, n_episodes=n_episodes,
                     v_threshold=v_threshold)

    # AC at different budgets (still no retraining!)
    for frac in [0.10, 0.50]:
        print(f"\nAC post-hoc at budget={frac*100:.0f}% (NO retraining):")
        eval_with_budget("HalfCheetah-v5", ac_ckpt, frac,
                         risk_aversion=2.0, n_episodes=n_episodes,
                         v_threshold=v_threshold)

    print(f"\nSaute RL would need RETRAINING for budget=10% and 50%.")
    print("Post-hoc AC handles all budget levels with zero additional compute.")


def main():
    parser = argparse.ArgumentParser(description="Saute RL baseline")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--budget-frac", type=float, default=0.25)
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--checkpoint", default="checkpoints/saute_cheetah_b25.pt")
    parser.add_argument("--ac-checkpoint", default="checkpoints/sac_cheetah.pt")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1)
    cli = parser.parse_args()

    if cli.train:
        print(f"Training Saute RL (budget={cli.budget_frac*100:.0f}%)...")
        train_saute(cli.budget_frac, cli.total_timesteps, save_path=cli.checkpoint,
                    seed=cli.seed)

    if cli.eval:
        eval_saute(cli.checkpoint, cli.n_episodes)

    if cli.compare:
        compare_all(cli.checkpoint, cli.ac_checkpoint, cli.n_episodes)


if __name__ == "__main__":
    main()
