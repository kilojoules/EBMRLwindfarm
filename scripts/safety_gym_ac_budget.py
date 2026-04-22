#!/usr/bin/env python3
"""
Adaptive Budget scheduling for Safety Gymnasium environments.

Applies the AC-inspired cumulative budget constraint to Safety Gym's
hazard/constraint environments. The agent has a limited budget of
"hazard exposure" timesteps per episode — it can enter costly zones
but must manage that budget across the episode.

The same urgency-based penalty schedule from the wind farm and
HalfCheetah domains composes post-hoc with a pre-trained policy.

Environments tested:
  - SafetyPointGoal1-v0: Point robot navigating to goal, hazards present
  - SafetyCarGoal1-v0: Car robot variant

Usage:
    # Train unconstrained policy
    python scripts/safety_gym_ac_budget.py --train --env SafetyPointGoal1-v0

    # Evaluate with AC budget constraint
    python scripts/safety_gym_ac_budget.py --eval --env SafetyPointGoal1-v0 \
        --checkpoint checkpoints/sac_safety_point.pt --budget-frac 0.10

    # Full comparison: unconstrained vs constant vs AC
    python scripts/safety_gym_ac_budget.py --compare --env SafetyPointGoal1-v0 \
        --checkpoint checkpoints/sac_safety_point.pt
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym


# =============================================================================
# HAZARD BUDGET SURROGATE
# =============================================================================

class HazardBudgetSurrogate:
    """
    AC-inspired budget surrogate for Safety Gym hazard constraints.

    Tracks cumulative cost (timesteps where cost > 0) and computes a
    time-varying penalty weight based on remaining budget and time.

    The cost signal comes from the environment's built-in cost function
    (proximity to hazards, entering restricted zones, etc.).

    This is the same urgency-based schedule as NegativeYawBudgetSurrogate
    and VelocityBudgetSurrogate, applied to spatial safety constraints.
    """

    def __init__(self, budget_steps, horizon_steps, risk_aversion=2.0,
                 steepness=3.0, wall_threshold=0.05):
        self.budget_steps = budget_steps
        self.horizon_steps = horizon_steps
        self.risk_aversion = risk_aversion
        self.steepness = steepness
        self.wall_threshold = wall_threshold

        self.current_step = 0
        self.cumulative_cost = 0

    def reset(self):
        self.current_step = 0
        self.cumulative_cost = 0

    def update(self, cost):
        """Update after environment step. cost > 0 means constraint violated."""
        if cost > 0:
            self.cumulative_cost += 1
        self.current_step += 1

    def compute_lambda(self):
        """Compute the time-varying penalty weight."""
        eps = 1e-6
        budget_remaining = max(self.budget_steps - self.cumulative_cost, 0)
        time_remaining = max(self.horizon_steps - self.current_step, 1)

        budget_fraction = budget_remaining / max(self.budget_steps, 1)
        time_fraction = time_remaining / max(self.horizon_steps, 1)

        urgency = budget_fraction / max(time_fraction, eps)
        safe_urgency = max(urgency, eps)

        ac_weight = np.exp(self.risk_aversion * (1.0 / safe_urgency - 1.0))

        depletion = max(1.0 - budget_fraction / self.wall_threshold, 0)
        hard_wall = np.exp(self.steepness * depletion)

        return min(ac_weight * hard_wall, 1e6)

    @property
    def budget_utilization(self):
        return self.cumulative_cost / max(self.budget_steps, 1)

    @property
    def budget_remaining_frac(self):
        return max(0, self.budget_steps - self.cumulative_cost) / max(self.budget_steps, 1)


# =============================================================================
# SAC AGENT (reused from safety_gym_budget.py)
# =============================================================================

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        return self.mu(h), self.log_std(h).clamp(-20, 2)

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(-1, keepdim=True), mu


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], -1))


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=200000):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.cap = 0, 0, capacity

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idx]),
            torch.FloatTensor(self.act[idx]),
            torch.FloatTensor(self.rew[idx]).unsqueeze(1),
            torch.FloatTensor(self.next_obs[idx]),
            torch.FloatTensor(self.done[idx]).unsqueeze(1),
        )


# =============================================================================
# TRAINING (unconstrained)
# =============================================================================

def train_sac(env_name, total_timesteps=200000, save_path="checkpoints/sac_safety.pt",
              seed=1):
    """Train standard SAC (unconstrained) on Safety Gym environment."""
    import safety_gymnasium

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = safety_gymnasium.make(env_name)
    obs_dim = env.observation_space.shape[0]
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
    gamma, tau = 0.99, 0.005
    learning_starts = 5000

    obs, _ = env.reset(seed=seed)
    ep_ret, ep_cost, ep_len, ep_count = 0.0, 0.0, 0, 0

    for step in range(1, total_timesteps + 1):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = a.squeeze(0).numpy() * act_limit

        step_result = env.step(action)
        # safety-gymnasium returns 6 values: obs, reward, cost, term, trunc, info
        if len(step_result) == 6:
            next_obs, reward, cost, term, trunc, info = step_result
        else:
            next_obs, reward, term, trunc, info = step_result
            cost = info.get("cost", 0.0)
        done = term or trunc

        buf.add(obs, action / act_limit, reward, next_obs, float(term))
        obs = next_obs
        ep_ret += reward
        ep_cost += cost
        ep_len += 1

        if done:
            ep_count += 1
            if ep_count % 10 == 0:
                print(f"Step {step}: ep_return={ep_ret:.1f}, ep_cost={ep_cost:.0f}, ep_len={ep_len}")
            obs, _ = env.reset()
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

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

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "actor": actor.state_dict(), "obs_dim": obs_dim,
        "act_dim": act_dim, "act_limit": act_limit,
        "env_name": env_name,
    }, save_path)
    print(f"Saved to {save_path}")
    env.close()


# =============================================================================
# EVALUATION WITH BUDGET CONSTRAINT
# =============================================================================

def compute_hazard_repulsion(obs, n_lidar=16):
    """
    Compute a repulsive correction vector from hazard lidar readings.

    The hazard lidar (last 16 dims of obs) encodes hazard proximity in
    16 angular bins around the agent. High values = close hazard.

    Returns (repulse_forward, repulse_rotation, hazard_intensity):
      - repulse_forward: how much to reduce forward motion (positive = slow down)
      - repulse_rotation: which way to turn away (positive = turn left/CCW)
      - hazard_intensity: max lidar reading (0 = no hazards nearby)
    """
    hazard_lidar = obs[-n_lidar:]
    hazard_lidar = np.clip(hazard_lidar, 0, 1)

    hazard_intensity = float(np.max(hazard_lidar))
    if hazard_intensity < 0.01:
        return 0.0, 0.0, 0.0

    angles = np.linspace(0, 2 * np.pi * (1 - 1 / n_lidar), n_lidar)

    # Weighted hazard direction in ego frame (bin 0 = ahead)
    hx = np.sum(hazard_lidar * np.cos(angles))  # forward component
    hy = np.sum(hazard_lidar * np.sin(angles))  # lateral component

    return float(hx), float(hy), hazard_intensity


def eval_with_budget(env_name, checkpoint, budget_frac=0.10, risk_aversion=2.0,
                     steepness=3.0, n_episodes=10, horizon=1000,
                     correction_scale=0.3):
    """Evaluate with AC budget constraint using directional hazard avoidance."""
    import safety_gymnasium

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    env = safety_gymnasium.make(env_name)
    budget = int(horizon * budget_frac)

    rewards, costs, goal_reached = [], [], []

    for ep in range(n_episodes):
        surr = HazardBudgetSurrogate(
            budget_steps=budget, horizon_steps=horizon,
            risk_aversion=risk_aversion, steepness=steepness,
        )
        surr.reset()

        obs, _ = env.reset()
        ep_rew, ep_cost = 0.0, 0.0
        reached_goal = False

        for t in range(horizon):
            lam = surr.compute_lambda()

            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = a.squeeze(0).numpy() * act_limit

            hx, hy, h_intensity = compute_hazard_repulsion(obs)

            if surr.cumulative_cost >= budget:
                # Budget exhausted: stop completely (zero cost when stationary)
                action = np.zeros_like(action)
            else:
                # Two-layer correction: magnitude scaling + directional push
                # Layer 1: scale down speed proportional to lambda and proximity
                speed_scale = 1.0 / (1.0 + min(lam, 200.0) * correction_scale * h_intensity)
                action = action * speed_scale

                # Layer 2: push away from hazards if very close
                if h_intensity > 0.3:
                    push = min(lam, 50.0) * correction_scale * h_intensity
                    action[0] -= push * hx
                    action[1] -= push * hy

            action = np.clip(action, -act_limit, act_limit)

            step_result = env.step(action)
            if len(step_result) == 6:
                obs, reward, cost, term, trunc, info = step_result
            else:
                obs, reward, term, trunc, info = step_result
                cost = info.get("cost", 0.0)
            surr.update(cost)

            ep_rew += reward
            ep_cost += cost

            if info.get("goal_met", False):
                reached_goal = True

            if term or trunc:
                break

        rewards.append(ep_rew)
        costs.append(surr.cumulative_cost)
        goal_reached.append(reached_goal)

    env.close()
    return {
        "reward": np.mean(rewards), "reward_std": np.std(rewards),
        "cost": np.mean(costs), "cost_std": np.std(costs),
        "budget": budget, "budget_frac": budget_frac,
        "utilization": 100 * np.mean(costs) / max(budget, 1),
        "goal_rate": 100 * np.mean(goal_reached),
    }


def compare_methods(env_name, checkpoint, n_episodes=10, horizon=1000,
                    output_json=None):
    """Full comparison: unconstrained vs constant vs AC at multiple budgets."""
    try:
        import safety_gymnasium  # noqa: F401
    except ImportError:
        pass

    print(f"\n{'='*75}")
    print(f"  Safety Gym AC Budget Comparison: {env_name}")
    print(f"{'='*75}")

    all_results = {}

    # Unconstrained
    res = eval_with_budget(env_name, checkpoint, budget_frac=1.0,
                            risk_aversion=0.0, n_episodes=n_episodes, horizon=horizon)
    print(f"\n  Unconstrained: Reward={res['reward']:.1f}±{res['reward_std']:.1f}, "
          f"Cost={res['cost']:.0f}, GoalRate={res['goal_rate']:.0f}%")
    uncon_reward = res['reward']
    uncon_cost = res['cost']
    all_results["unconstrained"] = res

    print(f"\n  Budget levels (as % of unconstrained cost {uncon_cost:.0f}):")
    print(f"  {'Method':<25s} {'Budget':>7s} {'Reward':>10s} {'Cost':>8s} "
          f"{'%Uncon':>7s} {'Used':>6s} {'GoalRate':>8s}")
    print(f"  {'-'*25} {'-'*7} {'-'*10} {'-'*8} {'-'*7} {'-'*6} {'-'*8}")

    # Set budgets relative to unconstrained cost (not horizon)
    # Standard Safety-Gym cost limit is 25. Unconstrained cost ~56.
    # Use absolute cost budgets that are genuinely binding.
    cost_budgets = [10, 25, 40, 56]  # 18%, 45%, 71%, 100% of unconstrained
    print(f"  Testing absolute cost budgets: {cost_budgets}")
    print(f"  (Unconstrained cost = {uncon_cost:.0f})")

    for cost_budget in cost_budgets:
        budget_frac = cost_budget / horizon  # convert to fraction of horizon
        pct_of_uncon = 100 * cost_budget / max(uncon_cost, 1)

        for ra, label in [(0.0, f"Const (C≤{cost_budget})"),
                           (2.0, f"AC η=2 (C≤{cost_budget})"),
                           (5.0, f"AC η=5 (C≤{cost_budget})")]:
            res = eval_with_budget(env_name, checkpoint, budget_frac=budget_frac,
                                    risk_aversion=ra, n_episodes=n_episodes,
                                    horizon=horizon)
            pct_uncon = 100 * res['reward'] / uncon_reward if uncon_reward != 0 else 0
            budget_used = 100 * res['cost'] / cost_budget if cost_budget > 0 else 0
            print(f"  {label:<25s} {cost_budget:>4d}/{int(uncon_cost)} "
                  f"{res['reward']:>10.1f} {res['cost']:>8.0f} "
                  f"{pct_uncon:>6.1f}% {budget_used:>5.0f}% {res['goal_rate']:>7.0f}%")
            all_results[f"ra{ra}_budget{cost_budget}"] = {
                **res, "pct_unconstrained": pct_uncon, "budget_used_pct": budget_used,
            }

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n  Results saved to {output_json}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Safety Gym AC Budget")
    parser.add_argument("--env", default="SafetyPointGoal1-v0")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--checkpoint", default="checkpoints/sac_safety_point.pt")
    parser.add_argument("--total-timesteps", type=int, default=200000)
    parser.add_argument("--budget-frac", type=float, default=0.10)
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output-json", default=None,
                        help="Save comparison results to JSON (for multi-seed aggregation)")
    cli = parser.parse_args()

    if cli.train:
        print(f"Training SAC on {cli.env} (seed={cli.seed})...")
        train_sac(cli.env, cli.total_timesteps, cli.checkpoint, cli.seed)

    if cli.eval:
        print(f"Evaluating with budget={cli.budget_frac*100:.0f}%...")
        for ra in [0.0, 2.0, 5.0]:
            res = eval_with_budget(cli.env, cli.checkpoint, cli.budget_frac,
                                    risk_aversion=ra, n_episodes=cli.n_episodes,
                                    horizon=cli.horizon)
            print(f"  η={ra}: Reward={res['reward']:.1f}±{res['reward_std']:.1f}, "
                  f"Cost={res['cost']:.0f}/{res['budget']} ({res['utilization']:.0f}%), "
                  f"GoalRate={res['goal_rate']:.0f}%")

    if cli.compare:
        output = cli.output_json
        if output is None and cli.seed != 1:
            output = f"results/safety_gym_seed_{cli.seed}.json"
        compare_methods(cli.env, cli.checkpoint, cli.n_episodes, cli.horizon,
                        output_json=output)

    if not any([cli.train, cli.eval, cli.compare]):
        print("Specify --train, --eval, or --compare")


if __name__ == "__main__":
    main()
