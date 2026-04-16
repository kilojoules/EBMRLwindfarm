#!/usr/bin/env python3
"""
Safety Gym with learned cost predictor + predictive urgency.

Pipeline:
  1. Roll out unconstrained policy, collect (s, a, c_soft) tuples
  2. Train a one-step cost predictor c_hat(s, a) -> E[c_soft(s')]
  3. Evaluate with AC budget schedule using nabla_a c_hat for gradient correction
     and predictive urgency u_tilde

The soft cost is max(hazard_lidar) -- a continuous [0,1] signal that has
gradients everywhere, unlike the binary cost which is 0 almost everywhere
and 1 only inside hazard zones.

Usage:
    python scripts/safety_gym_cost_predictor.py \
        --checkpoint checkpoints/sac_safety_point.pt \
        --collect --train --compare
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


# =============================================================================
# MODELS
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


class CostPredictor(nn.Module):
    """Predicts soft cost of next state given current (obs, action)."""

    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1), nn.Sigmoid(),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], dim=-1))


# =============================================================================
# BUDGET SURROGATE WITH PREDICTIVE URGENCY
# =============================================================================

class PredictiveBudgetSurrogate:
    """
    AC-inspired budget surrogate with learned cost predictor.

    Two improvements over the naive urgency ratio:
    1. Gradient correction uses nabla_a c_hat(s, a) instead of action scaling
    2. Predictive urgency subtracts expected future cost from remaining budget
    """

    def __init__(self, cost_predictor, budget_steps, horizon_steps,
                 risk_aversion=2.0, steepness=3.0, correction_lr=0.1,
                 correction_steps=3, lookahead_discount=0.95):
        self.cost_pred = cost_predictor
        self.budget_steps = budget_steps
        self.horizon_steps = horizon_steps
        self.risk_aversion = risk_aversion
        self.steepness = steepness
        self.correction_lr = correction_lr
        self.correction_steps = correction_steps
        self.lookahead_discount = lookahead_discount

        self.current_step = 0
        self.cumulative_cost = 0

    def reset(self):
        self.current_step = 0
        self.cumulative_cost = 0

    def update(self, cost):
        if cost > 0:
            self.cumulative_cost += 1
        self.current_step += 1

    def _predicted_future_cost(self, obs_tensor, action_tensor):
        """Estimate expected future cost if continuing current behavior."""
        with torch.no_grad():
            c_hat = self.cost_pred(obs_tensor.unsqueeze(0),
                                    action_tensor.unsqueeze(0)).item()
        time_remaining = max(self.horizon_steps - self.current_step, 1)
        # Geometric sum: c_hat * (1 + γ + γ² + ... + γ^(T-t-1))
        if self.lookahead_discount < 1.0:
            geo = (1 - self.lookahead_discount ** time_remaining) / (1 - self.lookahead_discount)
        else:
            geo = time_remaining
        return c_hat * geo

    def compute_lambda(self, obs_tensor=None, action_tensor=None):
        """Compute penalty weight with optional predictive urgency."""
        eps = 1e-6
        budget_remaining = max(self.budget_steps - self.cumulative_cost, 0)
        time_remaining = max(self.horizon_steps - self.current_step, 1)

        # Predictive urgency: subtract expected future cost from budget
        if obs_tensor is not None and action_tensor is not None:
            predicted_cost = self._predicted_future_cost(obs_tensor, action_tensor)
            # Effective budget = remaining - what we expect to spend
            effective_remaining = max(budget_remaining - predicted_cost * 0.1, 0)
        else:
            effective_remaining = budget_remaining

        budget_fraction = effective_remaining / max(self.budget_steps, 1)
        time_fraction = time_remaining / max(self.horizon_steps, 1)

        urgency = budget_fraction / max(time_fraction, eps)
        safe_urgency = max(urgency, eps)

        ac_weight = np.exp(self.risk_aversion * (1.0 / safe_urgency - 1.0))

        depletion = max(1.0 - budget_fraction / 0.05, 0)
        hard_wall = np.exp(self.steepness * depletion)

        return min(ac_weight * hard_wall, 1e6)

    def correct_action(self, obs, action_np, act_limit):
        """Apply gradient correction using nabla_a c_hat."""
        obs_t = torch.FloatTensor(obs)
        a_t = torch.FloatTensor(action_np / act_limit)  # normalize to [-1,1]

        lam = self.compute_lambda(obs_t, a_t)

        if self.cumulative_cost >= self.budget_steps:
            # Budget exhausted: maximum correction
            effective_lam = 100.0
        else:
            effective_lam = min(lam, 100.0)

        # Gradient descent on predicted cost
        a_corr = a_t.clone().detach()
        for _ in range(self.correction_steps):
            a_corr.requires_grad_(True)
            c_pred = self.cost_pred(obs_t.unsqueeze(0), a_corr.unsqueeze(0))
            grad = torch.autograd.grad(c_pred, a_corr)[0]
            a_corr = a_corr.detach() - effective_lam * self.correction_lr * grad
            a_corr = a_corr.clamp(-1.0, 1.0)

        return a_corr.detach().numpy() * act_limit


# =============================================================================
# DATA COLLECTION
# =============================================================================

def compute_soft_cost(obs, n_lidar=16):
    """Soft cost from hazard lidar: max reading in [0,1]."""
    hazard_lidar = obs[-n_lidar:]
    return float(np.clip(hazard_lidar, 0, 1).max())


def collect_data(env_name, checkpoint, n_episodes=100, horizon=1000,
                 save_path="data/safety_gym_transitions.npz"):
    """Roll out unconstrained policy, collect (s, a, c_soft, c_binary)."""
    import safety_gymnasium

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    env = safety_gymnasium.make(env_name)

    all_obs, all_act, all_cost_soft, all_cost_binary = [], [], [], []
    total_cost = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for t in range(horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = a.squeeze(0).numpy() * act_limit

            # Record current state and action
            all_obs.append(obs.copy())
            all_act.append(action / act_limit)  # normalized

            step_result = env.step(action)
            if len(step_result) == 6:
                next_obs, reward, cost, term, trunc, info = step_result
            else:
                next_obs, reward, term, trunc, info = step_result
                cost = info.get("cost", 0.0)

            # Soft cost from NEXT state's lidar (what we want to predict)
            all_cost_soft.append(compute_soft_cost(next_obs))
            all_cost_binary.append(float(cost > 0))
            total_cost += cost

            obs = next_obs
            if term or trunc:
                break

        if (ep + 1) % 20 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes, "
                  f"{len(all_obs)} transitions, "
                  f"avg cost/ep={total_cost/(ep+1):.1f}")

    env.close()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path,
             obs=np.array(all_obs, dtype=np.float32),
             act=np.array(all_act, dtype=np.float32),
             cost_soft=np.array(all_cost_soft, dtype=np.float32),
             cost_binary=np.array(all_cost_binary, dtype=np.float32))

    n = len(all_obs)
    pos_frac = np.mean(all_cost_binary)
    print(f"Saved {n} transitions to {save_path}")
    print(f"  Soft cost: mean={np.mean(all_cost_soft):.3f}, "
          f"max={np.max(all_cost_soft):.3f}")
    print(f"  Binary cost: {pos_frac*100:.1f}% positive")
    return save_path


# =============================================================================
# TRAINING
# =============================================================================

def train_cost_predictor(data_path, save_path="checkpoints/cost_predictor.pt",
                         epochs=200, batch_size=512, lr=3e-4, hidden=128):
    """Train cost predictor MLP on collected transitions."""
    data = np.load(data_path)
    obs = torch.FloatTensor(data["obs"])
    act = torch.FloatTensor(data["act"])
    cost_soft = torch.FloatTensor(data["cost_soft"]).unsqueeze(1)

    obs_dim = obs.shape[1]
    act_dim = act.shape[1]
    n = len(obs)

    # Train/val split
    perm = torch.randperm(n)
    n_val = max(n // 10, 1000)
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    model = CostPredictor(obs_dim, act_dim, hidden)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        model.train()
        shuffle = torch.randperm(len(train_idx))
        epoch_loss = 0.0
        n_batches = 0

        for i in range(0, len(train_idx), batch_size):
            idx = train_idx[shuffle[i:i+batch_size]]
            pred = model(obs[idx], act[idx])
            loss = F.mse_loss(pred, cost_soft[idx])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(obs[val_idx], act[val_idx])
            val_loss = F.mse_loss(val_pred, cost_soft[val_idx]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}/{epochs}: "
                  f"train={epoch_loss/n_batches:.4f}, val={val_loss:.4f}")

    model.load_state_dict(best_state)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "model": model.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim,
        "hidden": hidden, "val_loss": best_val_loss,
    }, save_path)
    print(f"Saved cost predictor to {save_path} (val_loss={best_val_loss:.4f})")

    # Quick accuracy check
    model.eval()
    with torch.no_grad():
        all_pred = model(obs, act).squeeze()
        binary_pred = (all_pred > 0.3).float()
        binary_true = torch.FloatTensor(data["cost_binary"])
        acc = (binary_pred == binary_true).float().mean()
        print(f"  Binary classification accuracy (threshold=0.3): {acc:.3f}")

    return save_path


# =============================================================================
# EVALUATION
# =============================================================================

def eval_with_predictor(env_name, actor_checkpoint, predictor_checkpoint,
                        budget_frac=0.10, risk_aversion=2.0,
                        n_episodes=20, horizon=1000,
                        correction_lr=0.1, correction_steps=3):
    """Evaluate with cost predictor + predictive urgency."""
    import safety_gymnasium

    ckpt = torch.load(actor_checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    pred_ckpt = torch.load(predictor_checkpoint, map_location="cpu",
                            weights_only=False)
    cost_pred = CostPredictor(pred_ckpt["obs_dim"], pred_ckpt["act_dim"],
                               pred_ckpt["hidden"])
    cost_pred.load_state_dict(pred_ckpt["model"])
    cost_pred.eval()

    env = safety_gymnasium.make(env_name)
    budget = int(horizon * budget_frac)

    rewards, costs, goal_reached = [], [], []

    for ep in range(n_episodes):
        surr = PredictiveBudgetSurrogate(
            cost_predictor=cost_pred, budget_steps=budget,
            horizon_steps=horizon, risk_aversion=risk_aversion,
            correction_lr=correction_lr, correction_steps=correction_steps,
        )
        surr.reset()

        obs, _ = env.reset()
        ep_rew, ep_cost = 0.0, 0.0
        reached_goal = False

        for t in range(horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            raw_action = a.squeeze(0).numpy() * act_limit

            action = surr.correct_action(obs, raw_action, act_limit)

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
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "budget": budget,
        "utilization": float(100 * np.mean(costs) / max(budget, 1)),
        "goal_rate": float(100 * np.mean(goal_reached)),
    }


def compare_all(env_name, actor_checkpoint, predictor_checkpoint,
                n_episodes=20, horizon=1000, output_json=None):
    """Full comparison: unconstrained vs naive vs cost-predictor methods."""
    import safety_gymnasium

    print(f"\n{'='*80}")
    print(f"  Safety Gym: Cost Predictor + Predictive Urgency")
    print(f"{'='*80}")

    # Unconstrained baseline
    res_uncon = eval_with_predictor(
        env_name, actor_checkpoint, predictor_checkpoint,
        budget_frac=1.0, risk_aversion=0.0, n_episodes=n_episodes,
        horizon=horizon, correction_steps=0)
    uncon_reward = res_uncon["reward"]
    uncon_cost = res_uncon["cost"]
    print(f"\n  Unconstrained: Reward={uncon_reward:.1f}±{res_uncon['reward_std']:.1f}, "
          f"Cost={uncon_cost:.0f}, GoalRate={res_uncon['goal_rate']:.0f}%")

    cost_budgets = [10, 25, 40, int(uncon_cost)]
    results = {"unconstrained": res_uncon}

    header = (f"  {'Method':<40s} {'Budget':>6s} {'Reward':>8s} "
              f"{'Cost':>6s} {'%Uncon':>7s} {'Used':>6s} {'Goal':>5s}")
    print(f"\n{header}")
    print(f"  {'-'*78}")

    for cost_budget in cost_budgets:
        budget_frac = cost_budget / horizon
        for ra in [0.0, 2.0, 5.0]:
            for corr_steps, label_prefix in [(0, "No correction"),
                                              (3, "Grad corr")]:
                res = eval_with_predictor(
                    env_name, actor_checkpoint, predictor_checkpoint,
                    budget_frac=budget_frac, risk_aversion=ra,
                    n_episodes=n_episodes, horizon=horizon,
                    correction_steps=corr_steps)

                pct = 100 * res["reward"] / uncon_reward if uncon_reward > 0 else 0
                used = 100 * res["cost"] / cost_budget if cost_budget > 0 else 0
                ok = "✓" if res["cost"] <= cost_budget * 1.1 else "✗"
                label = f"{label_prefix} η={ra}"
                print(f"  {label:<40s} {cost_budget:>4d} {res['reward']:>8.1f} "
                      f"{res['cost']:>6.0f} {pct:>6.1f}% {used:>5.0f}% "
                      f"{res['goal_rate']:>4.0f}% {ok}")

                key = f"B{cost_budget}_ra{ra}_corr{corr_steps}"
                results[key] = {**res, "pct_unconstrained": pct,
                                "budget_used_pct": used}

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {output_json}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Safety Gym cost predictor")
    parser.add_argument("--env", default="SafetyPointGoal1-v0")
    parser.add_argument("--checkpoint", default="checkpoints/sac_safety_point.pt")
    parser.add_argument("--collect", action="store_true",
                        help="Collect transition data from unconstrained policy")
    parser.add_argument("--train", action="store_true",
                        help="Train cost predictor on collected data")
    parser.add_argument("--compare", action="store_true",
                        help="Run full comparison")
    parser.add_argument("--n-collect-episodes", type=int, default=100)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-path", default="data/safety_gym_transitions.npz")
    parser.add_argument("--predictor-path",
                        default="checkpoints/cost_predictor.pt")
    parser.add_argument("--output-json", default=None)
    cli = parser.parse_args()

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    if cli.collect:
        print(f"Collecting {cli.n_collect_episodes} episodes of transitions...")
        collect_data(cli.env, cli.checkpoint, cli.n_collect_episodes,
                     cli.horizon, cli.data_path)

    if cli.train:
        print(f"\nTraining cost predictor...")
        train_cost_predictor(cli.data_path, cli.predictor_path)

    if cli.compare:
        print(f"\nRunning comparison...")
        compare_all(cli.env, cli.checkpoint, cli.predictor_path,
                    cli.n_eval_episodes, cli.horizon, cli.output_json)

    if not any([cli.collect, cli.train, cli.compare]):
        print("Specify --collect, --train, --compare, or all three")


if __name__ == "__main__":
    main()
