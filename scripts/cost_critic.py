#!/usr/bin/env python3
"""
Cost critic Q_c for state-dependent budget constraints.

Trains Q_c(s,a) = E[Σ γ^k c_{t+k} | s,a] via offline Bellman updates on
rollouts from an unconstrained policy. At deployment, ∇_a Q_c provides a
long-horizon cost gradient that accounts for dynamics and propagation delays.

Works for both Safety Gym (hazard proximity) and wind farm (fatigue loading).

Usage:
    # Safety Gym: collect + train + evaluate
    python scripts/cost_critic.py \
        --domain safety_gym \
        --checkpoint checkpoints/sac_safety_point.pt \
        --collect --train --compare

    # Wind farm: collect + train + evaluate
    python scripts/cost_critic.py \
        --domain windfarm \
        --checkpoint runs/ebt_sac_windfarm/checkpoints/step_100000.pt \
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
# COST CRITIC
# =============================================================================

class CostCritic(nn.Module):
    """Q_c(s,a) = E[Σ γ^k c_{t+k} | s_t=s, a_t=a]"""

    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)

    def predict(self, obs, action):
        """Conservative estimate: max of two critics."""
        q1, q2 = self.forward(obs, action)
        return torch.max(q1, q2)


class CostReplayBuffer:
    """Stores (s, a, c, s', done) transitions."""

    def __init__(self, obs_dim, act_dim, capacity=500000):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.cost = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.cap = 0, 0, capacity

    def add(self, obs, act, cost, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.cost[self.ptr] = cost
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idx]),
            torch.FloatTensor(self.act[idx]),
            torch.FloatTensor(self.cost[idx]).unsqueeze(1),
            torch.FloatTensor(self.next_obs[idx]),
            torch.FloatTensor(self.done[idx]).unsqueeze(1),
        )

    def save(self, path):
        np.savez(path, obs=self.obs[:self.size], act=self.act[:self.size],
                 cost=self.cost[:self.size], next_obs=self.next_obs[:self.size],
                 done=self.done[:self.size])

    def load(self, path):
        data = np.load(path)
        n = len(data["obs"])
        self.obs[:n] = data["obs"]
        self.act[:n] = data["act"]
        self.cost[:n] = data["cost"]
        self.next_obs[:n] = data["next_obs"]
        self.done[:n] = data["done"]
        self.size = n
        self.ptr = n % self.cap


# =============================================================================
# SAFETY GYM DOMAIN
# =============================================================================

class SafetyGymActor(nn.Module):
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
        return action, mu


def collect_safety_gym(env_name, checkpoint, n_episodes, horizon, buf):
    """Collect (s, a, c, s', done) from unconstrained Safety Gym policy."""
    import safety_gymnasium

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = SafetyGymActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    env = safety_gymnasium.make(env_name)
    total_cost = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        for t in range(horizon):
            with torch.no_grad():
                a, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            action_norm = a.squeeze(0).numpy()
            action = action_norm * act_limit

            step_result = env.step(action)
            if len(step_result) == 6:
                next_obs, reward, cost, term, trunc, info = step_result
            else:
                next_obs, reward, term, trunc, info = step_result
                cost = info.get("cost", 0.0)

            buf.add(obs, action_norm, float(cost > 0), next_obs, float(term))
            total_cost += cost
            obs = next_obs
            if term or trunc:
                break

        if (ep + 1) % 20 == 0:
            print(f"  Collected {ep+1}/{n_episodes} episodes, "
                  f"{buf.size} transitions, cost/ep={total_cost/(ep+1):.1f}")

    env.close()
    return ckpt


def eval_safety_gym_with_qc(env_name, checkpoint, qc_path,
                             budget_frac, risk_aversion, n_episodes, horizon,
                             correction_lr=0.05, correction_steps=5):
    """Evaluate Safety Gym with Q_c-based gradient correction."""
    import safety_gymnasium

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = SafetyGymActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    qc_ckpt = torch.load(qc_path, map_location="cpu", weights_only=False)
    qc = CostCritic(qc_ckpt["obs_dim"], qc_ckpt["act_dim"], qc_ckpt["hidden"])
    qc.load_state_dict(qc_ckpt["model"])
    qc.eval()

    env = safety_gymnasium.make(env_name)
    budget = int(horizon * budget_frac)

    rewards, costs, goals = [], [], []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        ep_rew, cum_cost = 0.0, 0
        reached_goal = False

        for t in range(horizon):
            # Compute urgency-based lambda
            eps = 1e-6
            budget_remaining = max(budget - cum_cost, 0)
            time_remaining = max(horizon - t, 1)
            bf = budget_remaining / max(budget, 1)
            tf = time_remaining / max(horizon, 1)
            u = bf / max(tf, eps)
            lam = min(np.exp(risk_aversion * (1.0 / max(u, eps) - 1.0)), 1e4)

            # Budget exhausted: maximum correction
            if cum_cost >= budget:
                lam = 100.0

            with torch.no_grad():
                a, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            a_corr = a.squeeze(0).clone().detach()

            # Gradient correction via ∇_a Q_c
            obs_t = torch.FloatTensor(obs)
            for _ in range(correction_steps):
                a_corr.requires_grad_(True)
                qc_val = qc.predict(obs_t.unsqueeze(0), a_corr.unsqueeze(0))
                grad = torch.autograd.grad(qc_val, a_corr)[0]
                a_corr = a_corr.detach() - lam * correction_lr * grad
                a_corr = a_corr.clamp(-1.0, 1.0)

            action = a_corr.detach().numpy() * act_limit

            step_result = env.step(action)
            if len(step_result) == 6:
                next_obs, reward, cost, term, trunc, info = step_result
            else:
                next_obs, reward, term, trunc, info = step_result
                cost = info.get("cost", 0.0)

            if cost > 0:
                cum_cost += 1
            ep_rew += reward
            obs = next_obs

            if info.get("goal_met", False):
                reached_goal = True
            if term or trunc:
                break

        rewards.append(ep_rew)
        costs.append(cum_cost)
        goals.append(reached_goal)

    env.close()
    return {
        "reward": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cost": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "budget": budget,
        "utilization": float(100 * np.mean(costs) / max(budget, 1)),
        "goal_rate": float(100 * np.mean(goals)),
    }


# =============================================================================
# TRAINING Q_c
# =============================================================================

def train_cost_critic(buf, actor_ckpt, save_path,
                      gamma_c=0.99, epochs=200, batch_size=256,
                      lr=3e-4, hidden=256, tau=0.005):
    """Train Q_c via offline Bellman updates."""
    obs_dim = buf.obs.shape[1]
    act_dim = buf.act.shape[1]

    # Load actor for computing a' ~ π(s')
    actor = SafetyGymActor(actor_ckpt["obs_dim"], actor_ckpt["act_dim"])
    actor.load_state_dict(actor_ckpt["actor"])
    actor.eval()

    qc = CostCritic(obs_dim, act_dim, hidden)
    qc_target = CostCritic(obs_dim, act_dim, hidden)
    qc_target.load_state_dict(qc.state_dict())

    optimizer = torch.optim.Adam(qc.parameters(), lr=lr)

    steps_per_epoch = max(buf.size // batch_size, 100)
    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            o, a, c, no, d = buf.sample(batch_size)

            with torch.no_grad():
                na, _ = actor.sample(no)
                tq1, tq2 = qc_target(no, na)
                target_q = torch.min(tq1, tq2)
                target = c + gamma_c * (1 - d) * target_q

            q1, q2 = qc(o, a)
            loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            for p, pt in zip(qc.parameters(), qc_target.parameters()):
                pt.data.lerp_(p.data, tau)

        avg_loss = epoch_loss / steps_per_epoch
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in qc.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            # Check Q_c predictions
            with torch.no_grad():
                o, a, c, _, _ = buf.sample(1000)
                q_pred = qc.predict(o, a).squeeze()
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                      f"Q_c mean={q_pred.mean():.2f}, "
                      f"Q_c max={q_pred.max():.2f}, "
                      f"cost mean={c.mean():.3f}")

    qc.load_state_dict(best_state)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "model": qc.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim,
        "hidden": hidden, "gamma_c": gamma_c, "best_loss": best_loss,
    }, save_path)
    print(f"Saved Q_c to {save_path} (loss={best_loss:.4f})")


# =============================================================================
# COMPARISON
# =============================================================================

def compare_safety_gym(env_name, checkpoint, qc_path,
                       n_episodes=20, horizon=1000, output_json=None):
    """Compare unconstrained vs Q_c-corrected at multiple budgets."""
    print(f"\n{'='*80}")
    print(f"  Cost Critic Q_c on {env_name}")
    print(f"{'='*80}")

    # Unconstrained
    res_uncon = eval_safety_gym_with_qc(
        env_name, checkpoint, qc_path,
        budget_frac=1.0, risk_aversion=0.0, n_episodes=n_episodes,
        horizon=horizon, correction_steps=0)
    uncon_r = res_uncon["reward"]
    uncon_c = res_uncon["cost"]
    print(f"\n  Unconstrained: Reward={uncon_r:.1f}±{res_uncon['reward_std']:.1f}, "
          f"Cost={uncon_c:.0f}, Goal={res_uncon['goal_rate']:.0f}%")

    results = {"unconstrained": res_uncon}

    cost_budgets = [10, 25, 40, int(uncon_c)]
    header = (f"  {'Method':<35s} {'Budget':>6s} {'Reward':>8s} "
              f"{'Cost':>6s} {'%Uncon':>7s} {'Used':>6s} {'Goal':>5s}")
    print(f"\n{header}")
    print(f"  {'-'*75}")

    for cost_budget in cost_budgets:
        budget_frac = cost_budget / horizon
        for ra in [0.0, 2.0, 5.0]:
            for steps, lr, label in [(0, 0.0, "No correction"),
                                      (5, 0.05, "Q_c correction")]:
                res = eval_safety_gym_with_qc(
                    env_name, checkpoint, qc_path,
                    budget_frac=budget_frac, risk_aversion=ra,
                    n_episodes=n_episodes, horizon=horizon,
                    correction_steps=steps, correction_lr=lr)

                pct = 100 * res["reward"] / uncon_r if uncon_r > 0 else 0
                used = 100 * res["cost"] / cost_budget if cost_budget > 0 else 0
                ok = "✓" if res["cost"] <= cost_budget * 1.1 else "✗"
                print(f"  {label} η={ra:<5s}  {cost_budget:>4d} "
                      f"{res['reward']:>8.1f} {res['cost']:>6.0f} "
                      f"{pct:>6.1f}% {used:>5.0f}% {res['goal_rate']:>4.0f}% {ok}")

                key = f"B{cost_budget}_ra{ra}_steps{steps}"
                results[key] = {**res, "pct_unconstrained": pct}

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {output_json}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Cost critic Q_c")
    parser.add_argument("--domain", choices=["safety_gym", "windfarm"],
                        default="safety_gym")
    parser.add_argument("--env", default=None)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--n-collect-episodes", type=int, default=200)
    parser.add_argument("--n-eval-episodes", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--gamma-c", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-path", default="data/cost_critic_data.npz")
    parser.add_argument("--qc-path", default="checkpoints/cost_critic.pt")
    parser.add_argument("--output-json", default=None)
    cli = parser.parse_args()

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    if cli.env is None:
        cli.env = ("SafetyPointGoal1-v0" if cli.domain == "safety_gym"
                    else "3turb")

    if cli.domain == "safety_gym":
        obs_dim, act_dim = 60, 2

        if cli.collect:
            print(f"Collecting {cli.n_collect_episodes} episodes...")
            buf = CostReplayBuffer(obs_dim, act_dim)
            actor_ckpt = collect_safety_gym(
                cli.env, cli.checkpoint, cli.n_collect_episodes,
                cli.horizon, buf)
            buf.save(cli.data_path)
            print(f"Saved {buf.size} transitions to {cli.data_path}")

        if cli.train:
            print(f"\nTraining cost critic Q_c (γ={cli.gamma_c})...")
            buf = CostReplayBuffer(obs_dim, act_dim)
            buf.load(cli.data_path)
            print(f"Loaded {buf.size} transitions")
            actor_ckpt = torch.load(cli.checkpoint, map_location="cpu",
                                     weights_only=False)
            train_cost_critic(buf, actor_ckpt, cli.qc_path,
                              gamma_c=cli.gamma_c, epochs=cli.epochs)

        if cli.compare:
            print(f"\nComparing methods...")
            compare_safety_gym(cli.env, cli.checkpoint, cli.qc_path,
                               cli.n_eval_episodes, cli.horizon,
                               cli.output_json)

    elif cli.domain == "windfarm":
        print("Wind farm cost critic: use windfarm_cost_critic.py")

    if not any([cli.collect, cli.train, cli.compare]):
        print("Specify --collect, --train, --compare, or all three")


if __name__ == "__main__":
    main()
