"""
SAC-Lagrangian baseline on Safety Gymnasium (CMDP retraining).

Trains a constrained policy from scratch for a fixed cost budget d:
    maximize E[Σ r] s.t. E[Σ c] ≤ d
via Lagrangian relaxation:
    L(π, μ) = E[Σ r] − μ·(E[Σ c] − d)
with dual ascent on μ.

Use as baseline to compare against post-hoc blend (zero retraining).

Usage:
  python scripts/sac_lag_sg.py --budget 10 --total-steps 100000 --seed 1 \
      --out runs/sac_lag_sg_B10_s1
"""
import argparse
import json
import os
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import safety_gymnasium

import sys
sys.path.insert(0, str(Path(__file__).parent))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SACActor(nn.Module):
    """Gaussian-squashed policy with proper log-prob for SAC."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, s):
        h = self.net(s)
        return self.mu(h), self.log_std(h).clamp(-20, 2)

    def sample(self, s):
        mu, log_std = self(s)
        dist = torch.distributions.Normal(mu, log_std.exp())
        u = dist.rsample()
        a = torch.tanh(u)
        logp = dist.log_prob(u) - torch.log(1 - a.pow(2) + 1e-6)
        logp = logp.sum(dim=-1, keepdim=True)
        return a, logp


class QNet(nn.Module):
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

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


class ReplayBuffer:
    def __init__(self, size, obs_dim, act_dim):
        self.size = size
        self.obs = np.zeros((size, obs_dim), dtype=np.float32)
        self.act = np.zeros((size, act_dim), dtype=np.float32)
        self.rew = np.zeros(size, dtype=np.float32)
        self.cost = np.zeros(size, dtype=np.float32)
        self.obs2 = np.zeros((size, obs_dim), dtype=np.float32)
        self.done = np.zeros(size, dtype=np.float32)
        self.ptr, self.n = 0, 0

    def add(self, o, a, r, c, o2, d):
        self.obs[self.ptr] = o; self.act[self.ptr] = a
        self.rew[self.ptr] = r; self.cost[self.ptr] = c
        self.obs2[self.ptr] = o2; self.done[self.ptr] = d
        self.ptr = (self.ptr + 1) % self.size
        self.n = min(self.n + 1, self.size)

    def sample(self, batch):
        idx = np.random.randint(0, self.n, batch)
        return (torch.tensor(self.obs[idx], device=DEVICE),
                torch.tensor(self.act[idx], device=DEVICE),
                torch.tensor(self.rew[idx], device=DEVICE),
                torch.tensor(self.cost[idx], device=DEVICE),
                torch.tensor(self.obs2[idx], device=DEVICE),
                torch.tensor(self.done[idx], device=DEVICE))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=100000)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--mu-lr", type=float, default=0.01)
    p.add_argument("--mu-init", type=float, default=0.1)
    p.add_argument("--alpha-ent", type=float, default=0.2)
    p.add_argument("--warmup", type=int, default=5000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    obs, _ = env.reset(seed=args.seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = SACActor(obs_dim, act_dim).to(DEVICE)
    qr = QNet(obs_dim, act_dim).to(DEVICE); qr_t = QNet(obs_dim, act_dim).to(DEVICE)
    qc = QNet(obs_dim, act_dim).to(DEVICE); qc_t = QNet(obs_dim, act_dim).to(DEVICE)
    qr_t.load_state_dict(qr.state_dict()); qc_t.load_state_dict(qc.state_dict())

    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_qr = torch.optim.Adam(qr.parameters(), lr=args.lr)
    opt_qc = torch.optim.Adam(qc.parameters(), lr=args.lr)

    log_mu = torch.tensor(np.log(max(args.mu_init, 1e-6)),
                          device=DEVICE, requires_grad=True, dtype=torch.float32)
    opt_mu = torch.optim.Adam([log_mu], lr=args.mu_lr)

    # Budget → per-step target cost rate (for dual constraint)
    target_cost_per_step = args.budget / args.horizon

    rb = ReplayBuffer(max(args.total_steps, 100_000), obs_dim, act_dim)

    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
    ep_returns = deque(maxlen=20)
    ep_costs = deque(maxlen=20)

    for step in range(1, args.total_steps + 1):
        s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            a, _ = actor.sample(s)
        a_exec = a.squeeze(0).cpu().numpy() * act_limit
        ret = env.step(a_exec)
        if len(ret) == 6:
            obs2, r, c, term, trunc, info = ret
        else:
            obs2, r, term, trunc, info = ret
            c = info.get("cost", 0.0)
        done = float(term or trunc)
        rb.add(obs, a.squeeze(0).cpu().numpy(), float(r), float(c), obs2, done)
        obs = obs2
        ep_ret += r; ep_cost += c; ep_len += 1
        if term or trunc:
            ep_returns.append(ep_ret); ep_costs.append(ep_cost)
            obs, _ = env.reset()
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

        if rb.n >= args.warmup and step % 1 == 0:
            s_b, a_b, r_b, c_b, s2_b, d_b = rb.sample(args.batch)
            # Critic update
            with torch.no_grad():
                a2, logp2 = actor.sample(s2_b)
                qr1, qr2 = qr_t(s2_b, a2)
                qc1, qc2 = qc_t(s2_b, a2)
                target_qr = r_b.unsqueeze(-1) + args.gamma * (1 - d_b.unsqueeze(-1)) * (torch.minimum(qr1, qr2) - args.alpha_ent * logp2)
                target_qc = c_b.unsqueeze(-1) + args.gamma * (1 - d_b.unsqueeze(-1)) * torch.maximum(qc1, qc2)
            q1, q2 = qr(s_b, a_b)
            loss_qr = F.mse_loss(q1, target_qr) + F.mse_loss(q2, target_qr)
            opt_qr.zero_grad(); loss_qr.backward(); opt_qr.step()
            q1c, q2c = qc(s_b, a_b)
            loss_qc = F.mse_loss(q1c, target_qc) + F.mse_loss(q2c, target_qc)
            opt_qc.zero_grad(); loss_qc.backward(); opt_qc.step()

            # Actor update with Lagrangian penalty
            a_new, logp_new = actor.sample(s_b)
            q1n, q2n = qr(s_b, a_new)
            qr_min = torch.minimum(q1n, q2n).squeeze(-1)
            q1cn, q2cn = qc(s_b, a_new)
            qc_max = torch.maximum(q1cn, q2cn).squeeze(-1)
            mu = torch.exp(log_mu).detach()
            actor_loss = -(qr_min - args.alpha_ent * logp_new.squeeze(-1) - mu * qc_max).mean()
            opt_a.zero_grad(); actor_loss.backward(); opt_a.step()

            # Dual update: μ ← clip(μ + lr·(E[Q_c] − target), 0)
            with torch.no_grad():
                q1cn2, q2cn2 = qc(s_b, a_new.detach())
                qc_eval = torch.maximum(q1cn2, q2cn2).mean().item()
            mu_loss = -log_mu * (qc_eval - target_cost_per_step * args.horizon)
            opt_mu.zero_grad(); mu_loss.backward(); opt_mu.step()
            with torch.no_grad():
                log_mu.clamp_(max=np.log(100.0))

            # Soft target update
            for p_t, p in zip(qr_t.parameters(), qr.parameters()):
                p_t.data.mul_(1 - args.tau).add_(p.data, alpha=args.tau)
            for p_t, p in zip(qc_t.parameters(), qc.parameters()):
                p_t.data.mul_(1 - args.tau).add_(p.data, alpha=args.tau)

        if step % 5000 == 0 and len(ep_returns) > 0:
            mu_cur = float(torch.exp(log_mu).item())
            print(f"step={step:>6}  R̄={np.mean(ep_returns):6.1f}  C̄={np.mean(ep_costs):6.1f}  μ={mu_cur:.3f}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(),
                "obs_dim": obs_dim, "act_dim": act_dim,
                "act_limit": act_limit, "env_name": "SafetyPointGoal1-v0",
                "budget": args.budget,
                "mu_final": float(torch.exp(log_mu).item()),
                }, args.out)

    # Eval
    print("\nEval 20 episodes...")
    costs, rews = [], []
    for ep in range(20):
        obs, _ = env.reset(seed=2000 + ep)
        C, R = 0.0, 0.0
        for t in range(args.horizon):
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a, _ = actor.sample(s)
            a_exec = a.squeeze(0).cpu().numpy() * act_limit
            ret = env.step(a_exec)
            if len(ret) == 6:
                obs, r, c, term, trunc, info = ret
            else:
                obs, r, term, trunc, info = ret
                c = info.get("cost", 0.0)
            C += c; R += r
            if term or trunc: break
        costs.append(C); rews.append(R)
    env.close()

    out_json = str(args.out).replace(".pt", "_eval.json")
    with open(out_json, "w") as f:
        json.dump({
            "budget": args.budget, "seed": args.seed,
            "cost_mean": float(np.mean(costs)),
            "cost_se": float(np.std(costs, ddof=1) / np.sqrt(len(costs))),
            "reward_mean": float(np.mean(rews)),
            "reward_se": float(np.std(rews, ddof=1) / np.sqrt(len(rews))),
            "sat_rate": float(np.mean([c <= args.budget for c in costs])),
            "per_ep_cost": costs, "per_ep_reward": rews,
            "mu_final": float(torch.exp(log_mu).item()),
        }, f, indent=2)
    print(f"SAC-Lag B={args.budget}: cost={np.mean(costs):.1f}±{np.std(costs)/np.sqrt(20):.1f} "
          f"R={np.mean(rews):.1f} sat={np.mean([c<=args.budget for c in costs])*100:.0f}%")


if __name__ == "__main__":
    main()
