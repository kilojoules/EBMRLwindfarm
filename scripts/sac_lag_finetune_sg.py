"""Lagrangian finetune from pretrained unconstrained SAC actor.

Fair baseline: operator starts from the deployed unconstrained actor
(200k SAC steps) and finetunes under budget constraint via Lagrangian
dual ascent. Eval at multiple checkpoints — time-to-deployment curve.

Usage:
  python scripts/sac_lag_finetune_sg.py --pretrained-seed 1 --budget 25 \
      --eval-at 0 50000 200000 500000 --out results/saclag_finetune.json
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
from uncertainty_gated_qc import SafetyGymActor
from sac_lag_sg import QNet, ReplayBuffer, SACActor

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_pretrained_actor(seed, obs_dim, act_dim):
    """Load unconstrained SAC actor, copy weights into SACActor."""
    src = torch.load(CKPT / f"sac_safety_point_seed{seed}.pt",
                     map_location=DEVICE, weights_only=False)
    # Original SafetyGymActor has .net, .mu, .log_std; matches SACActor structure
    actor = SACActor(obs_dim, act_dim).to(DEVICE)
    # Remap keys
    sd = src["actor"]
    actor_sd = {}
    for k, v in sd.items():
        actor_sd[k] = v
    actor.load_state_dict(actor_sd, strict=False)
    print(f"  loaded pretrained actor from seed {seed}")
    return actor


def evaluate(actor, env_name, act_limit, n_eps=20, horizon=1000):
    env = safety_gymnasium.make(env_name)
    costs, rews = [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=2000 + ep)
        C, R = 0.0, 0.0
        for t in range(horizon):
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
    return {
        "cost_mean": float(np.mean(costs)),
        "cost_se": float(np.std(costs, ddof=1)/np.sqrt(len(costs))),
        "reward_mean": float(np.mean(rews)),
        "reward_se": float(np.std(rews, ddof=1)/np.sqrt(len(rews))),
        "sat_rate": float(np.mean([c <= 1e9 for c in costs])),  # filled below
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrained-seed", type=int, default=1)
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=1e-4)   # lower for finetune
    p.add_argument("--mu-lr", type=float, default=0.005)
    p.add_argument("--mu-init", type=float, default=0.1)
    p.add_argument("--mu-warmup", type=int, default=0,
                   help="hold μ at 0 for first K steps (warm start reward learning)")
    p.add_argument("--alpha-ent", type=float, default=0.2)
    p.add_argument("--reward-scale", type=float, default=1.0,
                   help="scale reward to roughly equal cost magnitudes for stable Lagrangian")
    p.add_argument("--eval-at", type=int, nargs="+", default=[0, 50000, 200000, 500000])
    p.add_argument("--out", default="results/saclag_finetune.json")
    args = p.parse_args()

    seed = args.pretrained_seed
    np.random.seed(seed); torch.manual_seed(seed)
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    obs, _ = env.reset(seed=seed)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = load_pretrained_actor(seed, obs_dim, act_dim)
    qr = QNet(obs_dim, act_dim).to(DEVICE); qr_t = QNet(obs_dim, act_dim).to(DEVICE)
    qc = QNet(obs_dim, act_dim).to(DEVICE); qc_t = QNet(obs_dim, act_dim).to(DEVICE)
    qr_t.load_state_dict(qr.state_dict()); qc_t.load_state_dict(qc.state_dict())

    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_qr = torch.optim.Adam(qr.parameters(), lr=args.lr)
    opt_qc = torch.optim.Adam(qc.parameters(), lr=args.lr)
    log_mu = torch.tensor(np.log(max(args.mu_init, 1e-6)),
                          device=DEVICE, requires_grad=True, dtype=torch.float32)
    opt_mu = torch.optim.Adam([log_mu], lr=args.mu_lr)

    rb = ReplayBuffer(max(max(args.eval_at), 100_000), obs_dim, act_dim)
    results = {"budget": args.budget, "seed": seed, "eval_at": args.eval_at, "evals": []}

    # Eval at step 0 (pure pretrained actor, no finetune)
    ev = evaluate(actor, "SafetyPointGoal1-v0", act_limit, n_eps=20, horizon=args.horizon)
    ev["sat_rate"] = float(ev["cost_mean"] <= args.budget)  # quick proxy
    ev["steps"] = 0
    results["evals"].append(ev)
    print(f"step=0   R={ev['reward_mean']:.2f} C={ev['cost_mean']:.1f}")

    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
    max_steps = max(args.eval_at)
    eval_targets = set(args.eval_at) - {0}
    warmup = 2000

    for step in range(1, max_steps + 1):
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
        rb.add(obs, a.squeeze(0).cpu().numpy(), float(r) * args.reward_scale, float(c), obs2, done)
        obs = obs2
        ep_ret += r; ep_cost += c; ep_len += 1
        if term or trunc:
            obs, _ = env.reset()
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0

        if rb.n >= warmup:
            s_b, a_b, r_b, c_b, s2_b, d_b = rb.sample(args.batch)
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

            a_new, logp_new = actor.sample(s_b)
            q1n, q2n = qr(s_b, a_new)
            qr_min = torch.minimum(q1n, q2n).squeeze(-1)
            q1cn, q2cn = qc(s_b, a_new)
            qc_max = torch.maximum(q1cn, q2cn).squeeze(-1)
            # μ warmup: hold at 0 (pure reward learning) for first K steps
            if step < args.mu_warmup:
                mu = torch.zeros((), device=DEVICE)
            else:
                mu = torch.exp(log_mu).detach()
            actor_loss = -(qr_min - args.alpha_ent * logp_new.squeeze(-1) - mu * qc_max).mean()
            opt_a.zero_grad(); actor_loss.backward(); opt_a.step()

            with torch.no_grad():
                q1cn2, q2cn2 = qc(s_b, a_new.detach())
                qc_eval = torch.maximum(q1cn2, q2cn2).mean().item()
            if step >= args.mu_warmup:
                mu_loss = -log_mu * (qc_eval - args.budget)
                opt_mu.zero_grad(); mu_loss.backward(); opt_mu.step()
            with torch.no_grad():
                log_mu.clamp_(max=np.log(50.0))

            for p_t, p in zip(qr_t.parameters(), qr.parameters()):
                p_t.data.mul_(1 - args.tau).add_(p.data, alpha=args.tau)
            for p_t, p in zip(qc_t.parameters(), qc.parameters()):
                p_t.data.mul_(1 - args.tau).add_(p.data, alpha=args.tau)

        if step in eval_targets:
            ev = evaluate(actor, "SafetyPointGoal1-v0", act_limit, n_eps=20, horizon=args.horizon)
            ev["sat_rate"] = float(ev["cost_mean"] <= args.budget)
            ev["steps"] = step
            ev["mu"] = float(torch.exp(log_mu).item())
            results["evals"].append(ev)
            print(f"step={step}  R={ev['reward_mean']:.2f} C={ev['cost_mean']:.1f} μ={ev['mu']:.3f}")
            obs, _ = env.reset()

    env.close()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if Path(args.out).exists():
        try: data = json.load(open(args.out))
        except Exception: pass
    key = f"B{args.budget}_seed{seed}"
    data[key] = results
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
