"""SAUTE-SAC baseline on Safety Gymnasium.

State augmentation with remaining-budget fraction ρ_t:
   s_aug = [s ; ρ_t]   where ρ_t = (d − C_t) / d
SAC trains on safety-MDP where reward is masked when ρ_t < 0:
   r_safe(s, a) = r(s, a) · 𝟙[ρ_t ≥ 0] + r_unsafe · 𝟙[ρ_t < 0]
Following Sootla et al. 2022, r_unsafe = -1/(1-γ) (terminal penalty proxy).

Usage:
  python scripts/saute_sac_sg.py --budget 25 --total-steps 500000 --seed 1 \
      --out runs/saute_sg_B25
"""
import argparse
import json
from pathlib import Path
from collections import deque
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import safety_gymnasium

import sys
sys.path.insert(0, str(Path(__file__).parent))
from sac_lag_sg import QNet, SACActor, ReplayBuffer

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--total-steps", type=int, default=500000)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--tau", type=float, default=0.005)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--alpha-ent", type=float, default=0.2)
    p.add_argument("--unsafe-reward", type=float, default=None,
                   help="reward when ρ < 0 (default: -1/(1-γ))")
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    np.random.seed(args.seed); torch.manual_seed(args.seed)
    env = safety_gymnasium.make("SafetyPointGoal1-v0")
    obs_raw, _ = env.reset(seed=args.seed)
    obs_dim_raw = env.observation_space.shape[0]
    obs_dim = obs_dim_raw + 1   # augmented with ρ
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])
    if args.unsafe_reward is None:
        args.unsafe_reward = -1.0 / (1.0 - args.gamma)

    actor = SACActor(obs_dim, act_dim).to(DEVICE)
    qr = QNet(obs_dim, act_dim).to(DEVICE); qr_t = QNet(obs_dim, act_dim).to(DEVICE)
    qr_t.load_state_dict(qr.state_dict())
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr)
    opt_qr = torch.optim.Adam(qr.parameters(), lr=args.lr)

    rb = ReplayBuffer(args.total_steps, obs_dim, act_dim)

    C = 0.0
    rho_t = 1.0   # start with full budget
    obs = np.concatenate([obs_raw, [rho_t]]).astype(np.float32)
    ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
    ep_returns = deque(maxlen=20); ep_costs = deque(maxlen=20)

    for step in range(1, args.total_steps + 1):
        s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        with torch.no_grad():
            a, _ = actor.sample(s)
        a_exec = a.squeeze(0).cpu().numpy() * act_limit
        ret = env.step(a_exec)
        if len(ret) == 6:
            obs2_raw, r, c, term, trunc, info = ret
        else:
            obs2_raw, r, term, trunc, info = ret
            c = info.get("cost", 0.0)
        # SAUTE: augment state with ρ_{t+1}, mask reward if budget violated
        C_new = C + float(c)
        rho_new = (args.budget - C_new) / max(args.budget, 1e-9)
        r_safe = float(r) if rho_new >= 0 else args.unsafe_reward
        obs2 = np.concatenate([obs2_raw, [rho_new]]).astype(np.float32)
        done = float(term or trunc)
        rb.add(obs, a.squeeze(0).cpu().numpy(), r_safe, float(c), obs2, done)

        ep_ret += r; ep_cost += c; ep_len += 1
        if term or trunc:
            ep_returns.append(ep_ret); ep_costs.append(ep_cost)
            obs_raw, _ = env.reset()
            C = 0.0; rho_t = 1.0
            obs = np.concatenate([obs_raw, [rho_t]]).astype(np.float32)
            ep_ret, ep_cost, ep_len = 0.0, 0.0, 0
        else:
            C = C_new
            rho_t = rho_new
            obs = obs2

        if rb.n >= args.warmup:
            s_b, a_b, r_b, _, s2_b, d_b = rb.sample(args.batch)
            with torch.no_grad():
                a2, logp2 = actor.sample(s2_b)
                qr1, qr2 = qr_t(s2_b, a2)
                target_qr = r_b.unsqueeze(-1) + args.gamma * (1 - d_b.unsqueeze(-1)) * (
                    torch.minimum(qr1, qr2) - args.alpha_ent * logp2)
            q1, q2 = qr(s_b, a_b)
            loss_qr = F.mse_loss(q1, target_qr) + F.mse_loss(q2, target_qr)
            opt_qr.zero_grad(); loss_qr.backward(); opt_qr.step()

            a_new, logp_new = actor.sample(s_b)
            q1n, q2n = qr(s_b, a_new)
            qr_min = torch.minimum(q1n, q2n).squeeze(-1)
            actor_loss = -(qr_min - args.alpha_ent * logp_new.squeeze(-1)).mean()
            opt_a.zero_grad(); actor_loss.backward(); opt_a.step()

            for p_t, p in zip(qr_t.parameters(), qr.parameters()):
                p_t.data.mul_(1 - args.tau).add_(p.data, alpha=args.tau)

        if step % 5000 == 0 and len(ep_returns) > 0:
            print(f"step={step:>6}  R̄={np.mean(ep_returns):6.2f}  C̄={np.mean(ep_costs):6.1f}  "
                  f"ρ̄={rho_t:.2f}", flush=True)

    # Evaluation: 20 eps with augmented state
    print("\nEval 20 episodes...")
    costs, rews = [], []
    for ep in range(20):
        obs_raw, _ = env.reset(seed=2000 + ep)
        C = 0.0; rho_t = 1.0
        obs = np.concatenate([obs_raw, [rho_t]]).astype(np.float32)
        R, Cep = 0.0, 0.0
        for t in range(args.horizon):
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a, _ = actor.sample(s)
            a_exec = a.squeeze(0).cpu().numpy() * act_limit
            ret = env.step(a_exec)
            if len(ret) == 6:
                obs2_raw, r, c, term, trunc, info = ret
            else:
                obs2_raw, r, term, trunc, info = ret
                c = info.get("cost", 0.0)
            R += r; Cep += c; C += float(c)
            rho_t = (args.budget - C) / max(args.budget, 1e-9)
            obs = np.concatenate([obs2_raw, [rho_t]]).astype(np.float32)
            if term or trunc: break
        costs.append(Cep); rews.append(R)
    env.close()

    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)
    torch.save({"actor": actor.state_dict(), "obs_dim": obs_dim,
                "act_dim": act_dim, "act_limit": act_limit,
                "budget": args.budget}, out_path / "actor.pt")
    eval_data = {
        "budget": args.budget, "seed": args.seed,
        "total_steps": args.total_steps,
        "cost_mean": float(np.mean(costs)),
        "cost_se": float(np.std(costs, ddof=1)/np.sqrt(len(costs))),
        "reward_mean": float(np.mean(rews)),
        "reward_se": float(np.std(rews, ddof=1)/np.sqrt(len(rews))),
        "sat_rate": float(np.mean([c <= args.budget for c in costs])),
        "per_ep_cost": costs, "per_ep_reward": rews,
    }
    with open(out_path / "actor_eval.json", "w") as f:
        json.dump(eval_data, f, indent=2)
    print(f"SAUTE B={args.budget}: cost={np.mean(costs):.1f}±{np.std(costs)/np.sqrt(20):.1f} "
          f"R={np.mean(rews):.1f} sat={np.mean([c<=args.budget for c in costs])*100:.0f}%")


if __name__ == "__main__":
    main()
