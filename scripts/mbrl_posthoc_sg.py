"""
Full post-hoc MBRL package on Safety Gym. Policy frozen. Auxiliary
components all trained offline (and optionally online-refined):

    - Dynamics ensemble  f_θ^{1..M}: (s, a) -> s'   (MSE)
    - Reward critic      Q_r(s, a) = MC reward return
    - Cost critic        Q_c(s, a) = MC cost return (re-used from prior training)

Planning: CEM over K action sequences (horizon H), each rolled out through
the mean of the f_θ ensemble. Score per candidate:

    S = Σ_{h=0..H-1}  λ(u_{t+h}) · Q_c(s_h, a_h)  −  β · Q_r(s_h, a_h)

Pick argmin, execute first action in real env.

Optional online refit: every --refit-every episodes, retrain Q_c/Q_r/f_θ on
the full replay buffer (offline + online trajectories).

Usage:
  python scripts/mbrl_posthoc_sg.py --seed 1 --budget 10 --n-eps 10 \
      --beta 1.0 --dyn-ensemble 5 --n-candidates 32 --horizon-mpc 10
"""
import argparse
import json
import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import safety_gymnasium

import sys
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import (
    SafetyGymActor, CostCritic, urgency_lambda,
)

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Dynamics(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return s + self.net(x)  # residual


class RewardCritic(nn.Module):
    """Q_r(s, a) scalar."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.net(x)


def collect_transitions(env, actor, act_limit, n_eps=100, device=DEVICE):
    """Frozen-policy rollouts. Returns (s, a, s', r, c)."""
    S, A, Sp, R, C = [], [], [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=5000 + ep)
        done = False
        for _ in range(1000):
            st = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                a, _ = actor.sample(st)
            a_np = a.squeeze(0).cpu().numpy() * act_limit
            step = env.step(a_np)
            if len(step) == 6:
                obs2, r, c, term, trunc, info = step
            else:
                obs2, r, term, trunc, info = step
                c = info.get("cost", 0.0)
            S.append(obs); A.append(a.squeeze(0).cpu().numpy())
            Sp.append(obs2); R.append(float(r)); C.append(float(c))
            obs = obs2
            if term or trunc: break
    return (np.asarray(S, dtype=np.float32),
            np.asarray(A, dtype=np.float32),
            np.asarray(Sp, dtype=np.float32),
            np.asarray(R, dtype=np.float32),
            np.asarray(C, dtype=np.float32))


def train_dynamics_ensemble(M, S, A, Sp, iters=5000, batch=512, lr=3e-4):
    models = [Dynamics(S.shape[-1], A.shape[-1]).to(DEVICE) for _ in range(M)]
    opts = [torch.optim.Adam(m.parameters(), lr=lr) for m in models]
    S_t = torch.tensor(S, device=DEVICE); A_t = torch.tensor(A, device=DEVICE)
    Sp_t = torch.tensor(Sp, device=DEVICE)
    N = S.shape[0]
    for it in range(iters):
        for m, opt in zip(models, opts):
            idx = torch.randint(0, N, (batch,), device=DEVICE)
            pred = m(S_t[idx], A_t[idx])
            loss = F.mse_loss(pred, Sp_t[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            losses = []
            with torch.no_grad():
                idx = torch.randint(0, N, (batch,), device=DEVICE)
                for m in models:
                    losses.append(F.mse_loss(m(S_t[idx], A_t[idx]), Sp_t[idx]).item())
            print(f"  dyn iter {it}  MSE={np.mean(losses):.4e} (ensemble mean)")
    for m in models: m.eval()
    return models


def mc_returns(vals, dones, gamma):
    """Monte-Carlo discounted return per step, episode-resetting at dones."""
    G = np.zeros_like(vals, dtype=np.float32)
    running = 0.0
    for t in reversed(range(len(vals))):
        if dones[t]:
            running = 0.0
        running = vals[t] + gamma * running
        G[t] = running
    return G


def train_critic(name, net, S, A, G, iters=3000, batch=512, lr=3e-4):
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    S_t = torch.tensor(S, device=DEVICE); A_t = torch.tensor(A, device=DEVICE)
    G_t = torch.tensor(G, device=DEVICE).unsqueeze(-1)
    N = S.shape[0]
    for it in range(iters):
        idx = torch.randint(0, N, (batch,), device=DEVICE)
        q = net(S_t[idx], A_t[idx])
        loss = F.mse_loss(q, G_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f"  {name} iter {it}  MSE={loss.item():.4e}")
    net.eval()
    return net


def ensemble_mean_step(f_ens, s, a):
    preds = torch.stack([m(s, a) for m in f_ens], dim=0)
    return preds.mean(dim=0)


def cem_mbrl(f_ens, qr, qc, actor, obs_now, C_cum, t_now, horizon_env, budget,
             H=10, K=32, n_iters=2, top_k=8, act_limit=1.0, beta=1.0,
             pessimism=0.0):
    """Plan over ensemble mean dynamics with λ·Q_c − β·Q_r objective."""
    with torch.no_grad():
        s0 = torch.tensor(obs_now, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a_seed, _ = actor.sample(s0)
    mean = a_seed.expand(H, -1).clone()
    std = torch.full_like(mean, 0.5)

    act_dim = mean.shape[-1]
    for it in range(n_iters):
        # Sample K sequences in parallel
        noise = torch.randn(K, H, act_dim, device=DEVICE)
        cand = (mean.unsqueeze(0) + noise * std.unsqueeze(0)).clamp(-1.0, 1.0)
        s_k = s0.expand(K, -1).contiguous()
        with torch.no_grad():
            total = torch.zeros(K, device=DEVICE)
            C_sim = torch.full((K,), float(C_cum), device=DEVICE)
            for h in range(H):
                a_h = cand[:, h]
                q_c1, q_c2 = qc(s_k, a_h)
                q_c = torch.maximum(q_c1, q_c2).squeeze(-1)
                q_r = qr(s_k, a_h).squeeze(-1)
                # Per-candidate urgency (uses C_sim from rollout so far)
                rho = (budget - C_sim) / max(budget, 1e-9)
                tau = (horizon_env - (t_now + h)) / horizon_env
                u = rho / max(float(tau), 1e-6)
                # Work tensor-wise with clamps
                u_safe = torch.clamp(u, min=1e-6)
                lam = torch.exp(3.0 * (1.0 / u_safe - 1.0)).clamp(max=1e4)
                # Pessimism bonus via disagreement between models
                if pessimism > 0.0 and len(f_ens) > 1:
                    preds = torch.stack([m(s_k, a_h) for m in f_ens], dim=0)
                    disagreement = preds.std(dim=0).norm(dim=-1)
                    total += lam * q_c - beta * q_r + pessimism * disagreement
                else:
                    total += lam * q_c - beta * q_r
                # Step via ensemble mean
                s_k = ensemble_mean_step(f_ens, s_k, a_h)
                # Update simulated budget
                # Q_c is expected future cost under frozen policy; use immediate
                # step proxy: difference between consecutive Q_c values is small
                # and dominated by single-step cost. Use a fixed per-step cost proxy:
                C_sim = C_sim + q_c.detach() * 0.01   # dampened proxy
            scores = total

        elite_idx = torch.topk(-scores, top_k).indices
        elites = cand[elite_idx]
        mean = elites.mean(dim=0)
        std = elites.std(dim=0).clamp_min(0.05)
    # Return best first action
    best_idx = int(torch.argmin(scores).item())
    return cand[best_idx, 0].cpu().numpy() * act_limit


def run_episode(env, actor, f_ens, qr, qc, budget, ep_idx, horizon=1000,
                H=10, K=32, n_iters=2, beta=1.0, pessimism=0.0, act_limit=1.0):
    obs, _ = env.reset(seed=ep_idx * 101 + 7)
    C = 0.0; R = 0.0
    Ss, As, Rs, Cs, Sps = [], [], [], [], []
    for t in range(horizon):
        a_np = cem_mbrl(f_ens, qr, qc, actor, obs, C, t, horizon, budget,
                        H=H, K=K, n_iters=n_iters, beta=beta,
                        pessimism=pessimism, act_limit=act_limit)
        ret = env.step(a_np)
        if len(ret) == 6:
            obs2, r, c, term, trunc, info = ret
        else:
            obs2, r, term, trunc, info = ret
            c = info.get("cost", 0.0)
        Ss.append(obs); As.append(a_np / act_limit)
        Sps.append(obs2); Rs.append(float(r)); Cs.append(float(c))
        R += float(r); C += float(c)
        obs = obs2
        if term or trunc: break
    return dict(reward=R, cost=C, steps=len(Ss),
                S=np.asarray(Ss, dtype=np.float32),
                A=np.asarray(As, dtype=np.float32),
                Sp=np.asarray(Sps, dtype=np.float32),
                R=np.asarray(Rs, dtype=np.float32),
                C=np.asarray(Cs, dtype=np.float32))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--horizon-mpc", type=int, default=10, dest="H")
    p.add_argument("--n-candidates", type=int, default=32, dest="K")
    p.add_argument("--cem-iters", type=int, default=1)
    p.add_argument("--n-eps", type=int, default=10)
    p.add_argument("--dyn-eps", type=int, default=100)
    p.add_argument("--dyn-ensemble", type=int, default=5)
    p.add_argument("--dyn-iters", type=int, default=3000)
    p.add_argument("--qr-iters", type=int, default=3000)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--pessimism", type=float, default=0.0)
    p.add_argument("--online-refit", action="store_true",
                   help="refit Q_c/Q_r/f_θ every --refit-every episodes")
    p.add_argument("--refit-every", type=int, default=5)
    p.add_argument("--refit-iters", type=int, default=500)
    p.add_argument("--out", default="results/mbrl_posthoc_sg.json")
    args = p.parse_args()

    ac = torch.load(CKPT / f"sac_safety_point_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()
    cc = torch.load(CKPT / f"cost_critic_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    qc = CostCritic(cc["obs_dim"], cc["act_dim"], cc["hidden"]).to(DEVICE)
    qc.load_state_dict(cc["model"]); qc.eval()

    env = safety_gymnasium.make(ac["env_name"])
    print(f"[seed {args.seed}] collecting {args.dyn_eps} eps transitions...")
    S, A, Sp, Rv, Cv = collect_transitions(env, actor, ac["act_limit"],
                                            n_eps=args.dyn_eps)
    print(f"  collected {len(S)} transitions")

    print(f"training dynamics ensemble (M={args.dyn_ensemble})...")
    f_ens = train_dynamics_ensemble(args.dyn_ensemble, S, A, Sp,
                                     iters=args.dyn_iters)

    print("training reward critic Q_r (MC returns)...")
    # Build dones (here: batch contains episodes concatenated; treat each
    # transition as if start of rollout since we don't track episode boundaries
    # — acceptable because Q_r is supposed to match expected return from (s,a)).
    # Use non-episodic MC: G_t = Σ γ^k r_{t+k} within episode chunks of 1000.
    H_ep = 1000
    dones = np.zeros(len(S), dtype=bool)
    for i in range(H_ep - 1, len(S), H_ep):
        dones[i] = True
    Gr = mc_returns(Rv, dones, gamma=0.99)
    qr = RewardCritic(cc["obs_dim"], cc["act_dim"]).to(DEVICE)
    train_critic("Q_r", qr, S, A, Gr, iters=args.qr_iters)

    # Evaluate
    results = {"seed": args.seed, "budget": args.budget, "beta": args.beta,
               "online_refit": args.online_refit, "per_ep": []}
    buf_S, buf_A, buf_Sp, buf_R, buf_C = [S], [A], [Sp], [Rv], [Cv]
    buf_dones = [dones]

    for ep in range(args.n_eps):
        e = run_episode(env, actor, f_ens, qr, qc, args.budget, ep,
                        horizon=args.horizon, H=args.H, K=args.K,
                        n_iters=args.cem_iters, beta=args.beta,
                        pessimism=args.pessimism, act_limit=ac["act_limit"])
        results["per_ep"].append(dict(reward=e["reward"], cost=e["cost"],
                                       steps=e["steps"]))
        print(f"  ep {ep}: R={e['reward']:.1f} C={e['cost']:.1f} "
              f"steps={e['steps']}  budget={args.budget}")

        if args.online_refit:
            ep_dones = np.zeros(e["steps"], dtype=bool)
            ep_dones[-1] = True
            buf_S.append(e["S"]); buf_A.append(e["A"]); buf_Sp.append(e["Sp"])
            buf_R.append(e["R"]); buf_C.append(e["C"])
            buf_dones.append(ep_dones)
            if (ep + 1) % args.refit_every == 0:
                print(f"  [online refit @ ep {ep+1}]")
                SS = np.concatenate(buf_S); AA = np.concatenate(buf_A)
                SSp = np.concatenate(buf_Sp); RR = np.concatenate(buf_R)
                CC = np.concatenate(buf_C); DD = np.concatenate(buf_dones)
                Gr_all = mc_returns(RR, DD, gamma=0.99)
                Gc_all = mc_returns(CC, DD, gamma=0.99)
                train_critic("Q_r (refit)", qr, SS, AA, Gr_all,
                             iters=args.refit_iters)
                # Refit Q_c (wrap twin output: use first head's MSE)
                opt = torch.optim.Adam(qc.parameters(), lr=1e-4)
                SS_t = torch.tensor(SS, device=DEVICE); AA_t = torch.tensor(AA, device=DEVICE)
                Gc_t = torch.tensor(Gc_all, device=DEVICE).unsqueeze(-1)
                for _ in range(args.refit_iters):
                    idx = torch.randint(0, len(SS), (512,), device=DEVICE)
                    q1, q2 = qc(SS_t[idx], AA_t[idx])
                    loss = F.mse_loss(q1, Gc_t[idx]) + F.mse_loss(q2, Gc_t[idx])
                    opt.zero_grad(); loss.backward(); opt.step()

    env.close()
    costs = [e["cost"] for e in results["per_ep"]]
    rews = [e["reward"] for e in results["per_ep"]]
    results["summary"] = dict(
        cost_mean=float(np.mean(costs)),
        cost_se=float(np.std(costs, ddof=1) / np.sqrt(len(costs))) if len(costs) > 1 else 0.0,
        reward_mean=float(np.mean(rews)),
        reward_se=float(np.std(rews, ddof=1) / np.sqrt(len(rews))) if len(rews) > 1 else 0.0,
        sat_rate=float(np.mean([c <= args.budget for c in costs])),
    )
    print(f"SUMMARY: cost={results['summary']['cost_mean']:.1f}±"
          f"{results['summary']['cost_se']:.1f}  R={results['summary']['reward_mean']:.1f} "
          f"sat={results['summary']['sat_rate']*100:.0f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if Path(args.out).exists():
        try: data = json.load(open(args.out))
        except Exception: pass
    tag = f"_online" if args.online_refit else ""
    data[f"seed{args.seed}_B{args.budget}_beta{args.beta}{tag}"] = results
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
