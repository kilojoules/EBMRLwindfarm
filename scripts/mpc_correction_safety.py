"""
MPC-based budget correction on Safety Gym. Works at any coupling strength κ
by rolling out a learned dynamics model and optimizing action sequences
under the λ(u)-weighted cost critic Q_c.

Pipeline:
  1. Collect logged (s, a, s', c) transitions from frozen-policy rollouts
     (re-use the same 200 eps used for Q_c training).
  2. Train f_θ: (s, a) -> s' (MLP, MSE).
  3. At inference, CEM over H-step action sequences; score via
     Σ_{k=0..H-1} λ(u_{t+k}) · Q_c(s_{t+k}, a_k) with s_{t+k+1} = f_θ(s_{t+k}, a_k).
  4. Execute first action, observe real (s', c), update C and u.

Optional: online Q_c refit at episode boundaries (--online-qc).

Usage:
    python scripts/mpc_correction_safety.py --budget 10 --seed 1 \
        --horizon 5 --n-candidates 64 --iters 2 --n-eps 20
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
    SafetyGymActor, CostCritic, urgency_lambda, LAM_CLAMP, LAM_HARD_GUARD,
)

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DynamicsModel(nn.Module):
    """f_θ: (s, a) -> Δs. Predicts state delta (residual)."""
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, obs_dim),
        )
        self.obs_dim = obs_dim

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return s + self.net(x)  # residual


def collect_transitions(env, actor, act_limit, n_eps=200, device=DEVICE):
    """Frozen-policy rollouts. Returns arrays (s, a, s', c)."""
    S, A, Sp, C = [], [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=4000 + ep)
        done = False
        t = 0
        while not done and t < 1000:
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
            S.append(obs); A.append(a.squeeze(0).cpu().numpy()); Sp.append(obs2); C.append(float(c))
            obs = obs2
            done = term or trunc
            t += 1
    return (np.asarray(S, dtype=np.float32),
            np.asarray(A, dtype=np.float32),
            np.asarray(Sp, dtype=np.float32),
            np.asarray(C, dtype=np.float32))


def train_dynamics(f, S, A, Sp, iters=5000, batch=512, lr=3e-4):
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    S_t = torch.tensor(S, device=DEVICE)
    A_t = torch.tensor(A, device=DEVICE)
    Sp_t = torch.tensor(Sp, device=DEVICE)
    N = S.shape[0]
    for it in range(iters):
        idx = torch.randint(0, N, (batch,), device=DEVICE)
        pred = f(S_t[idx], A_t[idx])
        loss = F.mse_loss(pred, Sp_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            print(f"  dyn iter {it}  MSE={loss.item():.4e}")
    return f


def cem_action(f, qc, actor, s_now, C_cum, t_now, horizon_env, budget,
               H=5, K=64, n_iters=2, top_k=8, act_limit=1.0):
    """CEM over H-step action sequences. Returns best first action.

    Score = Σ_{k=0..H-1} λ(u_{t+k}) · Q_c(s_{t+k}, a_k).  Dynamics via f.
    Seed the CEM distribution with the actor's proposal at s_now.
    """
    with torch.no_grad():
        s_t = torch.tensor(s_now, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a_seed, _ = actor.sample(s_t)
        mean = a_seed.expand(H, -1).clone()   # (H, act_dim)
        std = torch.full_like(mean, 0.5)
        for it in range(n_iters):
            # Sample K sequences, shape (K, H, act_dim)
            noise = torch.randn(K, H, mean.shape[-1], device=DEVICE)
            cand = (mean.unsqueeze(0) + noise * std.unsqueeze(0)).clamp(-1.0, 1.0)

            # Rollout under f; score
            s_roll = s_t.expand(K, -1).contiguous()
            scores = torch.zeros(K, device=DEVICE)
            C_sim = C_cum
            for k in range(H):
                a_k = cand[:, k]
                q = qc(s_roll, a_k)
                if isinstance(q, tuple):
                    q = torch.maximum(q[0], q[1])
                q = q.squeeze(-1)
                lam_k = urgency_lambda(t_now + k, horizon_env, C_sim, budget)
                scores += lam_k * q
                s_roll = f(s_roll, a_k)
                # Rough per-step cost estimate via Q_c (MC one-step diff inaccurate;
                # use mean Q as budget-accumulation proxy)
                C_sim = C_sim + (q.mean().item() * 0.05)   # very loose proxy
            # Elite selection
            elite_idx = torch.topk(-scores, top_k).indices
            elites = cand[elite_idx]       # (top_k, H, act_dim)
            mean = elites.mean(dim=0)
            std = elites.std(dim=0).clamp_min(0.05)
    # Return mean[0] as the chosen first action
    return mean[0].cpu().numpy() * act_limit


def online_qc_refit(qc_online, qc_anchor, S, A, Gc, iters=200,
                    batch=256, anchor_weight=0.1, lr=1e-4):
    """Short online refit with L2-to-anchor penalty."""
    opt = torch.optim.Adam(qc_online.parameters(), lr=lr)
    S_t = torch.tensor(S, device=DEVICE)
    A_t = torch.tensor(A, device=DEVICE)
    Gc_t = torch.tensor(Gc, device=DEVICE).unsqueeze(-1)
    N = S.shape[0]
    if N < batch:
        batch = N
    for it in range(iters):
        idx = torch.randint(0, N, (batch,), device=DEVICE)
        q1, q2 = qc_online(S_t[idx], A_t[idx])
        q1a, q2a = qc_anchor(S_t[idx], A_t[idx])
        loss_fit = F.mse_loss(q1, Gc_t[idx]) + F.mse_loss(q2, Gc_t[idx])
        loss_anc = F.mse_loss(q1, q1a.detach()) + F.mse_loss(q2, q2a.detach())
        loss = loss_fit + anchor_weight * loss_anc
        opt.zero_grad(); loss.backward(); opt.step()
    return qc_online


def run_episode(env, actor, qc, f, budget, ep_idx, horizon=1000,
                H=5, K=64, n_iters=2, act_limit=1.0):
    obs, _ = env.reset(seed=ep_idx * 101 + 7)
    C = 0.0; R = 0.0
    traj_S, traj_A, traj_C, traj_St = [], [], [], []
    for t in range(horizon):
        a_np = cem_action(f, qc, actor, obs, C, t, horizon, budget,
                          H=H, K=K, n_iters=n_iters, act_limit=act_limit)
        step = env.step(a_np)
        if len(step) == 6:
            obs2, r, c, term, trunc, info = step
        else:
            obs2, r, term, trunc, info = step
            c = info.get("cost", 0.0)
        traj_S.append(obs)
        traj_A.append(a_np / act_limit)
        traj_C.append(float(c))
        traj_St.append(t)
        R += float(r); C += float(c)
        obs = obs2
        if term or trunc:
            break
    return dict(reward=R, cost=C, steps=t + 1,
                S=np.asarray(traj_S, dtype=np.float32),
                A=np.asarray(traj_A, dtype=np.float32),
                c_vec=np.asarray(traj_C, dtype=np.float32))


def compute_mc_returns(c_vec, gamma=0.99):
    """Monte Carlo returns for Q_c training."""
    G = np.zeros_like(c_vec)
    running = 0.0
    for t in reversed(range(len(c_vec))):
        running = c_vec[t] + gamma * running
        G[t] = running
    return G


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--mpc-horizon", type=int, default=5, dest="H")
    p.add_argument("--n-candidates", type=int, default=64, dest="K")
    p.add_argument("--cem-iters", type=int, default=2)
    p.add_argument("--n-eps", type=int, default=20)
    p.add_argument("--dyn-iters", type=int, default=5000)
    p.add_argument("--dyn-eps", type=int, default=100)
    p.add_argument("--online-qc", action="store_true")
    p.add_argument("--anchor-weight", type=float, default=0.1)
    p.add_argument("--refit-iters", type=int, default=200)
    p.add_argument("--out", default="results/mpc_safety.json")
    args = p.parse_args()

    # Load actor + Q_c
    ac = torch.load(CKPT / f"sac_safety_point_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()
    cc = torch.load(CKPT / f"cost_critic_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    qc = CostCritic(cc["obs_dim"], cc["act_dim"], cc["hidden"]).to(DEVICE)
    qc.load_state_dict(cc["model"]); qc.eval()
    qc_anchor = CostCritic(cc["obs_dim"], cc["act_dim"], cc["hidden"]).to(DEVICE)
    qc_anchor.load_state_dict(cc["model"]); qc_anchor.eval()

    # Collect transitions + train dynamics
    env = safety_gymnasium.make(ac["env_name"])
    print(f"[seed {args.seed}] collecting {args.dyn_eps} eps transitions...")
    S, A, Sp, C_trans = collect_transitions(env, actor, ac["act_limit"],
                                             n_eps=args.dyn_eps)
    print(f"  collected {len(S)} transitions")
    f = DynamicsModel(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    print("training f_θ...")
    train_dynamics(f, S, A, Sp, iters=args.dyn_iters)
    f.eval()

    # Evaluate
    results = {"budget": args.budget, "seed": args.seed,
               "online_qc": args.online_qc, "per_ep": []}
    qc_live = qc
    if args.online_qc:
        qc_live = CostCritic(cc["obs_dim"], cc["act_dim"], cc["hidden"]).to(DEVICE)
        qc_live.load_state_dict(cc["model"])
    buf_S, buf_A, buf_G = [], [], []

    for ep in range(args.n_eps):
        ep_res = run_episode(env, actor, qc_live, f, args.budget, ep,
                             horizon=args.horizon, H=args.H, K=args.K,
                             n_iters=args.cem_iters, act_limit=ac["act_limit"])
        results["per_ep"].append(dict(reward=ep_res["reward"],
                                       cost=ep_res["cost"],
                                       steps=ep_res["steps"]))
        print(f"  ep {ep}: R={ep_res['reward']:.1f} C={ep_res['cost']:.1f} "
              f"steps={ep_res['steps']}  budget={args.budget}")
        if args.online_qc:
            Gc = compute_mc_returns(ep_res["c_vec"])
            buf_S.append(ep_res["S"]); buf_A.append(ep_res["A"])
            buf_G.append(Gc)
            S_all = np.concatenate(buf_S)
            A_all = np.concatenate(buf_A)
            G_all = np.concatenate(buf_G)
            qc_live = online_qc_refit(qc_live, qc_anchor, S_all, A_all, G_all,
                                       iters=args.refit_iters,
                                       anchor_weight=args.anchor_weight)

    env.close()
    costs = [e["cost"] for e in results["per_ep"]]
    rews = [e["reward"] for e in results["per_ep"]]
    results["summary"] = dict(
        cost_mean=float(np.mean(costs)), cost_se=float(np.std(costs, ddof=1) / np.sqrt(len(costs))),
        reward_mean=float(np.mean(rews)), reward_se=float(np.std(rews, ddof=1) / np.sqrt(len(rews))),
        budget_satisfied_rate=float(np.mean([c <= args.budget for c in costs])),
    )
    print(f"SUMMARY: cost={results['summary']['cost_mean']:.1f}±"
          f"{results['summary']['cost_se']:.2f}  budget_sat="
          f"{results['summary']['budget_satisfied_rate']*100:.0f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_data = {}
    if Path(args.out).exists():
        try: out_data = json.load(open(args.out))
        except Exception: pass
    key = f"seed{args.seed}_B{args.budget}" + ("_online" if args.online_qc else "")
    out_data[key] = results
    with open(args.out, "w") as fh:
        json.dump(out_data, fh, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
