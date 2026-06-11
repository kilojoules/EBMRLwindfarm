"""
CBF post-hoc shield on Safety Gym with CLEAN kinematic predictor.
No env.step/restore during planning — uses a trained MLP f_θ(pos, vel, a)
-> (pos', vel'). Bypasses all safety_gymnasium state-leak bugs.

Training data: logged (pos, vel, a, pos', vel') from 100 frozen-policy eps.

Shield at each real step:
  1. Read current agent pos, vel.
  2. Sample K actions = [actor_proposal, actor_proposal + noise, ...].
  3. Predict pos_{t+1} for each via f_θ.
  4. Evaluate CBF h_i(x_{t+1}) ≥ (1-α)·h_i(x_t) for all hazards.
  5. If actor_proposal feasible → execute it. Else pick nearest-to-actor
     feasible candidate. Else (none feasible) pick min-violation.
  6. Execute in real env.

Diagnostic: α → 0 must give reward ≈ uncon +27. Otherwise bug in
action-passing.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import safety_gymnasium

import sys
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import SafetyGymActor

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Kinematic(nn.Module):
    """(pos, vel, a) -> (Δpos, Δvel).  pos,vel are 2D; a is act_dim."""
    def __init__(self, act_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4 + act_dim, hidden), nn.SiLU(),
            nn.Linear(hidden, hidden), nn.SiLU(),
            nn.Linear(hidden, 4),
        )

    def forward(self, pos, vel, a):
        x = torch.cat([pos, vel, a], dim=-1)
        delta = self.net(x)
        return pos + delta[..., :2], vel + delta[..., 2:4]


def collect_kinematic_data(env, actor, act_limit, n_eps=50):
    """Collect (pos, vel, a, pos', vel') transitions."""
    Ps, Vs, As, Pps, Vps = [], [], [], [], []
    for ep in range(n_eps):
        obs, _ = env.reset(seed=6000 + ep)
        for _ in range(1000):
            task = env.unwrapped.task
            pos = np.asarray(task.agent.pos[:2]).copy()
            vel = np.asarray(task.agent.vel[:2]).copy()
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a, _ = actor.sample(s)
            a_norm = a.squeeze(0).cpu().numpy()
            a_exec = a_norm * act_limit
            ret = env.step(a_exec)
            if len(ret) == 6:
                obs, r, c, term, trunc, info = ret
            else:
                obs, r, term, trunc, info = ret
                c = info.get("cost", 0.0)
            pos_n = np.asarray(env.unwrapped.task.agent.pos[:2]).copy()
            vel_n = np.asarray(env.unwrapped.task.agent.vel[:2]).copy()
            Ps.append(pos); Vs.append(vel); As.append(a_norm)
            Pps.append(pos_n); Vps.append(vel_n)
            if term or trunc: break
    return (np.asarray(Ps, dtype=np.float32), np.asarray(Vs, dtype=np.float32),
            np.asarray(As, dtype=np.float32),
            np.asarray(Pps, dtype=np.float32), np.asarray(Vps, dtype=np.float32))


def train_kinematic(f, P, V, A, Pp, Vp, iters=3000, batch=512, lr=3e-4):
    opt = torch.optim.Adam(f.parameters(), lr=lr)
    P_t = torch.tensor(P, device=DEVICE); V_t = torch.tensor(V, device=DEVICE)
    A_t = torch.tensor(A, device=DEVICE)
    Pp_t = torch.tensor(Pp, device=DEVICE); Vp_t = torch.tensor(Vp, device=DEVICE)
    N = P.shape[0]
    for it in range(iters):
        idx = torch.randint(0, N, (batch,), device=DEVICE)
        pp, vp = f(P_t[idx], V_t[idx], A_t[idx])
        loss = F.mse_loss(pp, Pp_t[idx]) + F.mse_loss(vp, Vp_t[idx])
        opt.zero_grad(); loss.backward(); opt.step()
        if it % 1000 == 0:
            with torch.no_grad():
                pp, vp = f(P_t[:1024], V_t[:1024], A_t[:1024])
                mse_p = F.mse_loss(pp, Pp_t[:1024]).item()
                mse_v = F.mse_loss(vp, Vp_t[:1024]).item()
            print(f"  kin iter {it}  pos-MSE={mse_p:.4e}  vel-MSE={mse_v:.4e}")
    f.eval()
    return f


def eval_h(xy, haz_pos, r_haz):
    d = np.linalg.norm(haz_pos - xy[None, :], axis=1)
    return d - r_haz


def hazards_info(env):
    t = env.unwrapped.task
    pos = np.asarray([h[:2] for h in t.hazards.pos])
    r = float(t.hazards.size)
    return pos, r


def urgency_weight(t, T, C, budget, eta=3.0):
    if t >= T or budget <= 0: return 1.0
    rho = max((budget - C) / max(budget, 1e-9), 1e-6)
    tau = max((T - t) / T, 1e-6)
    u = rho / tau
    w = np.exp(eta * (1.0 / max(u, 1e-6) - 1.0))
    return float(np.clip(w, 1e-3, 1e4))


def cbf_action(f_kin, actor, obs_now, env, C_cum, t_now, horizon_env, budget,
               K=16, H=1, alpha_base=0.5, alpha_min=0.05, alpha_max=0.95,
               schedule="constant", eta=3.0, act_limit=1.0, noise_std=0.5,
               struct_cand=True):
    """CBF-shielded action with H-step kinematic lookahead over K candidates.
    Each candidate is a SINGLE first action; subsequent steps simulated by
    repeating that action (coarse constant-action rollout)."""
    if schedule == "urgency":
        alpha = float(np.clip(alpha_base * urgency_weight(t_now, horizon_env, C_cum,
                                                           budget, eta=eta),
                               alpha_min, alpha_max))
    else:
        alpha = alpha_base

    task = env.unwrapped.task
    haz_pos, r_haz = hazards_info(env)
    pos_t = np.asarray(task.agent.pos[:2]).copy()
    vel_t = np.asarray(task.agent.vel[:2]).copy()
    h_t = eval_h(pos_t, haz_pos, r_haz)

    # Actor proposal
    with torch.no_grad():
        s = torch.tensor(obs_now, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a_seed, _ = actor.sample(s)
    a0 = a_seed.squeeze(0).cpu().numpy()

    # Candidate bank: actor + structured cardinal + noisy perturbations
    n_noise = max(K - (9 if struct_cand else 1), 0)
    pieces = [a0[None, :]]
    if struct_cand:
        # 8 cardinal directions for thrust-dominated scan
        ang = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        cardinal = np.stack([np.cos(ang), np.sin(ang)], axis=-1)  # (8, 2)
        if a0.shape[0] >= 2:
            cardinal = cardinal[:, :a0.shape[0]]
        else:
            cardinal = cardinal[:, :1]
        pieces.append(cardinal)
    if n_noise > 0:
        noise = np.random.randn(n_noise, a0.shape[0]) * noise_std
        pieces.append(np.clip(a0[None, :] + noise, -1.0, 1.0))
    cand_norm = np.concatenate(pieces, axis=0)
    K_actual = cand_norm.shape[0]

    # Multi-step rollout under constant action per candidate
    with torch.no_grad():
        pos_k = torch.tensor(pos_t, dtype=torch.float32, device=DEVICE).expand(K_actual, -1).contiguous()
        vel_k = torch.tensor(vel_t, dtype=torch.float32, device=DEVICE).expand(K_actual, -1).contiguous()
        a_k = torch.tensor(cand_norm, dtype=torch.float32, device=DEVICE)
        all_pos = [pos_k.cpu().numpy()]
        for _ in range(H):
            pos_k, vel_k = f_kin(pos_k, vel_k, a_k)
            all_pos.append(pos_k.cpu().numpy())
    # h over entire horizon; require CBF holds at every step
    # For multistep with decay rate α per step, h_{t+h+1} >= (1-α)*h_{t+h}
    feasible = np.ones(K_actual, dtype=bool)
    h_prev = np.broadcast_to(h_t[None, :], (K_actual, len(haz_pos))).copy()
    margin_min = np.full(K_actual, np.inf)
    for h_i in range(1, H + 1):
        h_curr = np.stack([eval_h(all_pos[h_i][k], haz_pos, r_haz) for k in range(K_actual)], axis=0)
        rhs = np.where(h_prev >= 0, (1.0 - alpha) * h_prev, h_prev + 0.02)
        feasible &= np.all(h_curr >= rhs, axis=1)
        margin_min = np.minimum(margin_min, (h_curr - rhs).min(axis=1))
        h_prev = h_curr

    # Selection
    if feasible[0]:
        return cand_norm[0] * act_limit
    if feasible.any():
        dists = np.linalg.norm(cand_norm - a0[None, :], axis=1)
        dists = np.where(feasible, dists, np.inf)
        best = int(np.argmin(dists))
        return cand_norm[best] * act_limit
    best = int(np.argmax(margin_min))
    return cand_norm[best] * act_limit


def run_episode(env, actor, f_kin, budget, ep_idx, horizon=1000, K=16, H=1,
                alpha_base=0.5, schedule="constant", eta=3.0,
                act_limit=1.0, noise_std=0.5, struct_cand=True):
    obs, _ = env.reset(seed=ep_idx * 101 + 7)
    C = 0.0; R = 0.0
    for t in range(horizon):
        a_np = cbf_action(f_kin, actor, obs, env, C, t, horizon, budget,
                           K=K, H=H, alpha_base=alpha_base, schedule=schedule,
                           eta=eta, act_limit=act_limit, noise_std=noise_std,
                           struct_cand=struct_cand)
        ret = env.step(a_np)
        if len(ret) == 6:
            obs, r, c, term, trunc, info = ret
        else:
            obs, r, term, trunc, info = ret
            c = info.get("cost", 0.0)
        R += float(r); C += float(c)
        if term or trunc: break
    return dict(reward=R, cost=C, steps=t + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--n-eps", type=int, default=20)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--dyn-eps", type=int, default=50)
    p.add_argument("--dyn-iters", type=int, default=3000)
    p.add_argument("--n-candidates", type=int, default=16, dest="K")
    p.add_argument("--horizon-cbf", type=int, default=1, dest="H",
                   help="CBF lookahead horizon (constant action per candidate)")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--schedule", choices=["constant", "urgency"], default="constant")
    p.add_argument("--eta", type=float, default=3.0)
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--out", default="results/cbf_shield_kinematic_sg.json")
    args = p.parse_args()

    ac = torch.load(CKPT / f"sac_safety_point_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()

    env = safety_gymnasium.make(ac["env_name"])
    print(f"[seed {args.seed}] collecting {args.dyn_eps} eps kin data...")
    P, V, A, Pp, Vp = collect_kinematic_data(env, actor, ac["act_limit"],
                                             n_eps=args.dyn_eps)
    print(f"  N={len(P)}  pos std={P.std(axis=0)}  vel std={V.std(axis=0)}")
    f_kin = Kinematic(ac["act_dim"]).to(DEVICE)
    print("training kinematic f_θ...")
    train_kinematic(f_kin, P, V, A, Pp, Vp, iters=args.dyn_iters)

    print(f"eval: α={args.alpha} schedule={args.schedule} noise={args.noise_std}")
    np.random.seed(8000 + args.seed)
    per_ep = []
    for ep in range(args.n_eps):
        r = run_episode(env, actor, f_kin, args.budget, ep,
                        horizon=args.horizon, K=args.K, H=args.H,
                        alpha_base=args.alpha, schedule=args.schedule,
                        eta=args.eta, act_limit=ac["act_limit"],
                        noise_std=args.noise_std)
        per_ep.append(r)
        print(f"  ep {ep}: R={r['reward']:.1f} C={r['cost']:.1f} "
              f"steps={r['steps']}", flush=True)
    env.close()

    costs = [e["cost"] for e in per_ep]
    rews = [e["reward"] for e in per_ep]
    summary = {
        "seed": args.seed, "budget": args.budget,
        "alpha": args.alpha, "schedule": args.schedule,
        "cost_mean": float(np.mean(costs)),
        "cost_se": float(np.std(costs, ddof=1) / np.sqrt(len(costs))) if len(costs) > 1 else 0.0,
        "reward_mean": float(np.mean(rews)),
        "reward_se": float(np.std(rews, ddof=1) / np.sqrt(len(rews))) if len(rews) > 1 else 0.0,
        "sat_rate": float(np.mean([c <= args.budget for c in costs])),
        "per_ep": per_ep,
    }
    print(f"SUMMARY: cost={summary['cost_mean']:.1f}±{summary['cost_se']:.1f} "
          f"R={summary['reward_mean']:.1f}  sat={summary['sat_rate']*100:.0f}%")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if Path(args.out).exists():
        try: data = json.load(open(args.out))
        except Exception: pass
    key = f"seed{args.seed}_B{args.budget}_α{args.alpha}_H{args.H}_K{args.K}_ns{args.noise_std}_{args.schedule}"
    data[key] = summary
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
