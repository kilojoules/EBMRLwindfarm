"""
CBF-based post-hoc shield on Safety Gymnasium.

Uses geometric knowledge of hazards: h_i(x) = ||x − hazard_i|| − r_hazard.
Discrete-time CBF constraint:   h_i(x_{t+1}) ≥ (1 − α) · h_i(x_t)  for all i.

At each env step:
  1. Sample K candidate actions around the frozen actor's proposal.
  2. One-step lookahead via env snapshot/restore → get x_{t+1} for each.
  3. Pick max-reward candidate that satisfies CBF for all hazards. If none
     feasible, pick the one with smallest violation.
  4. Execute chosen action in the real env.

Budget-aware: α(u) = clamp(α_base * f(u), α_min, α_max). f(u) options:
  - "constant": f(u) = 1  (critic's baseline test)
  - "urgency":  f(u) = w(u) = exp(η(1/u - 1))  — tighter shield when deficit

Usage:
  python scripts/cbf_shield_sg.py --seed 1 --budget 10 --n-eps 20 \
      --alpha 0.5 --schedule constant --n-candidates 16
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
import safety_gymnasium

import sys
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import SafetyGymActor
from mpc_true_env_sg import save_mj_state, restore_mj_state

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def agent_xy(env):
    return np.asarray(env.unwrapped.task.agent.pos)[:2]


def hazards_info(env):
    t = env.unwrapped.task
    pos = np.asarray([h[:2] for h in t.hazards.pos])
    r = float(t.hazards.size)
    return pos, r


def step_true(env, a):
    ret = env.step(a)
    if len(ret) == 6:
        obs, r, c, term, trunc, info = ret
    else:
        obs, r, term, trunc, info = ret
        c = info.get("cost", 0.0)
    return obs, float(r), float(c), term, trunc


def eval_h(xy, haz_pos, r_haz):
    """h_i = ||xy - haz_i|| - r_haz per hazard. Positive = outside."""
    d = np.linalg.norm(haz_pos - xy[None, :], axis=1)
    return d - r_haz


def urgency_weight(t, T, C, budget, eta=3.0):
    if t >= T or budget <= 0:
        return 1.0
    rho = max((budget - C) / max(budget, 1e-9), 1e-6)
    tau = max((T - t) / T, 1e-6)
    u = rho / tau
    w = np.exp(eta * (1.0 / max(u, 1e-6) - 1.0))
    return float(np.clip(w, 1e-3, 1e4))


def cbf_shield_action(env, actor, obs_now, C_cum, t_now, horizon_env, budget,
                      K=16, alpha_base=0.5, alpha_min=0.05, alpha_max=0.95,
                      schedule="constant", eta=3.0, act_limit=1.0, noise_std=0.5):
    """Return a CBF-shielded action. One-step env lookahead for K candidates."""
    # Budget-aware alpha
    if schedule == "urgency":
        alpha = np.clip(alpha_base * urgency_weight(t_now, horizon_env,
                                                    C_cum, budget, eta=eta),
                        alpha_min, alpha_max)
    else:
        alpha = alpha_base

    # Current h values
    haz_pos, r_haz = hazards_info(env)
    x_t = agent_xy(env)
    h_t = eval_h(x_t, haz_pos, r_haz)       # (n_haz,)
    # Classical discrete-time CBF: h_{t+1} ≥ (1-α)·h_t when h_t ≥ 0
    # When h_t < 0 (already in hazard), require strict improvement: h_{t+1} ≥ h_t + δ
    delta_exit = 0.02   # min forward progress out of hazard
    cbf_rhs = np.where(h_t >= 0,
                       (1.0 - alpha) * h_t,
                       h_t + delta_exit)

    # Candidate actions: actor mean + Gaussian noise
    with torch.no_grad():
        s = torch.tensor(obs_now, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a_seed, _ = actor.sample(s)
    a_seed_np = a_seed.squeeze(0).cpu().numpy()
    noise = np.random.randn(K, a_seed_np.shape[0]) * noise_std
    cand = np.clip(a_seed_np[None, :] + noise, -1.0, 1.0) * act_limit
    cand[0] = a_seed_np * act_limit  # always include actor proposal

    # Snapshot and evaluate each candidate 1-step
    snap = save_mj_state(env)
    h_next = np.zeros((K, len(haz_pos)))
    for k in range(K):
        restore_mj_state(env, snap)
        _, _, _, _, _ = step_true(env, cand[k])
        xn = agent_xy(env)
        h_next[k] = eval_h(xn, haz_pos, r_haz)
    restore_mj_state(env, snap)

    # Feasibility per classical CBF inequality
    feasible = np.all(h_next >= cbf_rhs[None, :], axis=1)

    # cand[0] is always actor's raw proposal — preserve it if feasible
    if feasible[0]:
        return cand[0]
    if feasible.any():
        # Pick feasible candidate closest to actor proposal (projection)
        a_ref = cand[0]
        dists = np.linalg.norm(cand - a_ref[None, :], axis=1)
        dists = np.where(feasible, dists, np.inf)
        best = int(np.argmin(dists))
        return cand[best]
    # No feasible: pick min-violation (maximize worst-hazard margin)
    margin = (h_next - cbf_rhs[None, :]).min(axis=1)
    best = int(np.argmax(margin))
    return cand[best]


def run_episode(env, actor, budget, ep_idx, horizon=1000,
                K=16, alpha_base=0.5, schedule="constant", eta=3.0,
                act_limit=1.0, noise_std=0.5):
    obs, _ = env.reset(seed=ep_idx * 101 + 7)
    C = 0.0; R = 0.0
    for t in range(horizon):
        a_np = cbf_shield_action(env, actor, obs, C, t, horizon, budget,
                                  K=K, alpha_base=alpha_base, schedule=schedule,
                                  eta=eta, act_limit=act_limit, noise_std=noise_std)
        obs, r, c, term, trunc = step_true(env, a_np)
        R += r; C += c
        if term or trunc:
            break
    return dict(reward=R, cost=C, steps=t + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--n-eps", type=int, default=20)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--n-candidates", type=int, default=16, dest="K")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="CBF decay rate (0<α<1, higher = tighter shield)")
    p.add_argument("--schedule", choices=["constant", "urgency"], default="constant")
    p.add_argument("--eta", type=float, default=3.0, help="urgency schedule sharpness")
    p.add_argument("--noise-std", type=float, default=0.5)
    p.add_argument("--out", default="results/cbf_shield_sg.json")
    args = p.parse_args()

    ac = torch.load(CKPT / f"sac_safety_point_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()

    env = safety_gymnasium.make(ac["env_name"])
    # Validate snapshot works
    env.reset(seed=args.seed * 17)
    _ = save_mj_state(env)
    print(f"[seed {args.seed}] snapshot OK. α={args.alpha} schedule={args.schedule}")

    per_ep = []
    np.random.seed(7000 + args.seed)
    for ep in range(args.n_eps):
        r = run_episode(env, actor, args.budget, ep,
                        horizon=args.horizon, K=args.K,
                        alpha_base=args.alpha, schedule=args.schedule,
                        eta=args.eta, act_limit=ac["act_limit"],
                        noise_std=args.noise_std)
        per_ep.append(r)
        print(f"  ep {ep}: R={r['reward']:.1f} C={r['cost']:.1f} "
              f"steps={r['steps']}  budget={args.budget}", flush=True)
    env.close()

    costs = [e["cost"] for e in per_ep]
    rews = [e["reward"] for e in per_ep]
    summary = {
        "seed": args.seed, "budget": args.budget, "n_eps": args.n_eps,
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
    key = f"seed{args.seed}_B{args.budget}_α{args.alpha}_{args.schedule}"
    data[key] = summary
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
