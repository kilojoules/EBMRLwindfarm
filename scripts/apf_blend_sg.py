"""
Safety-controller blending for Safety Gymnasium.

Two-controller composition under urgency schedule:

    a(s) = (1 − σ(λ(u))) · π_task(s)  +  σ(λ(u)) · π_safe(s)

- π_task   : frozen unconstrained actor (SAC/MLP, goal-seeking)
- π_safe   : hand-coded artificial-potential-field (APF) navigator
             with repulsion from hazards and attraction to goal
- λ(u)     : urgency schedule; σ(·) maps λ to blend weight in [0, 1]

APF in world frame:
    F_goal  = (goal_pos − agent_pos) / ||.||           (unit attractor)
    F_rep   = Σ_i k_r · (agent − hazard_i) / d_i³     (inverse-square repulsor)
    a_apf   = α · F_goal  +  (1 − α) · F_rep          (a 2D world-frame vector)

World-frame vector is converted to Point-robot action (thrust, turn) by
rotating into body frame (body heading read from agent orientation).

Usage:
  python scripts/apf_blend_sg.py --seed 1 --budget 10 --n-eps 20 \
      --sigma-sharpness 3.0 --r-repel 0.6 --alpha-apf 0.5
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

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def agent_state(env):
    task = env.unwrapped.task
    pos = np.asarray(task.agent.pos[:2]).copy()
    vel = np.asarray(task.agent.vel[:2]).copy()
    # orientation: read body x-axis in world frame (mat_to_euler)
    try:
        xmat = np.asarray(task.agent.mat).reshape(3, 3)
        heading = np.arctan2(xmat[1, 0], xmat[0, 0])
    except Exception:
        # fallback: heading from velocity direction (unreliable when static)
        vn = np.linalg.norm(vel)
        heading = np.arctan2(vel[1], vel[0]) if vn > 1e-3 else 0.0
    return pos, vel, heading


def goal_pos(env):
    return np.asarray(env.unwrapped.task.goal.pos[:2]).copy()


def hazards(env):
    t = env.unwrapped.task
    pos = np.asarray([h[:2] for h in t.hazards.pos])
    r = float(t.hazards.size)
    return pos, r


def apf_world_direction(pos, goal, haz_pos, r_haz, r_repel=0.6,
                        alpha=0.5, k_rep=1.0):
    """Return desired world-frame direction vector from APF."""
    # Attractor toward goal
    d_goal = goal - pos
    ng = np.linalg.norm(d_goal)
    a_goal = d_goal / max(ng, 1e-6)

    # Repulsor from hazards within influence radius r_repel
    rep = np.zeros(2)
    for h in haz_pos:
        dv = pos - h
        d = np.linalg.norm(dv)
        # Influence distance: r_haz + r_repel
        if d < r_haz + r_repel and d > 1e-6:
            # 1/d³ falloff, normalized
            rep += k_rep * dv / (d ** 3)
    # Combine
    v = alpha * a_goal + (1.0 - alpha) * rep
    vn = np.linalg.norm(v)
    if vn > 1e-6:
        v = v / vn
    return v


def world_to_action(dir_world, heading, act_limit=1.0):
    """Map unit world-frame direction to Point-robot action (thrust, turn).
    Thrust = projection on body forward (cos(heading), sin(heading)).
    Turn   = proportional to signed angle between heading and direction."""
    fwd = np.array([np.cos(heading), np.sin(heading)])
    thrust = float(np.dot(dir_world, fwd))  # ∈ [-1, 1]
    # Angular offset (−π, π]
    target_angle = np.arctan2(dir_world[1], dir_world[0])
    d_angle = (target_angle - heading + np.pi) % (2 * np.pi) - np.pi
    turn = float(np.clip(d_angle / (np.pi / 2), -1.0, 1.0))
    return np.array([thrust, turn]) * act_limit


def urgency_blend_sigma(t, T, C, budget, sharpness=3.0, sigma_max=1.0):
    """σ ∈ [0, σ_max]. Small σ → trust actor. Large σ → trust safety controller.
    Uses the urgency ratio u = ρ/τ from the paper; σ = σ_max · (1 − exp(−s · max(0, 1/u − 1))).
    At u ≥ 1 (on pace or ahead): σ = 0. At u → 0 (exhausted): σ → σ_max."""
    if t >= T:
        return 0.0
    rho = (budget - C) / max(budget, 1e-9)
    tau = (T - t) / T
    if tau <= 1e-6 or rho >= 1e6:
        return 0.0
    u = rho / tau
    if u >= 1:
        return 0.0
    val = 1.0 - np.exp(-sharpness * (1.0 / max(u, 1e-6) - 1.0))
    return float(np.clip(sigma_max * val, 0.0, sigma_max))


def run_episode(env, actor, budget, ep_idx, horizon=1000,
                sharpness=3.0, r_repel=0.6, alpha_apf=0.5,
                sigma_max=1.0, k_rep=1.0, act_limit=1.0):
    obs, _ = env.reset(seed=ep_idx * 101 + 7)
    C = 0.0; R = 0.0
    for t in range(horizon):
        pos, vel, heading = agent_state(env)
        goal = goal_pos(env)
        haz_pos, r_haz = hazards(env)

        # Actor proposal
        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a, _ = actor.sample(s)
        a_task = a.squeeze(0).cpu().numpy() * act_limit

        # Safety controller (APF)
        dir_w = apf_world_direction(pos, goal, haz_pos, r_haz,
                                     r_repel=r_repel, alpha=alpha_apf, k_rep=k_rep)
        a_safe = world_to_action(dir_w, heading, act_limit=act_limit)

        # Urgency-based blend
        sigma = urgency_blend_sigma(t, horizon, C, budget,
                                     sharpness=sharpness, sigma_max=sigma_max)
        a_exec = (1.0 - sigma) * a_task + sigma * a_safe
        a_exec = np.clip(a_exec, -act_limit, act_limit)

        ret = env.step(a_exec)
        if len(ret) == 6:
            obs, r, c, term, trunc, info = ret
        else:
            obs, r, term, trunc, info = ret
            c = info.get("cost", 0.0)
        R += float(r); C += float(c)
        if term or trunc:
            break
    return dict(reward=R, cost=C, steps=t + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--n-eps", type=int, default=20)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--sigma-sharpness", type=float, default=3.0)
    p.add_argument("--sigma-max", type=float, default=1.0)
    p.add_argument("--r-repel", type=float, default=0.6,
                   help="influence radius of each hazard (m)")
    p.add_argument("--alpha-apf", type=float, default=0.5,
                   help="weight of goal attractor vs repulsor (0=pure-repel)")
    p.add_argument("--k-rep", type=float, default=1.0)
    p.add_argument("--out", default="results/apf_blend_sg.json")
    args = p.parse_args()

    ac = torch.load(CKPT / f"sac_safety_point_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()

    env = safety_gymnasium.make(ac["env_name"])
    print(f"[seed {args.seed}] APF blend. sharpness={args.sigma_sharpness} "
          f"r_repel={args.r_repel} α_apf={args.alpha_apf}")

    per_ep = []
    for ep in range(args.n_eps):
        r = run_episode(env, actor, args.budget, ep,
                        horizon=args.horizon,
                        sharpness=args.sigma_sharpness,
                        r_repel=args.r_repel, alpha_apf=args.alpha_apf,
                        sigma_max=args.sigma_max, k_rep=args.k_rep,
                        act_limit=ac["act_limit"])
        per_ep.append(r)
        print(f"  ep {ep}: R={r['reward']:.1f} C={r['cost']:.1f} "
              f"steps={r['steps']}", flush=True)
    env.close()

    costs = [e["cost"] for e in per_ep]
    rews = [e["reward"] for e in per_ep]
    summary = {
        "seed": args.seed, "budget": args.budget, "n_eps": args.n_eps,
        "sharpness": args.sigma_sharpness, "sigma_max": args.sigma_max,
        "r_repel": args.r_repel, "alpha_apf": args.alpha_apf,
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
    key = f"seed{args.seed}_B{args.budget}_sh{args.sigma_sharpness}_r{args.r_repel}_α{args.alpha_apf}"
    data[key] = summary
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
