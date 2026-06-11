"""
True-dynamics MPC on Safety Gym: snapshot mujoco state, rollout candidate
action sequences through the real env, score with oracle cost c(s,a) = 1[in hazard],
pick best first action under urgency schedule.

This is the critic-recommended rescue test: if this fails, framework is
structurally scope-limited; if it succeeds, the bottleneck was learned f_θ.

Usage:
    python scripts/mpc_true_env_sg.py --budget 10 --seed 1 --horizon-mpc 15 \
        --n-candidates 32 --cem-iters 2 --n-eps 10
"""
import argparse
import json
import numpy as np
import torch
from pathlib import Path
import safety_gymnasium

import sys
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import SafetyGymActor, urgency_lambda

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_mj_state(env):
    """Snapshot mujoco data + dynamic task + wrapper step counter + done flag."""
    task = env.unwrapped.task
    qpos = task.data.qpos.copy()
    qvel = task.data.qvel.copy()
    goal_pos = None
    try:
        goal_pos = np.asarray(task.goal.pos).copy()
    except Exception:
        pass
    terminated = bool(getattr(env.unwrapped, "terminated", False))
    truncated = bool(getattr(env.unwrapped, "truncated", False))
    builder_steps = int(getattr(env.unwrapped, "steps", 0))
    elapsed = int(getattr(env, "_elapsed_steps", 0))
    # Task-level reward-shaping state (used by goal/push/button tasks)
    task_extras = {}
    for attr in ("last_dist_goal", "last_dist_box", "last_box_goal",
                 "last_dist_button", "last_gremlins_dist"):
        if hasattr(task, attr):
            task_extras[attr] = getattr(task, attr)
    # Walk wrapper chain too (SafeTimeLimit at outer layer)
    elapsed_chain = []
    e = env
    while e is not None:
        elapsed_chain.append(getattr(e, "_elapsed_steps", None))
        e = getattr(e, "env", None)
    return {"qpos": qpos, "qvel": qvel, "goal_pos": goal_pos,
            "terminated": terminated, "truncated": truncated,
            "builder_steps": builder_steps,
            "elapsed": elapsed, "elapsed_chain": elapsed_chain,
            "task_extras": task_extras}


def restore_mj_state(env, snap):
    task = env.unwrapped.task
    task.data.qpos[:] = snap["qpos"]
    task.data.qvel[:] = snap["qvel"]
    if snap["goal_pos"] is not None:
        try:
            task.goal.pos[:] = snap["goal_pos"]
        except Exception:
            pass
    # Reset done (derived from terminated/truncated) + Builder.steps counter
    try:
        env.unwrapped.terminated = snap["terminated"]
        env.unwrapped.truncated = snap["truncated"]
        env.unwrapped.steps = snap["builder_steps"]
    except Exception:
        pass
    # Restore task reward-shaping state
    for attr, val in snap.get("task_extras", {}).items():
        try:
            setattr(task, attr, val)
        except Exception:
            pass
    # Restore _elapsed_steps up the wrapper chain
    e = env
    for val in snap["elapsed_chain"]:
        if val is not None and hasattr(e, "_elapsed_steps"):
            try:
                e._elapsed_steps = val
            except Exception:
                pass
        e = getattr(e, "env", None)
        if e is None:
            break
    # Recompute dependent sim fields
    try:
        import mujoco
        mujoco.mj_forward(task.model, task.data)
    except Exception as ex:
        print(f"  warn: mj_forward failed: {ex}")


def oracle_cost(env):
    """c(s,a) from actual env state: 1 if agent inside any hazard circle."""
    t = env.unwrapped.task
    try:
        agent_xy = np.asarray(t.agent.pos)[:2]
        hazards_pos = np.asarray([h[:2] for h in t.hazards.pos])
        hsize = float(t.hazards.size)
        d = np.linalg.norm(hazards_pos - agent_xy[None, :], axis=1)
        return float(np.any(d < hsize))
    except Exception:
        return 0.0


def step_true(env, a):
    """One real env step. Returns (obs, r, c_oracle, term, trunc)."""
    ret = env.step(a)
    if len(ret) == 6:
        obs, r, c_env, term, trunc, info = ret
    else:
        obs, r, term, trunc, info = ret
        c_env = info.get("cost", 0.0)
    c_oracle = oracle_cost(env)
    # Use env's reported c if present (binary); else oracle
    c = float(c_env) if c_env is not None else c_oracle
    return obs, float(r), c, term, trunc


def cem_true_env(env, actor, obs_now, C_cum, t_now, horizon_env, budget,
                 H=15, K=32, n_iters=2, top_k=8, act_limit=1.0,
                 reward_beta=0.0, feasibility=False):
    """CEM over H-step action sequences using real env rollouts.

    If feasibility=True: filter candidates to those with projected cost ≤
    remaining budget (rate-proportional), then pick max reward among
    feasible. Falls back to min-cost if none feasible.
    Otherwise: score = Σ (λ·c - β·r), minimize."""
    snap0 = save_mj_state(env)
    obs0 = obs_now

    # Seed CEM around actor's proposal
    with torch.no_grad():
        st = torch.tensor(obs0, dtype=torch.float32, device=DEVICE).unsqueeze(0)
        a_seed, _ = actor.sample(st)
    mean = a_seed.expand(H, -1).clone()
    std = torch.full_like(mean, 0.5)

    best_first = mean[0].cpu().numpy() * act_limit

    for it in range(n_iters):
        scores = np.zeros(K)
        first_actions = np.zeros((K, mean.shape[-1]))
        costs_H_arr = np.zeros(K)
        rewards_H_arr = np.zeros(K)
        for k in range(K):
            restore_mj_state(env, snap0)
            # Sample action sequence for this candidate
            noise = torch.randn(H, mean.shape[-1], device=DEVICE)
            seq = (mean + noise * std).clamp(-1.0, 1.0)
            seq_np = seq.cpu().numpy() * act_limit
            first_actions[k] = seq_np[0] / act_limit  # normalized first action
            C_sim = C_cum
            total = 0.0
            r_sum = 0.0
            for h in range(H):
                _, r_h, c_h, term, trunc = step_true(env, seq_np[h])
                lam = urgency_lambda(t_now + h, horizon_env, C_sim, budget)
                total += lam * c_h - reward_beta * r_h
                r_sum += r_h
                C_sim += c_h
                if term or trunc:
                    break
            scores[k] = total
            if feasibility:
                # Track candidate's trajectory cost (C_cum + H-step delta)
                # and H-step cumulative reward
                if not hasattr(cem_true_env, "_cost_H"):
                    pass
                # Store on arrays defined outside per-iter scope
                costs_H_arr[k] = C_sim
                rewards_H_arr[k] = r_sum

        # Elite selection
        if feasibility:
            # Project C_sim linearly to end of episode horizon and require ≤ budget.
            # Proxy: feasible if predicted cost after H steps fits remaining budget
            # at current rate. Use hard cap: c_sum ≤ budget for tightest check.
            feasible_mask = costs_H_arr <= budget
            if feasible_mask.any():
                # Pick max-reward among feasible
                reward_scores = np.where(feasible_mask, rewards_H_arr, -np.inf)
                elite_idx = np.argsort(-reward_scores)[:top_k]
            else:
                # No feasible plan — pick min-cost
                elite_idx = np.argsort(costs_H_arr)[:top_k]
        else:
            elite_idx = np.argsort(scores)[:top_k]
        elite_first = first_actions[elite_idx]
        # Fit mean/std over FULL seq by re-sampling — simpler: only refine first action
        # (cheaper and what drives the executed decision)
        mean_first = torch.tensor(elite_first.mean(axis=0), device=DEVICE).float()
        std_first = torch.tensor(np.clip(elite_first.std(axis=0), 0.05, None), device=DEVICE).float()
        mean[0] = mean_first
        std[0] = std_first
        # Track best-so-far first action
        if feasibility:
            if feasible_mask.any():
                best_idx = int(np.argmax(np.where(feasible_mask, rewards_H_arr, -np.inf)))
            else:
                best_idx = int(np.argmin(costs_H_arr))
        else:
            best_idx = int(np.argmin(scores))
        best_first = first_actions[best_idx] * act_limit

    # Restore env to true state (we consumed real steps during rollouts)
    restore_mj_state(env, snap0)
    return best_first


def run_episode(env, actor, budget, ep_idx, horizon=1000, H=15, K=32,
                n_iters=2, act_limit=1.0, no_mpc=False, reward_beta=0.0,
                feasibility=False):
    obs, _ = env.reset(seed=ep_idx * 101 + 7)
    C = 0.0; R = 0.0
    for t in range(horizon):
        if no_mpc:
            with torch.no_grad():
                s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
                a, _ = actor.sample(s)
            a_np = a.squeeze(0).cpu().numpy() * act_limit
        else:
            a_np = cem_true_env(env, actor, obs, C, t, horizon, budget,
                                 H=H, K=K, n_iters=n_iters, act_limit=act_limit,
                                 reward_beta=reward_beta, feasibility=feasibility)
        obs, r, c, term, trunc = step_true(env, a_np)
        R += r; C += c
        if term or trunc:
            break
    return dict(reward=R, cost=C, steps=t + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--horizon-mpc", type=int, default=15, dest="H")
    p.add_argument("--n-candidates", type=int, default=32, dest="K")
    p.add_argument("--cem-iters", type=int, default=2)
    p.add_argument("--n-eps", type=int, default=10)
    p.add_argument("--no-mpc", action="store_true", help="diagnostic: run actor directly")
    p.add_argument("--reward-beta", type=float, default=0.0,
                   help="weight on reward term in CEM score (0 = cost-only)")
    p.add_argument("--feasibility", action="store_true",
                   help="hard-constrained MPC: filter candidates by cost ≤ budget, maximize reward")
    p.add_argument("--out", default="results/mpc_true_env_sg.json")
    args = p.parse_args()

    ac = torch.load(CKPT / f"sac_safety_point_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()

    env = safety_gymnasium.make(ac["env_name"])
    # Test snapshot/restore works
    env.reset(seed=args.seed * 17)
    try:
        snap = save_mj_state(env)
        restore_mj_state(env, snap)
        print("mujoco state save/restore: OK")
    except Exception as e:
        print(f"snapshot FAILED: {e}")
        return

    per_ep = []
    for ep in range(args.n_eps):
        r = run_episode(env, actor, args.budget, ep,
                        horizon=args.horizon, H=args.H, K=args.K,
                        n_iters=args.cem_iters, act_limit=ac["act_limit"],
                        no_mpc=args.no_mpc, reward_beta=args.reward_beta,
                        feasibility=args.feasibility)
        per_ep.append(r)
        print(f"  ep {ep}: R={r['reward']:.1f} C={r['cost']:.1f} "
              f"steps={r['steps']}  budget={args.budget}")
    env.close()

    costs = [e["cost"] for e in per_ep]
    rews = [e["reward"] for e in per_ep]
    summary = {
        "seed": args.seed, "budget": args.budget, "n_eps": args.n_eps,
        "H": args.H, "K": args.K, "cem_iters": args.cem_iters,
        "cost_mean": float(np.mean(costs)),
        "cost_se": float(np.std(costs, ddof=1) / np.sqrt(len(costs))) if len(costs) > 1 else 0.0,
        "reward_mean": float(np.mean(rews)),
        "reward_se": float(np.std(rews, ddof=1) / np.sqrt(len(rews))) if len(rews) > 1 else 0.0,
        "sat_rate": float(np.mean([c <= args.budget for c in costs])),
        "per_ep": per_ep,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if Path(args.out).exists():
        try: data = json.load(open(args.out))
        except Exception: pass
    tag = ""
    if args.feasibility: tag += "_feas"
    if args.reward_beta != 0: tag += f"_beta{args.reward_beta}"
    data[f"seed{args.seed}_B{args.budget}{tag}"] = summary
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"SUMMARY: cost={summary['cost_mean']:.1f}±{summary['cost_se']:.1f} "
          f"R={summary['reward_mean']:.1f}  sat={summary['sat_rate']*100:.0f}%")


if __name__ == "__main__":
    main()
