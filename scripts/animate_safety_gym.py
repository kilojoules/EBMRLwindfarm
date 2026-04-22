"""
Birds-eye animation of Safety Gym episode.
- Left: robot + hazards + goal (rendered frame)
- Right top: cumulative cost vs budget
- Right bottom: cumulative reward

Usage: python scripts/animate_safety_gym.py --budget 40 --seed 1
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import safety_gymnasium
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import (
    SafetyGymActor, CostCritic, urgency_lambda,
    CORRECTION_STEPS, CORRECTION_LR, MAX_STEP, pess_q,
)

CKPT = Path("checkpoints")


def run(budget, seed, horizon=500, kappa=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_ck = torch.load(CKPT / f"sac_safety_point_seed{seed}.pt",
                           map_location=device, weights_only=False)
    actor = SafetyGymActor(actor_ck["obs_dim"], actor_ck["act_dim"]).to(device)
    actor.load_state_dict(actor_ck["actor"])
    actor.eval()

    ckpt = torch.load(CKPT / f"cost_critic_seed{seed}.pt",
                      map_location=device, weights_only=False)
    qc = CostCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"]).to(device)
    qc.load_state_dict(ckpt["model"])
    qc.eval()

    env = safety_gymnasium.make(actor_ck["env_name"])
    obs, _ = env.reset(seed=seed * 13 + 7)

    task = env.task
    hazards_pos = np.array([h[:2] for h in task.hazards.pos])
    hazards_size = float(task.hazards.size) if hasattr(task.hazards, 'size') else 0.2
    goal_size = float(task.goal.size) if hasattr(task.goal, 'size') else 0.3

    agent_xy, goal_xy, costs, rewards, lambdas = [], [], [], [], []
    goal_reached = []
    C, R = 0.0, 0.0
    last_goal = np.array(task.goal.pos[:2])

    for t in range(horizon):
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a_raw, _ = actor.sample(s)
        lam = urgency_lambda(t, horizon, C, budget)
        a = a_raw.clone().detach()
        for _ in range(CORRECTION_STEPS):
            a.requires_grad_(True)
            q = pess_q([qc], s, a, kappa)
            g = torch.autograd.grad(q.sum(), a)[0]
            step = lam * CORRECTION_LR * g
            sn = step.norm()
            if sn > MAX_STEP:
                step = step * (MAX_STEP / sn)
            a = (a.detach() - step).clamp(-1.0, 1.0)
        a_np = a.squeeze(0).cpu().numpy() * actor_ck["act_limit"]
        step_ret = env.step(a_np)
        if len(step_ret) == 6:
            obs, r, c, term, trunc, info = step_ret
        else:
            obs, r, term, trunc, info = step_ret
            c = info.get("cost", 0.0)
        R += float(r); C += float(c)
        try:
            xy = np.array(task.agent.pos[:2])
        except Exception:
            xy = np.array([0.0, 0.0])
        try:
            gxy = np.array(task.goal.pos[:2])
        except Exception:
            gxy = last_goal
        reached = not np.allclose(gxy, last_goal, atol=1e-3)
        last_goal = gxy
        agent_xy.append(xy); goal_xy.append(gxy); goal_reached.append(reached)
        rewards.append(R); costs.append(C); lambdas.append(lam)
        if term or trunc:
            break
    env.close()
    return {
        "agent_xy": np.array(agent_xy),
        "goal_xy": np.array(goal_xy),
        "goal_reached": np.array(goal_reached),
        "hazards_pos": hazards_pos,
        "hazards_size": hazards_size,
        "goal_size": goal_size,
        "costs": np.array(costs),
        "rewards": np.array(rewards),
        "lambdas": np.array(lambdas),
    }


def make_animation(data, budget, out_path):
    from matplotlib.patches import Circle
    agent_xy = data["agent_xy"]; goal_xy = data["goal_xy"]
    haz = data["hazards_pos"]; hsize = data["hazards_size"]
    gsize = data["goal_size"]
    reached = data["goal_reached"]
    costs = data["costs"]; rewards = data["rewards"]; lambdas = data["lambdas"]
    T = len(agent_xy)

    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.25)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])

    # Birds-eye layout
    all_xy = np.vstack([agent_xy, haz, goal_xy])
    pad = 0.5
    xmin, xmax = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    ymin, ymax = all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad
    ax_xy.set_aspect("equal"); ax_xy.set_xlim(xmin, xmax); ax_xy.set_ylim(ymin, ymax)
    ax_xy.grid(alpha=0.3); ax_xy.set_xlabel("x [m]"); ax_xy.set_ylabel("y [m]")

    for hx, hy in haz:
        ax_xy.add_patch(Circle((hx, hy), hsize, color="C3", alpha=0.35))
    goal_patch = Circle(goal_xy[0], gsize, color="C2", alpha=0.5, zorder=3)
    ax_xy.add_patch(goal_patch)
    trail, = ax_xy.plot([], [], "C0-", lw=1.2, alpha=0.7)
    agent = Circle(agent_xy[0], 0.1, color="C0", zorder=5)
    ax_xy.add_patch(agent)
    title = ax_xy.set_title("t=0")
    goals_hit = int(reached.sum())

    ts = np.arange(T)
    ax_c.plot(ts, costs, "C3-", lw=1.5, alpha=0.4)
    ax_c.axhline(budget, color="k", ls="--", lw=1, label=f"budget d={budget}")
    pt_c, = ax_c.plot([0], [costs[0]], "C3o", ms=7)
    ax_c.set_ylabel("cumulative cost"); ax_c.set_xlim(0, T)
    ax_c.legend(fontsize=8, loc="upper left"); ax_c.grid(alpha=0.3)

    ax_r.plot(ts, rewards, "C0-", lw=1.5, alpha=0.4)
    pt_r, = ax_r.plot([0], [rewards[0]], "C0o", ms=7)
    ax_r.set_xlabel("step"); ax_r.set_ylabel("cumulative reward")
    ax_r.set_xlim(0, T); ax_r.grid(alpha=0.3)

    def update(t):
        agent.center = agent_xy[t]
        goal_patch.center = goal_xy[t]
        hit = costs[t] > costs[t - 1] if t > 0 else False
        agent.set_color("C3" if hit else "C0")
        trail.set_data(agent_xy[:t + 1, 0], agent_xy[:t + 1, 1])
        n_reached = int(reached[:t + 1].sum())
        status = " BUDGET EXCEEDED" if costs[t] > budget else ""
        title.set_text(f"t={t}  λ={lambdas[t]:.1f}  C={costs[t]:.1f}/{budget}{status}  "
                        f"R={rewards[t]:.1f}  goals={n_reached}")
        pt_c.set_data([t], [costs[t]])
        pt_r.set_data([t], [rewards[t]])
        # Red-tinge everything after budget exceeded
        if costs[t] > budget:
            agent.set_edgecolor("red")
            agent.set_linewidth(2)
        else:
            agent.set_linewidth(0)
        return agent, goal_patch, trail, pt_c, pt_r, title

    ani = animation.FuncAnimation(fig, update, frames=T, interval=33, blit=False)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        ani.save(out_path, writer="ffmpeg", fps=30, dpi=100, bitrate=2000)
    except Exception as e:
        print(f"ffmpeg failed ({e}); saving as gif")
        gif_path = str(out_path).replace(".mp4", ".gif")
        ani.save(gif_path, writer="pillow", fps=15, dpi=80)
        out_path = gif_path
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--out", default="results/safety_gym_ep.mp4")
    args = p.parse_args()

    data = run(args.budget, args.seed, args.horizon, args.kappa)
    print(f"episode len={len(data['agent_xy'])}, "
          f"final C={data['costs'][-1]:.1f}, R={data['rewards'][-1]:.1f}")
    Path(args.out).parent.mkdir(exist_ok=True)
    make_animation(data, args.budget, args.out)


if __name__ == "__main__":
    main()
