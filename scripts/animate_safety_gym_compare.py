"""
Side-by-side Safety Gym: uncon (left) vs AC-corrected (right), same seed.
Each AC frame shows action-gradient arrow ‖∇_a Q_c‖ — visually tiny because
κ_SG ≈ 0.02. Bottom panel: cumulative cost both runs vs budget d.

Also dumps final-frame PNG (for 2-panel κ-diagnostic figure, Option 8).

Usage: python scripts/animate_safety_gym_compare.py --budget 10 --seed 1
"""
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle, FancyArrow
import safety_gymnasium
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import (
    SafetyGymActor, CostCritic, urgency_lambda,
    CORRECTION_STEPS, CORRECTION_LR, MAX_STEP, pess_q,
)

CKPT = Path("checkpoints")


def rollout(budget, seed, horizon, correct, kappa=0.0):
    """One episode. correct=True applies AC; False leaves action raw."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ac = torch.load(CKPT / f"sac_safety_point_seed{seed}.pt",
                    map_location=device, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(device)
    actor.load_state_dict(ac["actor"]); actor.eval()
    cc = torch.load(CKPT / f"cost_critic_seed{seed}.pt",
                    map_location=device, weights_only=False)
    qc = CostCritic(cc["obs_dim"], cc["act_dim"], cc["hidden"]).to(device)
    qc.load_state_dict(cc["model"]); qc.eval()

    env = safety_gymnasium.make(ac["env_name"])
    obs, _ = env.reset(seed=seed * 13 + 7)
    task = env.task
    haz = np.array([h[:2] for h in task.hazards.pos])
    hs = float(task.hazards.size) if hasattr(task.hazards, "size") else 0.2
    gs = float(task.goal.size) if hasattr(task.goal, "size") else 0.3

    xs, gs_xy, cs, rs, lams = [], [], [], [], []
    g_a_norms, g_s_norms, arrows = [], [], []
    C, R = 0.0, 0.0
    last_goal = np.array(task.goal.pos[:2])
    torch.manual_seed(3000 + seed); np.random.seed(3000 + seed)

    for t in range(horizon):
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a_raw, _ = actor.sample(s)

        s_g = s.clone().detach().requires_grad_(True)
        a_g = a_raw.clone().detach().requires_grad_(True)
        q = pess_q([qc], s_g, a_g, kappa)
        g_a = torch.autograd.grad(q.sum(), a_g, retain_graph=True)[0]
        g_s = torch.autograd.grad(q.sum(), s_g)[0]
        na = float(g_a.norm()); ns = float(g_s.norm())
        g_a_norms.append(na); g_s_norms.append(ns)
        arrows.append(g_a.squeeze(0).cpu().numpy())

        lam = urgency_lambda(t, horizon, C, budget)
        a = a_raw.clone().detach()
        if correct:
            for _ in range(CORRECTION_STEPS):
                a.requires_grad_(True)
                q = pess_q([qc], s, a, kappa)
                g = torch.autograd.grad(q.sum(), a)[0]
                step = lam * CORRECTION_LR * g
                sn = step.norm()
                if sn > MAX_STEP:
                    step = step * (MAX_STEP / sn)
                a = (a.detach() - step).clamp(-1.0, 1.0)
        a_np = a.squeeze(0).cpu().numpy() * ac["act_limit"]
        step_ret = env.step(a_np)
        if len(step_ret) == 6:
            obs, r, c, term, trunc, info = step_ret
        else:
            obs, r, term, trunc, info = step_ret
            c = info.get("cost", 0.0)
        R += float(r); C += float(c)
        try: xy = np.array(task.agent.pos[:2])
        except Exception: xy = np.array([0.0, 0.0])
        try: gxy = np.array(task.goal.pos[:2])
        except Exception: gxy = last_goal
        last_goal = gxy
        xs.append(xy); gs_xy.append(gxy); cs.append(C); rs.append(R); lams.append(lam)
        if term or trunc:
            break
    env.close()
    return dict(xy=np.array(xs), goal=np.array(gs_xy),
                haz=haz, hs=hs, gs=gs, costs=np.array(cs),
                rewards=np.array(rs), lams=np.array(lams),
                g_a=np.array(g_a_norms), g_s=np.array(g_s_norms),
                arrows=np.array(arrows))


def draw_scene(ax, d, t, title, budget, show_arrow=False, arrow_scale=5.0):
    haz = d["haz"]; hs = d["hs"]; gs = d["gs"]
    xy = d["xy"]; goal = d["goal"]; costs = d["costs"]
    for hx, hy in haz:
        ax.add_patch(Circle((hx, hy), hs, color="C3", alpha=0.35))
    gp = Circle(goal[t], gs, color="C2", alpha=0.5, zorder=3)
    ax.add_patch(gp)
    ax.plot(xy[:t+1, 0], xy[:t+1, 1], "C0-", lw=1.2, alpha=0.7)
    hit = costs[t] > costs[t-1] if t > 0 else False
    color = "C3" if hit else "C0"
    edge = "red" if costs[t] > budget else "none"
    lw = 2 if costs[t] > budget else 0
    ag = Circle(xy[t], 0.1, color=color, zorder=5, ec=edge, lw=lw)
    ax.add_patch(ag)
    if show_arrow:
        g = d["arrows"][t]
        # project 2D action gradient onto heading — point-robot has
        # act=[thrust, turn]; use -g[0] as forward push direction in agent frame
        # render arrow in world frame by using velocity (diff xy) as heading
        if t > 0:
            v = xy[t] - xy[t-1]
            vn = np.linalg.norm(v)
            if vn > 1e-3:
                heading = v / vn
                perp = np.array([-heading[1], heading[0]])
                # -grad direction = correction direction
                dvec = -(g[0] * heading + g[1] * perp) * arrow_scale
                ax.arrow(xy[t, 0], xy[t, 1], dvec[0], dvec[1],
                         head_width=0.05, head_length=0.07,
                         fc="red", ec="red", lw=1.2, zorder=6, alpha=0.9)
    exceed = " EXCEEDED" if costs[t] > budget else ""
    ax.set_title(f"{title}  C={costs[t]:.1f}/{budget}{exceed}", fontsize=10)


def make_animation(du, dc, budget, out_path, kappa_val=0.02):
    T = min(len(du["xy"]), len(dc["xy"]))
    fig = plt.figure(figsize=(12, 6))
    gs = fig.add_gridspec(3, 2, height_ratios=[3, 3, 1.2], hspace=0.35, wspace=0.15)
    ax_u = fig.add_subplot(gs[:2, 0])
    ax_c = fig.add_subplot(gs[:2, 1])
    ax_cost = fig.add_subplot(gs[2, :])

    all_xy = np.vstack([du["xy"], dc["xy"], du["haz"], dc["haz"],
                         du["goal"], dc["goal"]])
    pad = 0.5
    xmin, xmax = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    ymin, ymax = all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad
    for ax in (ax_u, ax_c):
        ax.set_aspect("equal"); ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)

    ts = np.arange(T)
    ax_cost.plot(ts, du["costs"][:T], "C3-", lw=1.5, label="uncon", alpha=0.8)
    ax_cost.plot(ts, dc["costs"][:T], "C0-", lw=1.5, label="AC", alpha=0.8)
    ax_cost.axhline(budget, color="k", ls="--", lw=1, label=f"d={budget}")
    pt_u, = ax_cost.plot([0], [du["costs"][0]], "C3o", ms=7)
    pt_c, = ax_cost.plot([0], [dc["costs"][0]], "C0o", ms=7)
    ax_cost.set_xlabel("step"); ax_cost.set_ylabel("cum. cost")
    ax_cost.set_xlim(0, T); ax_cost.grid(alpha=0.3)
    ax_cost.legend(fontsize=9, loc="upper left")

    fig.suptitle(f"Safety Gym: uncon vs AC  (κ={kappa_val:.2f}, weak coupling)",
                 fontsize=12)

    def update(t):
        ax_u.clear(); ax_c.clear()
        ax_u.set_aspect("equal"); ax_u.set_xlim(xmin, xmax); ax_u.set_ylim(ymin, ymax)
        ax_c.set_aspect("equal"); ax_c.set_xlim(xmin, xmax); ax_c.set_ylim(ymin, ymax)
        ax_u.grid(alpha=0.3); ax_c.grid(alpha=0.3)
        draw_scene(ax_u, du, t, "uncon", budget, show_arrow=False)
        draw_scene(ax_c, dc, t, f"AC λ={dc['lams'][t]:.1f}", budget, show_arrow=True)
        pt_u.set_data([t], [du["costs"][t]])
        pt_c.set_data([t], [dc["costs"][t]])
        return ()

    ani = animation.FuncAnimation(fig, update, frames=T, interval=33, blit=False)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        ani.save(out_path, writer="ffmpeg", fps=30, dpi=100, bitrate=2000)
    except Exception as e:
        print(f"ffmpeg failed ({e}); gif fallback")
        gif = str(out_path).replace(".mp4", ".gif")
        ani.save(gif, writer="pillow", fps=15, dpi=80); out_path = gif
    print(f"wrote {out_path}")

    # final-frame snapshot (Option 8 SG panel)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    ax2.set_aspect("equal"); ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
    ax2.grid(alpha=0.3); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    for i, (hx, hy) in enumerate(dc["haz"]):
        ax2.add_patch(Circle((hx, hy), dc["hs"], color="C3", alpha=0.35,
                              label="hazard" if i == 0 else None))
    ax2.plot(du["xy"][:T, 0], du["xy"][:T, 1], color="gray", ls="--",
             lw=1.2, alpha=0.55, label="uncon")
    ax2.plot(dc["xy"][:T, 0], dc["xy"][:T, 1], "C0-", lw=1.8, alpha=0.85,
             label="AC")
    step_ids = np.linspace(5, T - 1, 15).astype(int)
    arrow_scale = 12.0
    for ti in step_ids:
        if ti > 0 and ti < T:
            g = dc["arrows"][ti]
            v = dc["xy"][ti] - dc["xy"][ti-1]
            vn = np.linalg.norm(v)
            if vn > 1e-3:
                h = v / vn; p = np.array([-h[1], h[0]])
                dvec = -(g[0] * h + g[1] * p) * arrow_scale
                ax2.arrow(dc["xy"][ti, 0], dc["xy"][ti, 1], dvec[0], dvec[1],
                          head_width=0.07, head_length=0.10, fc="red",
                          ec="red", alpha=0.9, lw=1.5, zorder=10)
    ax2.set_title(f"Safety Gym  — action-gradient arrows (red)  "
                  f"scaled {arrow_scale:.0f}$\\times$", fontsize=10)
    ax2.legend(fontsize=9, loc="upper right")
    snap_path = str(out_path).replace(".mp4", "_snap.png").replace(".gif", "_snap.png")
    fig2.tight_layout(); fig2.savefig(snap_path, dpi=140)
    print(f"wrote {snap_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--out", default="results/safety_gym_compare.mp4")
    args = p.parse_args()

    print("rollout uncon…")
    du = rollout(args.budget, args.seed, args.horizon, correct=False)
    print(f"  uncon: C={du['costs'][-1]:.1f} R={du['rewards'][-1]:.1f}")
    print("rollout AC…")
    dc = rollout(args.budget, args.seed, args.horizon, correct=True)
    print(f"  AC:    C={dc['costs'][-1]:.1f} R={dc['rewards'][-1]:.1f}")
    kv = dc["g_a"].mean() / max(dc["g_s"].mean(), 1e-9)
    print(f"  kappa (live): {kv:.4f}")
    Path(args.out).parent.mkdir(exist_ok=True)
    make_animation(du, dc, args.budget, args.out, kappa_val=kv)


if __name__ == "__main__":
    main()
