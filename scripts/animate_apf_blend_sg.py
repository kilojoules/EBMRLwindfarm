"""
Animate Safety Gymnasium episode under APF + urgency blend.

Birds-eye:
  - Unconstrained actor (gray dashed, reference run on same seed)
  - APF-blend actor (blue solid, with σ color coding per step)
  - Red arrows per step: APF desired direction a_safe (world frame)
  - Cumulative cost panel with budget line
  - σ(u) track panel

Usage:
  python scripts/animate_apf_blend_sg.py --budget 40 --eta 3.0 --seed 1 \
      --out results/sg_apf_blend_B40.mp4
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Circle
import safety_gymnasium

import sys
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import SafetyGymActor
from apf_blend_sg import (
    agent_state, goal_pos, hazards as hazards_info,
    apf_world_direction, world_to_action, urgency_blend_sigma,
)

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def rollout(mode, seed, budget, horizon, sharpness, r_repel, alpha_apf, k_rep):
    """mode in {'uncon', 'blend', 'safe'}."""
    ac = torch.load(CKPT / f"sac_safety_point_seed{seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()

    env = safety_gymnasium.make(ac["env_name"])
    obs, _ = env.reset(seed=seed * 101 + 7)
    xs, gs, sigmas, cs, rs = [], [], [], [], []
    arrows = []
    C, R = 0.0, 0.0
    last_goal = np.array(env.unwrapped.task.goal.pos[:2])
    haz_pos, r_haz = hazards_info(env)
    hsize = r_haz

    for t in range(horizon):
        pos, vel, heading = agent_state(env)
        goal = goal_pos(env)

        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a, _ = actor.sample(s)
        a_task = a.squeeze(0).cpu().numpy() * ac["act_limit"]

        dir_w = apf_world_direction(pos, goal, haz_pos, r_haz,
                                     r_repel=r_repel, alpha=alpha_apf, k_rep=k_rep)
        a_safe = world_to_action(dir_w, heading, act_limit=ac["act_limit"])

        if mode == "uncon":
            sigma = 0.0
        elif mode == "safe":
            sigma = 1.0
        else:   # blend
            sigma = urgency_blend_sigma(t, horizon, C, budget,
                                         sharpness=sharpness, sigma_max=1.0)
        a_exec = (1.0 - sigma) * a_task + sigma * a_safe
        a_exec = np.clip(a_exec, -ac["act_limit"], ac["act_limit"])

        xs.append(pos); gs.append(goal.copy()); sigmas.append(sigma)
        arrows.append(dir_w)
        ret = env.step(a_exec)
        if len(ret) == 6:
            obs, r, c, term, trunc, info = ret
        else:
            obs, r, term, trunc, info = ret
            c = info.get("cost", 0.0)
        C += float(c); R += float(r)
        cs.append(C); rs.append(R)
        if term or trunc:
            break

    env.close()
    return dict(xs=np.array(xs), goals=np.array(gs),
                haz=haz_pos, hsize=hsize,
                sigmas=np.array(sigmas), arrows=np.array(arrows),
                costs=np.array(cs), rewards=np.array(rs))


def make_animation(data_u, data_b, data_s, budget, out_path):
    T = min(len(data_u["xs"]), len(data_b["xs"]), len(data_s["xs"]))
    haz = data_b["haz"]; hsize = data_b["hsize"]

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[3, 3, 1.2],
                           hspace=0.35, wspace=0.2)
    ax_u = fig.add_subplot(gs[:2, 0])
    ax_b = fig.add_subplot(gs[:2, 1])
    ax_s = fig.add_subplot(gs[:2, 2])
    ax_c = fig.add_subplot(gs[2, :])

    xy_all = np.vstack([data_u["xs"], data_b["xs"], data_s["xs"], haz,
                         data_u["goals"], data_b["goals"], data_s["goals"]])
    pad = 0.5
    xmin, xmax = xy_all[:, 0].min()-pad, xy_all[:, 0].max()+pad
    ymin, ymax = xy_all[:, 1].min()-pad, xy_all[:, 1].max()+pad

    titles = ["Unconstrained (π_perf only)",
              f"Blend (η=3, d={budget})",
              "Fully safe (π_safe only)"]
    axes = [ax_u, ax_b, ax_s]
    datas = [data_u, data_b, data_s]

    for ax, d, title in zip(axes, datas, titles):
        ax.set_aspect("equal")
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
        ax.grid(alpha=0.3)
        ax.set_title(title, fontsize=10)
        for hx, hy in haz:
            ax.add_patch(Circle((hx, hy), hsize, color="C3", alpha=0.3))

    # Cumulative cost panel
    ts = np.arange(T)
    ax_c.plot(ts, data_u["costs"][:T], color="gray", lw=1.5, ls="--", label="uncon", alpha=0.7)
    ax_c.plot(ts, data_b["costs"][:T], "C0-", lw=2.0, label=f"blend (d={budget})")
    ax_c.plot(ts, data_s["costs"][:T], "C2-", lw=1.5, label="safe", alpha=0.7)
    ax_c.axhline(budget, color="k", ls=":", lw=1, label=f"budget d={budget}")
    ax_c.set_xlabel("step"); ax_c.set_ylabel("cumulative cost")
    ax_c.set_xlim(0, T); ax_c.grid(alpha=0.3); ax_c.legend(fontsize=8, loc="upper left")

    patches = []
    goals = []
    trails = []
    for ax, d in zip(axes, datas):
        for hx, hy in haz:
            pass  # already drawn
        gp = Circle(d["goals"][0], 0.3, color="C2", alpha=0.4, zorder=3)
        ax.add_patch(gp); goals.append(gp)
        trail, = ax.plot([], [], "-", lw=1.3, alpha=0.75)
        trails.append(trail)
        ag = Circle(d["xs"][0], 0.1, color="C0", zorder=5)
        ax.add_patch(ag); patches.append(ag)

    # Color trail differently per mode
    trails[0].set_color("gray")
    trails[1].set_color("C0")
    trails[2].set_color("C2")

    def update(t):
        for i, (ax, d, patch, gp, trail) in enumerate(zip(axes, datas, patches, goals, trails)):
            patch.center = d["xs"][t]
            gp.center = d["goals"][t]
            trail.set_data(d["xs"][:t+1, 0], d["xs"][:t+1, 1])
            # Color agent by current cost status (blend panel)
            if i == 1 and d["costs"][t] > budget:
                patch.set_edgecolor("red"); patch.set_linewidth(2)
            else:
                patch.set_linewidth(0)
            # Color by sigma intensity
            if i == 1:
                s = d["sigmas"][t]
                patch.set_color((s, 0.3, 1.0-s))  # blue→red as σ increases
        return patches + goals + trails

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


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--eta", type=float, default=3.0)
    p.add_argument("--r-repel", type=float, default=0.6)
    p.add_argument("--alpha-apf", type=float, default=0.5)
    p.add_argument("--k-rep", type=float, default=1.0)
    p.add_argument("--out", default="results/sg_apf_blend.mp4")
    args = p.parse_args()

    kw = dict(seed=args.seed, budget=args.budget, horizon=args.horizon,
              sharpness=args.eta, r_repel=args.r_repel,
              alpha_apf=args.alpha_apf, k_rep=args.k_rep)
    print("rollout uncon…")
    du = rollout("uncon", **kw)
    print(f"  uncon: C={du['costs'][-1]:.0f} R={du['rewards'][-1]:.1f}")
    print("rollout blend…")
    db = rollout("blend", **kw)
    print(f"  blend: C={db['costs'][-1]:.0f} R={db['rewards'][-1]:.1f} σ̄={db['sigmas'].mean():.2f}")
    print("rollout safe…")
    ds = rollout("safe", **kw)
    print(f"  safe:  C={ds['costs'][-1]:.0f} R={ds['rewards'][-1]:.1f}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    make_animation(du, db, ds, args.budget, args.out)


if __name__ == "__main__":
    main()
