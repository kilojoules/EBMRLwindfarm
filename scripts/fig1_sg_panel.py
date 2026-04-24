"""Clean SG panel for Figure 1, post-critic revision.

Single square panel:
  - Hazards: red fill + thin dark edge
  - Goal: green star
  - Faded gray uncon trajectory (mechanism baseline, plows through hazards)
  - Colored blend trajectory (lw=5, navy → crimson via σ)
  - Slim horizontal colorbar at bottom for σ(u): 0 → 1

No σ or cost subplots. Mechanism reads in 2 seconds.

Usage: python scripts/fig1_sg_panel.py --budget 25 --eta 3 --seed 1 \
    --out latex_paper/figures/fig1_sg_clean.pdf
"""
import argparse
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
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


def rollout(seed, budget, horizon, eta, mode,
            r_repel=0.6, alpha_apf=0.5, k_rep=1.0):
    """mode: 'blend' or 'uncon'."""
    ac = torch.load(CKPT / f"sac_safety_point_seed{seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()

    env = safety_gymnasium.make(ac["env_name"])
    obs, _ = env.reset(seed=seed * 101 + 7)
    xs, sigmas = [], []
    C = 0.0
    haz_pos, r_haz = hazards_info(env)
    goal0 = goal_pos(env).copy()

    for t in range(horizon):
        pos, vel, heading = agent_state(env)
        goal = goal_pos(env)

        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a, _ = actor.sample(s)
        a_task = a.squeeze(0).cpu().numpy() * ac["act_limit"]

        if mode == "uncon":
            sigma = 0.0
            a_exec = a_task
        else:
            dir_w = apf_world_direction(pos, goal, haz_pos, r_haz,
                                         r_repel=r_repel, alpha=alpha_apf, k_rep=k_rep)
            a_safe = world_to_action(dir_w, heading, act_limit=ac["act_limit"])
            sigma = urgency_blend_sigma(t, horizon, C, budget,
                                         sharpness=eta, sigma_max=1.0)
            a_exec = np.clip((1.0 - sigma) * a_task + sigma * a_safe,
                             -ac["act_limit"], ac["act_limit"])

        xs.append(pos); sigmas.append(sigma)
        ret = env.step(a_exec)
        if len(ret) == 6: obs, r, c, term, trunc, info = ret
        else: obs, r, term, trunc, info = ret; c = info.get("cost", 0.0)
        C += float(c)
        if term or trunc: break
    env.close()
    return dict(xs=np.array(xs), sigmas=np.array(sigmas),
                haz=haz_pos, hsize=r_haz, goal0=goal0, total_cost=C)


def plot(d_blend, d_uncon, budget, out_path):
    xs = d_blend["xs"]; sigmas = d_blend["sigmas"]
    xs_u = d_uncon["xs"]
    haz = d_blend["haz"]; hsize = d_blend["hsize"]
    goal0 = d_blend["goal0"]

    fig = plt.figure(figsize=(5.5, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_cb = fig.add_subplot(gs[1])

    ax.set_aspect("equal")
    pad = 0.25
    all_xy = np.vstack([xs, xs_u, haz])
    xmin, xmax = all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad
    ymin, ymax = all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#fafafa")

    # Hazards: red fill + thin dark edge
    for hx, hy in haz:
        ax.add_patch(Circle((hx, hy), hsize, facecolor="#e53935",
                             edgecolor="#7a0000", linewidth=1.0,
                             alpha=0.30, zorder=1))

    # Faded uncon trajectory underneath
    ax.plot(xs_u[:, 0], xs_u[:, 1], "-", color="#5a5a5a",
            lw=1.8, alpha=0.45, zorder=3)

    # Blended trajectory: thick line collection, σ-colored
    cmap = LinearSegmentedColormap.from_list(
        "blendmap",
        [(0.0, "#0b3d91"),     # deep navy = pure RL
         (0.5, "#888888"),     # mid-blend gray
         (1.0, "#d32f2f")])    # crimson = pure APF
    points = xs.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1),
                        linewidth=4.5, alpha=0.95, zorder=5,
                        capstyle="round", joinstyle="round")
    lc.set_array(sigmas[:-1])
    ax.add_collection(lc)

    # Start: black square
    ax.scatter(xs[0, 0], xs[0, 1], marker="s", s=110, c="black",
               zorder=8)
    # End of blend trajectory: black triangle
    if len(xs) >= 2:
        v = xs[-1] - xs[-2]
        ax.scatter(xs[-1, 0], xs[-1, 1], marker="^", s=120, c="black",
                   zorder=8)

    # Goal: green star
    ax.scatter(goal0[0], goal0[1], marker="*", s=420, c="#2ca02c",
               edgecolor="black", linewidth=0.8, zorder=7)

    # Annotations
    ax.text(xs[0, 0] + 0.05, xs[0, 1] + 0.08, "start",
            fontsize=10, color="black")
    ax.text(goal0[0] + 0.08, goal0[1] + 0.08, "goal",
            fontsize=10, color="#2ca02c")

    # Inline label for the uncon baseline (positioned mid-trajectory)
    mid_u = xs_u[len(xs_u) // 2]
    ax.text(mid_u[0], mid_u[1] - 0.2, "$\\pi_\\mathrm{perf}$ alone\n(plows through hazards)",
            fontsize=8.5, color="#444", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.7))

    # Slim horizontal colorbar
    import matplotlib as mpl
    cb = mpl.colorbar.ColorbarBase(ax_cb, cmap=cmap,
                                    norm=plt.Normalize(0, 1),
                                    orientation="horizontal")
    cb.set_label("blend weight $\\sigma(u)$:  0 = $\\pi_\\mathrm{perf}$ (RL)  $\\to$  1 = $\\pi_\\mathrm{safe}$ (APF)",
                 fontsize=9)
    cb.ax.tick_params(labelsize=8)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=160, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eta", type=float, default=3.0)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--out", default="latex_paper/figures/fig1_sg_clean.pdf")
    args = p.parse_args()

    print("rollout: blend...")
    db = rollout(args.seed, args.budget, args.horizon, args.eta, "blend")
    print(f"  blend: len={len(db['xs'])}  C={db['total_cost']:.0f}  σ̄={db['sigmas'].mean():.2f}")
    print("rollout: uncon (baseline overlay)...")
    du = rollout(args.seed, args.budget, args.horizon, args.eta, "uncon")
    print(f"  uncon: len={len(du['xs'])}  C={du['total_cost']:.0f}")

    plot(db, du, args.budget, args.out)


if __name__ == "__main__":
    main()
