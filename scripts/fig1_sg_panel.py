"""Clean SG panel for Figure 1: trajectory colored by blend weight σ(u).

Runs one APF-blend episode at a challenging budget, plots birds-eye
with:
  - Hazards (red filled)
  - Goal (green)
  - Trajectory as a color-gradient line (blue σ=0 → orange σ=1)
  - Small σ-track subplot below

Communicates: as budget runs low, σ rises; agent switches from
performance policy toward APF safety controller, bending path around
hazards.

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


def rollout(seed, budget, horizon, eta, r_repel=0.6, alpha_apf=0.5, k_rep=1.0):
    ac = torch.load(CKPT / f"sac_safety_point_seed{seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()

    env = safety_gymnasium.make(ac["env_name"])
    obs, _ = env.reset(seed=seed * 101 + 7)
    xs, gs, sigmas, cs, goal_reach_t = [], [], [], [], []
    C = 0.0
    haz_pos, r_haz = hazards_info(env)
    last_goal = np.asarray(env.unwrapped.task.goal.pos[:2]).copy()

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

        sigma = urgency_blend_sigma(t, horizon, C, budget,
                                     sharpness=eta, sigma_max=1.0)
        a_exec = np.clip((1.0 - sigma) * a_task + sigma * a_safe,
                          -ac["act_limit"], ac["act_limit"])

        xs.append(pos); gs.append(goal.copy()); sigmas.append(sigma)
        ret = env.step(a_exec)
        if len(ret) == 6: obs, r, c, term, trunc, info = ret
        else: obs, r, term, trunc, info = ret; c = info.get("cost", 0.0)
        C += float(c); cs.append(C)
        new_goal = np.asarray(env.unwrapped.task.goal.pos[:2]).copy()
        if not np.allclose(new_goal, last_goal, atol=1e-3):
            goal_reach_t.append(t)
            last_goal = new_goal
        if term or trunc: break
    env.close()
    return dict(xs=np.array(xs), goals=np.array(gs), haz=haz_pos, hsize=r_haz,
                sigmas=np.array(sigmas), costs=np.array(cs),
                goal_reach=goal_reach_t)


def plot(data, budget, out_path):
    xs = data["xs"]; sigmas = data["sigmas"]; haz = data["haz"]
    hsize = data["hsize"]; goals = data["goals"]; costs = data["costs"]

    fig = plt.figure(figsize=(6.5, 6.5))
    gs = fig.add_gridspec(3, 1, height_ratios=[4, 1, 1], hspace=0.12)
    ax = fig.add_subplot(gs[0])
    ax_s = fig.add_subplot(gs[1], sharex=None)
    ax_c = fig.add_subplot(gs[2])

    # --- Birds-eye with trajectory color-coded by sigma
    ax.set_aspect("equal")
    pad = 0.3
    xmin = min(xs[:, 0].min(), haz[:, 0].min() - hsize) - pad
    xmax = max(xs[:, 0].max(), haz[:, 0].max() + hsize) + pad
    ymin = min(xs[:, 1].min(), haz[:, 1].min() - hsize) - pad
    ymax = max(xs[:, 1].max(), haz[:, 1].max() + hsize) + pad
    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(alpha=0.25)

    # Hazards
    for hx, hy in haz:
        ax.add_patch(Circle((hx, hy), hsize, color="#d62728", alpha=0.28,
                             linewidth=0.6, edgecolor="#a00"))
    # First hazard gets a label
    ax.add_patch(Circle((haz[0, 0], haz[0, 1]), hsize, color="#d62728",
                         alpha=0.0, label="hazard"))  # legend entry only

    # Goal markers (start + any subsequent respawns)
    unique_goals = [goals[0]]
    for t in data["goal_reach"]:
        if t < len(goals):
            unique_goals.append(goals[t])
    for i, g in enumerate(unique_goals):
        ax.add_patch(Circle(g, 0.12, facecolor="#2ca02c",
                             edgecolor="#0a5", alpha=0.6, zorder=4,
                             label="goal" if i == 0 else None))

    # Trajectory as a colored line collection
    cmap = LinearSegmentedColormap.from_list(
        "blend", [(0.0, "#1f77b4"), (0.5, "#888888"), (1.0, "#ff7f0e")])
    points = xs.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap=cmap, norm=plt.Normalize(0, 1),
                        linewidth=2.2, alpha=0.9, zorder=5)
    lc.set_array(sigmas[:-1])
    ax.add_collection(lc)

    # Start/end markers
    ax.scatter(xs[0, 0], xs[0, 1], marker="s", s=60, c="white",
               edgecolor="black", zorder=6, label="start")
    ax.scatter(xs[-1, 0], xs[-1, 1], marker="o", s=60, c="white",
               edgecolor="black", zorder=6, label="end")

    # Legend + colorbar
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    handles = [
        Patch(facecolor="#d62728", alpha=0.35, label="hazard"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#2ca02c",
               markersize=9, label="goal"),
        Line2D([0], [0], color="#1f77b4", lw=2.5, label="$\\sigma\\!=\\!0$ (pure RL)"),
        Line2D([0], [0], color="#ff7f0e", lw=2.5, label="$\\sigma\\!=\\!1$ (pure APF)"),
    ]
    ax.legend(handles=handles, loc="upper right", fontsize=9, framealpha=0.95)
    ax.set_title(f"Safety Gym APF blend ($d\\!=\\!{budget}$, $\\eta\\!=\\!3$)", fontsize=11)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")

    # --- σ track
    ts = np.arange(len(sigmas))
    ax_s.plot(ts, sigmas, "-", color="#555555", lw=1.5)
    ax_s.fill_between(ts, 0, sigmas, color="#ff7f0e", alpha=0.25)
    ax_s.set_ylim(-0.05, 1.1)
    ax_s.set_ylabel("$\\sigma(u)$", fontsize=10)
    ax_s.grid(alpha=0.3)
    ax_s.tick_params(axis='x', labelsize=8)
    for t in data["goal_reach"]:
        ax_s.axvline(t, color="#2ca02c", alpha=0.35, lw=1, ls=":")

    # --- cumulative cost track with budget
    ax_c.plot(ts, costs, "-", color="#d62728", lw=1.5)
    ax_c.axhline(budget, color="k", ls="--", lw=1, label=f"budget $d={budget}$")
    ax_c.set_ylabel("cum. cost", fontsize=10)
    ax_c.set_xlabel("step", fontsize=10)
    ax_c.grid(alpha=0.3); ax_c.legend(fontsize=8, loc="lower right")

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    fig.savefig(str(out_path).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--eta", type=float, default=3.0)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--out", default="latex_paper/figures/fig1_sg_clean.pdf")
    args = p.parse_args()
    d = rollout(args.seed, args.budget, args.horizon, args.eta)
    print(f"episode: len={len(d['xs'])} C={d['costs'][-1]:.0f} "
          f"goals_reached={len(d['goal_reach'])} σ̄={d['sigmas'].mean():.2f}")
    plot(d, args.budget, args.out)


if __name__ == "__main__":
    main()
