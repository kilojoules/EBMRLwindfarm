"""Comparison plot: reward vs cost for SG methods.

  - Unconstrained actor (no budget)
  - APF blend (η ∈ {1, 3, 10}) at d ∈ {10, 25, 40}
  - SAC-Lagrangian baseline at d ∈ {10, 25, 40}

Saves to latex_paper/figures/fig_sg_comparison.pdf.
"""
import json, re, glob
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_apf_blend():
    """Aggregate APF blend results."""
    d = json.load(open('results/apf_blend_sg.json'))
    rows = {}
    for k, v in d.items():
        m = re.match(r'seed(\d+)_B(\d+)_sh([\d.]+)_r[\d.]+_α([\d.]+)', k)
        if not m: continue
        s, B, sh, _ = m.groups()
        key = (int(B), float(sh))
        costs = [e["cost"] for e in v["per_ep"]]
        rews = [e["reward"] for e in v["per_ep"]]
        rows.setdefault(key, {"cost": [], "rew": []})
        rows[key]["cost"].extend(costs); rows[key]["rew"].extend(rews)
    return rows


def load_sac_lag():
    """Prefer 500k-step results; fall back to 100k."""
    out = {}
    for B in [10, 25, 40]:
        for tag in ("_long", ""):
            f = f'runs/sac_lag_sg_B{B}{tag}/actor_eval.json'
            if Path(f).exists():
                d = json.load(open(f))
                d["steps"] = 500000 if tag == "_long" else 100000
                out[B] = d
                break
    return out


def main():
    apf = load_apf_blend()
    sac = load_sac_lag()

    fig, ax = plt.subplots(figsize=(7.5, 5.5))

    # Unconstrained reference (η=0, d=10 "control" in apf — cost=49.4, R=27.1)
    uncon = apf.get((10, 0.0), None)
    if uncon:
        uc = np.mean(uncon["cost"]); ur = np.mean(uncon["rew"])
        ax.scatter([uc], [ur], marker="X", s=200, c="#222222",
                   label="Unconstrained", zorder=6)
        ax.annotate("uncon", (uc, ur), xytext=(6, 6), textcoords="offset points",
                    fontsize=10, color="#222222")

    # APF blend points per (B, η)
    etas = [1.0, 3.0, 10.0]
    budgets = [10, 25, 40]
    colors = {10: "#1f77b4", 25: "#ff7f0e", 40: "#2ca02c"}
    markers = {1.0: "o", 3.0: "s", 10.0: "^"}

    for B in budgets:
        xs, ys, xe, ye = [], [], [], []
        for eta in etas:
            k = (B, eta)
            if k not in apf: continue
            c = apf[k]["cost"]; r = apf[k]["rew"]
            n = len(c)
            ax.scatter([np.mean(c)], [np.mean(r)],
                       marker=markers[eta], s=110, facecolor=colors[B],
                       edgecolor="black", linewidth=0.8,
                       alpha=0.9, zorder=5)
            xs.append(np.mean(c)); ys.append(np.mean(r))
        if len(xs) > 1:
            ax.plot(xs, ys, ":", color=colors[B], alpha=0.5, lw=1.0, zorder=3)
        # Budget line
        ax.axvline(B, color=colors[B], ls="--", lw=0.8, alpha=0.5, zorder=1)
        ax.text(B + 0.5, 2.5 + (budgets.index(B) * 2), f"d={B}",
                color=colors[B], fontsize=9, rotation=90, va="bottom")

    # SAC-Lag points
    for B, d in sac.items():
        ax.scatter([d["cost_mean"]], [d["reward_mean"]],
                   marker="*", s=260, facecolor="red", edgecolor="black",
                   linewidth=0.8, zorder=7)
        ax.errorbar([d["cost_mean"]], [d["reward_mean"]],
                    xerr=[d["cost_se"]], yerr=[d["reward_se"]],
                    ecolor="red", elinewidth=0.8, capsize=3, alpha=0.6, zorder=6)
        ax.annotate(f"SAC-Lag d={B}", (d["cost_mean"], d["reward_mean"]),
                    xytext=(6, -14), textcoords="offset points",
                    fontsize=9, color="red")

    # Pareto frontier hint: connect best points
    ax.set_xlabel("Episode cost (mean, lower is safer)", fontsize=11)
    ax.set_ylabel("Episode reward (mean, higher is better)", fontsize=11)
    ax.set_title("Safety Gymnasium PointGoal1-v0: post-hoc blend vs.\nretrained CMDP baseline (5 seeds × 20 ep APF; 1 seed × 500k steps SAC-Lag)",
                 fontsize=11)
    ax.grid(alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="X", color="w", markerfacecolor="#222222",
               markeredgecolor="#222222", markersize=12, label="Unconstrained"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=10, label="APF blend η=1"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=10, label="APF blend η=3"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor="#888",
               markeredgecolor="black", markersize=10, label="APF blend η=10"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=14, label="SAC-Lag (CMDP retrained)"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.95)

    ax.set_xlim(-3, 70)
    ax.set_ylim(-5, 33)

    Path("latex_paper/figures").mkdir(parents=True, exist_ok=True)
    out = "latex_paper/figures/fig_sg_comparison.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    # PNG backup for viewing
    fig.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
