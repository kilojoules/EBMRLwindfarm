"""Time-to-deployment comparison plot.

X-axis: extra training steps after budget constraint revealed (log scale)
Y-axis: reward (mean over 20 eval eps)

Methods (per budget):
  - APF blend (zero retraining): horizontal line
  - Lagrangian finetune original: cost-only objective from pretrained actor
  - Lagrangian finetune + warmup + reward-scale (tuned)
  - Lagrangian finetune aggressive (high μ-init, μ-lr)
  - SAUTE state augmentation (single point at 500k)

Focus: budget d=25 (most informative; all retraining variants underperform).
"""
import json, re
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def load_finetune_json(path):
    """Return list of (steps, R, C) from one finetune json."""
    if not Path(path).exists(): return None
    j = json.load(open(path))
    # Each json has one budget, one seed, one config
    for k, v in j.items():
        return [(e["steps"], e["reward_mean"], e["cost_mean"]) for e in v["evals"]]
    return None


def load_blend_rewards():
    d = json.load(open('results/apf_blend_sg.json'))
    rewards = {}
    for k, v in d.items():
        m = re.match(r'seed(\d+)_B(\d+)_sh([\d.]+)_r[\d.]+_α([\d.]+)', k)
        if not m: continue
        B = int(m.group(2))
        sh = float(m.group(3))
        if sh != 3.0: continue
        rewards.setdefault(B, []).extend([e["reward"] for e in v["per_ep"]])
    return {B: np.mean(r) for B, r in rewards.items()}


def main():
    blend_R = load_blend_rewards()
    print(f"blend: {blend_R}")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.0), sharey=True)
    budgets = [10, 25]

    for ax, B in zip(axes, budgets):
        # Blend horizontal line
        ax.axhline(blend_R[B], color="#0b3d91", lw=2.0, ls="--",
                   label=f"APF blend (zero retrain): R={blend_R[B]:.1f}")

        # Unconstrained line
        ax.axhline(26.61, color="#222", lw=1, ls=":", alpha=0.6,
                   label="Unconstrained (R=27)")

        # Original finetune (cost-only)
        orig_data = json.load(open('results/saclag_finetune.json'))
        for k, v in orig_data.items():
            if v["budget"] == B:
                pts = [(e["steps"], e["reward_mean"]) for e in v["evals"]]
                xs = [max(p[0], 1e2) for p in pts]
                ys = [p[1] for p in pts]
                ax.plot(xs, ys, "o-", color="#d62728", lw=1.6, ms=7, alpha=0.8,
                        label="Lagrangian finetune (cost-only)")
                break

        # Tuned warmup variant
        warmup = load_finetune_json(f'results/saclag_finetune_warmup_B{B}.json')
        if warmup:
            xs = [max(p[0], 1e2) for p in warmup]
            ys = [p[1] for p in warmup]
            ax.plot(xs, ys, "s-", color="#2ca02c", lw=1.8, ms=8, alpha=0.9,
                    label="Lagrangian + warmup + reward-scale")

        # Aggressive variant
        aggr = load_finetune_json(f'results/saclag_finetune_aggressive_B{B}.json')
        if aggr:
            xs = [max(p[0], 1e2) for p in aggr]
            ys = [p[1] for p in aggr]
            ax.plot(xs, ys, "^-", color="#ff7f0e", lw=1.8, ms=8, alpha=0.9,
                    label="Lagrangian aggressive (μ-init=0.5)")

        # SAUTE single point at 500k
        saute_path = f'runs/saute_sg_B{B}/actor_eval.json'
        if Path(saute_path).exists():
            sd = json.load(open(saute_path))
            ax.scatter([500_000], [sd["reward_mean"]], marker="*", s=240,
                       facecolor="#9467bd", edgecolor="black", linewidth=0.8,
                       zorder=10, label=f"SAUTE (state aug, 500k)")
            ax.errorbar([500_000], [sd["reward_mean"]],
                        yerr=[sd["reward_se"]], ecolor="#9467bd",
                        elinewidth=0.8, capsize=3, alpha=0.6, zorder=9)

        ax.set_xscale("log")
        ax.set_xlim(80, 8e5)
        ax.set_xlabel("Extra training steps after budget revealed", fontsize=11)
        ax.set_title(f"$d\\!=\\!{B}$", fontsize=12)
        ax.grid(alpha=0.3, which="both")
        ax.set_ylim(-12, 32)

    axes[0].set_ylabel("Reward (mean over 20 eval eps)", fontsize=11)
    axes[0].legend(loc="lower left", fontsize=8.5, framealpha=0.95)

    fig.suptitle("Time-to-deployment: post-hoc blend (zero retrain) vs.\\ retraining baselines",
                 fontsize=12, y=1.0)

    out = "latex_paper/figures/fig_time_to_deployment.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
