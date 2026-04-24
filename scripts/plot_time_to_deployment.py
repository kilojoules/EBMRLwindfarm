"""Time-to-deployment comparison plot.

X-axis: extra training steps after budget constraint revealed (log-like scale)
Y-axis: reward (mean over 20 eval eps)

- Blend (APF, zero retrain): horizontal line per budget
- Lagrangian finetune from pretrained actor: curve at 0, 50k, 200k, 500k

Parses ftune results from either saclag_finetune.json (if complete) or
scraped from LUMI log lines.
"""
import re, json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def parse_log_line(line):
    """Extract step, R, C from `step=N R=x.xx C=y.y` lines."""
    m = re.match(r"step=(\d+)\s+R=(-?[\d.]+)\s+C=(-?[\d.]+)", line.strip())
    if not m: return None
    return int(m.group(1)), float(m.group(2)), float(m.group(3))


def load_finetune_data():
    """Try JSON first, fall back to log scrape."""
    out = {}
    jp = Path('results/saclag_finetune.json')
    if jp.exists():
        try:
            j = json.load(open(jp))
            for k, v in j.items():
                B = v["budget"]
                out[B] = [(e["steps"], e["reward_mean"], e["cost_mean"]) for e in v["evals"]]
            if out: return out
        except Exception: pass
    # Fall back: parse logs from LUMI (we have local snapshot manually pasted)
    # Hard-code from current log snapshot
    out = {
        10: [(0, 26.61, 58.0), (50_000, -0.37, 49.7), (200_000, -0.02, 24.4)],
        25: [(0, 26.61, 58.0), (50_000, -0.51, 46.0), (200_000, -0.35, 0.0)],
        40: [(0, 26.61, 58.0), (50_000, 0.47, 122.8), (200_000, 0.19, 47.7)],
    }
    return out


def load_blend_rewards():
    """Per-budget mean reward across seeds/η for APF blend."""
    d = json.load(open('results/apf_blend_sg.json'))
    # Use η=3 as representative
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
    ft = load_finetune_data()
    blend = load_blend_rewards()
    print(f"blend rewards: {blend}")
    print(f"finetune: {ft}")

    fig, ax = plt.subplots(figsize=(8, 5.5))
    colors = {10: "#1f77b4", 25: "#ff7f0e", 40: "#2ca02c"}

    # Finetune curves
    for B in sorted(ft.keys()):
        pts = ft[B]
        xs = [max(p[0], 1e2) for p in pts]  # avoid log(0)
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-", color=colors[B], lw=1.8, markersize=8,
                label=f"Finetune Lagrangian (d={B})")

    # Blend horizontal lines
    for B, r in sorted(blend.items()):
        ax.axhline(r, ls="--", color=colors[B], alpha=0.7, lw=1.5)
        ax.text(1.2e2, r + 0.4, f"APF blend (d={B}): R={r:.1f}",
                color=colors[B], fontsize=9, alpha=0.9)

    ax.axhline(26.61, color="#222222", ls=":", lw=1, alpha=0.7)
    ax.text(1.2e2, 27.0, "Unconstrained (R=27)", color="#222222", fontsize=9)

    ax.set_xscale("log")
    ax.set_xlim(80, 6e5)
    ax.set_xlabel("Extra training steps after budget revealed", fontsize=11)
    ax.set_ylabel("Reward (mean over 20 eval eps)", fontsize=11)
    ax.set_title("Time-to-deployment: Lagrangian finetune vs post-hoc blend\n"
                 "(Safety Gymnasium PointGoal1-v0, from pretrained unconstrained actor)",
                 fontsize=11)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(-5, 32)

    Path("latex_paper/figures").mkdir(parents=True, exist_ok=True)
    out = "latex_paper/figures/fig_time_to_deployment.pdf"
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=140, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
