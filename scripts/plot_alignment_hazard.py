"""Headline figure: alignment hazard Pareto plot.

Two-panel comparison.
Panel A (left): multi_modal DEL-aware. Power vs DEL_total for linear, rejection,
  argmin across pi_safe target sweep. Linear bulges anti-Pareto at y_tgt=0;
  rejection/argmin recover.
Panel B (right): Safety Gym PointGoal1. Reward vs Cost across theta rotation
  for the same three modes.

Outputs: latex_paper/figures/alignment_hazard.pdf
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_modes(path):
    d = json.load(open(path))
    rows = d["results"]
    out = {}
    for r in rows:
        mode = r["mode"]
        out.setdefault(mode, []).append(r)
    for mode in out:
        out[mode].sort(key=lambda r: r.get("safe_target", r.get("theta", 0.0)))
    return out


def plot_wf(ax, path):
    d = load_modes(path)
    colors = {"linear": "C3", "rejection": "C0", "argmin": "C2"}
    markers = {"linear": "o", "rejection": "s", "argmin": "^"}
    for mode in ("linear", "rejection", "argmin"):
        rows = d[mode]
        del_tot = np.array([sum(r["del_per_turb_mean"]) for r in rows])
        pwr = np.array([r["power_mean"] for r in rows]) / 1e9
        targets = np.array([r["safe_target"] for r in rows])
        ax.plot(del_tot, pwr, marker=markers[mode], color=colors[mode],
                 label=mode, lw=1.5, ms=6, alpha=0.85)
        # Annotate y_tgt=0
        idx0 = np.argmin(np.abs(targets))
        if mode == "linear":
            ax.annotate(r"$y^{\rm tgt}=0^\circ$",
                          (del_tot[idx0], pwr[idx0]),
                          textcoords="offset points", xytext=(8, -10),
                          color=colors[mode], fontsize=8)
            ax.scatter([del_tot[idx0]], [pwr[idx0]], facecolors="none",
                        edgecolors=colors[mode], s=180, lw=2,
                        label="_nolegend_")
    ax.set_xlabel(r"$\sum_i$ DEL$_i$ (kNm cumulative)")
    ax.set_ylabel("Farm power (GW-step)")
    ax.set_title("(a) Wind farm: multi\\_modal, DEL-aware actor")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)


def plot_sg(ax, path):
    d = load_modes(path)
    colors = {"linear": "C3", "rejection": "C0", "argmin": "C2"}
    markers = {"linear": "o", "rejection": "s", "argmin": "^"}
    for mode in ("linear", "rejection", "argmin"):
        rows = d[mode]
        cost = np.array([r["cost_mean"] for r in rows])
        rew = np.array([r["reward_mean"] for r in rows])
        thetas = np.array([r["theta"] for r in rows])
        ax.plot(cost, rew, marker=markers[mode], color=colors[mode],
                 label=mode, lw=1.5, ms=6, alpha=0.85)
        # Annotate worst-case theta = 3pi/4 for linear
        idx_bad = np.argmax(cost) if mode == "linear" else None
        if idx_bad is not None:
            ax.annotate(rf"$\theta\approx{thetas[idx_bad]:.2f}$",
                          (cost[idx_bad], rew[idx_bad]),
                          textcoords="offset points", xytext=(-30, -15),
                          color=colors[mode], fontsize=8)
            ax.scatter([cost[idx_bad]], [rew[idx_bad]], facecolors="none",
                        edgecolors=colors[mode], s=180, lw=2,
                        label="_nolegend_")
    ax.set_xscale("symlog", linthresh=10)
    ax.set_xlabel("Cumulative cost (hazard entries)")
    ax.set_ylabel("Episode reward")
    ax.set_title("(b) Safety Gym PointGoal1")
    ax.legend(loc="lower left", frameon=False, fontsize=9)
    ax.grid(alpha=0.3)


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                          "axes.spines.right": False})
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 4),
                                       constrained_layout=True)
    plot_wf(ax_l, ROOT / "results/blend_modes_b1p0_mm_n10.json")
    plot_sg(ax_r, ROOT / "results/sg_hazard_test.json")
    out = ROOT / "latex_paper/figures/alignment_hazard.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {out} and .png")


if __name__ == "__main__":
    main()
