"""Schematic SG panel for Figure 1.

Not a real rollout — illustrative. Two paths from start to goal:
  - pi_perf alone: straight line plowing through hazards (faded)
  - blended: curved path skirting around, sigma-colored (navy at safe segments,
    crimson at high-urgency bends)

Mechanism reads in one second.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib as mpl

OUT_PDF = Path("latex_paper/figures/fig1_sg_clean.pdf")
OUT_PNG = Path("latex_paper/figures/fig1_sg_clean.png")


def bezier(p0, p1, p2, p3, n=120):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)


def main():
    start = np.array([0.5, 0.5])
    goal = np.array([4.5, 4.3])

    # Hazards: tight cluster on the diagonal between start and goal
    hazards = [
        (np.array([2.0, 2.0]), 0.55),
        (np.array([3.2, 3.0]), 0.60),
        (np.array([1.4, 3.4]), 0.50),
        (np.array([2.6, 1.3]), 0.40),
    ]

    # pi_perf path: nearly straight, plows through middle hazards
    perf_path = np.linspace(start, goal, 60)

    # Blended path: cubic bezier swinging far right then up to goal,
    # clear of all hazards
    p0 = start
    p1 = np.array([3.4, 0.4])   # initial swing right (low along bottom)
    p2 = np.array([4.7, 1.2])   # bend up-right, clear of (2.6,1.3)
    p3 = goal
    blend_path = bezier(p0, p1, p2, p3, n=140)

    # sigma along blended path: rises toward middle (when hazards are most
    # threatening), falls near goal. Cosine bump.
    s = np.linspace(0, 1, len(blend_path))
    sigma = 0.85 * np.exp(-((s - 0.5) ** 2) / (2 * 0.18 ** 2)) + 0.05

    fig = plt.figure(figsize=(5.5, 5.8))
    gs = fig.add_gridspec(2, 1, height_ratios=[20, 1], hspace=0.05)
    ax = fig.add_subplot(gs[0])
    ax_cb = fig.add_subplot(gs[1])

    ax.set_aspect("equal")
    ax.set_xlim(0, 5); ax.set_ylim(0, 5)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_facecolor("#fafafa")
    for spine in ax.spines.values():
        spine.set_color("#888")

    # Hazards
    for c, r in hazards:
        ax.add_patch(Circle(c, r, facecolor="#e53935",
                             edgecolor="#7a0000", linewidth=1.0,
                             alpha=0.30, zorder=1))

    # pi_perf path: gray faded line + arrow at midpoint
    ax.plot(perf_path[:, 0], perf_path[:, 1], "-", color="#5a5a5a",
            lw=2.0, alpha=0.55, zorder=3)
    mid = len(perf_path) // 2
    ax.add_patch(FancyArrowPatch(
        perf_path[mid] - 0.001 * (perf_path[mid + 1] - perf_path[mid]),
        perf_path[mid + 1],
        arrowstyle="-|>", mutation_scale=18, color="#5a5a5a",
        alpha=0.75, lw=0, zorder=3))

    # Blended path: sigma-colored line collection
    cmap = LinearSegmentedColormap.from_list(
        "blendmap",
        [(0.0, "#0b3d91"),
         (0.5, "#888888"),
         (1.0, "#d32f2f")])
    pts = blend_path.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=plt.Normalize(0, 1),
                        linewidth=4.8, alpha=0.95, zorder=5,
                        capstyle="round", joinstyle="round")
    lc.set_array(sigma[:-1])
    ax.add_collection(lc)

    # Direction arrows along blended path
    for frac in [0.25, 0.55, 0.85]:
        i = int(frac * (len(blend_path) - 2))
        col = cmap(sigma[i])
        ax.add_patch(FancyArrowPatch(
            blend_path[i], blend_path[i + 2],
            arrowstyle="-|>", mutation_scale=18, color=col,
            lw=0, zorder=6))

    # Start: black square
    ax.scatter(*start, marker="s", s=130, c="black", zorder=8)
    # Goal: green star
    ax.scatter(*goal, marker="*", s=480, c="#2ca02c",
               edgecolor="black", linewidth=0.8, zorder=7)

    # Annotations
    ax.text(start[0] + 0.12, start[1] - 0.18, "start",
            fontsize=10, color="black")
    ax.text(goal[0] - 0.05, goal[1] + 0.22, "goal",
            fontsize=10, color="#2ca02c", ha="right")
    ax.text(2.0, 0.95, r"$\pi_\mathrm{perf}$ alone" + "\n(plows through)",
            fontsize=9, color="#444", ha="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec="#bbb", alpha=0.9))
    ax.text(4.7, 2.5, "blend bends\naround", fontsize=9,
            color="#7a0000", ha="center", style="italic",
            bbox=dict(boxstyle="round,pad=0.2", fc="white",
                      ec="none", alpha=0.85))

    # Schematic tag (bottom-right corner, out of legend)
    ax.text(4.95, 0.08, "schematic", fontsize=8, color="#888",
            style="italic", ha="right", va="bottom")

    # Legend
    handles = [
        Circle((0, 0), 0.1, facecolor="#e53935", edgecolor="#7a0000",
               linewidth=1.0, alpha=0.30, label="hazard"),
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#2ca02c",
               markeredgecolor="black", markersize=14, label="goal"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
               markersize=8, label="start"),
        Line2D([0], [0], color="#888888", lw=4.5,
               label=r"blended ($\sigma$-colored)"),
        Line2D([0], [0], color="#5a5a5a", lw=2.0, alpha=0.55,
               label=r"$\pi_\mathrm{perf}$ alone"),
    ]
    leg = ax.legend(handles=handles, loc="upper left", fontsize=8.5,
                    framealpha=0.95, handlelength=1.6, borderpad=0.5,
                    labelspacing=0.4)
    leg.get_frame().set_edgecolor("#888")

    # Colorbar
    cb = mpl.colorbar.ColorbarBase(
        ax_cb, cmap=cmap, norm=plt.Normalize(0, 1),
        orientation="horizontal")
    cb.set_label(
        r"blend weight $\sigma(u)$:  0 = $\pi_\mathrm{perf}$ (RL)  $\to$  1 = $\pi_\mathrm{safe}$ (APF)",
        fontsize=9)
    cb.ax.tick_params(labelsize=8)

    OUT_PDF.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_PDF, dpi=160, bbox_inches="tight")
    fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight")
    print(f"wrote {OUT_PDF}, {OUT_PNG}")


if __name__ == "__main__":
    main()
