"""Overlay a legend onto existing fig1_sg_clean.png.

Used when local env lacks mujoco — cannot rerun rollouts.
Loads PNG, draws labeled legend box at upper-left, saves new PNG + PDF.
"""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.lines import Line2D
import matplotlib.image as mpimg

SRC = Path("latex_paper/figures/fig1_sg_clean.png")
OUT_PDF = Path("latex_paper/figures/fig1_sg_clean.pdf")
OUT_PNG = Path("latex_paper/figures/fig1_sg_clean.png")

img = mpimg.imread(SRC)
h, w = img.shape[:2]

fig = plt.figure(figsize=(w / 160, h / 160), dpi=160)
ax = fig.add_axes([0, 0, 1, 1])
ax.imshow(img)
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)

handles = [
    Circle((0, 0), 0.1, facecolor="#e53935", edgecolor="#7a0000",
           linewidth=1.0, alpha=0.30, label="hazard"),
    Line2D([0], [0], marker="*", color="w", markerfacecolor="#2ca02c",
           markeredgecolor="black", markersize=14, label="goal"),
    Line2D([0], [0], marker="s", color="w", markerfacecolor="black",
           markersize=8, label="start"),
    Line2D([0], [0], marker="^", color="w", markerfacecolor="black",
           markersize=9, label="end"),
    Line2D([0], [0], color="#888888", lw=4.5,
           label=r"blended traj. ($\sigma$-colored)"),
    Line2D([0], [0], color="#5a5a5a", lw=1.8, alpha=0.55,
           label=r"$\pi_\mathrm{perf}$ alone"),
]
leg = ax.legend(handles=handles, loc="upper left", fontsize=9,
                framealpha=0.95, handlelength=1.6, borderpad=0.5,
                labelspacing=0.4,
                bbox_to_anchor=(0.02, 0.98))
leg.get_frame().set_edgecolor("#888")

fig.savefig(OUT_PNG, dpi=160, bbox_inches="tight", pad_inches=0)
fig.savefig(OUT_PDF, dpi=160, bbox_inches="tight", pad_inches=0)
print(f"wrote {OUT_PNG}, {OUT_PDF}")
