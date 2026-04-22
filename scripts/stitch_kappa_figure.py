"""
Stitch WF + SG snapshot PNGs into 2-panel κ-diagnostic figure with
explicit magnitude annotations (so the weak-coupling SG panel reads as
evidence, not as a blurry figure).

Expects:
  results/windfarm_ep_B5_tight_snap.png   (κ=0.72, ||∇_a||=0.69, ||∇_s||=0.97)
  results/safety_gym_compare_B10_snap.png (κ=0.02, ||∇_a||=0.99, ||∇_s||=49.8)

Writes:
  latex_paper/figures/fig_kappa_diagnostic.png
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wf", default="results/windfarm_ep_B5_tight_snap.png")
    p.add_argument("--sg", default="results/safety_gym_compare_B10_snap.png")
    p.add_argument("--out", default="latex_paper/figures/fig_kappa_diagnostic.png")
    args = p.parse_args()

    wf = mpimg.imread(args.wf)
    sg = mpimg.imread(args.sg)

    fig = plt.figure(figsize=(13, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 0.5], wspace=0.15)

    ax_wf = fig.add_subplot(gs[0])
    ax_sg = fig.add_subplot(gs[1])
    ax_bar = fig.add_subplot(gs[2])

    ax_wf.imshow(wf); ax_wf.axis("off")
    ax_wf.set_title("(a) Wind farm  —  $\\kappa\\!\\approx\\!0.72$  (strong coupling)",
                     fontsize=12)
    ax_wf.text(0.5, -0.04,
               "$\\|\\nabla_a Q_c\\|=0.69$   $\\|\\nabla_s Q_c\\|=0.97$",
               ha="center", va="top", transform=ax_wf.transAxes,
               fontsize=11, color="#003366")

    ax_sg.imshow(sg); ax_sg.axis("off")
    ax_sg.set_title("(b) Safety Gym  —  $\\kappa\\!\\approx\\!0.02$  (weak coupling)",
                     fontsize=12)
    ax_sg.text(0.5, -0.04,
               "$\\|\\nabla_a Q_c\\|=0.99$   $\\|\\nabla_s Q_c\\|=49.8$",
               ha="center", va="top", transform=ax_sg.transAxes,
               fontsize=11, color="#663300")

    # Bar chart making the 36× gap visually explicit
    ax_bar.barh([1], [0.72], color="#004488", label="wind farm")
    ax_bar.barh([0], [0.02], color="#bb4400", label="safety gym")
    ax_bar.set_yticks([0, 1]); ax_bar.set_yticklabels(["SG", "WF"])
    ax_bar.set_xscale("log"); ax_bar.set_xlim(0.01, 1.5)
    ax_bar.set_xlabel("$\\kappa$ (log scale)", fontsize=11)
    ax_bar.axvline(1.0, color="gray", ls=":", lw=1, alpha=0.6)
    ax_bar.set_title("$36\\times$ gap", fontsize=12)
    ax_bar.grid(axis="x", alpha=0.3, which="both")
    for sp in ("top", "right"):
        ax_bar.spines[sp].set_visible(False)

    fig.suptitle("Coupling-strength diagnostic:  "
                 "$\\kappa = \\|\\nabla_a Q_c\\| / \\|\\nabla_s Q_c\\|$",
                 fontsize=14, y=1.02)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=160, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
