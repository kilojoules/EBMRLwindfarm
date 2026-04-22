"""
Stitch WF + SG snapshot PNGs into 2-panel κ-diagnostic figure.

Expects:
  results/windfarm_ep_B5_tight_snap.png   (κ=0.72)
  results/safety_gym_compare_B10_snap.png (κ=0.02)

Writes:
  latex_paper/figures/fig_kappa_diagnostic.png
"""
import argparse
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--wf", default="results/windfarm_ep_B5_tight_snap.png")
    p.add_argument("--sg", default="results/safety_gym_compare_B10_snap.png")
    p.add_argument("--out", default="latex_paper/figures/fig_kappa_diagnostic.png")
    args = p.parse_args()

    wf = mpimg.imread(args.wf)
    sg = mpimg.imread(args.sg)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(wf); axes[0].axis("off")
    axes[0].set_title("(a) Wind farm — κ ≈ 0.72  (action gradient strong)",
                      fontsize=11)
    axes[1].imshow(sg); axes[1].axis("off")
    axes[1].set_title("(b) Safety Gym — κ ≈ 0.02  (action gradient weak)",
                      fontsize=11)
    fig.suptitle("Coupling-strength diagnostic: κ = ‖∇ₐQc‖ / ‖∇ₛQc‖",
                 fontsize=13, y=1.02)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
