"""Stitch new clean Figure 1:
  Left: WF Jensen-wake snapshot (turbines + flow field)
  Right: SG APF-blend trajectory (color-coded by σ, with σ/cost tracks)

Communicates: same urgency schedule, two domains, two different
safety controllers (zero-yaw for WF; APF for SG).
"""
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path


def main():
    wf = mpimg.imread("results/windfarm_ep_B5_tight_snap.png")
    sg = mpimg.imread("latex_paper/figures/fig1_sg_clean.png")

    fig = plt.figure(figsize=(13.5, 6.5))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.1, 1.0], wspace=0.08)
    ax_wf = fig.add_subplot(gs[0])
    ax_sg = fig.add_subplot(gs[1])

    ax_wf.imshow(wf); ax_wf.axis("off")
    ax_wf.set_title("(a) Wind farm  —  $\\pi_\\mathrm{safe}\\!\\equiv\\!0$ (zero yaw) ; $\\kappa\\!\\approx\\!0.72$",
                     fontsize=12)

    ax_sg.imshow(sg); ax_sg.axis("off")
    ax_sg.set_title("(b) Safety Gymnasium  —  $\\pi_\\mathrm{safe}$ = APF navigator ; $\\kappa\\!\\approx\\!0.02$",
                     fontsize=12)

    fig.suptitle("One schedule, two safety controllers. Blend = $(1-\\sigma(u))\\,\\pi_\\mathrm{perf} + \\sigma(u)\\,\\pi_\\mathrm{safe}$",
                 fontsize=13, y=1.02)

    out = "latex_paper/figures/fig1_blend_overview.pdf"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
