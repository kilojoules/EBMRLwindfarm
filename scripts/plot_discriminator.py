"""Plot discriminator scatter: ensemble std and ||Δa|| vs cost_ahead."""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import spearmanr

OUT = Path(__file__).parent.parent / "latex_paper" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

d = np.load(Path(__file__).parent.parent / "results" / "discriminator_qc.npz")
ens, dnorm, cost = d["ens_std"], d["delta_norm"], d["cost_ahead"]
rho_e, _ = spearmanr(ens, cost)
rho_d, _ = spearmanr(dnorm, cost)

# Bin by predictor, show mean cost per bin
def bin_plot(ax, x, y, nbins=20, label="", color="C0"):
    bins = np.quantile(x, np.linspace(0, 1, nbins + 1))
    centers, means, ses = [], [], []
    for i in range(nbins):
        m = (x >= bins[i]) & (x < bins[i + 1])
        if m.sum() < 5:
            continue
        centers.append(np.median(x[m]))
        means.append(y[m].mean())
        ses.append(y[m].std() / np.sqrt(m.sum()))
    ax.errorbar(centers, means, yerr=ses, color=color, lw=1.5, capsize=2,
                marker="o", ms=4, label=label)

fig, axes = plt.subplots(1, 2, figsize=(7, 3.0))

bin_plot(axes[0], ens, cost, label=fr"$\rho_{{\rm Spearman}}={rho_e:+.3f}$", color="C0")
axes[0].set_xlabel(r"Ensemble std  $\sigma(Q_c)$")
axes[0].set_ylabel(r"Mean cost in next $H{=}20$ steps")
axes[0].legend(loc="upper left", fontsize=9)
axes[0].grid(alpha=0.3)

bin_plot(axes[1], dnorm, cost, label=fr"$\rho_{{\rm Spearman}}={rho_d:+.3f}$", color="C3")
axes[1].set_xlabel(r"$\|\Delta a\|$")
axes[1].legend(loc="upper left", fontsize=9)
axes[1].grid(alpha=0.3)

fig.suptitle(r"Predictors of $Q_c$ failure (Safety Gym, 20k state samples)",
             fontsize=10)
fig.tight_layout()
out = OUT / "fig_discriminator.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
print(f"ensemble std: rho={rho_e:+.3f}")
print(f"||Delta a||:  rho={rho_d:+.3f}")
