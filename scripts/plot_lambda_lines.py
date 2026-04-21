"""Line plots: u and lambda vs time remaining, per budget remaining level.

Replaces contour plot (fig_lambda_contours.pdf) with cleaner line plots.
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "latex_paper" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

tau = np.linspace(0.02, 1.0, 400)
rho_levels = [0.1, 0.25, 0.5, 0.75, 1.0]
eta_values = [1.0, 2.0, 5.0]

colors = plt.cm.viridis(np.linspace(0.15, 0.85, len(rho_levels)))

fig, axes = plt.subplots(1, 1 + len(eta_values), figsize=(4 * (1 + len(eta_values)), 3.2),
                         sharex=True)

# Panel 1: urgency ratio u = rho/tau
ax = axes[0]
for rho, c in zip(rho_levels, colors):
    u = rho / tau
    ax.plot(tau, u, color=c, lw=2, label=fr"$\rho = {rho:.2f}$")
ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
ax.set_xlim(1.0, 0.0)  # time decreasing left→right
ax.set_ylim(0, 3)
ax.set_xlabel(r"Time remaining $\tau$")
ax.set_ylabel(r"Urgency ratio $u = \rho / \tau$")
ax.legend(fontsize=8, loc="upper left", title="Budget remaining")
ax.grid(alpha=0.3)

# Panels 2+: lambda vs tau for each eta
for idx, eta in enumerate(eta_values):
    ax = axes[1 + idx]
    for rho, c in zip(rho_levels, colors):
        u = rho / tau
        w = np.exp(eta * (1.0 / u - 1.0))
        w = np.clip(w, 1e-3, 1e4)
        ax.plot(tau, w, color=c, lw=2, label=fr"$\rho = {rho:.2f}$")
    ax.axhline(1.0, color="gray", lw=1, ls="--", alpha=0.6)
    ax.set_xlim(1.0, 0.0)
    ax.set_yscale("log")
    ax.set_ylim(1e-2, 1e4)
    ax.set_xlabel(r"Time remaining $\tau$")
    if idx == 0:
        ax.set_ylabel(r"Penalty weight $w(u)$")
    ax.set_title(fr"$\eta = {eta}$")
    ax.grid(alpha=0.3, which="both")

fig.tight_layout()
out = OUT_DIR / "fig_lambda_lines.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"wrote {out}")
