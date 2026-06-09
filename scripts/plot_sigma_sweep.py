"""Plot cost-vs-sigma sweep. The figure that matches the Proposition.

Two-panel:
  (a) DEL_tot vs sigma for each configuration (hazardous + control[+ sweet-spot])
  (b) farm power vs sigma
Annotate where blend yaw crosses 0 in the hazardous configuration.

Output: latex_paper/figures/sigma_sweep.pdf + .png
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load(path):
    d = json.load(open(path))
    rows = d["results"]
    sigmas = np.array([r["sigma"] for r in rows])
    del_tot = np.array([r["del_total_mean"] for r in rows])
    del_std = np.array([r["del_total_std"] for r in rows])
    pwr = np.array([r["power_mean"] for r in rows]) / 1e9
    pwr_std = np.array([r["power_std"] for r in rows]) / 1e9
    yaw0 = np.array([r["yaw_mean_per_turb"][0] for r in rows])
    return d, sigmas, del_tot, del_std, pwr, pwr_std, yaw0


def main():
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                          "axes.spines.right": False})

    configs = []
    files = {
        "sigma_sweep_safe_plus7p5_n50.json": ("$\\pi_{\\mathrm{safe}}=+7.5^\\circ$, n=50 (hazardous)", "C3", "o"),
        "sigma_sweep_safe_minus25.json": ("$\\pi_{\\mathrm{safe}}=-25^\\circ$, n=10 (co-directional control)", "C0", "s"),
        "sigma_sweep_safe_plus15_n50.json": ("$\\pi_{\\mathrm{safe}}=+15^\\circ$, n=50 (high-DEL endpoint)", "C2", "^"),
    }
    for fname, (label, color, marker) in files.items():
        p = ROOT / "results" / fname
        if not p.exists():
            print(f"missing {p}, skipping")
            continue
        configs.append((label, color, marker, load(p)))

    fig, (ax_d, ax_p) = plt.subplots(1, 2, figsize=(10.5, 4),
                                       constrained_layout=True)

    for label, color, marker, (d, sigmas, del_tot, del_std, pwr, pwr_std, yaw0) in configs:
        n = d.get("n_episodes", 10)
        # Convert std to SE for mean estimates
        del_se = del_std / np.sqrt(n)
        pwr_se = pwr_std / np.sqrt(n)
        ax_d.errorbar(sigmas, del_tot, yerr=del_se, label=label,
                       color=color, marker=marker, capsize=3, lw=1.5, ms=5)
        ax_p.errorbar(sigmas, pwr, yerr=pwr_se, label=label,
                       color=color, marker=marker, capsize=3, lw=1.5, ms=5)
        # annotate zero-yaw crossing for hazardous configurations
        if "hazardous" in label or "low-DEL" in label:
            # find sigma where yaw_T0 crosses 0
            for i in range(len(sigmas)-1):
                if yaw0[i] * yaw0[i+1] < 0 or abs(yaw0[i]) < 1.5:
                    sig_cross = sigmas[i] if abs(yaw0[i]) < 1.5 else (sigmas[i] + sigmas[i+1]) / 2
                    ax_d.axvline(sig_cross, color=color, ls=":", lw=1, alpha=0.6)
                    ax_d.annotate(f"yaw=0", (sig_cross, ax_d.get_ylim()[1]),
                                    textcoords="offset points", xytext=(2, -10),
                                    color=color, fontsize=8)
                    break

    ax_d.set_xlabel(r"Blend weight $\sigma$")
    ax_d.set_ylabel("Cumulative DEL (kNm)")
    ax_d.set_title("(a) Cumulative DEL vs $\\sigma$")
    ax_d.legend(loc="upper left", frameon=False, fontsize=8)
    ax_d.grid(alpha=0.3)

    ax_p.set_xlabel(r"Blend weight $\sigma$")
    ax_p.set_ylabel("Farm power (GW-step)")
    ax_p.set_title("(b) Farm power vs $\\sigma$")
    ax_p.legend(loc="upper right", frameon=False, fontsize=8)
    ax_p.grid(alpha=0.3)

    out = ROOT / "latex_paper/figures/sigma_sweep.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"wrote {out} and .png")


if __name__ == "__main__":
    main()
