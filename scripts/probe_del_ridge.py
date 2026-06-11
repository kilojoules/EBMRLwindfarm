"""Where is the DEL ridge, actually? (Referee M2 / Q2)

Probe the DLC12 surrogate DEL-vs-yaw curve directly at fixed inflow.
Three inflow conditions:
  - freestream (saws=9.0 uniform, sati=0.07)
  - waked (lower ws, higher ti, asymmetric sectors typical of multi_modal T1/T2)
  - strongly waked

If the ridge peaks at yaw != 0 for waked inflow, that explains the
sigma-sweep peak at sigma=0.4 (mean yaw -8 deg) instead of the zero
crossing (sigma=0.5).

Local run; bundle in checkpoints/.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name] = m
    return m
ts = _load("helpers.teodor_surrogate", ROOT / "helpers/teodor_surrogate.py")

SENSOR = "wrot_Bl1Rad0FlpMnt"


def main():
    surr = ts.TeodorDLC12Surrogate.from_bundle(
        ROOT / "checkpoints/teodor_dlc12_torch.pt", outputs=[SENSOR])
    surr.eval()

    yaws = np.arange(-30, 30.5, 1.0)
    conditions = {
        # (saws L,R,U,D), (sati L,R,U,D)
        "freestream ws9 ti0.07": ([9.0, 9.0, 9.0, 9.0], [0.07, 0.07, 0.07, 0.07]),
        "waked ws7.5 ti0.12":    ([7.2, 7.8, 7.6, 7.4], [0.12, 0.11, 0.12, 0.13]),
        "strongly waked ws6 ti0.16": ([5.6, 6.4, 6.1, 5.9], [0.17, 0.15, 0.16, 0.16]),
        "asymmetric partial wake": ([6.0, 9.0, 7.5, 7.5], [0.16, 0.08, 0.12, 0.12]),
    }

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                          "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(7, 4.2), constrained_layout=True)

    print(f"{'condition':<30} {'argmax yaw':>10} {'DEL@peak':>9} {'DEL@0':>8} {'DEL@-15':>8} {'DEL@+7.5':>9}")
    for label, (saws, sati) in conditions.items():
        x = np.zeros((len(yaws), 10), dtype=np.float32)
        x[:, 0:4] = saws
        x[:, 4:8] = sati
        x[:, 8] = 0.93
        x[:, 9] = yaws
        with torch.no_grad():
            d = surr.predict_one(SENSOR, torch.from_numpy(x)).flatten().numpy()
        ax.plot(yaws, d, label=label, lw=1.8)
        i_pk = int(np.argmax(d))
        i0 = int(np.argmin(np.abs(yaws)))
        i_m15 = int(np.argmin(np.abs(yaws + 15)))
        i_p75 = int(np.argmin(np.abs(yaws - 7.5)))
        print(f"{label:<30} {yaws[i_pk]:>10.1f} {d[i_pk]:>9.1f} {d[i0]:>8.1f} "
              f"{d[i_m15]:>8.1f} {d[i_p75]:>9.1f}")

    ax.axvline(0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Yaw misalignment (deg)")
    ax.set_ylabel(f"Surrogate DEL ({SENSOR}) [kNm]")
    ax.set_title("DEL vs yaw at fixed inflow (DLC12 surrogate)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    out = ROOT / "latex_paper/figures/del_ridge_probe.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=180, bbox_inches="tight")
    print(f"\nwrote {out} and .png")


if __name__ == "__main__":
    main()
