"""Time the DEL-surface precompute as a function of farm size.

Builds a simple linear farm layout (row of N turbines at 5D spacing, wind aligned
along the row) and times how long ``build_del_grid`` takes at a fixed grid size
as ``N`` varies. Writes a CSV + log-log plot.

The cost scales as ``G**N`` PyWake sims. With the default ``G=7`` and ``N=7``
this is ~823k sims ≈ 96 min on one core, so be ready to interrupt at high N or
reduce ``--n-list`` accordingly.

Usage:
    python scripts/time_del_precompute.py
    python scripts/time_del_precompute.py --grid-size 5 --n-list 2,3,4,5,6,7
    python scripts/time_del_precompute.py --grid-size 11 --n-list 2,3,4
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

# Silence TF noise and force CPU before any TF imports
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # noqa: E402

from load_surrogates import build_del_grid  # noqa: E402
from helpers.surrogate_loads import (  # noqa: E402
    TorchDELSurrogate,
    make_rotor_template,
    sector_averages_reordered,
)


REPO = Path(__file__).resolve().parent.parent
MODEL_PATH = REPO / "surrogate/models/ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras"
SCALER_IN_PATH = REPO / "surrogate/scalers/scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl"
SCALER_OUT_PATH = REPO / "surrogate/scalers/scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl"


def build_linear_farm(n_turbines: int, D: float = 178.3, spacing_D: float = 5.0):
    """Return a PyWake WindFarmModel + positions for a single-row farm.

    Turbines are spaced ``spacing_D * D`` along x. Wind at 270° is aligned
    with the row, giving fully-waked downstream turbines — the hardest case
    for the DEL surrogate.
    """
    from py_wake.deflection_models.jimenez import JimenezWakeDeflection
    from py_wake.examples.data.dtu10mw import DTU10MW
    from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
    from py_wake.site import UniformSite
    from py_wake.turbulence_models import STF2017TurbulenceModel

    wt = DTU10MW()
    site = UniformSite()
    wfm = Blondel_Cathelain_2020(
        site, windTurbines=wt,
        turbulenceModel=STF2017TurbulenceModel(),
        deflectionModel=JimenezWakeDeflection(),
    )
    positions = np.array([[i * spacing_D * D, 0.0] for i in range(n_turbines)])
    return wfm, wt, positions


def build_wfm_fn(wfm, wt, positions, ws=9.0, wd=270.0, ti=0.07):
    """Return a ``wfm_fn(yaw_deg) -> del`` closure.

    Internally runs PyWake + rotor-sector averaging + the TorchDELSurrogate.
    """
    import torch

    torch_del = TorchDELSurrogate.from_keras(MODEL_PATH, SCALER_IN_PATH, SCALER_OUT_PATH)
    rotor_template = make_rotor_template(wt.diameter() / 2)
    hub_h = np.full(len(positions), float(wt.hub_height()))
    x_wt = positions[:, 0]
    y_wt = positions[:, 1]

    def wfm_fn(yaw_deg: np.ndarray) -> np.ndarray:
        WS_in, TI_in = sector_averages_reordered(
            wfm, x_wt, y_wt, hub_h,
            wd=wd, ws=ws, ti=ti,
            yaw=yaw_deg, template=rotor_template,
        )
        n_turb = WS_in.shape[0]
        pset = np.full((n_turb, 1), 1.0)
        yaw_col = yaw_deg.reshape(n_turb, 1)
        x = np.hstack([WS_in, TI_in, pset, yaw_col]).astype(np.float32)
        with torch.no_grad():
            return torch_del(torch.as_tensor(x)).cpu().numpy()

    return wfm_fn


def time_single_call(wfm_fn, n_turbines: int, n_samples: int = 5) -> float:
    """Warm up and return median per-call wall time in seconds."""
    # Warmup (JIT / cache)
    for _ in range(2):
        wfm_fn(np.zeros(n_turbines, dtype=float))
    samples = []
    for _ in range(n_samples):
        yaw = np.random.uniform(-30, 30, size=n_turbines)
        t0 = time.perf_counter()
        wfm_fn(yaw)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-list", type=str, default="2,3,4,5,6,7",
                   help="Comma-separated farm sizes to sweep (default: 2,3,4,5,6,7).")
    p.add_argument("--grid-size", type=int, default=7,
                   help="Grid points per turbine dimension (default: 7).")
    p.add_argument("--max-sims", type=int, default=2_000_000,
                   help="Skip any N where G**N exceeds this limit (default: 2M).")
    p.add_argument("--output", type=str, default="results/del_precompute_timing.csv",
                   help="Where to write the CSV.")
    p.add_argument("--plot", type=str, default="results/del_precompute_timing.png",
                   help="Where to write the log-log scaling plot.")
    p.add_argument("--no-plot", action="store_true",
                   help="Skip the plot (CSV only).")
    return p.parse_args()


def main():
    cli = parse_args()
    n_list = [int(x) for x in cli.n_list.split(",")]
    G = cli.grid_size

    output = Path(cli.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    print(f"DEL precompute timing: G={G}, N in {n_list}, max_sims={cli.max_sims:,}")
    print(f"{'N':>3} {'sims':>10} {'per-sim ms':>12} {'est min':>10} {'actual s':>10} {'actual min':>10}")
    print("-" * 64)

    for N in n_list:
        total_sims = G ** N
        wfm, wt, positions = build_linear_farm(N)
        wfm_fn = build_wfm_fn(wfm, wt, positions)
        per_sim = time_single_call(wfm_fn, N)
        est_min = total_sims * per_sim / 60.0

        if total_sims > cli.max_sims:
            print(f"{N:>3d} {total_sims:>10,d} {per_sim*1000:>11.2f}  "
                  f"{est_min:>9.2f}  (skipped — exceeds max_sims)")
            rows.append({
                "n_turbines": N,
                "grid_size": G,
                "total_sims": total_sims,
                "per_sim_ms": per_sim * 1000,
                "estimated_total_min": est_min,
                "actual_total_s": np.nan,
                "actual_total_min": np.nan,
            })
            continue

        t0 = time.perf_counter()
        grid = build_del_grid(
            wfm_fn, n_turbines=N, grid_size=G,
            yaw_range_deg=(-30.0, 30.0),
            verbose=False,
        )
        actual = time.perf_counter() - t0
        assert grid.shape == (N,) + (G,) * N, f"unexpected grid shape {grid.shape}"

        print(f"{N:>3d} {total_sims:>10,d} {per_sim*1000:>11.2f}  "
              f"{est_min:>9.2f}  {actual:>9.1f}  {actual/60:>9.2f}")
        rows.append({
            "n_turbines": N,
            "grid_size": G,
            "total_sims": total_sims,
            "per_sim_ms": per_sim * 1000,
            "estimated_total_min": est_min,
            "actual_total_s": actual,
            "actual_total_min": actual / 60,
        })

    with output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nCSV written: {output}")

    if not cli.no_plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        Ns = [r["n_turbines"] for r in rows]
        actual_s = [r["actual_total_s"] for r in rows]
        est_s = [r["estimated_total_min"] * 60 for r in rows]

        fig, ax = plt.subplots(figsize=(8, 5))
        valid = [(n, a) for n, a in zip(Ns, actual_s) if np.isfinite(a)]
        if valid:
            nv, av = zip(*valid)
            ax.plot(nv, av, "o-", label=f"actual (G={G})", color="C0")
        ax.plot(Ns, est_s, "x--", label=f"estimated (per-sim × G^N, G={G})", color="C1")
        ax.set_yscale("log")
        ax.set_xlabel("N (turbines)")
        ax.set_ylabel("Total precompute time (s, log scale)")
        ax.set_title(f"DEL precompute scaling: total sims = {G}^N")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()
        plot_path = Path(cli.plot)
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        print(f"Plot written: {plot_path}")


if __name__ == "__main__":
    main()
