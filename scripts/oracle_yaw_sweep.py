"""Oracle brute-force sweep over yaw vectors:
report total power AND per-turbine DEL for every yaw config in a discrete grid,
using Teodor's DLC12 surrogate (single-turbine local but captures both
power = wtur_W and DEL = wrot_Bl1Rad0FlpMnt).

Caveat: surrogate is per-turbine. To get the wake-coupling effect, must query
PyWake to get the post-yaw inflow (saws_*) at each turbine. Then surrogate
predicts power and DEL given that inflow. This is the "wake-aware oracle"
from the idea-critic discussion, used here for OFFLINE analysis (not deployment
controller).

Output: JSON with all yaw vectors + (power_per_turb, DEL_per_turb), and a
plot showing the Pareto frontier of (total_power, max_DEL).

Usage:
  pixi run python scripts/oracle_yaw_sweep.py \\
      --layout multi_modal --wd 268 --ws 9 --ti 0.07 \\
      --n-yaw-levels 9 --yaw-max 30 \\
      --out-json results/oracle_yaw_sweep_mm.json
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name] = m
    return m

ts = _load("helpers.teodor_surrogate", ROOT / "helpers/teodor_surrogate.py")
layouts_mod = _load("helpers.layouts", ROOT / "helpers/layouts.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--layout", default="multi_modal")
    p.add_argument("--wd", type=float, default=268.0)
    p.add_argument("--ws", type=float, default=9.0)
    p.add_argument("--ti", type=float, default=0.07)
    p.add_argument("--n-yaw-levels", type=int, default=7)
    p.add_argument("--yaw-max", type=float, default=30.0)
    p.add_argument("--out-json", default="results/oracle_yaw_sweep.json")
    p.add_argument("--out-fig",  default="results/oracle_yaw_sweep.png")
    args = p.parse_args()

    sensor_c = "wrot_Bl1Rad0FlpMnt"
    sensor_r = "wtur_W"
    surr = ts.TeodorDLC12Surrogate.from_bundle(
        args.bundle, outputs=[sensor_c, sensor_r])
    surr.eval()

    # Layout
    from py_wake.examples.data.iea37 import IEA37_WindTurbines
    turbine = IEA37_WindTurbines()
    x_arr, y_arr = layouts_mod.get_layout_positions(args.layout, turbine)
    n_turb = len(x_arr)
    rd = float(turbine.diameter())
    print(f"layout={args.layout} n_turb={n_turb} D={rd:.0f}m")

    # Build PyWake site for wake-aware inflow at each yaw config
    from py_wake.site import UniformSite
    from py_wake import NOJ
    site = UniformSite(p_wd=[1.0], ti=args.ti, ws=[args.ws])
    wfm = NOJ(site, turbine)

    yaw_levels = np.linspace(-args.yaw_max, args.yaw_max, args.n_yaw_levels)
    yaw_grid = np.array(list(itertools.product(yaw_levels, repeat=n_turb)),
                          dtype=np.float32)
    print(f"sweeping K={len(yaw_grid)} yaw configurations")

    results = []
    for i, yaws in enumerate(yaw_grid):
        # PyWake sim with this yaw config -> get rotor-effective WS per turbine
        try:
            sim = wfm(x=list(map(float, x_arr)),
                      y=list(map(float, y_arr)),
                      wd=[args.wd], ws=[args.ws],
                      yaw=yaws.reshape(n_turb, 1, 1))
            # WS_eff per turbine
            ws_eff = np.asarray(sim.WS_eff.values).flatten()  # (n_turb,)
            ti_eff = np.asarray(sim.TI_eff.values).flatten() if hasattr(sim, 'TI_eff') else np.full(n_turb, args.ti)
        except Exception as e:
            if i == 0: print(f"PyWake err: {e}")
            ws_eff = np.full(n_turb, args.ws)
            ti_eff = np.full(n_turb, args.ti)

        # Build surrogate inputs (per turbine):
        # sectors set to ws_eff (we don't have true 4-sector; broadcast)
        saws = np.tile(ws_eff[:, None], (1, 4))
        sati = np.tile(ti_eff[:, None], (1, 4))
        pset = np.full(n_turb, 0.93)
        x_in = np.concatenate(
            [saws, sati, pset[:, None], yaws[:, None].astype(np.float32)],
            axis=1).astype(np.float32)
        with torch.no_grad():
            p_per = surr.predict_one(sensor_r,
                                       torch.from_numpy(x_in)).numpy().flatten()
            d_per = surr.predict_one(sensor_c,
                                       torch.from_numpy(x_in)).numpy().flatten()

        results.append({
            "yaw_deg": yaws.tolist(),
            "ws_eff": ws_eff.tolist(),
            "power_per_turb": p_per.tolist(),
            "DEL_per_turb": d_per.tolist(),
            "P_total": float(p_per.sum()),
            "DEL_max": float(d_per.max()),
            "DEL_total": float(d_per.sum()),
        })

    # Identify zero-yaw baseline
    zero_idx = int(np.argmin(np.abs(yaw_grid).sum(axis=1)))
    zero = results[zero_idx]
    P0 = zero["P_total"]; D0 = zero["DEL_max"]; Dt0 = zero["DEL_total"]
    print(f"\nzero-yaw baseline: P_total={P0:.0f} kW, DEL_max={D0:.0f}, "
          f"DEL_total={Dt0:.0f}")

    # Find best by metrics
    best_P = max(results, key=lambda r: r["P_total"])
    best_DELmax = min(results, key=lambda r: r["DEL_max"])
    best_DELtot = min(results, key=lambda r: r["DEL_total"])

    print(f"\n=== Headroom vs zero yaw ===")
    print(f"best P_total: yaw={best_P['yaw_deg']}  P={best_P['P_total']:.0f} "
          f"({100*(best_P['P_total']/P0-1):+.2f}%)  DEL_max={best_P['DEL_max']:.0f}")
    print(f"best DEL_max (lowest worst-turbine DEL): yaw={best_DELmax['yaw_deg']}  "
          f"DEL_max={best_DELmax['DEL_max']:.0f} ({100*(best_DELmax['DEL_max']/D0-1):+.2f}%)  "
          f"P={best_DELmax['P_total']:.0f}")
    print(f"best DEL_total (lowest sum DEL): yaw={best_DELtot['yaw_deg']}  "
          f"DEL_total={best_DELtot['DEL_total']:.0f} ({100*(best_DELtot['DEL_total']/Dt0-1):+.2f}%)  "
          f"P={best_DELtot['P_total']:.0f}")

    # Save JSON
    out = {
        "layout": args.layout, "wd": args.wd, "ws": args.ws, "ti": args.ti,
        "n_yaw_levels": args.n_yaw_levels, "yaw_max": args.yaw_max,
        "n_turb": n_turb,
        "zero_baseline": zero,
        "best_P": best_P, "best_DELmax": best_DELmax,
        "best_DELtot": best_DELtot,
        "all_results": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out_json}")

    # Pareto plot: total power vs DEL_max
    Ps = np.array([r["P_total"] for r in results])
    Dmax = np.array([r["DEL_max"] for r in results])
    Dtot = np.array([r["DEL_total"] for r in results])

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    axs[0].scatter(Dmax, Ps, s=12, alpha=0.5, color="#888")
    axs[0].scatter([D0], [P0], color="black", s=80, marker="*",
                    label="zero-yaw", zorder=5)
    axs[0].scatter([best_P["DEL_max"]], [best_P["P_total"]],
                    color="green", s=80, marker="o", label="best P_total", zorder=5)
    axs[0].scatter([best_DELmax["DEL_max"]], [best_DELmax["P_total"]],
                    color="red", s=80, marker="o", label="best DEL_max", zorder=5)
    axs[0].set_xlabel("DEL_max [kNm] (worst-turbine 10-min DEL)")
    axs[0].set_ylabel("P_total [kW] (sum across turbines)")
    axs[0].set_title(f"{args.layout} oracle sweep: P vs DEL_max\n"
                      f"({len(results)} yaw configs at wd={args.wd}, ws={args.ws}, TI={args.ti})")
    axs[0].legend(); axs[0].grid(alpha=0.25)

    axs[1].scatter(Dtot, Ps, s=12, alpha=0.5, color="#888")
    axs[1].scatter([Dt0], [P0], color="black", s=80, marker="*", label="zero-yaw", zorder=5)
    axs[1].scatter([best_DELtot["DEL_total"]], [best_DELtot["P_total"]],
                    color="red", s=80, marker="o", label="best DEL_total", zorder=5)
    axs[1].set_xlabel("DEL_total [kNm] (sum across turbines)")
    axs[1].set_ylabel("P_total [kW]")
    axs[1].set_title("P vs DEL_total")
    axs[1].legend(); axs[1].grid(alpha=0.25)

    Path(args.out_fig).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out_fig, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out_fig}")


if __name__ == "__main__":
    main()
