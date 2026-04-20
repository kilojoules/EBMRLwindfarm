"""Compare frozen vs precomputed DEL farm_max sweeps.

Reads the two `summary.csv` files produced by `evaluate_constraints.py` and:
  - prints a side-by-side table of steady-state yaws and power at each λ;
  - overlays per-turbine yaw-vs-λ (frozen solid, precomputed dashed);
  - plots power-vs-λ for both modes;
  - computes L2 distance to the PyWake constrained optimum [-7.1, +24.3, +3.2].

Usage:
    python scripts/compare_frozen_vs_precomputed.py \
        --frozen-dir  results/eval_3turb_simple/del_farm_max_frozen \
        --precomputed-dir results/eval_3turb_simple/del_farm_max_precomputed \
        --output-dir results/eval_3turb_simple/comparison
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# PyWake ground truth (3turb, ws=10, wd=270, TI=0.07)
UNCONSTRAINED_TARGET = np.array([+21.5, -21.8, 0.0])          # mirror: [-21.5, +21.8, 0]
CONSTRAINED_TARGET = np.array([-7.1, +24.3, +3.2])            # farm_max ≤ 10%


def load_sweep(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    return df[df["constraint_type"] == "del_farm_max"].reset_index(drop=True)


def l2_to(row: pd.Series, target: np.ndarray) -> float:
    yaw = np.array([row["T0_yaw"], row["T1_yaw"], row["T2_yaw"]])
    return float(np.linalg.norm(yaw - target))


def l2_to_closer_basin(row: pd.Series, target: np.ndarray) -> float:
    """The constrained target is bimodal around T2≈0 — use the closer mirror."""
    yaw = np.array([row["T0_yaw"], row["T1_yaw"], row["T2_yaw"]])
    mirror = np.array([-target[0], -target[1], target[2]])
    return float(min(np.linalg.norm(yaw - target), np.linalg.norm(yaw - mirror)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frozen-dir", type=Path, required=True)
    p.add_argument("--precomputed-dir", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frozen = load_sweep(args.frozen_dir / "summary.csv")
    precomp = load_sweep(args.precomputed_dir / "summary.csv")

    merged = frozen.merge(
        precomp, on=["lambda", "steepness"], suffixes=("_frozen", "_precomp")
    ).sort_values("lambda").reset_index(drop=True)

    # --- L2 distances to the constrained target (closer of the two mirrors)
    merged["dist_frozen"] = [
        l2_to_closer_basin(r.rename(lambda s: s.removesuffix("_frozen")), CONSTRAINED_TARGET)
        for _, r in merged.iterrows()
    ]
    merged["dist_precomp"] = [
        l2_to_closer_basin(r.rename(lambda s: s.removesuffix("_precomp")), CONSTRAINED_TARGET)
        for _, r in merged.iterrows()
    ]

    # --- Table
    cols_to_print = [
        "lambda",
        "T0_yaw_frozen", "T1_yaw_frozen", "T2_yaw_frozen", "power_ratio_frozen",
        "T0_yaw_precomp", "T1_yaw_precomp", "T2_yaw_precomp", "power_ratio_precomp",
        "dist_frozen", "dist_precomp",
    ]
    print("=" * 128)
    print(f"Constrained PyWake target: [{CONSTRAINED_TARGET[0]:+.1f}, "
          f"{CONSTRAINED_TARGET[1]:+.1f}, {CONSTRAINED_TARGET[2]:+.1f}]  "
          f"(and its mirror)")
    print("=" * 128)
    with pd.option_context("display.float_format", "{:+.2f}".format,
                            "display.max_columns", None,
                            "display.width", 160):
        print(merged[cols_to_print].to_string(index=False))
    print("=" * 128)

    # Save table to CSV
    (args.output_dir / "comparison.csv").write_text(merged.to_csv(index=False))

    # --- Yaw-vs-λ overlay
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True, constrained_layout=True)
    lam = merged["lambda"].values
    turbines = ["T0", "T1", "T2"]
    for i, (ax, t) in enumerate(zip(axes, turbines)):
        ax.plot(lam, merged[f"{t}_yaw_frozen"], "o-", color="C0",
                label=f"frozen", linewidth=2, markersize=7)
        ax.plot(lam, merged[f"{t}_yaw_precomp"], "s--", color="C1",
                label=f"precomputed", linewidth=2, markersize=7)
        ax.axhline(CONSTRAINED_TARGET[i], color="red", linestyle=":", linewidth=1.2,
                   label=f"PyWake target ({CONSTRAINED_TARGET[i]:+.1f}°)")
        ax.axhline(-CONSTRAINED_TARGET[i] if i < 2 else CONSTRAINED_TARGET[i],
                   color="red", linestyle=":", linewidth=1.2, alpha=0.5,
                   label=f"mirror ({-CONSTRAINED_TARGET[i] if i<2 else CONSTRAINED_TARGET[i]:+.1f}°)"
                   if i < 2 else None)
        ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlabel("λ (log scale, linear below 0.1)")
        ax.set_title(t)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="best")
    axes[0].set_ylabel("steady-state yaw (°)")
    fig.suptitle("DEL farm_max (10% threshold) — frozen vs precomputed context",
                 y=1.04)
    fig.savefig(args.output_dir / "yaw_vs_lambda.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- Power-vs-λ overlay
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(lam, 100 * (merged["power_ratio_frozen"] - 1), "o-", color="C0",
            label="frozen", linewidth=2, markersize=8)
    ax.plot(lam, 100 * (merged["power_ratio_precomp"] - 1), "s--", color="C1",
            label="precomputed", linewidth=2, markersize=8)
    ax.axhline(0, color="k", linewidth=0.5, alpha=0.4)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("λ (log scale, linear below 0.1)")
    ax.set_ylabel("Δ power vs unconstrained (%)")
    ax.set_title("Power response under DEL farm_max (10%) — frozen vs precomputed")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(args.output_dir / "power_vs_lambda.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    # --- Distance-to-target vs λ
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.plot(lam, merged["dist_frozen"], "o-", color="C0", label="frozen",
            linewidth=2, markersize=8)
    ax.plot(lam, merged["dist_precomp"], "s--", color="C1", label="precomputed",
            linewidth=2, markersize=8)
    ax.set_xscale("symlog", linthresh=0.1)
    ax.set_xlabel("λ (log scale, linear below 0.1)")
    ax.set_ylabel("L2 distance to nearest PyWake target (°)")
    ax.set_title("Distance to constrained PyWake optimum [-7.1, +24.3, +3.2] (or mirror)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.savefig(args.output_dir / "distance_vs_lambda.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    print(f"\nWrote CSV + 3 PNGs to {args.output_dir}/")


if __name__ == "__main__":
    main()
