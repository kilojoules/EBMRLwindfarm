#!/usr/bin/env python3
"""
Plot wind farm yaw trajectories under different budget strategies.

Reads per-step trajectory data (yaw angles, lambda, power) from
windfarm_trajectories.json and produces time series figures showing
how the agent's yaw decisions change across budget strategies.

Usage:
    python scripts/plot_windfarm_trajectories.py \
        --input results/windfarm_trajectories.json \
        --output latex_paper/figures/fig_windfarm_timeseries.pdf
"""

import argparse
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_trajectories(path):
    with open(path) as f:
        return json.load(f)


def plot_yaw_timeseries(trajs, output_path):
    """
    Main figure: yaw angle time series for T0 (upstream turbine) under
    different budget strategies. Shows how the agent allocates negative
    yaw across the episode.
    """
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True,
                             gridspec_kw={"height_ratios": [1, 1, 0.6, 0.6]})

    configs = [
        ("unconstrained", "Unconstrained", "#999999", 1.0, "-"),
        ("constant_eta0", "Constant ($\\eta$=0, B=15)", "#d6604d", 0.8, "-"),
        ("ac_eta2", "AC ($\\eta$=2, B=15)", "#2171b5", 1.0, "-"),
        ("ac_eta5", "AC ($\\eta$=5, B=15)", "#042451", 0.8, "--"),
    ]

    # Panel 1: T0 yaw angle time series
    ax = axes[0]
    for key, label, color, alpha, ls in configs:
        if key not in trajs:
            continue
        steps = trajs[key]
        t = [s["t"] for s in steps]
        yaw_t0 = [s.get("yaw_T0", 0) for s in steps]
        ax.plot(t, yaw_t0, color=color, alpha=alpha, linewidth=0.8,
                linestyle=ls, label=label)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylabel("T0 Yaw Angle (deg)")
    ax.set_title("Wind Farm: Yaw Steering Under Different Budget Strategies", fontsize=11)
    ax.legend(fontsize=7, ncol=2, loc="lower left")
    ax.grid(True, alpha=0.1)

    # Panel 2: T1 yaw angle
    ax = axes[1]
    for key, label, color, alpha, ls in configs:
        if key not in trajs:
            continue
        steps = trajs[key]
        t = [s["t"] for s in steps]
        yaw_t1 = [s.get("yaw_T1", 0) for s in steps]
        ax.plot(t, yaw_t1, color=color, alpha=alpha, linewidth=0.8, linestyle=ls)

    ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
    ax.set_ylabel("T1 Yaw Angle (deg)")
    ax.grid(True, alpha=0.1)

    # Panel 3: Lambda evolution
    ax = axes[2]
    for key, label, color, alpha, ls in configs:
        if key not in trajs or key == "unconstrained":
            continue
        steps = trajs[key]
        t = [s["t"] for s in steps]
        lam = [s.get("lambda", 1.0) for s in steps]
        lam_clipped = np.clip(lam, 1e-2, 1e4)
        ax.semilogy(t, lam_clipped, color=color, alpha=alpha,
                     linewidth=1.0, linestyle=ls, label=label)

    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylabel("$\\lambda(t)$")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.1)

    # Panel 4: Cumulative negative yaw (T0)
    ax = axes[3]
    for key, label, color, alpha, ls in configs:
        if key not in trajs:
            continue
        steps = trajs[key]
        t = [s["t"] for s in steps]
        yaw_t0 = [s.get("yaw_T0", 0) for s in steps]
        cum_neg = np.cumsum([1 if y < 0 else 0 for y in yaw_t0])
        ax.plot(t, cum_neg, color=color, alpha=alpha, linewidth=1.5,
                linestyle=ls, label=label)

    ax.axhline(15, color="red", linestyle=":", alpha=0.5, label="Budget (15)")
    ax.set_ylabel("Cumulative\nNeg Yaw (T0)")
    ax.set_xlabel("Timestep")
    ax.legend(fontsize=7, ncol=3)
    ax.grid(True, alpha=0.1)

    for ax in axes:
        ax.set_xlim(0, max(len(trajs.get("unconstrained", [])), 200))

    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight")
    print(f"Saved {output_path}")
    plt.close()


def plot_yaw_heatmap(trajs, output_path):
    """
    Heatmap: yaw angles for all turbines over time, comparing unconstrained
    vs AC-constrained. Shows the spatial-temporal pattern.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))

    for idx, (key, title) in enumerate([
        ("unconstrained", "Unconstrained"),
        ("ac_eta2", "AC ($\\eta$=2, B=15)"),
        ("ac_eta5", "AC ($\\eta$=5, B=15)"),
    ]):
        ax = axes[idx]
        if key not in trajs:
            ax.set_title(f"{title} (no data)")
            continue

        steps = trajs[key]
        n_turb = sum(1 for k in steps[0] if k.startswith("yaw_T"))
        T = len(steps)

        yaw_matrix = np.zeros((n_turb, T))
        for t, s in enumerate(steps):
            for ti in range(n_turb):
                yaw_matrix[ti, t] = s.get(f"yaw_T{ti}", 0)

        im = ax.imshow(yaw_matrix, aspect="auto", cmap="RdBu_r",
                        vmin=-30, vmax=30, interpolation="nearest")
        ax.set_xlabel("Timestep")
        ax.set_ylabel("Turbine")
        ax.set_yticks(range(n_turb))
        ax.set_yticklabels([f"T{i}" for i in range(n_turb)])
        ax.set_title(title, fontsize=10)

    cbar = fig.colorbar(im, ax=axes, orientation="vertical",
                         fraction=0.02, pad=0.02)
    cbar.set_label("Yaw angle (deg)")

    plt.tight_layout()
    output_heatmap = output_path.replace(".pdf", "_heatmap.pdf")
    fig.savefig(output_heatmap, bbox_inches="tight")
    print(f"Saved {output_heatmap}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/windfarm_trajectories.json")
    parser.add_argument("--output", default="latex_paper/figures/fig_windfarm_timeseries.pdf")
    cli = parser.parse_args()

    trajs = load_trajectories(cli.input)
    print(f"Loaded trajectories: {list(trajs.keys())}")
    for key, steps in trajs.items():
        print(f"  {key}: {len(steps)} steps")

    plot_yaw_timeseries(trajs, cli.output)
    plot_yaw_heatmap(trajs, cli.output)


if __name__ == "__main__":
    main()
