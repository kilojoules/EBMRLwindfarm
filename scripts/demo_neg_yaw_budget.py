#!/usr/bin/env python3
"""
Almgren-Chriss budget allocation: baselines, multi-year analysis, and second domain.

Uses 10 years of daily wind from Energy Island to demonstrate that an
Almgren-Chriss-inspired time-varying penalty captures more value from a
fixed negative-yaw budget than oracle-free alternatives.

Analyses (selected via --analysis):
  single_year   One-year deep dive with all baselines (default)
  multi_year    Consistency of AC advantage across all 10 years
  pareto        Risk-aversion Pareto frontier (value vs. utilization)
  heterogeneous Per-turbine budgets (e.g., worn bearing on turbine 3)
  battery       Second domain: battery cycle-life budget (generality demo)
  all           Run everything and save a multi-page PDF

Usage:
    python scripts/demo_neg_yaw_budget.py
    python scripts/demo_neg_yaw_budget.py --analysis all --output figures/ac_budget.pdf
    python scripts/demo_neg_yaw_budget.py --analysis pareto --budget-days 10 20 30
    python scripts/demo_neg_yaw_budget.py --analysis battery
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from load_surrogates import NegativeYawBudgetSurrogate


# =============================================================================
# DATA LOADING
# =============================================================================

def load_wind_data(csv_path: str):
    """Load Energy Island wind CSV (semicolon-delimited)."""
    dates, ws, wd = [], [], []
    with open(csv_path) as f:
        f.readline()  # skip header
        for line in f:
            parts = line.strip().split(";")
            if len(parts) != 3:
                continue
            dates.append(parts[0])
            ws.append(float(parts[1]))
            wd.append(float(parts[2]))
    return np.array(dates), np.array(ws), np.array(wd)


def filter_year(dates, ws, wd, year: int):
    """Extract one year of data."""
    mask = np.array([d.startswith(str(year)) for d in dates])
    return dates[mask], ws[mask], wd[mask]


# =============================================================================
# BENEFIT MODEL
# =============================================================================

def neg_yaw_benefit(ws: np.ndarray, wd: np.ndarray,
                    global_max: float = None) -> np.ndarray:
    """
    Estimate benefit of negative yaw from wind conditions.

    Negative yaw is most valuable when:
      - Wind speed is moderate (8-16 m/s): enough energy, not too much load
      - Wind direction is wake-heavy (270 deg or 90 deg for row-aligned farms)

    Args:
        ws: wind speeds (m/s)
        wd: wind directions (degrees)
        global_max: if provided, normalize by this value instead of the
            local max. This ensures threshold=0.3 means the same thing
            across different years/datasets.

    Returns [0, 1] benefit score per timestep.
    """
    ws_score = np.exp(-0.5 * ((ws - 12.0) / 4.0) ** 2)
    wd_rad = np.deg2rad(wd)
    wake_score = np.maximum(
        np.exp(-0.5 * ((np.cos(wd_rad) - np.cos(np.deg2rad(270))) / 0.3) ** 2),
        np.exp(-0.5 * ((np.cos(wd_rad) - np.cos(np.deg2rad(90))) / 0.3) ** 2),
    )
    benefit = ws_score * wake_score
    norm = global_max if global_max is not None else benefit.max()
    return benefit / (norm + 1e-8)


# =============================================================================
# BASELINES
# =============================================================================

def oracle_allocation(benefit: np.ndarray, budget: int) -> dict:
    """
    Hindsight-optimal: spend budget on the top-k highest-benefit days.
    This is the unreachable upper bound (requires perfect foresight).
    """
    horizon = len(benefit)
    top_k = np.argsort(benefit)[-budget:]
    spent = np.zeros(horizon)
    spent[top_k] = 1.0
    cumulative = np.cumsum(spent)
    return {
        "spent": spent,
        "cumulative_spent": cumulative,
        "total_value": float(np.sum(benefit[top_k])),
        "lambdas": np.ones(horizon),  # N/A
        "name": "Oracle (hindsight)",
    }


def greedy_allocation(benefit: np.ndarray, budget: int, threshold: float = 0.3) -> dict:
    """
    Myopic greedy: spend whenever benefit > threshold, first-come-first-served.
    No lookahead, no budget awareness. Exhausts budget on the first good days.
    """
    horizon = len(benefit)
    spent = np.zeros(horizon)
    total = 0
    for t in range(horizon):
        if benefit[t] > threshold and total < budget:
            spent[t] = 1.0
            total += 1
    return {
        "spent": spent,
        "cumulative_spent": np.cumsum(spent),
        "total_value": float(np.sum(benefit * spent)),
        "lambdas": np.ones(horizon),
        "name": "Greedy (myopic)",
    }


def fixed_penalty_allocation(
    benefit: np.ndarray, budget: int, penalty_weight: float = 1.0,
    threshold: float = 0.3,
) -> dict:
    """
    Fixed constant penalty: spend when benefit > threshold * penalty_weight.
    No time-varying adaptation. The penalty_weight is tuned to approximately
    match the budget (but has no mechanism to adapt if it over/underspends).
    """
    horizon = len(benefit)
    spent = np.zeros(horizon)
    total = 0
    for t in range(horizon):
        if benefit[t] > threshold * penalty_weight and total < budget:
            spent[t] = 1.0
            total += 1
    return {
        "spent": spent,
        "cumulative_spent": np.cumsum(spent),
        "total_value": float(np.sum(benefit * spent)),
        "lambdas": np.full(horizon, penalty_weight),
        "name": f"Fixed penalty (w={penalty_weight:.1f})",
    }


def twap_allocation(benefit: np.ndarray, budget: int) -> dict:
    """
    TWAP (Time-Weighted Average): spend budget uniformly.
    Every (horizon/budget)-th day, spend if benefit > 0 at all.
    """
    horizon = len(benefit)
    interval = max(horizon // budget, 1)
    spent = np.zeros(horizon)
    total = 0
    for t in range(horizon):
        if t % interval == 0 and total < budget:
            spent[t] = 1.0
            total += 1
    return {
        "spent": spent,
        "cumulative_spent": np.cumsum(spent),
        "total_value": float(np.sum(benefit * spent)),
        "lambdas": np.ones(horizon),
        "name": "TWAP (uniform)",
    }


def ac_allocation(
    benefit: np.ndarray, budget: int, horizon: int,
    risk_aversion: float, steepness: float = 10.0, threshold: float = 0.3,
) -> dict:
    """
    Almgren-Chriss schedule: time-varying lambda from the surrogate.
    Spend when benefit > threshold * lambda(t).
    """
    surr = NegativeYawBudgetSurrogate(
        budget_steps=budget, horizon_steps=horizon,
        risk_aversion=risk_aversion, steepness=steepness,
    )
    surr.reset()

    lambdas, spent_arr = [], []
    total = 0

    for t in range(horizon):
        lam = surr._compute_lambda()
        lam_val = float(lam.squeeze()) if lam.dim() > 0 else float(lam)
        lambdas.append(lam_val)

        effective_threshold = threshold * max(lam_val, 1e-8)
        use = benefit[t] > effective_threshold and total < budget

        spent_arr.append(1.0 if use else 0.0)
        if use:
            total += 1

        yaw_deg = torch.tensor([-15.0]) if use else torch.tensor([15.0])
        surr.update(yaw_deg)

    spent = np.array(spent_arr)
    return {
        "spent": spent,
        "cumulative_spent": np.cumsum(spent),
        "total_value": float(np.sum(benefit * spent)),
        "lambdas": np.array(lambdas),
        "name": f"AC (RA={risk_aversion})",
    }


# =============================================================================
# BATTERY DOMAIN (SECOND APPLICATION)
# =============================================================================

def generate_battery_scenario(horizon: int = 365, seed: int = 42):
    """
    Simulate a grid battery storage scenario over one year.

    The battery earns revenue from price arbitrage (charge low, discharge high).
    Deep discharge cycles cause degradation. The "budget" is the number of
    deep-discharge cycles allowed per year (e.g., 50 out of 365 days).

    Returns:
        prices: (horizon,) daily electricity price
        deep_discharge_benefit: (horizon,) benefit of deep discharge each day
    """
    rng = np.random.RandomState(seed)

    # Base price: seasonal pattern + weekly pattern + noise
    t = np.arange(horizon)
    seasonal = 10 * np.sin(2 * np.pi * t / 365)  # summer peak
    weekly = 5 * np.sin(2 * np.pi * t / 7)        # weekday/weekend
    spikes = rng.exponential(3, horizon) * (rng.random(horizon) > 0.92)  # rare price spikes
    prices = 50 + seasonal + weekly + spikes + rng.normal(0, 3, horizon)
    prices = np.maximum(prices, 10)

    # Deep discharge benefit: proportional to daily price spread
    # (high spread = more arbitrage revenue from deep cycling)
    spread = np.abs(prices - np.median(prices))
    deep_discharge_benefit = spread / (spread.max() + 1e-8)

    return prices, deep_discharge_benefit


# =============================================================================
# ANALYSIS: SINGLE YEAR
# =============================================================================

def analysis_single_year(dates, ws, wd, year, budget, threshold, output_dir,
                         global_benefit_max=None):
    """Deep dive into one year with all baselines."""
    import matplotlib.pyplot as plt

    dates_y, ws_y, wd_y = filter_year(dates, ws, wd, year)
    horizon = len(dates_y)
    benefit = neg_yaw_benefit(ws_y, wd_y, global_max=global_benefit_max)

    # Tune fixed penalty: sweep weights, pick the one that maximizes value
    # while staying within budget. This is the BEST a fixed-weight policy can do.
    best_fixed_val, fixed_weight = -1.0, 1.0
    for w in np.linspace(0.1, 5.0, 200):
        res = fixed_penalty_allocation(benefit, budget, w, threshold)
        if res["cumulative_spent"][-1] <= budget and res["total_value"] > best_fixed_val:
            best_fixed_val = res["total_value"]
            fixed_weight = w

    # Run all methods
    methods = [
        oracle_allocation(benefit, budget),
        greedy_allocation(benefit, budget, threshold),
        twap_allocation(benefit, budget),
        fixed_penalty_allocation(benefit, budget, 1.0, threshold),  # untuned (w=1)
        fixed_penalty_allocation(benefit, budget, fixed_weight, threshold),  # hindsight-tuned
        ac_allocation(benefit, budget, horizon, 0.5, threshold=threshold),
        ac_allocation(benefit, budget, horizon, 1.0, threshold=threshold),
        ac_allocation(benefit, budget, horizon, 2.0, threshold=threshold),
        ac_allocation(benefit, budget, horizon, 5.0, threshold=threshold),
    ]
    # Clarify names
    methods[3]["name"] = "Fixed penalty (w=1, no tune)"
    methods[4]["name"] = f"Fixed penalty (w={fixed_weight:.1f}, tuned*)"

    # Print table
    oracle_val = methods[0]["total_value"]
    print(f"\n{'='*75}")
    print(f"  Single Year Analysis: Energy Island {year}")
    print(f"  Budget: {budget} days | Horizon: {horizon} days | Threshold: {threshold}")
    print(f"{'='*75}")
    print(f"  {'Method':<28s} | {'Value':>7s} | {'%Oracle':>7s} | {'Days Used':>9s}")
    print(f"  {'-'*28}-+-{'-'*7}-+-{'-'*7}-+-{'-'*9}")
    for m in methods:
        pct = 100 * m["total_value"] / oracle_val if oracle_val > 0 else 0
        used = int(m["cumulative_spent"][-1])
        print(f"  {m['name']:<28s} | {m['total_value']:7.2f} | {pct:6.1f}% | {used:4d}/{budget}")
    print()

    # Figure 1: Main comparison
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True,
                             gridspec_kw={"height_ratios": [0.8, 1.2, 1.2, 0.8]})

    days = np.arange(horizon)
    month_ticks, month_labels = _month_ticks(dates_y, year)

    # Panel 1: Wind conditions + benefit
    ax = axes[0]
    ax2 = ax.twinx()
    ax.fill_between(days, ws_y, alpha=0.2, color="steelblue")
    ax.plot(days, ws_y, color="steelblue", linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Wind Speed (m/s)", color="steelblue")
    ax2.fill_between(days, benefit, alpha=0.15, color="orange")
    ax2.plot(days, benefit, color="orange", linewidth=0.5)
    ax2.set_ylabel("Neg-Yaw Benefit", color="orange")
    ax.set_title(f"Energy Island {year} — Budget Allocation Comparison "
                 f"(budget = {budget} days)")

    # Panel 2: Spending decisions
    ax = axes[1]
    ax.fill_between(days, benefit, alpha=0.1, color="gray")
    baseline_methods = [methods[0], methods[1], methods[2], methods[3]]
    ac_methods = [m for m in methods if m["name"].startswith("AC")]
    markers = ["*", "v", "s", "D"]
    for m, mk in zip(baseline_methods, markers):
        mask = m["spent"] > 0
        if mask.any():
            ax.scatter(days[mask], benefit[mask], s=25, marker=mk,
                       alpha=0.7, label=m["name"], zorder=3)
    colors_ac = plt.cm.viridis(np.linspace(0.3, 0.9, len(ac_methods)))
    for m, c in zip(ac_methods, colors_ac):
        mask = m["spent"] > 0
        if mask.any():
            ax.scatter(days[mask], benefit[mask], s=18, color=c,
                       alpha=0.7, label=m["name"], zorder=4)
    ax.set_ylabel("Benefit at Spending Points")
    ax.legend(fontsize=7, ncol=4, loc="upper right")

    # Panel 3: Cumulative spending
    ax = axes[2]
    twap_line = np.linspace(0, budget, horizon)
    ax.plot(days, twap_line, "k--", linewidth=1, alpha=0.3, label="TWAP line")
    ax.axhline(budget, color="red", linestyle=":", alpha=0.4, label="Budget limit")
    for m, mk in zip(baseline_methods, markers):
        ax.plot(days, m["cumulative_spent"], linewidth=1.2, alpha=0.6,
                linestyle="--", label=m["name"])
    for m, c in zip(ac_methods, colors_ac):
        ax.plot(days, m["cumulative_spent"], color=c, linewidth=1.8,
                label=f"{m['name']} (val={m['total_value']:.1f})")
    ax.set_ylabel("Cumulative Days Spent")
    ax.legend(fontsize=6, ncol=4, loc="upper left")

    # Panel 4: Lambda evolution (AC methods only)
    ax = axes[3]
    for m, c in zip(ac_methods, colors_ac):
        lam_clipped = np.clip(m["lambdas"], 1e-3, 1e4)
        ax.semilogy(days, lam_clipped, color=c, linewidth=1, label=m["name"])
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3, label="baseline (λ=1)")
    ax.set_ylabel("λ (penalty weight)")
    ax.set_xlabel("Day of Year")
    ax.legend(fontsize=7, ncol=4)

    for ax in axes:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.15)

    plt.tight_layout()
    _save_or_show(fig, output_dir, "single_year.png")
    return methods


# =============================================================================
# ANALYSIS: MULTI-YEAR
# =============================================================================

def analysis_multi_year(dates, ws, wd, budget, threshold, output_dir,
                        global_benefit_max=None):
    """Run AC vs baselines across all available years."""
    import matplotlib.pyplot as plt

    years = sorted(set(int(d[:4]) for d in dates))
    print(f"\n{'='*75}")
    print(f"  Multi-Year Analysis: {years[0]}-{years[-1]} ({len(years)} years)")
    print(f"  Budget: {budget} days/year")
    print(f"{'='*75}")

    rows = []
    for yr in years:
        d, w, wdir = filter_year(dates, ws, wd, yr)
        if len(d) < 300:
            continue
        h = len(d)
        b = neg_yaw_benefit(w, wdir, global_max=global_benefit_max)

        oracle = oracle_allocation(b, budget)
        greedy = greedy_allocation(b, budget, threshold)
        twap = twap_allocation(b, budget)
        ac1 = ac_allocation(b, budget, h, 1.0, threshold=threshold)
        ac2 = ac_allocation(b, budget, h, 2.0, threshold=threshold)

        rows.append({
            "year": yr,
            "oracle": oracle["total_value"],
            "greedy": greedy["total_value"],
            "twap": twap["total_value"],
            "ac1": ac1["total_value"],
            "ac2": ac2["total_value"],
            "greedy_used": int(greedy["cumulative_spent"][-1]),
            "ac2_used": int(ac2["cumulative_spent"][-1]),
        })

    # Print table
    print(f"  {'Year':<6s} | {'Oracle':>7s} | {'Greedy':>7s} | {'TWAP':>7s} | "
          f"{'AC(1)':>7s} | {'AC(2)':>7s} | {'AC(2)/%Orc':>10s}")
    print(f"  {'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}-+-{'-'*10}")
    for r in rows:
        pct = 100 * r["ac2"] / r["oracle"] if r["oracle"] > 0 else 0
        print(f"  {r['year']:<6d} | {r['oracle']:7.2f} | {r['greedy']:7.2f} | "
              f"{r['twap']:7.2f} | {r['ac1']:7.2f} | {r['ac2']:7.2f} | {pct:9.1f}%")

    # Summary statistics
    ac2_vs_greedy = [100 * (r["ac2"] - r["greedy"]) / r["greedy"]
                     for r in rows if r["greedy"] > 0]
    ac2_pct_oracle = [100 * r["ac2"] / r["oracle"] for r in rows if r["oracle"] > 0]
    print(f"\n  AC(RA=2) vs Greedy: {np.mean(ac2_vs_greedy):+.1f}% avg "
          f"({np.min(ac2_vs_greedy):+.1f}% to {np.max(ac2_vs_greedy):+.1f}%)")
    print(f"  AC(RA=2) % of Oracle: {np.mean(ac2_pct_oracle):.1f}% avg")

    # Figure
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    yrs = [r["year"] for r in rows]
    x = np.arange(len(yrs))
    w = 0.15

    ax = axes[0]
    ax.bar(x - 2*w, [r["oracle"] for r in rows], w, label="Oracle", color="gold", alpha=0.8)
    ax.bar(x - w, [r["greedy"] for r in rows], w, label="Greedy", color="salmon", alpha=0.8)
    ax.bar(x, [r["twap"] for r in rows], w, label="TWAP", color="silver", alpha=0.8)
    ax.bar(x + w, [r["ac1"] for r in rows], w, label="AC(RA=1)", color="mediumseagreen", alpha=0.8)
    ax.bar(x + 2*w, [r["ac2"] for r in rows], w, label="AC(RA=2)", color="steelblue", alpha=0.8)
    ax.set_ylabel("Total Value Captured")
    ax.set_title(f"Multi-Year Budget Allocation (budget = {budget} days/year)")
    ax.set_xticks(x)
    ax.set_xticklabels(yrs)
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.15, axis="y")

    ax = axes[1]
    oracle_vals = np.array([r["oracle"] for r in rows])
    for label, key, color in [
        ("Greedy", "greedy", "salmon"),
        ("TWAP", "twap", "silver"),
        ("AC(RA=1)", "ac1", "mediumseagreen"),
        ("AC(RA=2)", "ac2", "steelblue"),
    ]:
        pcts = [100 * r[key] / r["oracle"] for r in rows]
        ax.plot(yrs, pcts, "o-", label=label, color=color, linewidth=1.5)
    ax.axhline(100, color="gold", linestyle="--", alpha=0.5, label="Oracle (100%)")
    ax.set_ylabel("% of Oracle Value")
    ax.set_xlabel("Year")
    ax.legend(fontsize=8, ncol=5)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    _save_or_show(fig, output_dir, "multi_year.png")


# =============================================================================
# ANALYSIS: PARETO FRONTIER
# =============================================================================

def analysis_pareto(dates, ws, wd, year, budgets, threshold, output_dir,
                    global_benefit_max=None):
    """Risk-aversion Pareto frontier for multiple budget levels."""
    import matplotlib.pyplot as plt

    dates_y, ws_y, wd_y = filter_year(dates, ws, wd, year)
    horizon = len(dates_y)
    benefit = neg_yaw_benefit(ws_y, wd_y, global_max=global_benefit_max)

    ra_values = np.concatenate([np.linspace(0, 1, 11), np.linspace(1.5, 10, 18)])

    print(f"\n{'='*75}")
    print(f"  Pareto Frontier: Energy Island {year}")
    print(f"  Budgets: {budgets} days | RA sweep: {ra_values[0]:.1f} to {ra_values[-1]:.1f}")
    print(f"{'='*75}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(budgets)))

    for bdg, color in zip(budgets, colors):
        oracle_val = oracle_allocation(benefit, bdg)["total_value"]
        greedy_val = greedy_allocation(benefit, bdg, threshold)["total_value"]

        values, utilizations, ra_labels = [], [], []
        for ra in ra_values:
            res = ac_allocation(benefit, bdg, horizon, ra, threshold=threshold)
            values.append(res["total_value"])
            utilizations.append(res["cumulative_spent"][-1] / bdg * 100)
            ra_labels.append(ra)

        # Left: Value vs RA
        ax = axes[0]
        ax.plot(ra_values, values, "o-", color=color, markersize=3,
                linewidth=1.5, label=f"B={bdg}d")
        ax.axhline(oracle_val, color=color, linestyle=":", alpha=0.3)
        ax.axhline(greedy_val, color=color, linestyle="--", alpha=0.3)

        # Right: Value vs Utilization
        ax = axes[1]
        ax.plot(utilizations, values, "o-", color=color, markersize=3,
                linewidth=1.5, label=f"B={bdg}d")
        ax.axhline(oracle_val, color=color, linestyle=":", alpha=0.3)

        # Annotate best RA
        best_idx = np.argmax(values)
        best_ra = ra_values[best_idx]
        print(f"  Budget={bdg:3d}d: best RA={best_ra:.1f}, "
              f"value={values[best_idx]:.2f} ({100*values[best_idx]/oracle_val:.1f}% of oracle), "
              f"utilization={utilizations[best_idx]:.0f}%")

    axes[0].set_xlabel("Risk Aversion (RA)")
    axes[0].set_ylabel("Total Value Captured")
    axes[0].set_title(f"Value vs. Risk Aversion ({year})")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.15)

    axes[1].set_xlabel("Budget Utilization (%)")
    axes[1].set_ylabel("Total Value Captured")
    axes[1].set_title("Pareto Frontier: Value vs. Utilization")
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.15)

    plt.tight_layout()
    _save_or_show(fig, output_dir, "pareto.png")


# =============================================================================
# ANALYSIS: HETEROGENEOUS PER-TURBINE BUDGETS
# =============================================================================

def analysis_heterogeneous(dates, ws, wd, year, total_budget, threshold, output_dir,
                           global_benefit_max=None):
    """Demo: different turbines have different budgets."""
    import matplotlib.pyplot as plt

    dates_y, ws_y, wd_y = filter_year(dates, ws, wd, year)
    horizon = len(dates_y)
    benefit = neg_yaw_benefit(ws_y, wd_y, global_max=global_benefit_max)

    # Scenario: 4-turbine farm
    #   T0: worn bearing, only 5 days budget
    #   T1: normal, 15 days
    #   T2: recently replaced, 30 days
    #   T3: normal, 15 days
    budgets = [5, 15, 30, 15]
    n_turb = len(budgets)

    print(f"\n{'='*75}")
    print(f"  Heterogeneous Budget Demo: Energy Island {year}")
    print(f"  Turbine budgets: {budgets} days")
    print(f"{'='*75}")

    surr = NegativeYawBudgetSurrogate(
        budget_steps=max(budgets),  # not used when per_turbine_budgets is set
        horizon_steps=horizon,
        risk_aversion=2.0,
        steepness=10.0,
        per_turbine_budgets=budgets,
    )
    surr.reset()

    lambdas_per_turb = [[] for _ in range(n_turb)]
    cum_spent_per_turb = [[] for _ in range(n_turb)]
    spent_per_turb = [[] for _ in range(n_turb)]
    totals = [0] * n_turb

    # Seed the surrogate with an initial update so cumulative_neg_steps is initialized
    surr.update(torch.zeros(n_turb))
    surr.current_step = 0  # reset step counter (the update was just for initialization)
    surr.cumulative_neg_steps.zero_()  # reset counts

    for t in range(horizon):
        lam = surr._compute_lambda()  # (n_turb, 1)

        for i in range(n_turb):
            lam_i = float(lam[i].squeeze())
            lambdas_per_turb[i].append(lam_i)

            eff_thresh = threshold * max(lam_i, 1e-8)
            use = benefit[t] > eff_thresh and totals[i] < budgets[i]
            spent_per_turb[i].append(1.0 if use else 0.0)
            if use:
                totals[i] += 1
            cum_spent_per_turb[i].append(totals[i])

        # Build per-turbine yaw: negative if that turbine decided to spend
        yaw_deg = torch.tensor([
            -15.0 if spent_per_turb[i][-1] > 0 else 15.0
            for i in range(n_turb)
        ])
        surr.update(yaw_deg)

    # Print results
    for i in range(n_turb):
        val = float(np.sum(np.array(spent_per_turb[i]) * benefit))
        print(f"  T{i} (budget={budgets[i]:2d}d): spent={totals[i]:2d}d, value={val:.2f}")

    # Figure
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    days = np.arange(horizon)
    month_ticks, month_labels = _month_ticks(dates_y, year)
    turb_colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6"]
    turb_labels = [f"T{i} (budget={budgets[i]}d)" for i in range(n_turb)]

    ax = axes[0]
    for i in range(n_turb):
        ax.plot(days, cum_spent_per_turb[i], color=turb_colors[i],
                linewidth=1.8, label=turb_labels[i])
        ax.axhline(budgets[i], color=turb_colors[i], linestyle=":", alpha=0.3)
    ax.set_ylabel("Cumulative Days Spent")
    ax.set_title(f"Heterogeneous Per-Turbine Budgets — AC(RA=2.0), {year}")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    for i in range(n_turb):
        lam_clipped = np.clip(lambdas_per_turb[i], 1e-3, 1e4)
        ax.semilogy(days, lam_clipped, color=turb_colors[i],
                    linewidth=1, label=turb_labels[i])
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.3)
    ax.set_ylabel("λ (penalty weight)")
    ax.set_xlabel("Day of Year")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.15)

    for ax in axes:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, output_dir, "heterogeneous.png")


# =============================================================================
# ANALYSIS: BATTERY DOMAIN (SECOND APPLICATION)
# =============================================================================

def analysis_battery(budget_cycles, output_dir):
    """
    Second domain: battery cycle-life budget.

    A grid-scale battery earns revenue from price arbitrage. Deep discharge
    cycles are profitable but cause degradation. The operator limits deep
    cycles to `budget_cycles` per year.

    This demonstrates that the AC framework generalizes beyond wind:
    the "action" is deep-discharge, the "budget" is cycle-life, and
    the "benefit" is arbitrage revenue.
    """
    import matplotlib.pyplot as plt

    horizon = 365
    prices, benefit = generate_battery_scenario(horizon)

    print(f"\n{'='*75}")
    print(f"  Battery Domain: Cycle-Life Budget Allocation")
    print(f"  Budget: {budget_cycles} deep cycles/year | Horizon: {horizon} days")
    print(f"{'='*75}")

    threshold = 0.3
    methods = [
        oracle_allocation(benefit, budget_cycles),
        greedy_allocation(benefit, budget_cycles, threshold),
        twap_allocation(benefit, budget_cycles),
        ac_allocation(benefit, budget_cycles, horizon, 1.0, threshold=threshold),
        ac_allocation(benefit, budget_cycles, horizon, 2.0, threshold=threshold),
        ac_allocation(benefit, budget_cycles, horizon, 5.0, threshold=threshold),
    ]

    oracle_val = methods[0]["total_value"]
    print(f"  {'Method':<28s} | {'Value':>7s} | {'%Oracle':>7s} | {'Cycles':>7s}")
    print(f"  {'-'*28}-+-{'-'*7}-+-{'-'*7}-+-{'-'*7}")
    for m in methods:
        pct = 100 * m["total_value"] / oracle_val if oracle_val > 0 else 0
        used = int(m["cumulative_spent"][-1])
        print(f"  {m['name']:<28s} | {m['total_value']:7.2f} | {pct:6.1f}% | {used:4d}/{budget_cycles}")

    # Figure
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True,
                             gridspec_kw={"height_ratios": [0.8, 1, 1]})
    days = np.arange(horizon)

    ax = axes[0]
    ax.plot(days, prices, color="steelblue", linewidth=0.8)
    ax.fill_between(days, prices, alpha=0.2, color="steelblue")
    ax2 = ax.twinx()
    ax2.fill_between(days, benefit, alpha=0.15, color="orange")
    ax2.plot(days, benefit, color="orange", linewidth=0.5)
    ax.set_ylabel("Electricity Price ($/MWh)", color="steelblue")
    ax2.set_ylabel("Deep-Discharge Benefit", color="orange")
    ax.set_title(f"Battery Cycle-Life Budget — AC vs Baselines (budget = {budget_cycles} cycles/year)")

    ax = axes[1]
    ax.fill_between(days, benefit, alpha=0.1, color="gray")
    markers = ["*", "v", "s"]
    for m, mk in zip(methods[:3], markers):
        mask = m["spent"] > 0
        if mask.any():
            ax.scatter(days[mask], benefit[mask], s=20, marker=mk,
                       alpha=0.6, label=m["name"], zorder=3)
    ac_methods = [m for m in methods if m["name"].startswith("AC")]
    colors_ac = plt.cm.viridis(np.linspace(0.3, 0.9, len(ac_methods)))
    for m, c in zip(ac_methods, colors_ac):
        mask = m["spent"] > 0
        if mask.any():
            ax.scatter(days[mask], benefit[mask], s=15, color=c,
                       alpha=0.7, label=m["name"], zorder=4)
    ax.set_ylabel("Benefit at Cycle Points")
    ax.legend(fontsize=7, ncol=3, loc="upper right")

    ax = axes[2]
    twap_line = np.linspace(0, budget_cycles, horizon)
    ax.plot(days, twap_line, "k--", linewidth=1, alpha=0.3, label="TWAP")
    ax.axhline(budget_cycles, color="red", linestyle=":", alpha=0.4)
    for m, mk in zip(methods[:3], markers):
        ax.plot(days, m["cumulative_spent"], linewidth=1.2, alpha=0.5,
                linestyle="--", label=m["name"])
    for m, c in zip(ac_methods, colors_ac):
        ax.plot(days, m["cumulative_spent"], color=c, linewidth=1.8,
                label=f"{m['name']} (val={m['total_value']:.1f})")
    ax.set_ylabel("Cumulative Cycles Used")
    ax.set_xlabel("Day of Year")
    ax.legend(fontsize=6, ncol=3, loc="upper left")

    for ax in axes:
        ax.grid(True, alpha=0.15)
        # Month labels
        month_ticks = [i * 30 for i in range(12) if i * 30 < horizon]
        month_labels = ["Jan","Feb","Mar","Apr","May","Jun",
                        "Jul","Aug","Sep","Oct","Nov","Dec"][:len(month_ticks)]
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, output_dir, "battery.png")


# =============================================================================
# UTILITIES
# =============================================================================

def _month_ticks(dates, year):
    ticks, labels = [], []
    for m in range(1, 13):
        label = f"{year}-{m:02d}"
        idx = np.where([d.startswith(label) for d in dates])[0]
        if len(idx) > 0:
            ticks.append(idx[0])
            labels.append(label)
    return ticks, labels


def _save_or_show(fig, output_dir, filename):
    import matplotlib.pyplot as plt
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {path}")
        plt.close(fig)
    else:
        plt.show()


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Almgren-Chriss budget allocation with baselines and multi-domain demo"
    )
    parser.add_argument("--csv", default="energy_island_10y_daily_av_wind.csv",
                        help="Path to wind data CSV")
    parser.add_argument("--year", type=int, default=2015,
                        help="Year for single_year/pareto/heterogeneous analyses")
    parser.add_argument("--budget-days", type=int, nargs="+", default=[15],
                        help="Budget(s) in days (Pareto uses all, others use first)")
    parser.add_argument("--budget-cycles", type=int, default=50,
                        help="Battery deep-discharge cycles per year")
    parser.add_argument("--benefit-threshold", type=float, default=0.3,
                        help="Base benefit threshold for spending decisions")
    parser.add_argument("--analysis", nargs="+",
                        default=["single_year"],
                        choices=["single_year", "multi_year", "pareto",
                                 "heterogeneous", "battery", "all"],
                        help="Which analyses to run")
    parser.add_argument("--output", default=None,
                        help="Output directory for figures (default: show)")
    cli = parser.parse_args()

    # Resolve analyses
    analyses = set(cli.analysis)
    if "all" in analyses:
        analyses = {"single_year", "multi_year", "pareto", "heterogeneous", "battery"}

    # Load wind data
    csv_path = cli.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", csv_path)

    dates, ws, wd = load_wind_data(csv_path)
    available_years = sorted(set(int(d[:4]) for d in dates))
    print(f"Loaded {len(dates)} days from Energy Island ({available_years[0]}-{available_years[-1]})")

    # Compute global benefit normalization across ALL years so that
    # threshold=0.3 means the same thing in every year.
    global_benefit_max = neg_yaw_benefit(ws, wd, global_max=None).max()
    print(f"Global benefit max: {global_benefit_max:.4f} (used for normalization)")

    budget = cli.budget_days[0]
    threshold = cli.benefit_threshold

    if "single_year" in analyses:
        analysis_single_year(dates, ws, wd, cli.year, budget, threshold, cli.output,
                             global_benefit_max)

    if "multi_year" in analyses:
        analysis_multi_year(dates, ws, wd, budget, threshold, cli.output,
                            global_benefit_max)

    if "pareto" in analyses:
        budgets = cli.budget_days if len(cli.budget_days) > 1 else [10, 15, 20, 30]
        analysis_pareto(dates, ws, wd, cli.year, budgets, threshold, cli.output,
                        global_benefit_max)

    if "heterogeneous" in analyses:
        analysis_heterogeneous(dates, ws, wd, cli.year, budget, threshold, cli.output,
                               global_benefit_max)

    if "battery" in analyses:
        analysis_battery(cli.budget_cycles, cli.output)

    print("\nDone.")


if __name__ == "__main__":
    main()
