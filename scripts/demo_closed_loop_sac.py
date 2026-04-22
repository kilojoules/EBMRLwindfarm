#!/usr/bin/env python3
"""
Closed-loop SAC-style evaluation of the AC budget surrogate.

Bridges the gap between the threshold-policy demo and actual RL integration
by simulating the EBT actor's gradient-descent action optimization with
continuous yaw actions and composed energy landscapes.

Instead of a binary "spend or not" decision, the agent optimizes yaw angles
via gradient descent on:

    E_total(a) = E_power(a; wind) + guidance_scale * E_budget(a; lambda(t))

where E_power captures the benefit of yaw steering (negative yaw can increase
farm power in certain wind conditions), and E_budget is the time-varying
AC penalty from NegativeYawBudgetSurrogate.

This mirrors the EBT actor's _compose_per_turbine_energy() + optimize_actions()
loop (ebt.py lines 311-467), using a differentiable power surrogate in place
of the learned energy head.

Key differences from the threshold demo:
  - Actions are CONTINUOUS yaw angles in [-30, +30] degrees
  - The agent can choose "slightly negative" yaw, not just on/off
  - Gradient-based optimization (not threshold comparison)
  - The budget penalty's gradient competes with the power gradient

Usage:
    python scripts/demo_closed_loop_sac.py
    python scripts/demo_closed_loop_sac.py --n-turbines 5 --budget-days 15
    python scripts/demo_closed_loop_sac.py --output figures/closed_loop.png
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn.functional as F

from load_surrogates import NegativeYawBudgetSurrogate


# =============================================================================
# DIFFERENTIABLE POWER SURROGATE
# =============================================================================

class DifferentiablePowerModel(torch.nn.Module):
    """
    Simplified differentiable wake-steering power model.

    Models the power gain from yaw steering for a row-aligned wind farm.
    The key physics: upstream turbines deflect their wakes via yaw misalignment,
    increasing power at downstream turbines at the cost of some power loss
    at the yawed turbine.

    For negative yaw specifically:
      - Upstream turbine loses cos^3(yaw) of its own power
      - Downstream turbines gain power from wake deflection
      - Net farm benefit depends on wind speed, direction, and farm geometry

    This model captures the essential trade-off: negative yaw has a
    per-turbine power cost but a farm-level benefit in wake-aligned conditions.

    The energy is NEGATIVE power (lower energy = more power), matching the
    EBT convention where gradient descent minimizes energy to find the
    best action.
    """

    def __init__(self, n_turbines: int = 3, yaw_max_deg: float = 30.0):
        super().__init__()
        self.n_turbines = n_turbines
        self.yaw_max_deg = yaw_max_deg

    def forward(
        self,
        action_deg: torch.Tensor,
        ws: float,
        wd: float,
    ) -> torch.Tensor:
        """
        Compute per-turbine power energy (negative power) given yaw angles.

        Args:
            action_deg: (batch, n_turbines, 1) yaw angles in degrees
            ws: wind speed (m/s)
            wd: wind direction (degrees)

        Returns:
            (batch, n_turbines, 1) per-turbine energy (negative = more power)
        """
        yaw_rad = action_deg * (3.14159 / 180.0)

        # Base power: proportional to ws^3, reduced by cos^3(yaw)
        # (Pena & Rathmann model: P ~ cos^3(gamma))
        power_factor = torch.cos(yaw_rad).pow(3)
        base_power = (ws / 12.0) ** 3 * power_factor  # normalized to ~1 at 12 m/s

        # Wake deflection benefit: negative yaw on upstream turbines
        # deflects wake, increasing power on downstream turbines.
        # This benefit is strongest when wind is aligned with turbine rows.
        wd_rad = wd * (3.14159 / 180.0)
        # Wake alignment factor: cos^2 for 270 deg, cos^2 for 90 deg
        alignment = max(
            np.cos(wd_rad - np.radians(270)) ** 2,
            np.cos(wd_rad - np.radians(90)) ** 2,
        )

        # Negative yaw on upstream turbines (indices 0..n-2) benefits
        # downstream turbines (indices 1..n-1).
        # Benefit proportional to sin(yaw) * alignment * ws_factor
        ws_factor = np.exp(-0.5 * ((ws - 12.0) / 4.0) ** 2)  # moderate wind best
        wake_benefit = torch.zeros_like(action_deg)
        for i in range(self.n_turbines - 1):
            # Benefit at turbine i+1 from turbine i's yaw
            # Negative yaw (action < 0) gives positive deflection benefit
            deflection = -torch.sin(yaw_rad[:, i:i+1, :]) * alignment * ws_factor
            wake_benefit[:, i+1:i+2, :] = wake_benefit[:, i+1:i+2, :] + deflection * 0.3

        # Total power per turbine (higher is better)
        total_power = base_power + wake_benefit

        # Return negative power as energy (EBT convention: minimize energy)
        return -total_power


def optimize_yaw_actions(
    power_model: DifferentiablePowerModel,
    budget_surrogate: NegativeYawBudgetSurrogate,
    ws: float,
    wd: float,
    guidance_scale: float,
    n_turbines: int,
    opt_steps: int = 50,
    opt_lr: float = 0.1,
    num_candidates: int = 8,
    yaw_max_deg: float = 30.0,
) -> np.ndarray:
    """
    Optimize yaw actions via gradient descent on composed energy.

    This mirrors EBT's optimize_actions() (ebt.py:343-467):
    1. Initialize random action candidates in [-1, 1]
    2. Gradient descent on E_power + guidance_scale * E_budget
    3. Pick lowest-energy candidate (self-verification)

    Returns:
        (n_turbines,) yaw angles in degrees
    """
    batch = 1

    with torch.enable_grad():
        # Initialize candidates
        actions = torch.randn(num_candidates, n_turbines, 1)
        actions = actions.clamp(-1.0, 1.0)

        for _ in range(opt_steps):
            actions = actions.detach().requires_grad_(True)

            # Scale to degrees for power model
            action_deg = actions * yaw_max_deg

            # Per-turbine power energy
            energy_per_turb = power_model(action_deg, ws, wd)

            # Per-turbine budget penalty (compose before aggregation)
            if guidance_scale > 0:
                budget_energy = budget_surrogate.per_turbine_energy(actions)
                energy_per_turb = energy_per_turb + guidance_scale * budget_energy

            # Aggregate: mean over turbines
            energy = energy_per_turb.mean(dim=1)  # (candidates, 1)

            # Gradient descent
            grad = torch.autograd.grad(energy.sum(), actions)[0]
            actions = actions - opt_lr * grad
            actions = actions.clamp(-1.0, 1.0)

    # Self-verification: pick lowest-energy candidate
    with torch.no_grad():
        action_deg = actions * yaw_max_deg
        final_energy = power_model(action_deg, ws, wd)
        if guidance_scale > 0:
            final_energy = final_energy + guidance_scale * budget_surrogate.per_turbine_energy(actions)
        final_scalar = final_energy.mean(dim=1)  # (candidates, 1)
        best_idx = final_scalar.argmin(dim=0).item()
        best_action = actions[best_idx]  # (n_turbines, 1)

    return (best_action.squeeze(-1) * yaw_max_deg).detach().numpy()


# =============================================================================
# DATA LOADING
# =============================================================================

def load_wind_data(csv_path: str):
    """Load Energy Island wind CSV."""
    dates, ws, wd = [], [], []
    with open(csv_path) as f:
        f.readline()
        for line in f:
            parts = line.strip().split(";")
            if len(parts) != 3:
                continue
            dates.append(parts[0])
            ws.append(float(parts[1]))
            wd.append(float(parts[2]))
    return np.array(dates), np.array(ws), np.array(wd)


# =============================================================================
# CLOSED-LOOP EPISODE
# =============================================================================

def run_closed_loop_episode(
    ws_series: np.ndarray,
    wd_series: np.ndarray,
    n_turbines: int,
    budget_days: int,
    risk_aversion: float,
    guidance_scale: float = 1.0,
    yaw_max_deg: float = 30.0,
) -> dict:
    """
    Run a full closed-loop episode with gradient-based action optimization.

    At each timestep:
    1. The power model provides the current energy landscape (wind-dependent)
    2. The budget surrogate provides the time-varying penalty
    3. Gradient descent optimizes yaw actions on the composed energy
    4. The chosen yaw angles are executed (updating budget state)
    5. Power is computed from the chosen angles

    Returns dict with per-step trajectories.
    """
    horizon = len(ws_series)
    power_model = DifferentiablePowerModel(n_turbines, yaw_max_deg)

    surr = NegativeYawBudgetSurrogate(
        budget_steps=budget_days,
        horizon_steps=horizon,
        risk_aversion=risk_aversion,
        steepness=10.0,
        yaw_max_deg=yaw_max_deg,
    )
    surr.reset()

    yaw_trajectory = []
    power_trajectory = []
    lambda_trajectory = []
    neg_yaw_count = np.zeros(n_turbines)

    for t in range(horizon):
        ws, wd = float(ws_series[t]), float(wd_series[t])

        # Record lambda before action
        lam = surr._compute_lambda()
        lam_val = float(lam.squeeze()) if lam.numel() == 1 else lam.squeeze().numpy().tolist()
        lambda_trajectory.append(lam_val)

        # Optimize yaw via gradient descent on composed energy
        yaw_deg = optimize_yaw_actions(
            power_model, surr, ws, wd,
            guidance_scale=guidance_scale,
            n_turbines=n_turbines,
        )
        yaw_trajectory.append(yaw_deg.copy())

        # Compute actual power at chosen yaw
        with torch.no_grad():
            action_t = torch.tensor(yaw_deg, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            power = -power_model(action_t, ws, wd).squeeze().numpy()  # negate: energy = -power
        power_trajectory.append(power)

        # Count negative yaw per turbine
        neg_yaw_count += (yaw_deg < 0).astype(float)

        # Update surrogate state
        surr.update(torch.tensor(yaw_deg, dtype=torch.float32))

    yaw_arr = np.array(yaw_trajectory)      # (horizon, n_turbines)
    power_arr = np.array(power_trajectory)   # (horizon, n_turbines)

    return {
        "yaw": yaw_arr,
        "power": power_arr,
        "farm_power": power_arr.sum(axis=1),
        "lambdas": lambda_trajectory,
        "neg_yaw_days": neg_yaw_count,
        "total_farm_power": float(power_arr.sum()),
        "mean_yaw": float(yaw_arr.mean()),
    }


def run_unconstrained_episode(
    ws_series: np.ndarray,
    wd_series: np.ndarray,
    n_turbines: int,
    yaw_max_deg: float = 30.0,
) -> dict:
    """Baseline: optimize power with no budget constraint (guidance_scale=0)."""
    horizon = len(ws_series)
    power_model = DifferentiablePowerModel(n_turbines, yaw_max_deg)

    # Dummy surrogate (won't be used)
    dummy = NegativeYawBudgetSurrogate(budget_steps=horizon, horizon_steps=horizon)
    dummy.reset()

    yaw_trajectory = []
    power_trajectory = []
    neg_yaw_count = np.zeros(n_turbines)

    for t in range(horizon):
        ws, wd = float(ws_series[t]), float(wd_series[t])

        yaw_deg = optimize_yaw_actions(
            power_model, dummy, ws, wd,
            guidance_scale=0.0,  # no constraint
            n_turbines=n_turbines,
        )
        yaw_trajectory.append(yaw_deg.copy())

        with torch.no_grad():
            action_t = torch.tensor(yaw_deg, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
            power = -power_model(action_t, ws, wd).squeeze().numpy()
        power_trajectory.append(power)
        neg_yaw_count += (yaw_deg < 0).astype(float)

        dummy.update(torch.tensor(yaw_deg, dtype=torch.float32))

    yaw_arr = np.array(yaw_trajectory)
    power_arr = np.array(power_trajectory)

    return {
        "yaw": yaw_arr,
        "power": power_arr,
        "farm_power": power_arr.sum(axis=1),
        "neg_yaw_days": neg_yaw_count,
        "total_farm_power": float(power_arr.sum()),
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Closed-loop SAC-style evaluation of AC budget surrogate"
    )
    parser.add_argument("--csv", default="energy_island_10y_daily_av_wind.csv")
    parser.add_argument("--year", type=int, default=2015)
    parser.add_argument("--n-turbines", type=int, default=3)
    parser.add_argument("--budget-days", type=int, default=15)
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--output", default=None)
    cli = parser.parse_args()

    # Load wind data
    csv_path = cli.csv
    if not os.path.isabs(csv_path):
        csv_path = os.path.join(os.path.dirname(__file__), "..", csv_path)
    dates, ws, wd = load_wind_data(csv_path)
    mask = np.array([d.startswith(str(cli.year)) for d in dates])
    dates_y, ws_y, wd_y = dates[mask], ws[mask], wd[mask]
    horizon = len(dates_y)

    n_turb = cli.n_turbines
    budget = cli.budget_days
    gs = cli.guidance_scale

    print(f"Closed-Loop SAC Evaluation — Energy Island {cli.year}")
    print(f"  {n_turb} turbines | budget={budget} days | guidance_scale={gs}")
    print(f"  Horizon: {horizon} days")
    print()

    # Run unconstrained baseline
    print("Running unconstrained baseline...", flush=True)
    unconstrained = run_unconstrained_episode(ws_y, wd_y, n_turb)
    print(f"  Unconstrained: farm_power={unconstrained['total_farm_power']:.1f}, "
          f"neg_yaw_days={unconstrained['neg_yaw_days'].astype(int).tolist()}")

    # Run AC at different risk aversions
    ra_values = [0.0, 0.5, 1.0, 2.0, 5.0]
    results = {}
    for ra in ra_values:
        print(f"Running AC (RA={ra})...", flush=True)
        res = run_closed_loop_episode(ws_y, wd_y, n_turb, budget, ra, gs)
        results[ra] = res
        pwr_pct = 100 * res["total_farm_power"] / unconstrained["total_farm_power"]
        print(f"  AC(RA={ra}): farm_power={res['total_farm_power']:.1f} ({pwr_pct:.1f}%), "
              f"neg_yaw_days={res['neg_yaw_days'].astype(int).tolist()}")

    # Summary table
    print(f"\n{'='*80}")
    print(f"  {'Method':<18s} | {'Farm Power':>10s} | {'%Unconstr':>9s} | "
          f"{'Neg-Yaw Days (per turbine)':>30s}")
    print(f"  {'-'*18}-+-{'-'*10}-+-{'-'*9}-+-{'-'*30}")
    print(f"  {'Unconstrained':<18s} | {unconstrained['total_farm_power']:10.1f} | "
          f"{'100.0%':>9s} | {unconstrained['neg_yaw_days'].astype(int).tolist()!s:>30s}")
    for ra, res in results.items():
        pct = 100 * res["total_farm_power"] / unconstrained["total_farm_power"]
        neg_days = res["neg_yaw_days"].astype(int).tolist()
        print(f"  AC(RA={ra:<4.1f})      | {res['total_farm_power']:10.1f} | "
              f"{pct:8.1f}% | {neg_days!s:>30s}")
    print(f"{'='*80}")
    print(f"  Budget: {budget} days per turbine")

    # Plot
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True,
                             gridspec_kw={"height_ratios": [0.7, 1, 1, 1]})
    days = np.arange(horizon)

    # Month ticks
    month_ticks, month_labels = [], []
    for m in range(1, 13):
        label = f"{cli.year}-{m:02d}"
        idx = np.where([d.startswith(label) for d in dates_y])[0]
        if len(idx) > 0:
            month_ticks.append(idx[0])
            month_labels.append(label)

    # Panel 1: Wind conditions
    ax = axes[0]
    ax.fill_between(days, ws_y, alpha=0.2, color="steelblue")
    ax.plot(days, ws_y, color="steelblue", linewidth=0.5)
    ax.set_ylabel("Wind Speed (m/s)")
    ax.set_title(f"Closed-Loop SAC — {n_turb} Turbines, Budget={budget} days, "
                 f"Energy Island {cli.year}")

    # Panel 2: Yaw angles chosen by the agent (turbine 0)
    ax = axes[1]
    ax.axhline(0, color="gray", linestyle="-", alpha=0.3)
    ax.fill_between(days, unconstrained["yaw"][:, 0], alpha=0.15, color="gray",
                    label="Unconstrained T0")
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(ra_values)))
    for (ra, res), c in zip(results.items(), colors):
        ax.plot(days, res["yaw"][:, 0], color=c, linewidth=0.7,
                alpha=0.8, label=f"AC(RA={ra}) T0")
    ax.set_ylabel("Yaw Angle T0 (deg)")
    ax.legend(fontsize=7, ncol=3, loc="lower right")

    # Panel 3: Farm power comparison
    ax = axes[2]
    ax.plot(days, unconstrained["farm_power"], color="gray", linewidth=0.5,
            alpha=0.5, label="Unconstrained")
    for (ra, res), c in zip(results.items(), colors):
        ax.plot(days, res["farm_power"], color=c, linewidth=0.7,
                alpha=0.8, label=f"AC(RA={ra})")
    ax.set_ylabel("Farm Power")
    ax.legend(fontsize=7, ncol=3, loc="upper right")

    # Panel 4: Cumulative negative-yaw days (turbine 0)
    ax = axes[3]
    ax.axhline(budget, color="red", linestyle=":", alpha=0.5, label=f"Budget ({budget}d)")
    neg_cum_uncon = np.cumsum(unconstrained["yaw"][:, 0] < 0)
    ax.plot(days, neg_cum_uncon, color="gray", linewidth=1.5, linestyle="--",
            label="Unconstrained")
    for (ra, res), c in zip(results.items(), colors):
        neg_cum = np.cumsum(res["yaw"][:, 0] < 0)
        ax.plot(days, neg_cum, color=c, linewidth=1.5,
                label=f"AC(RA={ra})")
    ax.set_ylabel("Cumulative Neg-Yaw Days (T0)")
    ax.set_xlabel("Day of Year")
    ax.legend(fontsize=7, ncol=3, loc="upper left")

    for ax in axes:
        ax.set_xticks(month_ticks)
        ax.set_xticklabels(month_labels, rotation=45, ha="right", fontsize=8)
        ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if cli.output:
        os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
        plt.savefig(cli.output, dpi=150, bbox_inches="tight")
        print(f"\nSaved: {cli.output}")
    else:
        plt.show()


if __name__ == "__main__":
    main()
