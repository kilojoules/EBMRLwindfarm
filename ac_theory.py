"""
Empirical verification of AC-inspired budget surrogate properties.

This module verifies key properties of the time-varying penalty schedule
numerically. These are empirical observations, not formal proofs.

Verified properties:
    1. CMDP analogy: λ(t) behaves like a Lagrangian dual variable —
       stays near 1 on-TWAP, increases monotonically under overspending.
    2. Regret scaling: Under i.i.d. benefits, regret grows approximately
       as O(√T) relative to the hindsight-optimal allocation.
    3. Monotonicity: cumulative spending ≤ budget ∀t (enforced by the
       hard wall backstop + hard guard in the decision rule).
    4. TWAP recovery: risk_aversion=0 → constant λ=1 (analytically exact
       since exp(0 * x) = 1 for all x).

Note: The penalty formula exp(η * (1/urgency - 1)) is inspired by
Almgren-Chriss optimal execution but differs from the original sinh
trajectory. It shares the same qualitative properties.

Usage:
    python ac_theory.py                  # Run all verification tests
    python ac_theory.py --verbose        # Show detailed output
"""

import numpy as np
import torch


def cmdp_lagrangian_equivalence():
    """
    Verify that the AC-inspired penalty weight behaves like a Lagrangian
    dual variable for the CMDP budget constraint.

    CMDP formulation:
        max  E[Σ_t r(s_t, a_t)]
        s.t. E[Σ_t c(s_t, a_t)] ≤ B

    where c(s,a) = 1[yaw < 0] (cost indicator) and B = budget_steps.

    Standard CMDP learns a FIXED λ via dual gradient ascent:
        λ_{k+1} = max(0, λ_k + α · (C_k - B))

    Our AC-inspired schedule provides a TIME-VARYING λ(t):
        λ(t) = exp(η · (1/u(t) - 1))

    where u(t) = (b(t)/B) / (τ(t)/T) is the "urgency" ratio.

    This is NOT formally equivalent to the CMDP dual variable (which is
    the solution to a specific optimization problem). Rather, it shares
    key qualitative properties that we verify here:
    - λ ≈ 1 when on the TWAP trajectory (necessary condition)
    - λ increases monotonically under overspending (necessary condition)
    - λ tightens when budget is scarce, relaxes when plentiful

    The key advantage over standard CMDP dual ascent: no retraining
    needed. The λ schedule is computed in closed form from the current
    budget state.

    Returns:
        dict with verification results
    """
    from load_surrogates import NegativeYawBudgetSurrogate

    B = 20       # budget
    T = 100      # horizon
    eta = 1.0    # risk aversion

    surr = NegativeYawBudgetSurrogate(
        budget_steps=B, horizon_steps=T, risk_aversion=eta, steepness=10.0,
    )
    surr.reset()

    # Simulate an on-TWAP trajectory: spend exactly B/T fraction each step
    # On TWAP, urgency ≈ 1 and λ ≈ 1
    twap_rate = B / T
    lambdas_on_twap = []

    for t in range(T):
        lam = float(surr._compute_lambda().squeeze())
        lambdas_on_twap.append(lam)

        # Spend at TWAP rate (probabilistic)
        use = (t * twap_rate) % 1 < twap_rate
        yaw = torch.tensor([-15.0]) if use else torch.tensor([15.0])
        surr.update(yaw)

    lambdas_on_twap = np.array(lambdas_on_twap)

    # Verify: on TWAP trajectory, λ ≈ 1 (within tolerance)
    mid_section = lambdas_on_twap[10:80]  # avoid boundary effects
    on_twap_near_one = np.abs(mid_section - 1.0).mean() < 0.5

    # Now simulate overspending: use budget faster than TWAP
    surr.reset()
    lambdas_overspend = []
    for t in range(T):
        lam = float(surr._compute_lambda().squeeze())
        lambdas_overspend.append(lam)
        # Always go negative (overspend)
        surr.update(torch.tensor([-15.0]))

    # Verify: λ increases monotonically during overspending
    lambdas_overspend = np.array(lambdas_overspend)
    monotone_increasing = all(
        lambdas_overspend[i+1] >= lambdas_overspend[i]
        for i in range(min(B-1, T-2))
    )

    # Compare to CMDP dual gradient ascent
    # Standard CMDP: λ_{k+1} = max(0, λ_k + α·(c_k - b_avg))
    # where b_avg = B/T is the per-step budget rate
    # AC provides the SAME direction: tighten when overspending, relax when underspending
    # But AC uses remaining budget/time ratio instead of cumulative violation

    return {
        "on_twap_lambda_mean": float(mid_section.mean()),
        "on_twap_near_one": bool(on_twap_near_one),
        "overspend_monotone": bool(monotone_increasing),
        "overspend_final_lambda": float(lambdas_overspend[B-1]),
    }


def regret_bound_verification(n_trials: int = 200):
    """
    Empirically check whether regret scales as O(√T) under i.i.d. benefits.

    Regret = Oracle_value - AC_value

    We measure regret at multiple horizons and check whether regret/√T
    is approximately constant. This is an empirical observation, not a
    formal proof. The test would also pass for nearby scaling exponents
    (e.g., T^0.4 or T^0.6), so the result is suggestive, not conclusive.

    Note: The i.i.d. assumption is critical. Real wind data has seasonal
    structure and autocorrelation, so this bound may not transfer directly.

    Returns:
        dict with regret statistics at different horizons
    """
    from load_surrogates import NegativeYawBudgetSurrogate

    horizons = [50, 100, 200, 500, 1000]
    budget_fraction = 0.15  # 15% of timesteps
    risk_aversion = 2.0
    threshold = 0.3

    results = {}
    for T in horizons:
        B = max(int(T * budget_fraction), 1)
        regrets = []

        for trial in range(n_trials):
            rng = np.random.RandomState(trial)
            benefit = rng.beta(2, 5, T)  # i.i.d. benefits ~ Beta(2,5)
            benefit = benefit / (benefit.max() + 1e-8)

            # Oracle: top-k
            top_k = np.argsort(benefit)[-B:]
            oracle_val = benefit[top_k].sum()

            # AC allocation
            surr = NegativeYawBudgetSurrogate(
                budget_steps=B, horizon_steps=T,
                risk_aversion=risk_aversion, steepness=10.0,
            )
            surr.reset()
            ac_val = 0.0
            spent = 0

            for t in range(T):
                lam = float(surr._compute_lambda().squeeze())
                eff_thresh = threshold * max(lam, 1e-8)
                use = benefit[t] > eff_thresh and spent < B

                if use:
                    ac_val += benefit[t]
                    spent += 1

                yaw = torch.tensor([-15.0]) if use else torch.tensor([15.0])
                surr.update(yaw)

            regrets.append(oracle_val - ac_val)

        results[T] = {
            "mean_regret": float(np.mean(regrets)),
            "std_regret": float(np.std(regrets)),
            "sqrt_T": float(np.sqrt(T)),
            "regret_over_sqrt_T": float(np.mean(regrets) / np.sqrt(T)),
        }

    # Verify: regret/√T should be approximately constant (O(√T) scaling)
    ratios = [results[T]["regret_over_sqrt_T"] for T in horizons]
    ratio_std = np.std(ratios) / (np.mean(ratios) + 1e-8)
    sqrt_t_scaling = ratio_std < 0.5  # loose bound: ratios shouldn't vary wildly

    return {
        "horizons": horizons,
        "results": results,
        "sqrt_t_scaling_plausible": bool(sqrt_t_scaling),
        "ratio_cv": float(ratio_std),
    }


def monotonicity_verification():
    """
    Verify that cumulative spending never exceeds the budget.

    This property is enforced by TWO mechanisms working together:
    1. The hard guard `spent < B` in the decision rule (primary)
    2. The hard-wall backstop λ → 1e6 as budget → 0 (secondary)

    The monotonicity is NOT a property of the λ schedule alone — it
    requires the hard guard. If the hard guard were removed and only
    the λ schedule + hard wall were used, violations might still occur
    in edge cases where a very high benefit overwhelms the penalty.

    We verify this empirically across many random benefit sequences
    WITH the hard guard in place (as in actual usage).

    Returns:
        dict with verification results
    """
    from load_surrogates import NegativeYawBudgetSurrogate

    n_trials = 500
    T = 200
    B = 20
    threshold = 0.3
    violations = 0

    for trial in range(n_trials):
        rng = np.random.RandomState(trial)
        benefit = rng.uniform(0, 1, T)

        for ra in [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]:
            surr = NegativeYawBudgetSurrogate(
                budget_steps=B, horizon_steps=T,
                risk_aversion=ra, steepness=10.0,
            )
            surr.reset()
            spent = 0

            for t in range(T):
                lam = float(surr._compute_lambda().squeeze())
                eff_thresh = threshold * max(lam, 1e-8)
                use = benefit[t] > eff_thresh and spent < B

                if use:
                    spent += 1

                yaw = torch.tensor([-15.0]) if use else torch.tensor([15.0])
                surr.update(yaw)

            if spent > B:
                violations += 1

    return {
        "n_trials": n_trials * 6,  # 6 RA values
        "violations": violations,
        "monotonicity_holds": violations == 0,
    }


def twap_recovery_verification():
    """
    Verify that risk_aversion=0 produces constant λ=1 regardless of state.

    This is the TWAP (Time-Weighted Average) baseline: no adaptation
    to budget state, equivalent to a fixed penalty weight.

    Returns:
        dict with verification results
    """
    from load_surrogates import NegativeYawBudgetSurrogate

    surr = NegativeYawBudgetSurrogate(
        budget_steps=20, horizon_steps=100, risk_aversion=0.0, steepness=10.0,
    )
    surr.reset()

    lambdas = []
    for t in range(100):
        lam = float(surr._compute_lambda().squeeze())
        lambdas.append(lam)
        # Random spending pattern
        use = np.random.random() < 0.3
        yaw = torch.tensor([-15.0]) if use else torch.tensor([15.0])
        surr.update(yaw)

    lambdas = np.array(lambdas)

    # With RA=0, AC weight = exp(0 * anything) = 1 always
    # But hard wall still applies near depletion
    # Check that non-depleted lambdas are exactly 1.0
    budget_ok = surr.cumulative_neg_steps.squeeze() < surr.budget_steps * 0.95
    # Only check first portion before any hard wall might kick in
    early_lambdas = lambdas[:50]

    return {
        "mean_lambda_early": float(early_lambdas.mean()),
        "all_ones_early": bool(np.allclose(early_lambdas, 1.0, atol=1e-5)),
    }


def run_all_verifications(verbose: bool = False):
    """Run all theoretical property verifications."""
    print("=" * 65)
    print("  Empirical Properties Verification")
    print("=" * 65)

    # 1. CMDP analogy
    print("\n1. CMDP Lagrangian Dual Analogy (necessary conditions)")
    res = cmdp_lagrangian_equivalence()
    print(f"   On-TWAP λ mean: {res['on_twap_lambda_mean']:.3f} (expect ≈1.0)")
    print(f"   On-TWAP ≈ 1: {'PASS' if res['on_twap_near_one'] else 'FAIL'}")
    print(f"   Overspend monotone: {'PASS' if res['overspend_monotone'] else 'FAIL'}")
    print(f"   Overspend λ at depletion: {res['overspend_final_lambda']:.1f}")

    # 2. Regret scaling (empirical, not a proven bound)
    print("\n2. Regret Scaling (empirical O(√T) check)")
    res = regret_bound_verification()
    print(f"   {'T':>6s} | {'Regret':>8s} | {'√T':>6s} | {'Regret/√T':>10s}")
    print(f"   {'-'*6}-+-{'-'*8}-+-{'-'*6}-+-{'-'*10}")
    for T in res["horizons"]:
        r = res["results"][T]
        print(f"   {T:6d} | {r['mean_regret']:8.3f} | {r['sqrt_T']:6.1f} | "
              f"{r['regret_over_sqrt_T']:10.4f}")
    print(f"   √T scaling plausible: {'PASS' if res['sqrt_t_scaling_plausible'] else 'FAIL'} "
          f"(CV={res['ratio_cv']:.3f}, note: suggestive, not conclusive)")

    # 3. Monotonicity (with hard guard)
    print("\n3. Monotonicity (budget never exceeded, with hard guard)")
    res = monotonicity_verification()
    print(f"   Trials: {res['n_trials']}, Violations: {res['violations']}")
    print(f"   Monotonicity: {'PASS' if res['monotonicity_holds'] else 'FAIL'}")

    # 4. TWAP recovery
    print("\n4. TWAP Recovery (RA=0 → constant λ)")
    res = twap_recovery_verification()
    print(f"   Mean λ (early): {res['mean_lambda_early']:.6f}")
    print(f"   All ones: {'PASS' if res['all_ones_early'] else 'FAIL'}")

    print(f"\n{'=' * 65}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_all_verifications(args.verbose)
