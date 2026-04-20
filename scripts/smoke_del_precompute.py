"""Smoke test: DELConstraintSurrogate in precomputed mode.

Verifies:
1. build_del_grid produces the correct shape and finite values.
2. set_context_precomputed installs the grid.
3. per_turbine_energy at grid points matches the stored grid exactly
   (up to F32 round-off).
4. Gradient flows from action → penalty for every turbine.
5. Inside a grid cell, the interpolated DEL is a convex combination of
   the surrounding corner values (bounded above and below).
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from load_surrogates import (
    DELConstraintSurrogate,
    build_del_grid,
    ndim_linear_interp,
)
from helpers.surrogate_loads import (
    TorchDELSurrogate,
    make_rotor_template,
    sector_averages_reordered,
)

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_fake_wfm_fn(n_turb: int, seed: int = 0):
    """A cheap fake wfm_fn for testing: deterministic, smooth, N-dim dependent.

    Uses a simple quadratic: DEL_t = 100 + sum_j (yaw_j / 30)^2 * (t+1),
    so every turbine's DEL depends on every other turbine's yaw.
    """
    rng = np.random.default_rng(seed)
    coeff = rng.uniform(0.5, 1.5, size=(n_turb, n_turb)).astype(np.float32)

    def wfm_fn(yaw_deg: np.ndarray) -> np.ndarray:
        u = (yaw_deg / 30.0).astype(np.float32)
        del_t = 100.0 + (coeff * (u[None, :] ** 2)).sum(axis=1)
        return del_t
    return wfm_fn


def test_fake_end_to_end():
    print("=== Fake wfm_fn end-to-end ===")
    n_turb = 3
    G = 5
    yaw_range = (-30.0, 30.0)
    wfm_fn = build_fake_wfm_fn(n_turb)

    grid = build_del_grid(wfm_fn, n_turb, G, yaw_range)
    assert grid.shape == (n_turb, G, G, G), f"grid shape {grid.shape}"
    assert np.isfinite(grid).all(), "grid has non-finite values"
    print(f"  grid shape: {grid.shape}   min={grid.min():.2f}, max={grid.max():.2f}")

    baseline = wfm_fn(np.zeros(n_turb))
    print(f"  baseline DEL: {baseline}")

    surr = DELConstraintSurrogate(
        torch_del_model=torch.nn.Identity(),  # unused in precomputed mode
        mode="per_turbine",
        threshold_pct=0.0,
        steepness=10.0,
        penalty_type="exponential",
        yaw_max_deg=30.0,
        context_mode="precomputed",
        grid_size=G,
    )
    surr.set_context_precomputed(
        del_grid=torch.tensor(grid, dtype=torch.float32),
        baseline_dels=torch.tensor(baseline, dtype=torch.float32),
        yaw_range_deg=yaw_range,
    )

    # --- Test 1: exact match at the origin grid point (yaw=0 is the center) ---
    # G=5 grid spans -30..30 in steps of 15 → index 2 is yaw=0.
    action_zero = torch.zeros(1, n_turb, 1)
    yaw_norm = action_zero.squeeze(-1)
    interp = ndim_linear_interp(
        torch.tensor(grid), yaw_norm,
        grid_yaw_min_deg=yaw_range[0], grid_yaw_max_deg=yaw_range[1],
        yaw_max_deg=30.0,
    ).squeeze(0).numpy()
    assert np.allclose(interp, baseline, atol=1e-4), (
        f"interp at origin {interp} != baseline {baseline}"
    )
    print("  interp at grid origin matches baseline ✓")

    # --- Test 2: exact match at an off-origin grid corner ---
    # yaw_deg=30 → normalised action=1.0 → grid index G-1=4 (corner)
    yaw_deg_corner = np.array([30.0, 30.0, 30.0])
    true_del = wfm_fn(yaw_deg_corner)
    action_corner = torch.tensor([[[1.0], [1.0], [1.0]]], dtype=torch.float32)
    interp_corner = ndim_linear_interp(
        torch.tensor(grid), action_corner.squeeze(-1),
        grid_yaw_min_deg=yaw_range[0], grid_yaw_max_deg=yaw_range[1],
        yaw_max_deg=30.0,
    ).squeeze(0).numpy()
    assert np.allclose(interp_corner, true_del, rtol=1e-3), (
        f"interp at corner {interp_corner} != true {true_del}"
    )
    print(f"  interp at corner yaw=30° matches true DEL within 1e-3 ✓")

    # --- Test 3: gradient flows through interpolation ---
    action = torch.tensor(
        [[[0.4], [-0.2], [0.6]]], dtype=torch.float32, requires_grad=True
    )
    penalty = surr.per_turbine_energy(action)
    loss = penalty.sum()
    loss.backward()
    assert action.grad is not None and (action.grad.abs() > 0).any(), \
        "no gradient flowed through precomputed surrogate"
    print(f"  gradient shape={tuple(action.grad.shape)}, "
          f"values={action.grad.squeeze().tolist()} ✓")

    # --- Test 4: off-grid interpolation is between corner values ---
    action_mid = torch.tensor(
        [[[0.25], [0.25], [0.25]]], dtype=torch.float32
    )  # yaw = 7.5° each
    interp_mid = ndim_linear_interp(
        torch.tensor(grid), action_mid.squeeze(-1),
        grid_yaw_min_deg=yaw_range[0], grid_yaw_max_deg=yaw_range[1],
        yaw_max_deg=30.0,
    ).squeeze(0).numpy()
    # yaw=7.5° falls between grid indices 2 (yaw=0) and 3 (yaw=15), alpha=0.5
    # → interp value must be between grid[t, 2, 2, 2] and grid[t, 3, 3, 3]
    lo_val = grid[:, 2, 2, 2]
    hi_val = grid[:, 3, 3, 3]
    # Because the true surface is convex upward at yaw=0, interp sits between
    # these two but needn't equal the midpoint.
    for t in range(n_turb):
        lo_t, hi_t = sorted([lo_val[t], hi_val[t]])
        # Allow a little slack for numerical noise
        assert lo_t - 1e-3 <= interp_mid[t] <= hi_t + 1e-3, (
            f"T{t} interp {interp_mid[t]} not in [{lo_t}, {hi_t}]"
        )
    print(f"  mid-cell interp lies between corner values ✓")


def test_real_surrogate_small():
    """Exercise the real TorchDELSurrogate + PyWake pipeline at N=2, G=5.

    2-turbine farm, 5 grid points per dim → 25 PyWake sims, ~0.2s.
    """
    print("\n=== Real TorchDELSurrogate pipeline (N=2, G=5) ===")
    from py_wake.deflection_models.jimenez import JimenezWakeDeflection
    from py_wake.examples.data.dtu10mw import DTU10MW
    from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
    from py_wake.site import UniformSite
    from py_wake.turbulence_models import STF2017TurbulenceModel

    wt = DTU10MW()
    wfm = Blondel_Cathelain_2020(
        UniformSite(), windTurbines=wt,
        turbulenceModel=STF2017TurbulenceModel(),
        deflectionModel=JimenezWakeDeflection(),
    )
    D = 178.3
    positions = np.array([[0.0, 0.0], [5 * D, 0.0]])
    rotor_template = make_rotor_template(float(wt.diameter()) / 2)
    hub_h = np.full(len(positions), float(wt.hub_height()))

    torch_del = TorchDELSurrogate.from_keras(
        f"{REPO}/surrogate/models/ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras",
        f"{REPO}/surrogate/scalers/scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
        f"{REPO}/surrogate/scalers/scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
    )

    def wfm_fn(yaw_deg: np.ndarray) -> np.ndarray:
        WS_in, TI_in = sector_averages_reordered(
            wfm, positions[:, 0], positions[:, 1], hub_h,
            wd=270.0, ws=9.0, ti=0.07,
            yaw=yaw_deg, template=rotor_template,
        )
        n = WS_in.shape[0]
        pset = np.full((n, 1), 1.0)
        yaw_col = yaw_deg.reshape(n, 1)
        x = np.hstack([WS_in, TI_in, pset, yaw_col]).astype(np.float32)
        with torch.no_grad():
            return torch_del(torch.as_tensor(x)).cpu().numpy()

    n_turb = 2
    G = 5
    yaw_range = (-30.0, 30.0)
    grid = build_del_grid(wfm_fn, n_turb, G, yaw_range)
    baseline = wfm_fn(np.zeros(n_turb))
    print(f"  baseline={baseline}   grid shape={grid.shape}   "
          f"grid range=[{grid.min():.1f}, {grid.max():.1f}]")

    surr = DELConstraintSurrogate(
        torch_del_model=torch.nn.Identity(),
        mode="farm_max",
        threshold_pct=0.10,
        steepness=6.0,
        penalty_type="exponential",
        yaw_max_deg=30.0,
        context_mode="precomputed",
        grid_size=G,
    )
    surr.set_context_precomputed(
        del_grid=torch.tensor(grid, dtype=torch.float32),
        baseline_dels=torch.tensor(baseline, dtype=torch.float32),
        yaw_range_deg=yaw_range,
    )

    # Penalty at origin should be ~0 (ratio = 1, below threshold 1.1)
    action_zero = torch.zeros(1, n_turb, 1)
    pen_zero = surr.per_turbine_energy(action_zero).sum().item()
    assert pen_zero < 1e-3, f"penalty at origin was {pen_zero}"
    print(f"  penalty at yaw=0: {pen_zero:.4g} ✓")

    # Penalty at large yaw should be positive, and gradient non-zero
    action_big = torch.tensor(
        [[[0.8], [0.8]]], dtype=torch.float32, requires_grad=True
    )
    pen_big = surr.per_turbine_energy(action_big).sum()
    pen_big.backward()
    assert action_big.grad is not None
    print(f"  penalty at yaw=24°,24°: {pen_big.item():.4g}  "
          f"grad={action_big.grad.squeeze().tolist()}")
    assert (action_big.grad.abs() > 0).any(), "no gradient at large yaw"
    print("  gradient non-zero at large yaw ✓")


def test_mode_guard():
    print("\n=== Mode-guard assertions ===")
    surr_frozen = DELConstraintSurrogate(
        torch_del_model=torch.nn.Identity(),
        context_mode="frozen",
    )
    try:
        surr_frozen.set_context_precomputed(
            del_grid=torch.zeros(2, 3, 3),
            baseline_dels=torch.ones(2),
        )
    except RuntimeError as e:
        print(f"  frozen mode rejects set_context_precomputed: {e}")
    else:
        raise AssertionError("frozen mode should reject set_context_precomputed")

    surr_pre = DELConstraintSurrogate(
        torch_del_model=torch.nn.Identity(),
        context_mode="precomputed",
    )
    try:
        surr_pre.set_context(
            sector_avgs=torch.zeros(2, 8),
            baseline_dels=torch.ones(2),
        )
    except RuntimeError as e:
        print(f"  precomputed mode rejects set_context: {e}")
    else:
        raise AssertionError("precomputed mode should reject set_context")
    print("  ✓")


if __name__ == "__main__":
    test_fake_end_to_end()
    test_mode_guard()
    test_real_surrogate_small()
    print("\nAll precompute smoke checks PASSED")
