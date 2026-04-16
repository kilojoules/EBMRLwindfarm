"""Smoke test: verify DELConstraintSurrogate produces correct penalties + gradients."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import torch
import numpy as np
from helpers.surrogate_loads import TorchDELSurrogate
from load_surrogates import DELConstraintSurrogate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
torch_model = TorchDELSurrogate.from_keras(
    f"{REPO}/surrogate/models/ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras",
    f"{REPO}/surrogate/scalers/scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
    f"{REPO}/surrogate/scalers/scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
)

# Fake sector averages and baseline DELs (3 turbines)
sector_avgs = torch.tensor([
    [6.5, 6.5, 6.5, 6.5, 0.07, 0.07, 0.07, 0.07],
    [5.2, 4.7, 5.3, 5.8, 0.16, 0.17, 0.16, 0.15],
    [6.2, 7.1, 6.1, 4.4, 0.17, 0.15, 0.17, 0.20],
], dtype=torch.float32)
baseline_dels = torch.tensor([435.0, 430.0, 690.0], dtype=torch.float32)

for mode in ("per_turbine", "farm_max"):
    for ptype in ("exponential", "quadratic"):
        # For farm_max: max(baseline)=690 is the reference — most turbines'
        # DELs are well below 690, so the penalty rarely fires. To test
        # gradient flow, use a lower baseline so the reference is smaller.
        if mode == "farm_max":
            test_baseline = torch.tensor([300.0, 300.0, 300.0], dtype=torch.float32)
            thresh = 0.05
        else:
            test_baseline = baseline_dels
            thresh = 0.05
        constraint = DELConstraintSurrogate(
            torch_del_model=torch_model,
            mode=mode,
            threshold_pct=thresh,
            steepness=6.0,
            penalty_type=ptype,
        )
        constraint.set_context(sector_avgs, test_baseline)

        # Action that should trigger penalty (large yaw)
        action = torch.tensor([[[0.8], [-0.3], [0.9]]], dtype=torch.float32, requires_grad=True)
        penalty = constraint.per_turbine_energy(action)
        total = penalty.sum()
        total.backward()

        print(f"{mode}/{ptype} (threshold={thresh:.0%}):")
        print(f"  penalty = {penalty.detach().squeeze().tolist()}")
        print(f"  grad    = {action.grad.squeeze().tolist()}")
        assert action.grad is not None, "No gradient!"
        assert (action.grad.abs() > 0).any(), f"Zero gradient in {mode}/{ptype}!"
        print(f"  PASSED")

        # Action near zero — should have zero or small penalty
        action0 = torch.tensor([[[0.0], [0.0], [0.0]]], dtype=torch.float32, requires_grad=True)
        penalty0 = constraint.per_turbine_energy(action0)
        print(f"  penalty at yaw=0: {penalty0.detach().squeeze().tolist()}")
        print()

print("All gradient-flow checks PASSED")
