"""End-to-end: WindGym env -> LoadWrapper -> DELConstraintSurrogate -> gradient check."""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import numpy as np
import torch
from py_wake.examples.data.dtu10mw import DTU10MW
from WindGym import WindFarmEnv

from helpers.env_configs import make_env_config
from helpers.layouts import get_layout_positions
from helpers.load_wrapper import LoadWrapper
from helpers.surrogate_loads import SurrogateLoadModel, TorchDELSurrogate
from load_surrogates import DELConstraintSurrogate

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Build env + wrapper ---
turbine = DTU10MW()
x_pos, y_pos = get_layout_positions("multi_modal", turbine)
config = make_env_config("multi_modal")
raw_env = WindFarmEnv(
    turbine=turbine, x_pos=x_pos, y_pos=y_pos, config=config,
    backend="pywake", dt_sim=1, dt_env=1, yaw_step_sim=1,
    n_passthrough=1, Baseline_comp=True, reset_init=True, seed=0,
)
keras_surr = SurrogateLoadModel(
    f"{REPO}/surrogate/models/ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras",
    f"{REPO}/surrogate/scalers/scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
    f"{REPO}/surrogate/scalers/scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
)
env = LoadWrapper(raw_env, keras_surr)

# --- Build PyTorch constraint ---
torch_del = TorchDELSurrogate.from_keras(
    f"{REPO}/surrogate/models/ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras",
    f"{REPO}/surrogate/scalers/scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
    f"{REPO}/surrogate/scalers/scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
)
constraint = DELConstraintSurrogate(
    torch_del_model=torch_del, mode="per_turbine",
    threshold_pct=0.05, steepness=6.0, penalty_type="exponential",
)

# --- Reset and set context ---
obs, info = env.reset(seed=0)
constraint.set_context(
    sector_avgs=torch.tensor(info["sector_averages"], dtype=torch.float32),
    baseline_dels=torch.tensor(info["loads_baseline"], dtype=torch.float32),
)

print(f"sector_averages shape: {info['sector_averages'].shape}")
print(f"loads_baseline: {np.round(info['loads_baseline'], 2)}")

# --- Test gradient flow with a fake action ---
action = torch.tensor([[[0.5], [0.0], [-0.7]]], dtype=torch.float32, requires_grad=True)
penalty = constraint.per_turbine_energy(action)
penalty.sum().backward()

print(f"\nAction (normalised): {action.detach().squeeze().tolist()}")
print(f"Penalty:             {penalty.detach().squeeze().tolist()}")
print(f"Gradient:            {action.grad.squeeze().tolist()}")
assert action.grad is not None and (action.grad.abs() > 0).any(), "No gradient!"

# --- Step the env and update context ---
obs, _, _, _, info = env.step(env.action_space.sample())
constraint.set_context(
    sector_avgs=torch.tensor(info["sector_averages"], dtype=torch.float32),
    baseline_dels=torch.tensor(info["loads_baseline"], dtype=torch.float32),
)
print(f"\nPost-step loads_baseline: {np.round(info['loads_baseline'], 2)}")
print(f"Post-step loads_current: {np.round(info['loads_current'], 2)}")

print("\nEnd-to-end smoke test PASSED")
