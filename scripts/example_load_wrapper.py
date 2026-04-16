"""End-to-end example: wrap a WindGym env with LoadWrapper and print DELs.

Builds a standalone WindFarmEnv on the `multi_modal` 3-turbine layout (the
hero layout used by the EBT training runs), wraps it with LoadWrapper, and
steps through a handful of random actions. At each step the wrapper adds
two arrays to the info dict:

    info["loads_baseline"] : DEL under the WindGym baseline controller
                             (info["yaw angles base"])
    info["loads_current"]  : DEL under whatever yaws the agent action led to

Both are plain (n_turbines,) numpy arrays in whatever units the ANN DEL
surrogate was trained on (blade-root flap moment, kNm-scale).

Run from the repo root:

    python scripts/example_load_wrapper.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running directly from the repo root without a package install.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from py_wake.examples.data.dtu10mw import DTU10MW
from WindGym import WindFarmEnv

from helpers.env_configs import make_env_config
from helpers.layouts import get_layout_positions
from helpers.load_wrapper import LoadWrapper
from helpers.surrogate_loads import SurrogateLoadModel


SURROGATE_MODEL = REPO_ROOT / "surrogate/models/ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras"
SCALER_IN = REPO_ROOT / "surrogate/scalers/scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl"
SCALER_OUT = REPO_ROOT / "surrogate/scalers/scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl"


def build_env(seed: int = 0):
    """Minimal standalone WindFarmEnv on the multi_modal layout."""
    turbine = DTU10MW()
    x_pos, y_pos = get_layout_positions("multi_modal", turbine)
    config = make_env_config("multi_modal")
    env = WindFarmEnv(
        turbine=turbine,
        x_pos=x_pos,
        y_pos=y_pos,
        config=config,
        backend="pywake",
        dt_sim=1,
        dt_env=1,
        yaw_step_sim=1,
        n_passthrough=1,
        Baseline_comp=True,  # so info["yaw angles base"] / "Power baseline" populate
        reset_init=True,
        seed=seed,
    )
    env.action_space.seed(seed)
    return env


def main():
    print("Loading surrogate...")
    surrogate = SurrogateLoadModel(SURROGATE_MODEL, SCALER_IN, SCALER_OUT)

    print("Building WindGym env (multi_modal layout)...")
    raw_env = build_env(seed=0)
    env = LoadWrapper(raw_env, surrogate, pset=1.0)

    print("Resetting...")
    obs, info = env.reset(seed=0)

    print()
    print("-" * 78)
    print(f"Farm state after reset: wd={raw_env.wd:.1f}  ws={raw_env.ws:.2f}  ti={raw_env.ti:.3f}")
    print(f"Positions (m): x={raw_env.x_pos.tolist()}  y={raw_env.y_pos.tolist()}")
    print(f"Baseline yaws:  {np.round(info['yaw angles base'], 2).tolist()}")
    print(f"Current yaws:   {np.round(raw_env.current_yaw, 2).tolist()}")
    print(f"loads_baseline: {np.round(info['loads_baseline'], 2).tolist()}")
    print(f"loads_current:  {np.round(info['loads_current'], 2).tolist()}")
    print("-" * 78)

    rng = np.random.default_rng(0)
    n_steps = 5
    for t in range(n_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)

        baseline_yaws = np.round(info["yaw angles base"], 2)
        current_yaws = np.round(raw_env.current_yaw, 2)
        lb = info["loads_baseline"]
        lc = info["loads_current"]
        delta = lc - lb

        print(f"step {t:2d}  reward={reward:+.4f}")
        print(f"         yaw_base   = {baseline_yaws.tolist()}")
        print(f"         yaw_agent  = {current_yaws.tolist()}")
        print(f"         load_base  = {np.round(lb, 2).tolist()}")
        print(f"         load_agent = {np.round(lc, 2).tolist()}")
        print(f"         delta      = {np.round(delta, 2).tolist()}   (agent - baseline)")

        if terminated or truncated:
            print("  episode ended; resetting")
            obs, info = env.reset(seed=rng.integers(0, 2**31 - 1).item())

    print()
    print("Done. Both `loads_baseline` and `loads_current` are available in `info`.")


if __name__ == "__main__":
    main()
