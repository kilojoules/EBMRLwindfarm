"""Compute per-turbine baseline DEL budget from a zero-yaw rollout.

Budget framing: each turbine cannot exceed the total DEL it incurs under
no-yaw-offset baseline operations. This script runs that baseline rollout
and emits the per-turbine cumulative DEL as comma-separated values, ready
to pass to ebt_sac_windfarm.py via:

    --flap_del_per_turbine_budgets <CSV>
    --flap_del_horizon_steps <T>

Usage (LUMI):
    pixi run python scripts/calibrate_flap_del_budget.py \\
        --bundle checkpoints/teodor_dlc12_torch.pt \\
        --layout square_2 --horizon 200 --n-episodes 5
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name] = m
    return m

ts = _load("helpers.teodor_surrogate", ROOT / "helpers/teodor_surrogate.py")
rdf = _load("helpers.rotor_disk_flow", ROOT / "helpers/rotor_disk_flow.py")
ec = _load("helpers.env_configs", ROOT / "helpers/env_configs.py")


def rollout(env, surr, n_turb, horizon, sensor):
    """Single zero-yaw rollout. Returns per-turbine cumulative DEL."""
    obs, _ = env.reset()
    pset = np.full(n_turb, 0.93, dtype=np.float32)
    yaw = np.zeros(n_turb, dtype=np.float32)
    cum_del = np.zeros(n_turb, dtype=np.float64)
    for t in range(horizon):
        per = []
        for i in range(n_turb):
            try:
                feats = rdf.disk_features_for_env(env, turbine_idx=i, yaw_deg=0.0)
            except Exception:
                feats = dict(saws_left=9.0, saws_right=9.0, saws_up=9.0,
                              saws_down=9.0, sati_left=0.07, sati_right=0.07,
                              sati_up=0.07, sati_down=0.07)
            per.append(feats)
        x = rdf.features_to_surrogate_input(per, pset.tolist(), yaw.tolist())
        out = surr.predict_one(sensor, torch.from_numpy(x)).flatten().numpy()
        cum_del += out
        try:
            ret = env.step(np.zeros(n_turb, dtype=np.float32))
            obs, _, term, trunc, info = ret if len(ret) == 5 else (
                ret[0], ret[1], ret[2], ret[3], ret[4])
            if term or trunc:
                break
        except Exception:
            break
    return cum_del


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--turbines", type=int, default=2,
                   help="ignored if --layout is given")
    p.add_argument("--spacing", type=float, default=800.0,
                   help="ignored if --layout is given")
    p.add_argument("--layout", default=None,
                   help="layout name from helpers/layouts.py (e.g. multi_modal)")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--sensor", default="wrot_Bl1Rad0FlpMnt")
    p.add_argument("--out-json", default="results/flap_del_budgets.json")
    args = p.parse_args()

    surr = ts.TeodorDLC12Surrogate.from_bundle(args.bundle, outputs=[args.sensor])
    surr.eval()

    import warnings; warnings.filterwarnings("ignore")
    from WindGym import WindFarmEnv
    from py_wake.examples.data.iea37 import IEA37_WindTurbines
    turbine = IEA37_WindTurbines()
    if args.layout is not None:
        layouts_mod = _load("helpers.layouts", ROOT / "helpers/layouts.py")
        x_arr, y_arr = layouts_mod.get_layout_positions(args.layout, turbine)
        xs, ys = list(map(float, x_arr)), list(map(float, y_arr))
        n_turbines = len(xs)
        print(f"layout={args.layout}: x={xs} y={ys}")
    else:
        xs = list(np.arange(args.turbines, dtype=float) * args.spacing)
        ys = [0.0] * args.turbines
        n_turbines = args.turbines
    args.turbines = n_turbines  # propagate so rollout uses correct count
    cfg = ec.make_env_config("default")

    cums = []
    for ep in range(args.n_episodes):
        env = WindFarmEnv(turbine=turbine, x_pos=xs, y_pos=ys,
                          config=cfg, backend="dynamiks", seed=ep + 1)
        c = rollout(env, surr, args.turbines, args.horizon, args.sensor)
        env.close()
        cums.append(c)
        print(f"  ep {ep}: cum DEL = {c}")

    cums = np.asarray(cums)
    mean_b = cums.mean(axis=0)
    std_b = cums.std(axis=0)

    print(f"\n=== per-turbine baseline DEL budget (kNm-step, T={args.horizon}) ===")
    for i, (m, s) in enumerate(zip(mean_b, std_b)):
        print(f"  turbine {i}: B = {m:.0f} ± {s:.0f}")
    csv = ",".join(f"{m:.1f}" for m in mean_b)
    print(f"\n--flap_del_per_turbine_budgets {csv}")
    print(f"--flap_del_horizon_steps {args.horizon}")

    out = {
        "horizon_steps": args.horizon,
        "n_episodes": args.n_episodes,
        "per_turbine_budgets": mean_b.tolist(),
        "per_turbine_std": std_b.tolist(),
        "csv": csv,
        "sensor": args.sensor,
    }
    Path(args.out_json).parent.mkdir(exist_ok=True, parents=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
