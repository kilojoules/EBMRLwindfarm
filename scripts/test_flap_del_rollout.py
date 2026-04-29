"""Smoke-test the full FlapDEL pipeline against a live WindGym env.

Per env step:
    - 4-sector u/TI per turbine via helpers/rotor_disk_flow
    - feed into FlapDELSurrogate via update_context
    - read DEL per turbine
Aggregate over the rollout to calibrate del_ref (mean DEL under unconstrained
zero-yaw operation).

Run on LUMI (needs WindGym + dynamiks backend).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Load helpers without triggering helpers/__init__'s mujoco-dependent chain.
import importlib.util
def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    sys.modules[name] = m
    return m

ts = _load("helpers.teodor_surrogate", ROOT / "helpers/teodor_surrogate.py")
rdf = _load("helpers.rotor_disk_flow", ROOT / "helpers/rotor_disk_flow.py")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--n-steps", type=int, default=100)
    p.add_argument("--turbines", type=int, default=2)
    p.add_argument("--spacing", type=float, default=800.0)
    args = p.parse_args()

    print(f"loading surrogate from {args.bundle}")
    surr = ts.TeodorDLC12Surrogate.from_bundle(
        args.bundle, outputs=["wrot_Bl1Rad0FlpMnt"])
    surr.eval()

    print("creating WindGym env...")
    import warnings
    warnings.filterwarnings("ignore")
    from WindGym import WindFarmEnv
    from py_wake.examples.data.iea37 import IEA37_WindTurbines
    turbine = IEA37_WindTurbines()
    xs = list(np.arange(args.turbines, dtype=float) * args.spacing)
    ys = [0.0] * args.turbines
    config = {"layouts": "custom", "n_turbines": args.turbines}
    try:
        env = WindFarmEnv(turbine=turbine, x_pos=xs, y_pos=ys,
                          config=config, backend="dynamiks", seed=1)
    except Exception as e:
        print(f"[fallback] dynamiks init failed: {e}\n  trying pywake...")
        env = WindFarmEnv(turbine=turbine, x_pos=xs, y_pos=ys,
                          config=config, backend="pywake", seed=1)
    obs, _ = env.reset(seed=1)
    print(f"  backend = {type(env.unwrapped.fs).__name__}")

    n_t = args.turbines
    yaw_deg = np.zeros(n_t)
    pset = np.full(n_t, 0.93)

    dels = []
    for t in range(args.n_steps):
        # Pull 4-sector flow per turbine.
        per = []
        for i in range(n_t):
            try:
                feats = rdf.disk_features_for_env(env, turbine_idx=i,
                                                    yaw_deg=yaw_deg[i])
            except Exception as e:
                if t == 0:
                    print(f"[turbine {i}] disk_features error: {e}")
                feats = dict(saws_left=9.0, saws_right=9.0, saws_up=9.0,
                              saws_down=9.0, sati_left=0.10, sati_right=0.10,
                              sati_up=0.10, sati_down=0.10)
            per.append(feats)

        x = rdf.features_to_surrogate_input(per, pset.tolist(),
                                              yaw_deg.tolist())
        x_t = torch.from_numpy(x)
        out = surr.predict_one("wrot_Bl1Rad0FlpMnt", x_t)
        del_per_turb = out.flatten().numpy()
        dels.append(del_per_turb)

        if t < 3 or t == args.n_steps - 1:
            print(f"  t={t:3d}  saws_avg(turb0)="
                  f"{np.mean([per[0][k] for k in ('saws_left','saws_right','saws_up','saws_down')]):.2f} m/s"
                  f"  DEL={del_per_turb}")

        # Step env with zero yaw (action = 0).
        action = np.zeros((n_t,), dtype=np.float32)
        try:
            obs, _, term, trunc, info = env.step(action)
            if term or trunc:
                print(f"  episode ended at t={t}")
                break
        except Exception as e:
            print(f"  env.step failed: {e}; stopping")
            break

    dels = np.asarray(dels)
    print(f"\n=== rollout summary ({len(dels)} steps, {n_t} turbines) ===")
    print(f"  DEL per turbine, mean: {dels.mean(axis=0)}")
    print(f"  DEL per turbine, std:  {dels.std(axis=0)}")
    print(f"  global mean DEL: {dels.mean():.2f} kNm")
    print(f"\nSuggested del_ref = {dels.mean():.1f}")


if __name__ == "__main__":
    main()
