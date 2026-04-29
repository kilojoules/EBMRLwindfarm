"""Sanity-check: canonical wind_farm_loads.predict_loads_sector_average vs
the internal torch path produce matching DEL.

Run on LUMI after wind-farm-loads is installed.
"""
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util
def _load(name, p):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m)
    sys.modules[name] = m
    return m

ts = _load("helpers.teodor_surrogate", ROOT / "helpers/teodor_surrogate.py")
wfl = _load("helpers.wfl_integration", ROOT / "helpers/wfl_integration.py")


def main():
    import argparse, os
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default=str(ROOT / "checkpoints" / "teodor_dlc12_torch.pt"))
    p.add_argument("--scalers-dir",
        default=os.environ.get("TEODOR_SCALERS_DIR",
            "/scratch/project_465002609/julian/Teodor_surrogates/scalers"))
    args = p.parse_args()
    bundle = Path(args.bundle)
    scalers_dir = Path(args.scalers_dir)
    sensor = "wrot_Bl1Rad0FlpMnt"

    # Test point: 3 turbines, varied yaw.
    saws_teodor = np.array([
        [9.10, 9.13, 9.83, 8.47],   # turb0  (L, R, U, D)
        [8.50, 8.50, 9.00, 8.00],
        [7.50, 7.50, 8.00, 7.00],
    ], dtype=np.float32)
    sati_teodor = np.array([
        [0.167, 0.164, 0.151, 0.176],
        [0.18, 0.18, 0.16, 0.20],
        [0.20, 0.20, 0.18, 0.22],
    ], dtype=np.float32)
    pset = np.array([0.93, 0.93, 0.93], dtype=np.float32)
    yaw = np.array([0.0, 10.0, -15.0], dtype=np.float32)

    # ----- Internal torch path -----
    surr_internal = ts.TeodorDLC12Surrogate.from_bundle(
        bundle, outputs=[sensor])
    surr_internal.eval()
    x_internal = np.concatenate(
        [saws_teodor, sati_teodor, pset[:, None], yaw[:, None]], axis=1)
    out_internal = surr_internal.predict_one(
        sensor, torch.from_numpy(x_internal)).flatten().numpy()

    # ----- Canonical path -----
    surrogates = wfl.load_teodor_surrogates(
        bundle, scalers_dir, outputs=[sensor], sectors_in="teodor")
    loads = wfl.predict_flap_del(
        surrogates, saws_teodor, sati_teodor, pset, yaw,
        sectors_order="teodor")
    sensor_dim = "name" if "name" in loads.dims else "sensor"
    out_canonical = np.asarray(loads.sel({sensor_dim: sensor}).data).flatten()

    print(f"sensor: {sensor}")
    print(f"  internal:  {out_internal}")
    print(f"  canonical: {out_canonical}")
    diff = np.abs(out_internal - out_canonical)
    print(f"  max abs diff: {diff.max():.4e}")
    print(f"  rel diff: {(diff / np.abs(out_internal)).max():.4e}")
    if (diff / np.abs(out_internal)).max() < 1e-3:
        print("OK — paths agree.")
    else:
        print("MISMATCH — investigate scaler/permutation wiring.")


if __name__ == "__main__":
    main()
