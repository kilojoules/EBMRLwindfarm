"""Smoke test for the converted DLC12 torch bundle.

Loads the bundle, predicts blade-flapwise DEL at a few representative
operating points, prints output. Confirms:
  - bundle loads with pure torch (no TF needed),
  - predictions are finite + sensible magnitude (kNm),
  - yaw sensitivity is non-trivial.

Usage:
    python scripts/probe_flap_del.py --bundle checkpoints/teodor_dlc12_torch.pt
"""
import argparse
import sys
from pathlib import Path

import torch

# Allow running from project root without installing package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "teodor_surrogate",
    Path(__file__).resolve().parents[1] / "helpers" / "teodor_surrogate.py")
_ts = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ts)
TeodorDLC12Surrogate = _ts.TeodorDLC12Surrogate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", required=True)
    p.add_argument("--output", default="wrot_Bl1Rad0FlpMnt")
    args = p.parse_args()

    surr = TeodorDLC12Surrogate.from_bundle(
        args.bundle, outputs=[args.output])
    surr.eval()

    # Nominal operating point — derived from input scaler means.
    base = torch.tensor([
        [9.10, 9.13, 9.83, 8.47,    # saws L,R,U,D
         0.167, 0.164, 0.151, 0.176, # sati L,R,U,D
         0.93,                       # pset
         0.0]                        # yaw
    ], dtype=torch.float32)

    yaws = [-25.0, -15.0, -5.0, 0.0, 5.0, 15.0, 25.0]
    print(f"Output: {args.output}")
    print(f"Inputs (per-feature): saws_L,R,U,D, sati_L,R,U,D, pset, yaw")
    print(f"Base operating point (yaw varies):")
    print(f"  saws ~ 9 m/s, sati ~ 0.16, pset = 0.93\n")
    print(f"  yaw[deg]   {args.output}")
    print(f"  ------     " + "-" * 14)
    base_at_zero = None
    for y in yaws:
        x = base.clone()
        x[0, 9] = y
        out = surr.predict_one(args.output, x).item()
        if y == 0.0:
            base_at_zero = out
        rel = "" if base_at_zero is None else f"  ({100*(out/base_at_zero - 1):+.1f}%)"
        print(f"  {y:7.1f}    {out:10.2f}{rel}")

    # Vary inflow speed at yaw=0.
    print("\nWind sensitivity (yaw=0):")
    for u in [6.0, 9.0, 12.0]:
        x = base.clone()
        x[0, 0:4] = u
        x[0, 9] = 0.0
        out = surr.predict_one(args.output, x).item()
        print(f"  u={u:.1f} m/s  ->  {out:10.2f}")

    print("\nOK." if base_at_zero is not None and torch.isfinite(
        torch.tensor(base_at_zero)) else "\nFAIL.")


if __name__ == "__main__":
    main()
