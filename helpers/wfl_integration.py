"""Canonical wind-farm-loads / surrogates_interface adapter for Teodor's DLC12.

Bridges Teodor's specific sector-averaged ANN surrogates to the canonical
interface from https://gitlab.windenergy.dtu.dk/surrogate-models/wind-farm-loads.

The canonical `predict_loads_sector_average(surrogates, sector_average, *theta)`
expects:
  - sectors ordered as (up, right, down, left)
  - TI optionally multiplied by 100 (`ti_in_percent`)
  - theta = additional control inputs (yaw, pitch, rotor speed for Hari's
    surrogates)

Teodor's DLC12 surrogates differ in two ways:
  - sectors are (left, right, up, down) per the saved StandardScaler
    `feature_names_in_`
  - theta = (pset, yaw); inputs total 10 = 4(WS) + 4(TI) + 2(theta)
  - TI in fraction (scaler mean ~0.167) → `ti_in_percent=False`

This module provides:
  - load_teodor_surrogates(bundle_path, scalers_dir): wraps our converted torch
    ANNs (`teodor_dlc12_torch.pt`) as `surrogates_interface.PyTorchModel`
    SurrogateModels with input/output StandardScalerTorch transformers loaded
    from the original .pkl scalers. Compatible with predict_loads_sector_average.
  - sector_average_xarray(saws, sati, ...): pack 4-sector u/TI into the
    xarray DataArray expected by predict_loads_sector_average. Reorders
    Teodor's (L, R, U, D) -> canonical (U, R, D, L) so canonical code reads
    sectors in a consistent order — Teodor's surrogate is then re-fed in its
    expected (L, R, U, D) order via a wrapping permutation in the loader.

NOTE: the canonical predict_loads_sector_average concatenates sectors in the
order they appear along the `sector` dim. To honor BOTH conventions we wrap
each surrogate's input scaler with a sector-permutation Transformer that
maps (U, R, D, L) -> (L, R, U, D) before applying StandardScaler. This means
calling code can pass sectors in canonical (U, R, D, L) order and get the
right answer.
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, Iterable, Optional

import numpy as np


# Canonical (wind-farm-loads) sector order.
CANONICAL_SECTORS = ("up", "right", "down", "left")
# Teodor's surrogate's expected sector order (per saved scaler).
TEODOR_SECTORS = ("left", "right", "up", "down")
# Indices into a length-4 array in Teodor order, given canonical input.
# x_teodor[i] = x_canonical[CANONICAL_TO_TEODOR[i]]
CANONICAL_TO_TEODOR = tuple(CANONICAL_SECTORS.index(s) for s in TEODOR_SECTORS)


def _build_teodor_torch_module(layer_list):
    """Mirror helpers.teodor_surrogate._build_torch_module here to avoid
    circular imports when this module is loaded standalone."""
    import torch
    import torch.nn as nn
    _ACT = {
        "relu": nn.ReLU, "tanh": nn.Tanh, "sigmoid": nn.Sigmoid,
        "swish": nn.SiLU, "silu": nn.SiLU, "gelu": nn.GELU,
        "elu": nn.ELU, "selu": nn.SELU, "softplus": nn.Softplus,
        "linear": nn.Identity,
    }
    layers = []
    for kind, params in layer_list:
        if kind == "dense":
            W, b = params["W"], params["b"]
            in_f, out_f = W.shape
            lin = nn.Linear(in_f, out_f)
            with torch.no_grad():
                lin.weight.copy_(W.t().contiguous())
                lin.bias.copy_(b)
            layers.append(lin)
        elif kind == "activation":
            cls = _ACT.get(params["name"])
            if cls is None:
                raise ValueError(f"unknown activation {params['name']!r}")
            layers.append(cls())
        elif kind == "leaky_relu":
            layers.append(nn.LeakyReLU(float(params["negative_slope"])))
        elif kind == "prelu":
            w = params["weight"]
            p = nn.PReLU(num_parameters=int(w.numel()))
            with torch.no_grad():
                p.weight.copy_(w.flatten())
            layers.append(p)
        else:
            raise ValueError(f"unknown layer kind {kind!r}")
    return nn.Sequential(*layers)


class _CanonicalToTeodorSectorPermuter:
    """surrogates_interface Transformer that reorders sectors before scaling.

    Canonical input row layout (concatenated by predict_loads_sector_average):
        [WS_up, WS_right, WS_down, WS_left,
         TI_up, TI_right, TI_down, TI_left,
         theta_0, theta_1, ...]

    Teodor's StandardScaler expects:
        [WS_left, WS_right, WS_up, WS_down,
         TI_left, TI_right, TI_up, TI_down,
         pset, yaw]

    This transformer permutes the first 8 columns; passes theta through.
    Stateless (no fit needed). Drop-in for surrogates_interface Transformer
    contract: must implement `transform(x)` and `inverse_transform(y)`.
    """

    def __init__(self):
        # Build a permutation index for the first 8 columns.
        # canonical order: U, R, D, L
        # teodor order:    L, R, U, D
        # so teodor[i] = canonical[CANONICAL_TO_TEODOR[i]]
        ws_perm = list(CANONICAL_TO_TEODOR)               # 4 idx
        ti_perm = [4 + i for i in CANONICAL_TO_TEODOR]    # 4 idx, offset 4
        self._perm = ws_perm + ti_perm                    # 8 idx

    def transform(self, x, inplace=False):
        # SurrogateModel.predict_output calls with inplace=True; we permute
        # columns so an inplace op isn't possible. Raise so the base class
        # falls back to inplace=False (catches ValueError).
        if inplace:
            raise ValueError("permutation cannot be done inplace")
        x = np.asarray(x)
        n_extra = x.shape[-1] - 8
        if n_extra < 0:
            raise ValueError(
                f"input has {x.shape[-1]} cols, expected >= 8 (4 WS + 4 TI)")
        out = np.empty_like(x, dtype=np.float32)
        out[..., :8] = x[..., self._perm]
        if n_extra > 0:
            out[..., 8:] = x[..., 8:]
        return out

    def inverse_transform(self, y, inplace=False):
        # Not meaningful for input transformer in this pipeline; identity-ish.
        return np.asarray(y)


def load_teodor_surrogates(
    bundle_path: str | Path,
    scalers_dir: str | Path,
    outputs: Optional[Iterable[str]] = None,
    sectors_in: str = "canonical",
    map_location: str = "cpu",
) -> Dict[str, "object"]:
    """Load Teodor's DLC12 surrogates as canonical SurrogateModel objects.

    Args:
        bundle_path: path to teodor_dlc12_torch.pt (output of convert_load_surrogates.py).
        scalers_dir: dir containing scaler_input_DLC12_*.pkl, scaler_output_DLC12_*.pkl.
        outputs: subset of output names to load; default = all 14.
        sectors_in: "canonical" → caller passes sectors in (U,R,D,L); a permuter
            adapts to Teodor's (L,R,U,D). "teodor" → caller passes sectors in
            (L,R,U,D); no permutation.
        map_location: torch device.

    Returns:
        dict[name, surrogates_interface.surrogates.SurrogateModel] suitable
        for `predict_loads_sector_average(surrogates, sector_average, pset, yaw)`.
    """
    import joblib
    import torch
    from surrogates_interface.surrogates import (
        PyTorchModel, StandardScalerTorch,
    )

    bundle = torch.load(bundle_path, map_location=map_location,
                         weights_only=False)
    names = list(outputs) if outputs is not None else bundle["outputs"]
    scalers_dir = Path(scalers_dir)

    surrogates: Dict[str, object] = {}
    for name in names:
        net = _build_teodor_torch_module(bundle["models"][name])
        net.eval()

        # Input scaler (StandardScaler in sklearn) -> StandardScalerTorch.
        s_in = joblib.load(scalers_dir / f"scaler_input_DLC12_{name}.pkl")
        s_out = joblib.load(scalers_dir / f"scaler_output_DLC12_{name}.pkl")
        in_scaler = StandardScalerTorch(
            mean=torch.from_numpy(s_in.mean_.astype(np.float32)),
            scale=torch.from_numpy(s_in.scale_.astype(np.float32)),
        )
        out_scaler = StandardScalerTorch(
            mean=torch.from_numpy(s_out.mean_.astype(np.float32)),
            scale=torch.from_numpy(s_out.scale_.astype(np.float32)),
        )

        in_transformers = []
        if sectors_in == "canonical":
            in_transformers.append(_CanonicalToTeodorSectorPermuter())
        in_transformers.append(in_scaler)

        sm = PyTorchModel(
            model=net,
            input_transformers=in_transformers,
            output_transformers=[out_scaler],
            input_names=list(s_in.feature_names_in_),
            output_names=list(s_out.feature_names_in_),
            n_inputs=10,
            n_outputs=1,
            dtype=torch.float32,
        )
        surrogates[name] = sm
    return surrogates


def make_sector_average(
    saws: np.ndarray,
    sati: np.ndarray,
    sectors_order: str = "canonical",
):
    """Pack per-turbine 4-sector flow into the xarray DataArray expected by
    `wind_farm_loads.tool_agnostic.predict_loads_sector_average`.

    Args:
        saws: (n_turb, 4) wind speed per sector [m/s].
        sati: (n_turb, 4) turbulence intensity per sector [-, fraction].
        sectors_order: order of the 4 sector columns. "canonical" =
            (up, right, down, left). "teodor" = (left, right, up, down).

    Returns:
        xarray.DataArray with dims (wt, wd, ws, sector, quantity), shape
        (n_turb, 1, 1, 4, 2). Sectors are stored in canonical order; if input
        was teodor, this function permutes accordingly.
    """
    import xarray as xr

    saws = np.asarray(saws, dtype=np.float32)
    sati = np.asarray(sati, dtype=np.float32)
    if saws.shape != sati.shape or saws.shape[-1] != 4:
        raise ValueError(
            f"expected (n_turb, 4); got saws={saws.shape}, sati={sati.shape}")

    if sectors_order == "teodor":
        sector_labels = list(TEODOR_SECTORS)
    elif sectors_order == "canonical":
        sector_labels = list(CANONICAL_SECTORS)
    else:
        raise ValueError("sectors_order must be 'canonical' or 'teodor'")

    n_turb = saws.shape[0]
    arr = np.zeros((n_turb, 1, 1, 4, 2), dtype=np.float32)
    arr[..., 0] = saws[:, None, None, :]   # quantity = WS_eff
    arr[..., 1] = sati[:, None, None, :]   # quantity = TI_eff
    return xr.DataArray(
        arr,
        dims=("wt", "wd", "ws", "sector", "quantity"),
        coords={
            "wt": np.arange(n_turb),
            "wd": [270.0],
            "ws": [10.0],
            "sector": sector_labels,
            "quantity": ["WS_eff", "TI_eff"],
        },
    )


def predict_flap_del(
    surrogates: Dict[str, object],
    saws: np.ndarray,
    sati: np.ndarray,
    pset: np.ndarray,
    yaw_deg: np.ndarray,
    sectors_order: str = "canonical",
):
    """Convenience wrapper: call canonical predict_loads_sector_average for
    Teodor's DLC12 surrogates. ti_in_percent=False because Teodor's scaler
    expects TI in fraction.

    Args:
        surrogates: dict from `load_teodor_surrogates(...)`.
        saws, sati: (n_turb, 4) per-sector flow.
        pset, yaw_deg: (n_turb,) control inputs (Teodor's theta).
        sectors_order: same as `make_sector_average`.

    Returns:
        xarray.DataArray of loads (wt, wd, ws, sensor).
    """
    from wind_farm_loads.tool_agnostic import predict_loads_sector_average

    sa = make_sector_average(saws, sati, sectors_order=sectors_order)
    pset = np.asarray(pset, dtype=np.float32).reshape(-1)
    yaw_deg = np.asarray(yaw_deg, dtype=np.float32).reshape(-1)
    return predict_loads_sector_average(
        surrogates, sa, pset, yaw_deg, ti_in_percent=False)
