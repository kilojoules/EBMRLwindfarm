"""Pure-torch consumer of Teodor's DLC12 ANN load surrogates.

Loads a bundle written by scripts/convert_load_surrogates.py and rebuilds
each ANN as a torch nn.Sequential with frozen weights. Differentiable end
to end.

Inputs (per turbine):
    saws_left, saws_right, saws_up, saws_down,    # 4-sector wind speed [m/s]
    sati_left, sati_right, sati_up, sati_down,    # 4-sector turbulence intensity [-]
    pset,                                          # power setpoint [-]
    yaw                                            # yaw angle [deg]

Outputs: 10-min DEL (or other quantity for non-load outputs).

Typical usage:
    surr = TeodorDLC12Surrogate.from_bundle("checkpoints/teodor_dlc12_torch.pt",
                                              outputs=["wrot_Bl1Rad0FlpMnt"])
    feats = torch.tensor([[9.1, 9.1, 9.8, 8.5, 0.17, 0.16, 0.15, 0.18,
                            0.93, 0.0]])
    del_flap = surr(feats)["wrot_Bl1Rad0FlpMnt"]   # shape (1,)
"""
from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn


_ACT_MAP = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "swish": nn.SiLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "elu": nn.ELU,
    "selu": nn.SELU,
    "softplus": nn.Softplus,
    "linear": nn.Identity,
    "softmax": lambda: nn.Softmax(dim=-1),
}


def _build_torch_module(layer_list):
    layers = []
    for kind, params in layer_list:
        if kind == "dense":
            W = params["W"]              # keras: (in, out)
            b = params["b"]              # keras: (out,)
            in_f, out_f = W.shape
            lin = nn.Linear(in_f, out_f, bias=True)
            with torch.no_grad():
                lin.weight.copy_(W.t().contiguous())  # torch: (out, in)
                lin.bias.copy_(b)
            layers.append(lin)
        elif kind == "activation":
            name = params["name"]
            cls = _ACT_MAP.get(name)
            if cls is None:
                raise ValueError(f"unknown activation {name!r}")
            layers.append(cls())
        elif kind == "leaky_relu":
            layers.append(nn.LeakyReLU(
                negative_slope=float(params["negative_slope"])))
        elif kind == "prelu":
            w = params["weight"]
            n = int(w.numel())
            p = nn.PReLU(num_parameters=n)
            with torch.no_grad():
                p.weight.copy_(w.flatten())
            layers.append(p)
        elif kind == "layernormalization":
            # Build LayerNorm matching keras shape.
            gamma = params.get("gamma")
            beta = params.get("beta")
            shape = gamma.shape if gamma is not None else (params.get("scale").shape)
            ln = nn.LayerNorm(shape[-1])
            if gamma is not None:
                with torch.no_grad():
                    ln.weight.copy_(gamma)
                    ln.bias.copy_(beta)
            layers.append(ln)
        elif kind == "batchnormalization":
            raise NotImplementedError(
                "BatchNorm in inference: use eval-mode running stats; "
                "add explicit branch if needed.")
        else:
            raise ValueError(f"unknown layer kind {kind!r}")
    return nn.Sequential(*layers)


class TeodorDLC12Surrogate(nn.Module):
    """Wraps the 14 (or a subset of) DLC12 ANN surrogates.

    Forward returns a dict {output_name: tensor (..., 1)} in physical units
    (de-scaled). Input is in physical units; the module applies the saved
    StandardScaler internally.
    """

    INPUT_FEATURES = [
        "in_saws_left", "in_saws_right", "in_saws_up", "in_saws_down",
        "in_sati_left", "in_sati_right", "in_sati_up", "in_sati_down",
        "in_pset", "in_yaw",
    ]

    def __init__(self,
                 nets: Dict[str, nn.Sequential],
                 in_mean: torch.Tensor,
                 in_scale: torch.Tensor,
                 out_means: Dict[str, torch.Tensor],
                 out_scales: Dict[str, torch.Tensor]):
        super().__init__()
        self.nets = nn.ModuleDict(nets)
        self.register_buffer("in_mean", in_mean)
        self.register_buffer("in_scale", in_scale)
        for name, m in out_means.items():
            self.register_buffer(f"out_mean__{name}", m)
            self.register_buffer(f"out_scale__{name}", out_scales[name])
        self.output_names = list(nets.keys())
        # Freeze: surrogate is not trained.
        for p in self.parameters():
            p.requires_grad = False

    @classmethod
    def from_bundle(cls,
                    path: str | Path,
                    outputs: Optional[List[str]] = None,
                    map_location: str = "cpu") -> "TeodorDLC12Surrogate":
        b = torch.load(path, map_location=map_location, weights_only=False)
        names = outputs if outputs is not None else b["outputs"]
        nets = {n: _build_torch_module(b["models"][n]) for n in names}
        out_means = {n: b["output_scalers"][n]["mean"] for n in names}
        out_scales = {n: b["output_scalers"][n]["scale"] for n in names}
        return cls(nets,
                    in_mean=b["input_scaler"]["mean"],
                    in_scale=b["input_scaler"]["scale"],
                    out_means=out_means,
                    out_scales=out_scales)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """x: (..., 10) in physical units.

        Returns dict of (..., 1) tensors in physical units.
        """
        if x.shape[-1] != 10:
            raise ValueError(
                f"expected last-dim 10, got {tuple(x.shape)}; "
                f"feature order = {self.INPUT_FEATURES}")
        x_n = (x - self.in_mean) / self.in_scale
        out = {}
        for name, net in self.nets.items():
            y_n = net(x_n)
            mean = getattr(self, f"out_mean__{name}")
            scale = getattr(self, f"out_scale__{name}")
            out[name] = y_n * scale + mean
        return out

    def predict_one(self, name: str, x: torch.Tensor) -> torch.Tensor:
        """Single-output convenience. Returns (..., 1) in physical units."""
        if name not in self.nets:
            raise KeyError(f"{name} not loaded; loaded={self.output_names}")
        x_n = (x - self.in_mean) / self.in_scale
        y_n = self.nets[name](x_n)
        mean = getattr(self, f"out_mean__{name}")
        scale = getattr(self, f"out_scale__{name}")
        return y_n * scale + mean
