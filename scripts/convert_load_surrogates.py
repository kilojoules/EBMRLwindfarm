"""Convert Teodor's keras DLC12 load surrogates to a single torch .pt file.

Run this ONCE on LUMI (or any AVX-capable host with TF + keras + sklearn +
torch installed). The output .pt is portable; downstream code only needs
torch + numpy.

Inputs (default --src):
    <SRC>/models/ann_dlc12_out_<group>_<name>_rank1.keras   x14
    <SRC>/scalers/scaler_input_DLC12_<group>_<name>.pkl     x14 (all identical)
    <SRC>/scalers/scaler_output_DLC12_<group>_<name>.pkl    x14

Output:
    <OUT> = e.g. checkpoints/teodor_dlc12_torch.pt
        {
          "outputs": list of 14 output names (in order),
          "input_features": list of 10 feature names,
          "input_scaler": {"mean": (10,), "scale": (10,)},
          "output_scalers": {name: {"mean": (1,), "scale": (1,)}},
          "models": {name: list of (layer_type, params) tuples},
        }

Each model is a stack of Dense(+activation) layers extracted from keras.
Torch consumer rebuilds with helpers/teodor_surrogate.py.

Usage on LUMI:
    pixi run python scripts/convert_load_surrogates.py \\
        --src /scratch/.../Teodor_surrogates \\
        --out checkpoints/teodor_dlc12_torch.pt
"""
import argparse
import os
import re
from pathlib import Path

import numpy as np
import torch


def _activation_name(layer):
    """Return canonical activation string for a keras Dense/Activation layer."""
    act = getattr(layer, "activation", None)
    if act is None:
        return "linear"
    name = getattr(act, "__name__", None) or str(act)
    name = name.lower()
    for k in ["relu", "tanh", "sigmoid", "swish", "gelu", "elu", "selu",
              "softplus", "linear", "softmax"]:
        if k in name:
            return k
    return name


def keras_to_layer_list(model):
    """Walk a keras Sequential/Functional model, extract dense layers + acts."""
    layers = []
    for L in model.layers:
        cls = type(L).__name__
        if cls == "InputLayer":
            continue
        if cls == "Dense":
            W, b = [w.numpy() if hasattr(w, "numpy") else np.array(w)
                    for w in L.get_weights()]
            layers.append(("dense", {"W": W, "b": b}))
            act = _activation_name(L)
            if act != "linear":
                layers.append(("activation", {"name": act}))
        elif cls == "Activation":
            layers.append(("activation",
                            {"name": _activation_name(L)}))
        elif cls in ("LeakyReLU", "PReLU"):
            # LeakyReLU: alpha (or negative_slope). PReLU: per-channel weight.
            if cls == "LeakyReLU":
                alpha = float(getattr(L, "negative_slope",
                                        getattr(L, "alpha", 0.01)))
                layers.append(("leaky_relu", {"negative_slope": alpha}))
            else:
                W = L.get_weights()[0]  # (..., features)
                layers.append(("prelu",
                                {"weight": np.asarray(W, dtype=np.float32)}))
        elif cls in ("ReLU", "ELU"):
            layers.append(("activation", {"name": cls.lower()}))
        elif cls in ("Dropout", "BatchNormalization", "LayerNormalization"):
            # Dropout: training-only, skip. Norm: keep weights.
            if cls == "Dropout":
                continue
            params = {n: w.numpy() for n, w in zip(L.weights,
                                                     L.get_weights())}
            layers.append((cls.lower(), params))
        else:
            raise NotImplementedError(
                f"Layer type {cls!r} not handled by converter. "
                f"Add a branch in keras_to_layer_list.")
    return layers


OUTPUT_NAMES = [
    "wrot_Bl1PthAngVal",
    "wrot_Bl1Rad0EdgMnt",
    "wrot_Bl1Rad0FlpMnt",
    "wrot_Bl1Rad0TorMnt",
    "wtow_H0FAMnt",
    "wtow_H0SSMnt",
    "wtow_H0TorMnt",
    "wtow_H100FAMnt",
    "wtow_H100SSMnt",
    "wtow_H100TorMnt",
    "wtrm_LoSpdShftRotXMnt",
    "wtrm_LoSpdShftRotYMnt",
    "wtrm_LoSpdShftRotZMnt",
    "wtur_W",
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True,
                   help="path to Teodor_surrogates dir (contains models/, scalers/)")
    p.add_argument("--out", required=True,
                   help="output .pt path")
    args = p.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
    import joblib
    import keras

    src = Path(args.src)
    models_dir = src / "models"
    scalers_dir = src / "scalers"

    # All input scalers are identical; load the first one as canonical.
    canonical_in = joblib.load(
        scalers_dir / f"scaler_input_DLC12_{OUTPUT_NAMES[0]}.pkl")
    in_mean = canonical_in.mean_.astype(np.float32)
    in_scale = canonical_in.scale_.astype(np.float32)
    in_features = list(canonical_in.feature_names_in_)
    print(f"input features: {in_features}")
    print(f"input mean: {in_mean}\nin_scale: {in_scale}")

    bundle = {
        "outputs": OUTPUT_NAMES,
        "input_features": in_features,
        "input_scaler": {"mean": torch.from_numpy(in_mean),
                          "scale": torch.from_numpy(in_scale)},
        "output_scalers": {},
        "models": {},
    }

    for name in OUTPUT_NAMES:
        kpath = models_dir / f"ann_dlc12_out_{name}_rank1.keras"
        spath_out = scalers_dir / f"scaler_output_DLC12_{name}.pkl"
        spath_in = scalers_dir / f"scaler_input_DLC12_{name}.pkl"
        if not kpath.exists():
            raise FileNotFoundError(kpath)
        # Sanity-check: per-output input scaler matches canonical.
        s_in = joblib.load(spath_in)
        assert np.allclose(s_in.mean_, in_mean, atol=1e-6), (
            f"input scaler for {name} differs from canonical")

        s_out = joblib.load(spath_out)
        bundle["output_scalers"][name] = {
            "mean": torch.from_numpy(s_out.mean_.astype(np.float32)),
            "scale": torch.from_numpy(s_out.scale_.astype(np.float32)),
        }

        m = keras.models.load_model(kpath, compile=False)
        in_shape = m.input_shape
        out_shape = m.output_shape
        assert in_shape[-1] == 10, f"{name}: in_shape={in_shape}"
        assert out_shape[-1] == 1, f"{name}: out_shape={out_shape}"

        layers = keras_to_layer_list(m)
        # Convert numpy arrays to torch tensors so the bundle is pure-torch.
        torch_layers = []
        for kind, params in layers:
            tp = {}
            for k, v in params.items():
                if isinstance(v, np.ndarray):
                    tp[k] = torch.from_numpy(v.astype(np.float32))
                else:
                    tp[k] = v
            torch_layers.append((kind, tp))
        bundle["models"][name] = torch_layers
        print(f"  {name}: {len(layers)} layers, "
              f"params={m.count_params()}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bundle, out)
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
