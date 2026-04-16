"""Reusable rotor-sector averaging and ANN DEL surrogate inference.

Lifted from test_surrogate.ipynb after the validation in
~/.claude/plans/abundant-soaring-lerdorf.md. The notebook and any wrappers
that need per-step fatigue-load predictions should import from here so the
left/right sector convention, the `in_`-prefixed scaler column names, and
the reorder from [top, right, bottom, left] -> [left, right, up, down] are
all defined in exactly one place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from py_wake.flow_map import Points


# -----------------------------------------------------------------------------
# Rotor sampling template
# -----------------------------------------------------------------------------

Template = Tuple[np.ndarray, np.ndarray, np.ndarray]


def make_rotor_template(R: float, n_r: int = 6, n_theta: int = 48) -> Template:
    """Polar sample template over the rotor disc. Build once, reuse across steps.

    Returns
    -------
    s_local, dz, W : arrays
        s_local : (P,) in-plane offset from hub along the rotor plane (m)
        dz      : (P,) vertical offset from hub (m)
        W       : (4, P) row-normalised sector weights for
                  [top, right, bottom, left] — each row sums to 1.
    """
    r = np.linspace(R / n_r / 2, R, n_r)
    th = np.linspace(0, 2 * np.pi, n_theta, endpoint=False)
    R_, T_ = np.meshgrid(r, th, indexing="ij")
    s_local = (R_ * np.sin(T_)).ravel()
    dz = (R_ * np.cos(T_)).ravel()
    area_w = R_.ravel()

    th_deg = (np.rad2deg(T_).ravel() + 360) % 360
    sector_id = np.select(
        [th_deg <= 45, th_deg <= 135, th_deg <= 225, th_deg <= 315],
        [0, 1, 2, 3],
        default=0,
    )
    W = np.zeros((4, s_local.size))
    for k in range(4):
        m = sector_id == k
        W[k, m] = area_w[m] / area_w[m].sum()
    return s_local, dz, W


def sector_averages(
    wfm,
    x_wt: np.ndarray,
    y_wt: np.ndarray,
    hub_h: np.ndarray,
    wd: float,
    ws: float,
    ti: float,
    yaw: np.ndarray,
    template: Template,
    sim_res=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute rotor sector averages of effective wind speed and turbulence.

    Parameters
    ----------
    wfm : py_wake.WindFarmModel
        A PyWake wind-farm model. Only used if `sim_res` is None.
    x_wt, y_wt : (T,) float arrays
        Turbine positions (m).
    hub_h : (T,) float array
        Turbine hub heights (m).
    wd, ws, ti : float
        Scalar wind direction (deg), wind speed (m/s), turbulence intensity.
    yaw : (T,) float array
        Yaw offsets in degrees.
    template : output of `make_rotor_template`
    sim_res : optional PyWake `SimulationResult`
        If provided, reuses it instead of re-running `wfm(...)`.

    Returns
    -------
    WS_sec, TI_sec : (T, 4) float arrays
        Columns ordered [top, right, bottom, left] — the raw sector IDs
        produced by `make_rotor_template`. Call sites should reorder to
        the surrogate's [left, right, up, down] via `SurrogateLoadModel`
        (which applies the verified L/R swap).
    """
    s_local, dz, W = template
    T, P = len(x_wt), s_local.size

    wd_rad = np.deg2rad(wd)
    ex, ey = -np.cos(wd_rad), np.sin(wd_rad)

    xq = (x_wt[:, None] + s_local[None, :] * ex).ravel()
    yq = (y_wt[:, None] + s_local[None, :] * ey).ravel()
    zq = (hub_h[:, None] + dz[None, :]).ravel()

    if sim_res is None:
        sim_res = wfm(x_wt, y_wt, wd=wd, ws=ws, tilt=0, TI=ti, yaw=yaw)
    fm = sim_res.flow_map(Points(x=xq, y=yq, h=zq))

    WS = np.asarray(fm.WS_eff).squeeze().reshape(T, P)
    TI = np.asarray(fm.TI_eff).squeeze().reshape(T, P)

    return WS @ W.T, TI @ W.T


def sector_averages_reordered(
    wfm, x_wt, y_wt, hub_h, wd, ws, ti, yaw, template, sim_res=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Like sector_averages but returns (WS, TI) already in the surrogate's
    input order [left, right, up, down] with the verified L/R swap applied.

    Returns (n_turbines, 4), (n_turbines, 4).
    """
    WS_sec, TI_sec = sector_averages(
        wfm, x_wt, y_wt, hub_h, wd, ws, ti, yaw, template, sim_res
    )
    WS_in = np.stack([WS_sec[:, 1], WS_sec[:, 3], WS_sec[:, 0], WS_sec[:, 2]], axis=1)
    TI_in = np.stack([TI_sec[:, 1], TI_sec[:, 3], TI_sec[:, 0], TI_sec[:, 2]], axis=1)
    return WS_in, TI_in


# -----------------------------------------------------------------------------
# Surrogate wrapper
# -----------------------------------------------------------------------------

# Unprefixed physical names, matching INPUT_VARS in test_surrogate.ipynb.
# The trained scaler stores the same names prefixed with `in_`.
INPUT_VARS = [
    "saws_left", "saws_right", "saws_up", "saws_down",
    "sati_left", "sati_right", "sati_up", "sati_down",
    "pset", "yaw",
]
TRAINING_COLUMNS = [f"in_{name}" for name in INPUT_VARS]


class SurrogateLoadModel:
    """ANN DEL surrogate bundled with its input/output scalers.

    Takes rotor-sector inputs, applies the verified left/right swap and
    [top, right, bottom, left] -> [left, right, up, down] reorder, and
    returns a per-turbine DEL prediction.
    """

    def __init__(
        self,
        model_path: str | Path,
        scaler_in_path: str | Path,
        scaler_out_path: str | Path,
    ) -> None:
        self.model = tf.keras.models.load_model(str(model_path))
        self.scaler_in = joblib.load(str(scaler_in_path))
        self.scaler_out = joblib.load(str(scaler_out_path))

        # Lock in the assumption that our positional reorder matches the
        # training column order. If someone swaps in a differently-trained
        # scaler, this assert fires at load time with a clear message.
        assert list(self.scaler_in.feature_names_in_) == TRAINING_COLUMNS, (
            "Surrogate scaler was fit with unexpected feature names: "
            f"{list(self.scaler_in.feature_names_in_)}. Expected {TRAINING_COLUMNS}."
        )

    # ------------------------------------------------------------------ utils

    @staticmethod
    def _reorder_sectors(WS_sec: np.ndarray, TI_sec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map the raw sector IDs [top=0, right=1, bottom=2, left=3] to the
        surrogate's input order [left, right, up, down].

        The left/right columns are the EMPIRICALLY VERIFIED swap: the surrogate
        was trained with sector 1 (south-at-wd=270) labelled "left" and
        sector 3 (north-at-wd=270) labelled "right", which is the opposite
        of what `plot_rotor_points` shows. See E2 cell in test_surrogate.ipynb
        for the spatial-mirror + yaw-flip evidence.
        """
        WS_in = np.stack(
            [WS_sec[:, 1], WS_sec[:, 3], WS_sec[:, 0], WS_sec[:, 2]], axis=1
        )
        TI_in = np.stack(
            [TI_sec[:, 1], TI_sec[:, 3], TI_sec[:, 0], TI_sec[:, 2]], axis=1
        )
        return WS_in, TI_in

    # --------------------------------------------------------------- predict

    def predict(
        self,
        WS_sec: np.ndarray,
        TI_sec: np.ndarray,
        yaws: np.ndarray,
        pset: float | np.ndarray = 1.0,
    ) -> np.ndarray:
        """Predict DEL for each turbine.

        Parameters
        ----------
        WS_sec, TI_sec : (T, 4) arrays
            Sector-averaged wind speed and TI, columns ordered
            [top, right, bottom, left] — as returned by `sector_averages`.
        yaws : (T,) array
            Yaw offsets in degrees, one per turbine.
        pset : float or (T,) array
            Power setpoint per turbine. Training mean is 0.925.

        Returns
        -------
        loads : (T,) ndarray
            DEL per turbine, in the output scaler's native units.
        """
        WS_in, TI_in = self._reorder_sectors(WS_sec, TI_sec)
        T = WS_in.shape[0]
        if isinstance(pset, (int, float)):
            pset_vec = np.full(T, float(pset))
        else:
            pset_vec = np.asarray(pset, dtype=float).reshape(-1)
            assert pset_vec.shape == (T,), f"pset shape {pset_vec.shape} != ({T},)"
        yaws = np.asarray(yaws, dtype=float).reshape(T, 1)

        X = np.hstack([WS_in, TI_in, pset_vec[:, None], yaws])
        df = pd.DataFrame(X, columns=TRAINING_COLUMNS)
        x_scaled = self.scaler_in.transform(df)
        y_scaled = self.model.predict(x_scaled, verbose=0)
        return self.scaler_out.inverse_transform(y_scaled).ravel()


# -----------------------------------------------------------------------------
# PyTorch port of the Keras DEL surrogate
# -----------------------------------------------------------------------------

import torch
import torch.nn as nn


class TorchDELSurrogate(nn.Module):
    """Pure-PyTorch mirror of the Keras ANN DEL surrogate.

    Architecture (identical to the Keras model):
        Linear(10→128) → LeakyReLU(0.3)
        Linear(128→128) → LeakyReLU(0.3)
        Linear(128→96)  → LeakyReLU(0.3)
        Linear(96→1)

    The forward pass applies input standardisation, runs the ANN, then
    inverse-standardises the output so the caller works in physical units
    (the same convention as ``SurrogateLoadModel.predict``).

    All scaler parameters are stored as registered buffers so they move
    with ``.to(device)`` and are included in ``state_dict`` but are never
    updated by the optimiser.
    """

    def __init__(self) -> None:
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(10, 128),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(128, 128),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(128, 96),
            nn.LeakyReLU(negative_slope=0.3),
            nn.Linear(96, 1),
        )

        # Placeholders — overwritten by ``from_keras``.
        self.register_buffer("mean_in", torch.zeros(10))
        self.register_buffer("scale_in", torch.ones(10))
        self.register_buffer("mean_out", torch.zeros(1))
        self.register_buffer("scale_out", torch.ones(1))

    # ------------------------------------------------------------------ forward

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map raw 10-d physical inputs to DEL predictions.

        Parameters
        ----------
        x : (*, 10) tensor
            Raw (un-scaled) inputs in the same column order as
            ``TRAINING_COLUMNS``.

        Returns
        -------
        del_pred : (*,) tensor
            Predicted DEL in physical units.
        """
        x_scaled = (x - self.mean_in) / self.scale_in
        y_scaled = self.net(x_scaled)
        y = y_scaled * self.scale_out + self.mean_out
        return y.squeeze(-1)

    # ----------------------------------------------------------- factory loader

    @classmethod
    def from_keras(
        cls,
        model_path: str | Path,
        scaler_in_path: str | Path,
        scaler_out_path: str | Path,
    ) -> "TorchDELSurrogate":
        """Load Keras weights + sklearn scalers and return an eval-mode model.

        Parameters
        ----------
        model_path : path to the ``.keras`` checkpoint.
        scaler_in_path, scaler_out_path : paths to joblib-pickled
            ``sklearn.preprocessing.StandardScaler`` objects.

        Returns
        -------
        model : TorchDELSurrogate
            Weights copied from Keras, scaler buffers populated, ``eval()``
            mode set.
        """
        import warnings

        # -- load Keras model (CPU-only, suppress TF noise) ----------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            keras_model = tf.keras.models.load_model(str(model_path))

        # -- load sklearn scalers ------------------------------------------
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scaler_in = joblib.load(str(scaler_in_path))
            scaler_out = joblib.load(str(scaler_out_path))

        # -- build PyTorch model -------------------------------------------
        model = cls()

        # Copy scaler parameters as float32 buffers
        model.mean_in.copy_(torch.tensor(scaler_in.mean_, dtype=torch.float32))
        model.scale_in.copy_(torch.tensor(scaler_in.scale_, dtype=torch.float32))
        model.mean_out.copy_(torch.tensor(scaler_out.mean_, dtype=torch.float32))
        model.scale_out.copy_(torch.tensor(scaler_out.scale_, dtype=torch.float32))

        # Identify the Linear layers in order
        linear_layers = [m for m in model.net if isinstance(m, nn.Linear)]

        # Keras Dense layers come in pairs: (kernel, bias).
        # kernel shape is (in_features, out_features) — PyTorch stores
        # weight as (out_features, in_features), so we transpose.
        dense_layers = [
            layer for layer in keras_model.layers
            if len(layer.get_weights()) == 2  # Dense layers have kernel + bias
        ]
        assert len(dense_layers) == len(linear_layers), (
            f"Expected {len(linear_layers)} Dense layers, got {len(dense_layers)}"
        )

        for pt_linear, keras_dense in zip(linear_layers, dense_layers):
            kernel, bias = keras_dense.get_weights()
            # Keras kernel: (in, out) → PyTorch weight: (out, in)
            pt_linear.weight.data.copy_(
                torch.tensor(kernel.T, dtype=torch.float32)
            )
            pt_linear.bias.data.copy_(
                torch.tensor(bias, dtype=torch.float32)
            )

        model.eval()
        return model
