"""4-sector rotor-disk averaging of wind speed and turbulence intensity.

Used to build inputs for Teodor's DLC12 ANN surrogate, which expects:
    saws_{left,right,up,down}, sati_{left,right,up,down}

Sectoring (rotor disk seen from upstream, wind aligned with x):
    left  = horizontal half y < 0
    right = horizontal half y > 0
    up    = vertical half z > hub_height
    down  = vertical half z < hub_height

If a turbine is yawed, sector axes rotate with the rotor (left = port side
of the rotor frame). saws/sati are area-averaged over each half-disk.

Two query paths:

  sample_disk(flow_callable, ...) -> (4, n_samples) array of (u, TI)
      flow_callable: fn (xs, ys, zs) -> (u_arr, ti_arr) in WORLD frame.

  WindGym integration: see windgym_disk_features() — wraps the above and
  reads the env's PyWake site/flow_field handle.
"""
from __future__ import annotations
from typing import Callable, Tuple

import numpy as np


def _disk_sample_points(rotor_diameter: float,
                          n_radial: int = 4,
                          n_angular: int = 16) -> np.ndarray:
    """Polar-grid sample points on a unit disk, scaled to rotor radius.

    Returns array of shape (n_radial * n_angular, 2) with columns (y_local,
    z_local) where y is horizontal (port-positive), z is vertical (up-positive).
    Uses concentric annular sampling so area-weighting is uniform when each
    point gets weight r_i (radial distance).
    """
    R = rotor_diameter / 2.0
    # Mid-radii of annuli — equal area when n_radial constant.
    r_edges = np.linspace(0, R, n_radial + 1)
    r_mids = 0.5 * (r_edges[:-1] + r_edges[1:])
    weights_r = (r_edges[1:] ** 2 - r_edges[:-1] ** 2)  # area per annulus
    thetas = np.linspace(0, 2 * np.pi, n_angular, endpoint=False)
    pts = []
    weights = []
    for r, w_r in zip(r_mids, weights_r):
        for th in thetas:
            pts.append((r * np.cos(th), r * np.sin(th)))
            weights.append(w_r / n_angular)   # uniform along theta
    return np.asarray(pts), np.asarray(weights)


def _rotate_wind_aligned(local_yz: np.ndarray,
                           yaw_deg: float) -> np.ndarray:
    """Rotate local (y, z) by yaw_deg around vertical axis.

    Returns local frame still with axes (y_along_rotor_normal_horizontal, z).
    For sector assignment we rotate the 2D query points; turbine itself sits
    at origin.
    """
    yaw = np.deg2rad(yaw_deg)
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s], [s, c]])
    return local_yz @ R.T


def disk_sector_features(
    turbine_xy: Tuple[float, float],
    hub_height: float,
    rotor_diameter: float,
    yaw_deg: float,
    flow_uti: Callable[[np.ndarray, np.ndarray, np.ndarray],
                         Tuple[np.ndarray, np.ndarray]],
    n_radial: int = 4,
    n_angular: int = 16,
) -> dict:
    """Compute 4-sector area-averaged u and TI over a turbine's rotor disk.

    Args:
        turbine_xy: (x, y) world position of the turbine [m].
        hub_height: hub elevation [m].
        rotor_diameter: rotor diameter [m].
        yaw_deg: turbine yaw [deg], 0 = facing +x. Positive = nose to +y.
        flow_uti: callable returning (u, TI) at world (xs, ys, zs).
        n_radial, n_angular: sampling density.

    Returns:
        dict with keys saws_left, saws_right, saws_up, saws_down,
        sati_left, sati_right, sati_up, sati_down  (all floats).
    """
    pts_local, w = _disk_sample_points(rotor_diameter, n_radial, n_angular)
    # Rotate disk-local (y, z) to align with yawed rotor.
    rot_yz = _rotate_wind_aligned(pts_local, yaw_deg)
    xs = np.full(len(pts_local), turbine_xy[0], dtype=float)
    ys = turbine_xy[1] + rot_yz[:, 0]
    zs = hub_height + pts_local[:, 1]   # vertical sector unaffected by yaw

    u, ti = flow_uti(xs, ys, zs)
    u = np.asarray(u, dtype=float)
    ti = np.asarray(ti, dtype=float)

    # Sector masks in *rotor-frame* local coords (pre-rotation):
    #   left  = y_local < 0, right = y_local > 0
    #   down  = z_local < 0, up    = z_local > 0  (z_local == pts_local[:,1])
    y_loc = pts_local[:, 0]
    z_loc = pts_local[:, 1]

    def _avg(mask):
        if not mask.any():
            return float("nan")
        return float((u[mask] * w[mask]).sum() / w[mask].sum())

    def _avg_ti(mask):
        if not mask.any():
            return float("nan")
        return float((ti[mask] * w[mask]).sum() / w[mask].sum())

    return {
        "saws_left":  _avg(y_loc < 0),
        "saws_right": _avg(y_loc > 0),
        "saws_up":    _avg(z_loc > 0),
        "saws_down":  _avg(z_loc < 0),
        "sati_left":  _avg_ti(y_loc < 0),
        "sati_right": _avg_ti(y_loc > 0),
        "sati_up":    _avg_ti(z_loc > 0),
        "sati_down":  _avg_ti(z_loc < 0),
    }


def features_to_surrogate_input(per_turbine: list[dict],
                                  pset: list[float],
                                  yaw_deg: list[float]) -> "np.ndarray":
    """Stack n turbines' sector dicts into (n, 10) array in surrogate order.

    Order matches TeodorDLC12Surrogate.INPUT_FEATURES.
    """
    n = len(per_turbine)
    arr = np.zeros((n, 10), dtype=np.float32)
    for i, d in enumerate(per_turbine):
        arr[i, 0] = d["saws_left"]
        arr[i, 1] = d["saws_right"]
        arr[i, 2] = d["saws_up"]
        arr[i, 3] = d["saws_down"]
        arr[i, 4] = d["sati_left"]
        arr[i, 5] = d["sati_right"]
        arr[i, 6] = d["sati_up"]
        arr[i, 7] = d["sati_down"]
        arr[i, 8] = pset[i]
        arr[i, 9] = yaw_deg[i]
    return arr


def windgym_flow_callable(env):
    """Build a flow_uti(xs, ys, zs) callable from a WindGym vector env.

    Tries multiple WindGym APIs in order:
      1. env.unwrapped.get_flow_at(xs, ys, zs)  -> (u, ti)
      2. env.unwrapped.flow_field(xs, ys, zs)   -> (u, ti)
      3. env.unwrapped.site.local_wind(xs, ys, zs) (PyWake fallback)

    If the env does not expose flow sampling, raises with a clear message
    pointing to the WindGym addition needed.
    """
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    for attr in ("get_flow_at", "flow_field", "sample_flow", "rotor_inflow"):
        fn = getattr(raw, attr, None)
        if callable(fn):
            return fn
    site = getattr(raw, "site", None)
    if site is not None and hasattr(site, "local_wind"):
        def _from_site(xs, ys, zs):
            lw = site.local_wind(x_i=xs, y_i=ys, h_i=zs)
            u = np.asarray(lw.WS).flatten()
            ti = np.asarray(lw.TI).flatten()
            return u, ti
        return _from_site
    raise RuntimeError(
        "WindGym env exposes no flow-sampling API. "
        "Add a method get_flow_at(xs, ys, zs) -> (u, TI) "
        "(see helpers/rotor_disk_flow.py for the contract).")
