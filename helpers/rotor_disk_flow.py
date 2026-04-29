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
    """Build a flow_uti(xs, ys, zs) callable from a WindGym env.

    WindGym 19a6644+ exposes `env.fs` (the flow simulator). Supported backends:
      - dynamiks DWMFlowSimulation: `fs.get_windspeed(xyz=(x,y,z))` works for
        arbitrary 3D points.
      - PyWakeFlowSimulationAdapter: only hub-height grid via a View — falls
        back to per-turbine rotor average broadcast across sectors. TI is
        the env-level scalar.

    Returns: callable f(xs, ys, zs) -> (u, ti) where each output is a 1D
    numpy array matching the input length. TI is currently a scalar
    broadcast — extend if WindGym backends gain per-point TI.
    """
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    fs = getattr(raw, "fs", None)
    if fs is None:
        raise RuntimeError(
            "WindGym env has no `.fs` simulator handle (call reset() first?)")

    # Pull a representative TI scalar; sites/configs differ.
    def _ti_scalar():
        for attr in ("ti", "TI", "current_ti"):
            v = getattr(raw, attr, None)
            if v is not None and np.isscalar(v):
                return float(v)
            if hasattr(v, "__len__") and len(v) > 0:
                return float(np.mean(v))
        # Try fs.ti
        v = getattr(fs, "ti", None)
        if v is not None:
            arr = np.asarray(v).flatten()
            if arr.size > 0:
                return float(np.mean(arr))
        return 0.10  # last-ditch default; warn elsewhere

    backend = type(fs).__name__
    if backend == "DWMFlowSimulation" or "DWM" in backend:
        def _dynamiks(xs, ys, zs):
            xs = np.asarray(xs); ys = np.asarray(ys); zs = np.asarray(zs)
            n = xs.size
            u = np.empty(n, dtype=float)
            for i in range(n):
                v = fs.get_windspeed(xyz=(xs[i], ys[i], zs[i]),
                                      include_wakes=True, xarray=False)
                arr = np.asarray(v).flatten()
                # Use magnitude of UVW vector (matches WindProbe.read_speed_magnitude).
                u[i] = float(np.linalg.norm(arr)) if arr.size > 0 else 0.0
            ti = np.full(n, _ti_scalar(), dtype=float)
            return u, ti
        return _dynamiks

    # PyWake adapter fallback: rotor-average broadcast.
    def _pywake_avg(xs, ys, zs):
        n = np.asarray(xs).size
        try:
            avgs = np.asarray(fs.windTurbines.rotor_avg_windspeed).flatten()
        except Exception:
            avgs = np.array([float(getattr(raw, "ws", 9.0))])
        u_mean = float(np.mean(avgs))
        return (np.full(n, u_mean), np.full(n, _ti_scalar()))
    return _pywake_avg


def disk_features_for_env(env, turbine_idx: int,
                            yaw_deg: float | None = None,
                            n_radial: int = 4, n_angular: int = 16) -> dict:
    """Convenience wrapper: 4-sector u/TI for one turbine in a WindGym env.

    Args:
        env: WindGym env (after reset).
        turbine_idx: which turbine.
        yaw_deg: yaw override; if None, reads from env.fs.windTurbines.yaw.
        n_radial, n_angular: sampling density.
    """
    raw = env.unwrapped if hasattr(env, "unwrapped") else env
    fs = raw.fs
    wts = fs.windTurbines
    xy = (float(wts.positions_xyz[0][turbine_idx]),
          float(wts.positions_xyz[1][turbine_idx]))

    def _scalar(obj, names, default):
        for n in names:
            v = getattr(obj, n, None)
            if v is None:
                continue
            if callable(v):
                try:
                    v = v()
                except Exception:
                    continue
            try:
                arr = np.asarray(v).flatten()
                if arr.size == 0:
                    continue
                return float(arr[turbine_idx if arr.size > 1 else 0])
            except Exception:
                continue
        return float(default)

    hub_h = _scalar(wts, ("hub_height", "hub_heights", "hub_height_"), 119.0)
    rd = _scalar(wts, ("rotor_diameter", "diameter", "D"), 178.0)
    if yaw_deg is None:
        yaw_deg = float(np.asarray(wts.yaw).flatten()[turbine_idx])
    flow = windgym_flow_callable(env)
    return disk_sector_features(xy, hub_h, rd, yaw_deg, flow,
                                  n_radial=n_radial, n_angular=n_angular)
