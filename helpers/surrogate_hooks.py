"""Glue between WindGym vector envs and stateful load surrogates that need
per-step rotor-disk flow context (e.g. FlapDELSurrogate, FlapDELBudgetSurrogate).

WindFarmEnv lives inside AsyncVectorEnv subprocesses, so we can't reach
`env.unwrapped.fs` from the main process. The fix:

1. `SectorFlowExposer` wraps WindFarmEnv with a method `get_sector_features()`
   that runs INSIDE the subprocess and returns serializable per-turbine
   sector u/TI + pset.
2. `refresh_surrogate_context(envs, surr)` calls the wrapped method via
   `envs.env.call(...)`, averages over sub-envs (only meaningful for
   num_envs=1; warns otherwise), and feeds the surrogate.
3. `update_surrogate_after_step(envs, surr, infos)` reads yaw_deg from
   `infos["yaw angles agent"]` and accumulates the realized DEL.

Usage in ebt_sac_windfarm.py:
    # at make_env_fn:
    env = SectorFlowExposer(env)
    # after creating the surrogate:
    needs_ctx = hasattr(load_surrogate, "update_context")
    # in training loop:
    if needs_ctx:
        refresh_surrogate_context(envs, load_surrogate)
    # ... agent.act ...
    next_obs, ..., infos = envs.step(actions)
    if needs_ctx:
        update_surrogate_after_step(envs, load_surrogate, infos)
"""
from __future__ import annotations
import warnings
from typing import Any, Dict, List

import numpy as np
import gymnasium as gym


class SectorFlowExposer(gym.Wrapper):
    """Add a `get_sector_features()` method to a WindFarmEnv-bearing env.

    Returns a serialisable dict the parent process consumes.
    """

    def get_layout_info(self) -> Dict[str, np.ndarray]:
        """Per-turbine positions, hub_height, rotor diameter, current yaw."""
        env = self.env
        if hasattr(env, "_get_base_env"):
            raw = env._get_base_env()
        elif hasattr(env, "_current_env"):
            cur = env._current_env
            raw = cur.unwrapped if hasattr(cur, "unwrapped") else cur
        else:
            raw = env.unwrapped if hasattr(env, "unwrapped") else env
        fs = getattr(raw, "fs", None)
        if fs is None:
            return {"err": "no_fs"}
        wts = fs.windTurbines
        pos_x = np.asarray(wts.positions_xyz[0], dtype=np.float32)
        pos_y = np.asarray(wts.positions_xyz[1], dtype=np.float32)
        try:
            hub = wts.hub_height() if callable(wts.hub_height) else wts.hub_height
            hub = float(np.asarray(hub).flatten()[0])
        except Exception:
            hub = 119.0
        try:
            rd = wts.diameter() if callable(wts.diameter) else wts.diameter
            rd = float(np.asarray(rd).flatten()[0])
        except Exception:
            rd = 178.3
        yaw = np.asarray(wts.yaw, dtype=np.float32).flatten()
        return {"pos_x": pos_x, "pos_y": pos_y,
                "hub_height": np.float32(hub),
                "rotor_diameter": np.float32(rd),
                "yaw_deg": yaw}

    def get_renderer_flow_field(self) -> Dict[str, np.ndarray]:
        """Return WindGym renderer's native flow field (uses fs.get_windspeed
        with the proper View object — works for both PyWake and dynamiks
        backends). Initialises the renderer lazily.
        """
        env = self.env
        if hasattr(env, "_get_base_env"):
            raw = env._get_base_env()
        elif hasattr(env, "_current_env"):
            cur = env._current_env
            raw = cur.unwrapped if hasattr(cur, "unwrapped") else cur
        else:
            raw = env.unwrapped if hasattr(env, "unwrapped") else env
        fs = getattr(raw, "fs", None)
        if fs is None:
            return {"err": "no_fs"}
        rdr = getattr(raw, "renderer", None)
        if rdr is None:
            return {"err": "no_renderer"}
        try:
            d = rdr.get_flow_field(fs, turbine=raw.turbine)
        except Exception as e:
            return {"err": f"render_err: {e}"}
        # uvw is xarray; serialise to numpy. Component 0 = u-magnitude.
        uvw = d["uvw"]
        try:
            arr = np.asarray(uvw.values, dtype=np.float32)
        except Exception:
            arr = np.asarray(uvw, dtype=np.float32)
        # Pick u (first component) magnitude. arr shape varies by backend.
        if arr.ndim == 3 and arr.shape[0] == 3:
            U = np.linalg.norm(arr, axis=0)
        elif arr.ndim == 3 and arr.shape[-1] == 3:
            U = np.linalg.norm(arr, axis=-1)
        else:
            U = np.asarray(arr).squeeze()
        return {
            "X": np.asarray(d["x"], dtype=np.float32),
            "Y": np.asarray(d["y"], dtype=np.float32),
            "U": U.astype(np.float32),
            "x_turb": np.asarray(d["x_turb"], dtype=np.float32),
            "y_turb": np.asarray(d["y_turb"], dtype=np.float32),
            "yaw_deg": np.asarray(d["yaw_plot"], dtype=np.float32).flatten(),
            "wd": float(d["wd"]),
            "diameter": float(d["diameter"]),
        }

    def get_flow_grid(self, x_min: float, x_max: float, nx: int,
                       y_min: float, y_max: float, ny: int,
                       hub_h: float) -> Dict[str, np.ndarray]:
        """Sample wind-speed magnitude on a hub-height xy grid INSIDE the
        subprocess (avoids hammering the AsyncVectorEnv pipe)."""
        env = self.env
        if hasattr(env, "_get_base_env"):
            raw = env._get_base_env()
        elif hasattr(env, "_current_env"):
            cur = env._current_env
            raw = cur.unwrapped if hasattr(cur, "unwrapped") else cur
        else:
            raw = env.unwrapped if hasattr(env, "unwrapped") else env
        fs = getattr(raw, "fs", None)
        if fs is None:
            return {"err": "no_fs"}
        xs = np.linspace(x_min, x_max, nx)
        ys = np.linspace(y_min, y_max, ny)
        U = np.zeros((ny, nx), dtype=np.float32)
        n_ok = 0
        first_err = None
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                try:
                    v = fs.get_windspeed(xyz=(float(x), float(y), float(hub_h)),
                                          include_wakes=True, xarray=False)
                    arr = np.asarray(v).flatten()
                    U[i, j] = float(np.linalg.norm(arr)) if arr.size > 0 else float("nan")
                    n_ok += 1
                except Exception as e:
                    U[i, j] = float("nan")
                    if first_err is None:
                        first_err = repr(e)
        return {"xs": xs.astype(np.float32),
                "ys": ys.astype(np.float32),
                "U": U,
                "n_ok": n_ok,
                "first_err": first_err}

    def get_sector_features(self) -> Dict[str, np.ndarray]:
        from helpers.rotor_disk_flow import disk_features_for_env
        # MultiLayoutEnv hides the active WindFarmEnv behind _get_base_env / _current_env.
        env = self.env
        if hasattr(env, "_get_base_env"):
            raw = env._get_base_env()
        elif hasattr(env, "_current_env"):
            cur = env._current_env
            raw = cur.unwrapped if hasattr(cur, "unwrapped") else cur
        else:
            raw = env.unwrapped if hasattr(env, "unwrapped") else env
        fs = getattr(raw, "fs", None)
        if fs is None:
            return {"err": "no_fs"}
        n_turb = int(np.asarray(fs.windTurbines.positions_xyz[0]).shape[0])
        saws = np.zeros((n_turb, 4), dtype=np.float32)
        sati = np.zeros((n_turb, 4), dtype=np.float32)
        for i in range(n_turb):
            try:
                f = disk_features_for_env(raw, turbine_idx=i)
                saws[i] = [f["saws_left"], f["saws_right"],
                            f["saws_up"], f["saws_down"]]
                sati[i] = [f["sati_left"], f["sati_right"],
                            f["sati_up"], f["sati_down"]]
            except Exception:
                saws[i] = 9.0
                sati[i] = 0.07
        # Yaw + pset for `update` after env.step.
        yaw_deg = np.asarray(fs.windTurbines.yaw, dtype=np.float32).flatten()
        # WindGym doesn't expose pset; default to 0.93 (training-data mean).
        pset = np.full(n_turb, 0.93, dtype=np.float32)
        return {"saws": saws, "sati": sati, "pset": pset,
                "yaw_deg": yaw_deg, "n_turb": n_turb}


def _aggregate_features(per_env_results: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Average sector features across sub-envs.

    Only meaningful for num_envs=1; warns otherwise. The surrogate is per-
    turbine but does not natively know which env it belongs to.
    """
    valid = [r for r in per_env_results
             if isinstance(r, dict) and "err" not in r]
    if not valid:
        raise RuntimeError(
            "no sub-env returned sector features (env not initialised yet?)")
    if len(per_env_results) > 1:
        warnings.warn(
            "DEL surrogate context aggregated by mean across "
            f"{len(per_env_results)} envs; recommended num_envs=1 with this "
            "surrogate type.")
    saws = np.mean([r["saws"] for r in valid], axis=0)
    sati = np.mean([r["sati"] for r in valid], axis=0)
    pset = np.mean([r["pset"] for r in valid], axis=0)
    yaw_deg = np.mean([r["yaw_deg"] for r in valid], axis=0)
    return {"saws": saws, "sati": sati, "pset": pset, "yaw_deg": yaw_deg}


def _safe_call(envs, method: str):
    """envs.env.call(method) but tolerant of broken subprocess pipes.

    AsyncVectorEnv subprocess can die during long eval sweeps (Linux pipe
    closure on shutdown / second-reset corner cases). Returns None instead
    of propagating BrokenPipeError so the surrogate hook becomes a no-op.
    """
    try:
        return envs.env.call(method)
    except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as _e:
        return None


def refresh_surrogate_context(envs, surr) -> None:
    """Read 4-sector flow per turbine from each sub-env, feed the surrogate."""
    if not hasattr(surr, "update_context"):
        return
    results = _safe_call(envs, "get_sector_features")
    if results is None:
        return
    feats = _aggregate_features(list(results))
    surr.update_context(feats["saws"], feats["sati"], feats["pset"])


def update_surrogate_after_step(envs, surr, infos) -> None:
    """Accumulate realised DEL using the yaw the env actually applied."""
    if not hasattr(surr, "update"):
        return
    # Re-read flow + yaw to capture post-step state.
    results = _safe_call(envs, "get_sector_features")
    if results is None:
        return
    feats = _aggregate_features(list(results))
    saws, sati, pset = feats["saws"], feats["sati"], feats["pset"]
    # Prefer the info-reported yaw if available (matches what WindGym applied).
    yaw_deg = feats["yaw_deg"]
    if isinstance(infos, dict) and "yaw angles agent" in infos:
        ya = np.array(infos["yaw angles agent"], dtype=np.float32)
        yf = ya[0] if ya.ndim > 1 else ya
        if yf.shape == yaw_deg.shape:
            yaw_deg = yf
    surr.update(saws, sati, pset, yaw_deg)
