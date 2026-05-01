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
