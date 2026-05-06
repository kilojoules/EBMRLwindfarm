"""Reward wrapper that subtracts a DEL penalty from the env's base reward.

r_new = r_orig - beta * sum_i(DEL_i) / del_ref

DEL_i is from Teodor's DLC12 sector-averaged surrogate, queried each step
using current 4-sector rotor-disk inflow + per-turbine yaw. This makes the
EBT-SAC actor learn a wake-steering policy that minimises blade-flap DEL
(while still rewarded for power).

Plug after PerTurbineObservationWrapper and before SectorFlowExposer or
MultiLayoutEnv. The wrapper expects to find sector_features by traversing
its own wrapped env, OR (preferred) calls a parent SectorFlowExposer.

Usage in make_env_fn:
    e = WindFarmEnv(...)
    e = PerTurbineObservationWrapper(e)
    e = DelRewardWrapper(e, surrogate, beta=1.0, del_ref=648.6)
    # ... continue wrapping (MultiLayoutEnv etc.)
"""
from __future__ import annotations
from typing import Any

import numpy as np
import torch
import gymnasium as gym


class DelRewardWrapper(gym.Wrapper):
    """Modifies env reward by subtracting a DEL penalty per step.

    Args:
        env: WindFarmEnv (or wrapper around it).
        surrogate: TeodorDLC12Surrogate with output 'wrot_Bl1Rad0FlpMnt'.
        beta: penalty weight; r_new = r - beta * sum(DEL) / del_ref.
        del_ref: normaliser [kNm]; default 648.6 (calibrated mean).
        sensor: which surrogate output to use as cost.
    """

    def __init__(self, env, surrogate, beta: float = 1.0,
                 del_ref: float = 648.6,
                 sensor: str = "wrot_Bl1Rad0FlpMnt",
                 default_pset: float = 0.93):
        super().__init__(env)
        self.surrogate = surrogate
        self.beta = float(beta)
        self.del_ref = float(del_ref)
        self.sensor = sensor
        self.default_pset = float(default_pset)
        # Track running mean DEL for diagnostics
        self._del_running_mean = None
        self._n_steps = 0

    def _resolve_fs(self):
        """Walk wrapper chain to find env.fs (the dynamiks/pywake simulator)."""
        env = self.env
        while env is not None:
            if hasattr(env, "fs"):
                return env.fs
            env = getattr(env, "env", None)
        return None

    def _get_disk_features_for_turbine(self, fs, turbine_idx: int):
        from helpers.rotor_disk_flow import disk_features_for_env
        try:
            base_env = self.env
            while base_env is not None and not hasattr(base_env, "fs"):
                base_env = getattr(base_env, "env", None)
            if base_env is None:
                return None
            return disk_features_for_env(base_env, turbine_idx=turbine_idx)
        except Exception:
            return None

    def step(self, action):
        out = self.env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
        else:
            obs, reward, done, info = out
            terminated, truncated = done, False

        fs = self._resolve_fs()
        if fs is None:
            return obs, reward, terminated, truncated, info

        try:
            n_turb = int(np.asarray(fs.windTurbines.positions_xyz[0]).shape[0])
            yaw_deg = np.asarray(fs.windTurbines.yaw,
                                  dtype=np.float32).flatten()
        except Exception:
            return obs, reward, terminated, truncated, info

        feats = []
        for i in range(n_turb):
            f = self._get_disk_features_for_turbine(fs, i)
            if f is None:
                f = dict(saws_left=9.0, saws_right=9.0, saws_up=9.0,
                          saws_down=9.0, sati_left=0.07, sati_right=0.07,
                          sati_up=0.07, sati_down=0.07)
            feats.append(f)

        x = np.zeros((n_turb, 10), dtype=np.float32)
        for i, f in enumerate(feats):
            x[i, 0] = f["saws_left"];  x[i, 1] = f["saws_right"]
            x[i, 2] = f["saws_up"];    x[i, 3] = f["saws_down"]
            x[i, 4] = f["sati_left"];  x[i, 5] = f["sati_right"]
            x[i, 6] = f["sati_up"];    x[i, 7] = f["sati_down"]
            x[i, 8] = self.default_pset
            x[i, 9] = float(yaw_deg[i]) if i < len(yaw_deg) else 0.0
        with torch.no_grad():
            del_per = self.surrogate.predict_one(
                self.sensor, torch.from_numpy(x)).flatten().numpy()
        del_total = float(del_per.sum())
        penalty = self.beta * del_total / self.del_ref
        reward = float(reward) - penalty

        self._n_steps += 1
        if self._del_running_mean is None:
            self._del_running_mean = del_total
        else:
            a = 0.99
            self._del_running_mean = a * self._del_running_mean + (1 - a) * del_total
        info = dict(info) if isinstance(info, dict) else {"info": info}
        info["del_total_step"] = del_total
        info["del_total_running"] = float(self._del_running_mean)
        info["del_penalty_applied"] = penalty
        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._del_running_mean = None
        self._n_steps = 0
        return self.env.reset(**kwargs)
