"""Gymnasium wrapper that adds per-step DEL predictions to a WindGym env.

On each `reset`/`step`, the wrapper:
  1. Reads the current farm state (wd, ws, ti, turbine positions, hub height)
     from the unwrapped base env.
  2. Runs an internal probe PyWake model twice:
       - once with the WindGym baseline-controller yaws  (info["yaw angles base"])
       - once with the current agent yaws                (env.current_yaw)
  3. Extracts rotor-sector-averaged wind speed and TI at each rotor via
     `helpers.surrogate_loads.sector_averages`.
  4. Feeds both through the ANN DEL surrogate and writes two per-turbine
     arrays into the returned `info` dict:
       - info["loads_baseline"] : DEL under the baseline controller
       - info["loads_current"]  : DEL under the agent's current yaws

The probe PyWake model is entirely independent of whatever backend WindGym is
using internally. This is deliberate: it decouples load inference from the
environment's simulator and lets us use the exact wake/turbulence/deflection
configuration the surrogate was trained against.
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

import gymnasium as gym
import numpy as np

from helpers.surrogate_loads import (
    SurrogateLoadModel,
    Template,
    make_rotor_template,
    sector_averages,
)


def _default_wake_model_factory(wind_turbine):
    """Build the probe PyWake farm model matching test_surrogate.ipynb:
    Blondel_Cathelain_2020 + STF2017TurbulenceModel + JimenezWakeDeflection
    on a UniformSite.
    """
    from py_wake.deflection_models.jimenez import JimenezWakeDeflection
    from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
    from py_wake.site import UniformSite
    from py_wake.turbulence_models import STF2017TurbulenceModel

    site = UniformSite()
    return Blondel_Cathelain_2020(
        site,
        windTurbines=wind_turbine,
        turbulenceModel=STF2017TurbulenceModel(),
        deflectionModel=JimenezWakeDeflection(),
    )


class LoadWrapper(gym.Wrapper):
    """Wrap a WindGym env and expose per-turbine DELs in `info`.

    Parameters
    ----------
    env : gym.Env
        Any WindGym `WindFarmEnv` (or a wrapped version thereof). The
        wrapper walks `env.unwrapped` to reach the base env attributes
        `wd`, `ws`, `ti`, `current_yaw`, `x_pos`, `y_pos`, `turbine`.
    surrogate : SurrogateLoadModel
        Already-loaded DEL surrogate.
    wake_model_factory : callable, optional
        ``wind_turbine -> py_wake.WindFarmModel``. Defaults to the
        Blondel_Cathelain_2020 + STF2017 + Jimenez stack used by
        test_surrogate.ipynb.
    rotor_template : tuple, optional
        Pre-built output of `make_rotor_template`. If None, one is built
        lazily on the first step using `turbine.diameter() / 2`.
    pset : float, default 1.0
        Power setpoint fed to the surrogate for both baseline and current
        passes. Training mean is ~0.925.
    baseline_yaw_key : str, default "yaw angles base"
        Key in the WindGym info dict that holds the baseline controller's
        yaw vector. Per user spec this is what the wrapper uses for the
        baseline load computation.
    """

    def __init__(
        self,
        env: gym.Env,
        surrogate: SurrogateLoadModel,
        wake_model_factory: Optional[Callable[[Any], Any]] = None,
        rotor_template: Optional[Template] = None,
        pset: float = 1.0,
        baseline_yaw_key: str = "yaw angles base",
    ) -> None:
        super().__init__(env)
        self.surrogate = surrogate
        self._wake_model_factory = wake_model_factory or _default_wake_model_factory
        self._rotor_template = rotor_template
        self.pset = float(pset)
        self.baseline_yaw_key = baseline_yaw_key

        base = self._base_env()
        self._wfm = self._wake_model_factory(base.turbine)
        if self._rotor_template is None:
            self._rotor_template = make_rotor_template(base.turbine.diameter() / 2)
        self._hub_height = float(base.turbine.hub_height())

    # -------------------------------------------------------------- internals

    def _base_env(self):
        """Return the innermost env that holds the farm-state attributes."""
        base = self.env
        while hasattr(base, "env") and not hasattr(base, "wd"):
            base = base.env
        return base

    def _compute_loads(self, yaws: np.ndarray) -> np.ndarray:
        """Run the probe farm + surrogate for one yaw vector. (T,) -> (T,)."""
        base = self._base_env()
        x = np.asarray(base.x_pos, dtype=float)
        y = np.asarray(base.y_pos, dtype=float)
        hub_h = np.full(len(x), self._hub_height)

        WS_sec, TI_sec = sector_averages(
            self._wfm,
            x_wt=x, y_wt=y, hub_h=hub_h,
            wd=float(base.wd), ws=float(base.ws), ti=float(base.ti),
            yaw=np.asarray(yaws, dtype=float),
            template=self._rotor_template,
        )
        return self.surrogate.predict(
            WS_sec, TI_sec, yaws=np.asarray(yaws, dtype=float), pset=self.pset
        )

    def _inject_loads(self, info: dict) -> dict:
        base = self._base_env()
        current_yaws = np.asarray(base.current_yaw, dtype=float)

        baseline_yaws = info.get(self.baseline_yaw_key)
        if baseline_yaws is None:
            # Fallback: if the env hasn't produced a baseline vector yet
            # (e.g. some resets don't populate info), assume zeros.
            baseline_yaws = np.zeros_like(current_yaws)
        else:
            baseline_yaws = np.asarray(baseline_yaws, dtype=float).ravel()

        info["loads_baseline"] = self._compute_loads(baseline_yaws)
        info["loads_current"] = self._compute_loads(current_yaws)
        return info

    # ------------------------------------------------------------------- API

    def reset(self, **kwargs) -> Tuple[Any, dict]:
        obs, info = self.env.reset(**kwargs)
        info = self._inject_loads(info)
        return obs, info

    def step(self, action) -> Tuple[Any, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = self._inject_loads(info)
        return obs, reward, terminated, truncated, info
