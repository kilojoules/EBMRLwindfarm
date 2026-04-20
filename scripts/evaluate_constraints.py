#!/usr/bin/env python3
"""
Comprehensive post-training evaluation of constraint composition for EBT actors.

Loads a trained EBT checkpoint and systematically sweeps over constraint types,
guidance scales (lambda), steepness values, and thresholds. Records per-turbine
yaw angles, power output, and reward, then computes derived metrics like
cooperative adaptation scores and power ratios.

This is the primary evaluation tool for the hero experiment: train once with
NO constraints, then compose arbitrary constraints at inference and observe
emergent cooperative adaptation (constraining T1 causes T2/T3 to reorganize).

Usage:
    # Hero test: t1_positive_only sweep
    python scripts/evaluate_constraints.py \\
        --checkpoint runs/ebt_histlen_1/checkpoints/step_30000.pt

    # Full sweep with multiple constraint types
    python scripts/evaluate_constraints.py \\
        --checkpoint runs/ebt_histlen_1/checkpoints/step_30000.pt \\
        --constraint-types t1_positive_only,exponential,per_turbine \\
        --lambdas 0.0,0.5,1.0,2.0,5.0,10.0,20.0 \\
        --steepness-values 6.0,10.0

    # Quick sanity check (unconstrained only)
    python scripts/evaluate_constraints.py \\
        --checkpoint runs/ebt_histlen_1/checkpoints/step_30000.pt \\
        --constraint-types none --num-episodes 1
"""

import argparse
import os
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from config import Args
from load_surrogates import (
    create_load_surrogate,
    YawTravelBudgetSurrogate,
)
from helpers.agent import WindFarmAgent
from helpers.surrogate_loads import (
    SurrogateLoadModel,
    make_rotor_template,
    sector_averages_reordered,
)
from helpers.constraint_viz import (
    plot_yaw_trajectory,
    plot_local_energy_landscape,
    plot_yaw_vs_lambda,
    plot_power_vs_lambda,
)


# =============================================================================
# DATA STRUCTURES
# =============================================================================


@dataclass
class SweepConfig:
    """One point in the constraint sweep space."""

    constraint_type: str  # "t1_positive_only", "exponential", etc. or "none"
    lambda_val: float
    steepness: float = 10.0
    threshold_deg: float = 15.0
    per_turbine_thresholds: Optional[List[float]] = None
    del_threshold_pct: float = 0.10
    del_penalty_type: str = "exponential"
    del_context_mode: str = "frozen"
    del_grid_size: int = 21
    label: str = ""

    def __post_init__(self):
        if not self.label:
            self.label = self._auto_label()

    def _auto_label(self) -> str:
        if self.constraint_type == "none":
            return "unconstrained"
        parts = [self.constraint_type, f"lam{self.lambda_val}"]
        if self.constraint_type in ("t1_positive_only", "t1_positive_quadratic", "t1_positive_linear", "exponential", "per_turbine"):
            parts.append(f"k{self.steepness}")
        if self.constraint_type in ("exponential", "threshold"):
            parts.append(f"t{self.threshold_deg}")
        if self.constraint_type == "per_turbine" and self.per_turbine_thresholds:
            thresh_str = "_".join(f"{t:.0f}" for t in self.per_turbine_thresholds)
            parts.append(f"thresh{thresh_str}")
        if self.constraint_type.startswith("del_"):
            parts.append(f"{self.del_threshold_pct:.0%}")
            parts.append(self.del_penalty_type[:3])
            if self.del_context_mode == "precomputed":
                parts.append(f"pc{self.del_grid_size}")
        return "_".join(parts)


@dataclass
class EpisodeResult:
    """Results from a single assessment episode."""

    yaw_trajectory: np.ndarray  # (num_steps, n_turb) in degrees
    power_trajectory: np.ndarray  # (num_steps,)
    reward_trajectory: np.ndarray  # (num_steps,)
    final_yaw: np.ndarray  # (n_turb,)
    mean_yaw: np.ndarray  # (n_turb,) mean across ALL steps
    steady_state_yaw: np.ndarray  # (n_turb,) mean across last N steps
    mean_power: float
    total_reward: float


@dataclass
class ConfigResult:
    """Aggregated results for one SweepConfig across multiple episodes."""

    config: SweepConfig
    episodes: List[EpisodeResult]
    # Aggregated across episodes (using steady-state yaw)
    mean_yaw_per_turbine: np.ndarray  # (n_turb,) steady-state mean
    std_yaw_per_turbine: np.ndarray
    mean_final_yaw: np.ndarray  # (n_turb,) steady-state mean
    mean_power: float
    std_power: float
    mean_reward: float
    std_reward: float
    # Derived metrics (filled post-hoc)
    power_ratio: Optional[float] = None
    cooperative_adaptation_score: Optional[float] = None
    constraint_satisfied: Optional[bool] = None
    convergence_step: Optional[int] = None


# =============================================================================
# CHECKPOINT LOADING
# =============================================================================


def load_ebt_checkpoint(
    path: str,
    device: torch.device,
    yaw_init: str = "zeros",
    ebt_opt_steps_eval_override: Optional[int] = None,
    ebt_langevin_noise_override: Optional[float] = None,
    ebt_num_candidates_override: Optional[int] = None,
) -> Tuple[Any, WindFarmAgent, nn.Module, Args, dict]:
    """
    Load an EBT checkpoint, reconstruct environment and actor.

    Based on scripts/visualize_energy_landscape.py:load_checkpoint_and_setup().

    Args:
        path: Path to .pt checkpoint
        device: Torch device
        yaw_init: "zeros" for deterministic zero init, "random" for random init
        ebt_opt_steps_eval_override: If set, override the checkpoint's
            eval-time EBT optimization step count (default uses checkpoint value).
        ebt_langevin_noise_override: If set, override eval-time Langevin noise.
        ebt_num_candidates_override: If set, override eval-time candidate count.

    Returns:
        (envs, agent, actor, args, env_info)
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    args = Args(**{k: v for k, v in args_dict.items() if hasattr(Args, k)})

    # Inference-time overrides for exploration knobs (Option 1 diagnostics).
    # These mutate the reconstructed args before the actor is built, so the
    # TransformerEBTActor picks them up in the constructor below.
    if ebt_opt_steps_eval_override is not None:
        print(f"  [override] ebt_opt_steps_eval: {args.ebt_opt_steps_eval} → "
              f"{ebt_opt_steps_eval_override}")
        args.ebt_opt_steps_eval = ebt_opt_steps_eval_override
    if ebt_langevin_noise_override is not None:
        print(f"  [override] ebt_langevin_noise: {args.ebt_langevin_noise} → "
              f"{ebt_langevin_noise_override}")
        args.ebt_langevin_noise = ebt_langevin_noise_override
    if ebt_num_candidates_override is not None:
        print(f"  [override] ebt_num_candidates: {args.ebt_num_candidates} → "
              f"{ebt_num_candidates_override}")
        args.ebt_num_candidates = ebt_num_candidates_override

    # Verify this is an EBT checkpoint
    if "ebt_opt_steps_train" not in args_dict:
        raise ValueError(
            "This checkpoint is not from an EBT actor. "
            "Use scripts/demo_per_turbine_constraints.py for diffusion checkpoints."
        )

    from ebt_sac_windfarm import setup_env
    from ebt import TransformerEBTActor
    from networks import create_profile_encoding

    # Override yaw init for evaluation (None = zeros in WindGym)
    config_overrides = None
    if yaw_init == "zeros":
        config_overrides = {"yaw_init": None}

    env_info = setup_env(args, config_overrides=config_overrides)

    # Profile encoders (if model was trained with profiles)
    use_profiles = env_info["use_profiles"]
    shared_recep_encoder, shared_influence_encoder = None, None
    if use_profiles:
        shared_recep_encoder, shared_influence_encoder = create_profile_encoding(
            profile_type=args.profile_encoding_type,
            embed_dim=args.embed_dim,
            hidden_channels=args.profile_encoder_hidden,
        )

    common_kwargs = {
        "obs_dim_per_turbine": env_info["obs_dim_per_turbine"],
        "action_dim_per_turbine": 1,
        "embed_dim": args.embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "pos_encoding_type": args.pos_encoding_type,
        "pos_embed_dim": args.pos_embed_dim,
        "pos_embedding_mode": args.pos_embedding_mode,
        "rel_pos_hidden_dim": args.rel_pos_hidden_dim,
        "rel_pos_per_head": args.rel_pos_per_head,
        "profile_encoding": args.profile_encoding_type,
        "shared_recep_encoder": shared_recep_encoder,
        "shared_influence_encoder": shared_influence_encoder,
        "args": args,
    }

    actor = TransformerEBTActor(
        action_scale=env_info["action_scale"],
        action_bias=env_info["action_bias"],
        energy_hidden_dim=args.ebt_energy_hidden_dim,
        energy_num_layers=args.ebt_energy_num_layers,
        opt_steps_train=args.ebt_opt_steps_train,
        opt_steps_eval=args.ebt_opt_steps_eval,
        opt_lr=args.ebt_opt_lr,
        num_candidates=args.ebt_num_candidates,
        langevin_noise=args.ebt_langevin_noise,
        random_steps=args.ebt_random_steps,
        random_lr=args.ebt_random_lr,
        **common_kwargs,
    ).to(device)

    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    agent = WindFarmAgent(
        actor=actor,
        device=device,
        rotor_diameter=env_info["rotor_diameter"],
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=getattr(args, "rotate_profiles", False),
    )

    return env_info["envs"], agent, actor, args, env_info


# =============================================================================
# SWEEP CONFIGURATION
# =============================================================================


def build_sweep_configs(
    constraint_types: List[str],
    lambdas: List[float],
    steepness_values: List[float],
    threshold_values: List[float],
    n_turbines: int,
    del_threshold_pct: float = 0.10,
    del_penalty_type: str = "exponential",
    del_context_mode: str = "frozen",
    del_grid_size: int = 21,
) -> List[SweepConfig]:
    """Build the full list of configurations to sweep."""
    configs: List[SweepConfig] = []

    # Always include unconstrained baseline
    configs.append(SweepConfig(constraint_type="none", lambda_val=0.0))

    for ctype in constraint_types:
        if ctype == "none":
            continue

        for lam in lambdas:
            if lam == 0.0:
                continue  # covered by unconstrained baseline

            if ctype in ("t1_positive_only", "t1_positive_quadratic", "t1_positive_linear"):
                for steep in steepness_values:
                    configs.append(
                        SweepConfig(
                            constraint_type=ctype,
                            lambda_val=lam,
                            steepness=steep,
                        )
                    )

            elif ctype == "exponential":
                for steep in steepness_values:
                    for thresh in threshold_values:
                        configs.append(
                            SweepConfig(
                                constraint_type=ctype,
                                lambda_val=lam,
                                steepness=steep,
                                threshold_deg=thresh,
                            )
                        )

            elif ctype == "threshold":
                for thresh in threshold_values:
                    configs.append(
                        SweepConfig(
                            constraint_type=ctype,
                            lambda_val=lam,
                            threshold_deg=thresh,
                        )
                    )

            elif ctype == "per_turbine":
                # Predefined heterogeneous configs
                per_turb_options = [
                    [5.0] + [20.0] * (n_turbines - 1),
                    [10.0] + [30.0] * (n_turbines - 1),
                ]
                for thresholds in per_turb_options:
                    for steep in steepness_values:
                        configs.append(
                            SweepConfig(
                                constraint_type=ctype,
                                lambda_val=lam,
                                steepness=steep,
                                per_turbine_thresholds=thresholds,
                            )
                        )

            elif ctype == "relu":
                configs.append(
                    SweepConfig(constraint_type=ctype, lambda_val=lam)
                )

            elif ctype == "travel_budget":
                configs.append(
                    SweepConfig(constraint_type=ctype, lambda_val=lam)
                )

            elif ctype.startswith("del_"):
                for steep in steepness_values:
                    configs.append(
                        SweepConfig(
                            constraint_type=ctype,
                            lambda_val=lam,
                            steepness=steep,
                            del_threshold_pct=del_threshold_pct,
                            del_penalty_type=del_penalty_type,
                            del_context_mode=del_context_mode,
                            del_grid_size=del_grid_size,
                        )
                    )

    return configs


def create_surrogate_from_config(
    config: SweepConfig, device: torch.device
) -> Optional[nn.Module]:
    """Instantiate the appropriate surrogate for a given config."""
    if config.constraint_type == "none":
        return None

    if config.constraint_type == "travel_budget":
        surrogate = YawTravelBudgetSurrogate(
            budget_deg=100.0,
            window_steps=100,
            yaw_max_deg=30.0,
            steepness=config.steepness,
        )
        return surrogate.to(device)

    if config.constraint_type.startswith("del_"):
        surrogate_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "surrogate",
        )
        surrogate = create_load_surrogate(
            surrogate_type=config.constraint_type,
            steepness=config.steepness,
            yaw_max_deg=30.0,
            del_model_dir=surrogate_dir,
            del_threshold_pct=config.del_threshold_pct,
            del_penalty_type=config.del_penalty_type,
            del_context_mode=config.del_context_mode,
            del_grid_size=config.del_grid_size,
        )
        return surrogate.to(device)

    surrogate = create_load_surrogate(
        surrogate_type=config.constraint_type,
        steepness=config.steepness,
        threshold_deg=config.threshold_deg,
        yaw_max_deg=30.0,
        per_turbine_thresholds=(
            ",".join(str(t) for t in config.per_turbine_thresholds)
            if config.per_turbine_thresholds
            else ""
        ),
    )
    return surrogate.to(device)


# =============================================================================
# DEL CONTEXT UPDATER (side-channel for vectorized envs)
# =============================================================================


class DELContextUpdater:
    """Computes DEL constraint context from the vectorized env state.

    Builds its own probe PyWake farm model (matching the surrogate's training
    setup) and rotor template. Supports both DELConstraintSurrogate context
    modes:

    - ``frozen``  → refreshes (sector_avgs, baseline_dels) every step.
    - ``precomputed`` → builds a full ``grid_size**n_turbines`` DEL grid once
      per episode reset and installs it via ``set_context_precomputed``.
      Step-time updates are no-ops in this mode.
    """

    def __init__(self, wind_turbine, keras_surrogate: SurrogateLoadModel):
        from py_wake.deflection_models.jimenez import JimenezWakeDeflection
        from py_wake.literature.gaussian_models import Blondel_Cathelain_2020
        from py_wake.site import UniformSite
        from py_wake.turbulence_models import STF2017TurbulenceModel

        self._wfm = Blondel_Cathelain_2020(
            UniformSite(),
            windTurbines=wind_turbine,
            turbulenceModel=STF2017TurbulenceModel(),
            deflectionModel=JimenezWakeDeflection(),
        )
        self._template = make_rotor_template(wind_turbine.diameter() / 2)
        self._hub_h = float(wind_turbine.hub_height())
        self._keras_surr = keras_surrogate
        # In-memory cache of the precomputed DEL grid. Key = (wd, ws, ti,
        # x_pos, y_pos, G, yaw_range). Reused across episodes and configs
        # whenever the wind state and layout are unchanged.
        self._grid_cache: Optional[Tuple[Any, np.ndarray, np.ndarray]] = None

    def _get_env_state(self, envs):
        """Read scalar env state from the first sub-env of a vectorized env."""
        get = envs.env.get_attr
        wd = float(get("wd")[0])
        ws = float(get("ws")[0])
        ti = float(get("ti")[0])
        x = np.asarray(get("x_pos")[0], dtype=float)
        y = np.asarray(get("y_pos")[0], dtype=float)
        yaw = np.asarray(get("current_yaw")[0], dtype=float)
        return wd, ws, ti, x, y, yaw

    def compute_context(self, envs) -> tuple:
        """Compute (sector_avgs, baseline_dels) from current env state.

        Returns numpy arrays: sector_avgs (n_turb, 8), baseline_dels (n_turb,).
        """
        wd, ws, ti, x, y, current_yaw = self._get_env_state(envs)
        hub_h = np.full(len(x), self._hub_h)

        # Sector averages for the current state
        WS_in, TI_in = sector_averages_reordered(
            self._wfm, x, y, hub_h, wd, ws, ti,
            yaw=current_yaw, template=self._template,
        )
        sector_avgs = np.hstack([WS_in, TI_in])  # (n_turb, 8)

        # Baseline DELs using zero yaws
        baseline_yaws = np.zeros_like(current_yaw)
        WS_bl, TI_bl = sector_averages_reordered(
            self._wfm, x, y, hub_h, wd, ws, ti,
            yaw=baseline_yaws, template=self._template,
        )
        baseline_dels = self._keras_surr.predict(
            WS_bl, TI_bl, yaws=baseline_yaws, pset=1.0,
        )
        return sector_avgs, baseline_dels

    def _build_wfm_fn(self, wd: float, ws: float, ti: float,
                      x: np.ndarray, y: np.ndarray, hub_h: np.ndarray):
        """Closure that maps ``yaw_deg (n_turb,) -> DEL (n_turb,)``."""
        def wfm_fn(yaw_deg: np.ndarray) -> np.ndarray:
            WS_in, TI_in = sector_averages_reordered(
                self._wfm, x, y, hub_h, wd, ws, ti,
                yaw=yaw_deg, template=self._template,
            )
            return self._keras_surr.predict(
                WS_in, TI_in, yaws=yaw_deg, pset=1.0,
            )
        return wfm_fn

    def rebuild_precomputed_grid(self, surrogate, envs, device, verbose: bool = False):
        """Build the full N-dim DEL grid for the current wind state and install it.

        Called at episode reset. The cost is ``grid_size ** n_turbines`` PyWake
        sims — small farms only.

        Cached: re-uses the previous grid if wind state, layout, grid size and
        yaw range are all identical to the last rebuild. The cache survives
        across episodes and across sweep configs.
        """
        from load_surrogates import build_del_grid

        wd, ws, ti, x, y, _ = self._get_env_state(envs)
        hub_h = np.full(len(x), self._hub_h)
        n_turb = int(len(x))
        yaw_max = float(getattr(surrogate, "yaw_max_deg", 30.0))
        G = int(getattr(surrogate, "grid_size", 21))
        yaw_range = (-yaw_max, +yaw_max)

        cache_key = (
            round(wd, 3), round(ws, 3), round(ti, 4),
            tuple(np.round(x, 3).tolist()), tuple(np.round(y, 3).tolist()),
            G, yaw_range,
        )

        if self._grid_cache is not None and self._grid_cache[0] == cache_key:
            _, grid_np, baseline_dels_np = self._grid_cache
            print(f"  [DEL grid cache HIT] skipping {G}^{n_turb} sim rebuild")
        else:
            wfm_fn = self._build_wfm_fn(wd, ws, ti, x, y, hub_h)
            print(f"  [DEL grid cache MISS] building {G}^{n_turb} = "
                  f"{G**n_turb} PyWake sims...")
            grid_np = build_del_grid(
                wfm_fn=wfm_fn,
                n_turbines=n_turb,
                grid_size=G,
                yaw_range_deg=yaw_range,
                verbose=verbose,
            )
            baseline_dels_np = wfm_fn(np.zeros(n_turb, dtype=float))
            self._grid_cache = (cache_key, grid_np, baseline_dels_np)

        surrogate.set_context_precomputed(
            del_grid=torch.tensor(grid_np, device=device, dtype=torch.float32),
            baseline_dels=torch.tensor(baseline_dels_np, device=device, dtype=torch.float32),
            yaw_range_deg=yaw_range,
        )

    def update_constraint(self, surrogate, envs, device):
        """Update surrogate context. Frozen: every step. Precomputed: no-op."""
        if getattr(surrogate, "context_mode", "frozen") == "precomputed":
            return  # grid is refreshed via rebuild_precomputed_grid at reset
        if not hasattr(surrogate, "set_context"):
            return
        sector_avgs, baseline_dels = self.compute_context(envs)
        surrogate.set_context(
            sector_avgs=torch.tensor(sector_avgs, device=device, dtype=torch.float32),
            baseline_dels=torch.tensor(baseline_dels, device=device, dtype=torch.float32),
        )

    def update_on_reset(self, surrogate, envs, device, verbose: bool = False):
        """Refresh context at episode reset. Rebuilds precomputed grid if needed."""
        if getattr(surrogate, "context_mode", "frozen") == "precomputed":
            self.rebuild_precomputed_grid(surrogate, envs, device, verbose=verbose)
        else:
            self.update_constraint(surrogate, envs, device)


# =============================================================================
# CORE LOOP
# =============================================================================


def run_constrained_episodes(
    agent: WindFarmAgent,
    envs: Any,
    surrogate: Optional[nn.Module],
    lambda_val: float,
    num_episodes: int,
    num_steps: int,
    device: torch.device,
    is_stateful: bool = False,
    steady_state_steps: int = 30,
    del_context_updater: Optional["DELContextUpdater"] = None,
) -> List[EpisodeResult]:
    """Run multiple episodes with a given constraint configuration."""
    results = []

    for ep in range(num_episodes):
        obs, _ = envs.reset()
        if is_stateful and surrogate is not None:
            surrogate.reset()

        # Initial DEL context update (uses env state right after reset).
        # In precomputed mode, this rebuilds the full N-dim DEL grid for the
        # new episode's wind conditions — potentially minutes of PyWake work.
        if del_context_updater is not None and surrogate is not None:
            del_context_updater.update_on_reset(surrogate, envs, device)

        yaw_history: List[np.ndarray] = []
        power_history: List[float] = []
        reward_history: List[float] = []

        for step in range(num_steps):
            gfn = surrogate if lambda_val > 0 else None

            with torch.no_grad():
                act = agent.act(
                    envs, obs, guidance_fn=gfn, guidance_scale=lambda_val
                )

            obs, rew, _, _, info = envs.step(act)
            reward_history.append(float(np.mean(rew)))

            # Update DEL constraint context after each env step
            if del_context_updater is not None and surrogate is not None:
                del_context_updater.update_constraint(surrogate, envs, device)

            if "yaw angles agent" in info:
                yaw = np.array(info["yaw angles agent"])
                yaw_flat = yaw[0] if yaw.ndim > 1 else yaw
                yaw_history.append(yaw_flat)

                if is_stateful and surrogate is not None:
                    yaw_t = torch.tensor(
                        yaw_flat, device=device, dtype=torch.float32
                    )
                    surrogate.update(yaw_t)

            if "Power agent" in info:
                power_history.append(float(np.mean(info["Power agent"])))

        yaw_arr = np.array(yaw_history) if yaw_history else np.array([])
        power_arr = np.array(power_history) if power_history else np.array([])
        reward_arr = np.array(reward_history)

        # Steady-state: mean over last N steps
        if yaw_arr.size and len(yaw_arr) > steady_state_steps:
            ss_yaw = yaw_arr[-steady_state_steps:].mean(axis=0)
        elif yaw_arr.size:
            ss_yaw = yaw_arr.mean(axis=0)
        else:
            ss_yaw = np.array([])

        results.append(
            EpisodeResult(
                yaw_trajectory=yaw_arr,
                power_trajectory=power_arr,
                reward_trajectory=reward_arr,
                final_yaw=yaw_arr[-1] if yaw_arr.size else np.array([]),
                mean_yaw=yaw_arr.mean(axis=0) if yaw_arr.size else np.array([]),
                steady_state_yaw=ss_yaw,
                mean_power=float(power_arr.mean()) if power_arr.size else 0.0,
                total_reward=float(reward_arr.sum()),
            )
        )

    return results


def aggregate_episodes(
    config: SweepConfig, episodes: List[EpisodeResult]
) -> ConfigResult:
    """Aggregate episode-level results into a ConfigResult."""
    ss_yaws = np.array([ep.steady_state_yaw for ep in episodes])
    powers = np.array([ep.mean_power for ep in episodes])
    rewards = np.array([ep.total_reward for ep in episodes])

    return ConfigResult(
        config=config,
        episodes=episodes,
        mean_yaw_per_turbine=ss_yaws.mean(axis=0),
        std_yaw_per_turbine=ss_yaws.std(axis=0),
        mean_final_yaw=ss_yaws.mean(axis=0),
        mean_power=float(powers.mean()),
        std_power=float(powers.std()),
        mean_reward=float(rewards.mean()),
        std_reward=float(rewards.std()),
    )


# =============================================================================
# DERIVED METRICS
# =============================================================================


def compute_derived_metrics(
    result: ConfigResult, baseline: ConfigResult
) -> None:
    """Compute derived metrics in-place, relative to the unconstrained baseline."""
    # Power ratio
    if baseline.mean_power > 0:
        result.power_ratio = result.mean_power / baseline.mean_power

    # Cooperative adaptation: how much unconstrained turbines shift
    baseline_yaw = baseline.mean_yaw_per_turbine
    current_yaw = result.mean_yaw_per_turbine
    yaw_shifts = np.abs(current_yaw - baseline_yaw)
    # T1 (index 0) is typically the constrained turbine; measure T2+ shift
    if len(yaw_shifts) > 1:
        result.cooperative_adaptation_score = float(yaw_shifts[1:].mean())
    else:
        result.cooperative_adaptation_score = 0.0

    # Constraint satisfaction (type-specific)
    ctype = result.config.constraint_type
    final_yaw = result.mean_final_yaw
    if ctype == "t1_positive_only":
        result.constraint_satisfied = bool(final_yaw[0] > 0)
    elif ctype in ("exponential", "threshold"):
        thresh = result.config.threshold_deg
        result.constraint_satisfied = bool(np.all(np.abs(final_yaw) <= thresh + 1.0))
    elif ctype == "per_turbine" and result.config.per_turbine_thresholds:
        thresholds = np.array(result.config.per_turbine_thresholds)
        result.constraint_satisfied = bool(
            np.all(np.abs(final_yaw[: len(thresholds)]) <= thresholds + 1.0)
        )
    elif ctype == "none":
        result.constraint_satisfied = True

    # Convergence step: first step where all turbines are within 2 deg of final
    if result.episodes:
        traj = result.episodes[0].yaw_trajectory
        if traj.size:
            final = traj[-1]
            for t in range(len(traj)):
                if np.all(np.abs(traj[t] - final) < 2.0):
                    result.convergence_step = t
                    break


# =============================================================================
# RESULTS OUTPUT
# =============================================================================


def save_results(results: List[ConfigResult], output_dir: str) -> str:
    """Save results as xarray Dataset (.nc) and CSV summary."""
    import xarray as xr

    labels = [r.config.label for r in results]
    n_turb = len(results[0].mean_yaw_per_turbine)

    ds = xr.Dataset(
        data_vars={
            "mean_yaw": (
                ["config", "turbine"],
                np.array([r.mean_yaw_per_turbine for r in results]),
            ),
            "std_yaw": (
                ["config", "turbine"],
                np.array([r.std_yaw_per_turbine for r in results]),
            ),
            "mean_final_yaw": (
                ["config", "turbine"],
                np.array([r.mean_final_yaw for r in results]),
            ),
            "mean_power": (["config"], [r.mean_power for r in results]),
            "std_power": (["config"], [r.std_power for r in results]),
            "mean_reward": (["config"], [r.mean_reward for r in results]),
            "std_reward": (["config"], [r.std_reward for r in results]),
            "power_ratio": (
                ["config"],
                [r.power_ratio if r.power_ratio is not None else np.nan for r in results],
            ),
            "coop_adapt_score": (
                ["config"],
                [r.cooperative_adaptation_score or 0.0 for r in results],
            ),
            "convergence_step": (
                ["config"],
                [r.convergence_step if r.convergence_step is not None else -1 for r in results],
            ),
            "constraint_satisfied": (
                ["config"],
                [r.constraint_satisfied if r.constraint_satisfied is not None else False for r in results],
            ),
            "lambda_val": (["config"], [r.config.lambda_val for r in results]),
            "steepness": (["config"], [r.config.steepness for r in results]),
            "constraint_type": (["config"], [r.config.constraint_type for r in results]),
        },
        coords={
            "config": labels,
            "turbine": np.arange(n_turb),
        },
    )

    nc_path = os.path.join(output_dir, "constraint_sweep_results.nc")
    ds.to_netcdf(nc_path)
    print(f"  Saved xarray Dataset: {nc_path}")

    # CSV summary
    csv_path = os.path.join(output_dir, "summary.csv")
    with open(csv_path, "w") as f:
        header = ["config", "constraint_type", "lambda", "steepness"]
        header += [f"T{t}_yaw" for t in range(n_turb)]
        header += ["power", "power_ratio", "coop_score", "satisfied", "converge_step"]
        f.write(",".join(header) + "\n")

        for r in results:
            row = [
                r.config.label,
                r.config.constraint_type,
                f"{r.config.lambda_val:.1f}",
                f"{r.config.steepness:.1f}",
            ]
            for t in range(n_turb):
                row.append(f"{r.mean_final_yaw[t]:.1f}")
            row.append(f"{r.mean_power:.0f}")
            row.append(f"{r.power_ratio:.3f}" if r.power_ratio is not None else "")
            row.append(f"{r.cooperative_adaptation_score:.2f}" if r.cooperative_adaptation_score is not None else "")
            row.append(str(r.constraint_satisfied) if r.constraint_satisfied is not None else "")
            row.append(str(r.convergence_step) if r.convergence_step is not None else "")
            f.write(",".join(row) + "\n")

    print(f"  Saved CSV summary: {csv_path}")
    return nc_path


def print_summary_table(results: List[ConfigResult], baseline: ConfigResult) -> None:
    """Print a formatted summary table to the console."""
    n_turb = len(baseline.mean_final_yaw)

    # Header
    turb_headers = "  ".join(f"{'T' + str(t) + ' yaw':>8s}" for t in range(n_turb))
    print(f"\n{'='*110}")
    print(f"  {'Config':<35s} | {turb_headers} | {'Power':>10s} | {'Pwr%':>6s} | {'Coop':>6s} | {'Ok?':>4s}")
    print(f"{'_'*110}")

    for r in results:
        yaw_strs = "  ".join(f"{r.mean_final_yaw[t]:+8.1f}" for t in range(n_turb))
        pwr_pct = (
            f"{(r.power_ratio - 1) * 100:+5.1f}%"
            if r.power_ratio is not None
            else "  ---"
        )
        coop = (
            f"{r.cooperative_adaptation_score:6.2f}"
            if r.cooperative_adaptation_score is not None
            else "   ---"
        )
        sat = ""
        if r.constraint_satisfied is not None:
            sat = " yes" if r.constraint_satisfied else "  NO"

        print(
            f"  {r.config.label:<35s} | {yaw_strs} | "
            f"{r.mean_power:10.0f} | {pwr_pct:>6s} | {coop:>6s} | {sat:>4s}"
        )

    print(f"{'='*110}\n")


# =============================================================================
# FIGURE GENERATION
# =============================================================================


def plot_cooperative_heatmap(
    results: List[ConfigResult],
    baseline: ConfigResult,
    constraint_type: str,
    output_dir: str,
) -> None:
    """
    Heatmap: x=lambda, y=turbine, color=yaw shift from unconstrained baseline.

    Shows how each turbine's yaw changes as constraint strength increases.
    """
    # Filter results for this constraint type
    filtered = [
        r for r in results
        if r.config.constraint_type == constraint_type
    ]
    if not filtered:
        return

    # Group by lambda (take first steepness/threshold combo for simplicity)
    seen_lambdas = {}
    for r in filtered:
        lam = r.config.lambda_val
        if lam not in seen_lambdas:
            seen_lambdas[lam] = r

    lambdas = sorted(seen_lambdas.keys())
    if not lambdas:
        return

    n_turb = len(baseline.mean_final_yaw)
    shifts = np.zeros((n_turb, len(lambdas)))
    for j, lam in enumerate(lambdas):
        r = seen_lambdas[lam]
        shifts[:, j] = r.mean_final_yaw - baseline.mean_final_yaw

    fig, ax = plt.subplots(figsize=(10, 4))
    im = ax.imshow(shifts, aspect="auto", cmap="RdBu_r", origin="lower")
    ax.set_xticks(range(len(lambdas)))
    ax.set_xticklabels([f"{l:.1f}" for l in lambdas], fontsize=9)
    ax.set_yticks(range(n_turb))
    ax.set_yticklabels([f"T{t}" for t in range(n_turb)])
    ax.set_xlabel("Guidance scale $\\lambda$")
    ax.set_ylabel("Turbine")
    ax.set_title(f"Yaw Shift from Baseline ({constraint_type})")

    # Annotate cells
    for i in range(n_turb):
        for j in range(len(lambdas)):
            ax.text(
                j, i, f"{shifts[i, j]:+.1f}",
                ha="center", va="center", fontsize=8,
                color="white" if abs(shifts[i, j]) > 10 else "black",
            )

    fig.colorbar(im, ax=ax, label="Yaw shift (deg)", shrink=0.8)
    fig.tight_layout()

    path = os.path.join(output_dir, f"cooperative_heatmap_{constraint_type}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_summary_bars(
    results: List[ConfigResult],
    output_dir: str,
    reference_lambda: float = 5.0,
) -> None:
    """
    Grouped bar chart comparing constraint types at a fixed lambda.

    Shows power ratio and cooperative adaptation score side-by-side.
    """
    # Pick one result per constraint type at the reference lambda
    type_results = {}
    for r in results:
        if (
            r.config.constraint_type != "none"
            and r.config.lambda_val == reference_lambda
        ):
            if r.config.constraint_type not in type_results:
                type_results[r.config.constraint_type] = r

    if not type_results:
        # Try any available lambda
        for r in results:
            if r.config.constraint_type != "none":
                if r.config.constraint_type not in type_results:
                    type_results[r.config.constraint_type] = r

    if not type_results:
        return

    names = list(type_results.keys())
    power_ratios = [
        type_results[n].power_ratio if type_results[n].power_ratio is not None else 1.0
        for n in names
    ]
    coop_scores = [
        type_results[n].cooperative_adaptation_score or 0.0 for n in names
    ]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    x = np.arange(len(names))
    ax1.bar(x, power_ratios, color="C0", alpha=0.8)
    ax1.axhline(1.0, color="gray", linestyle="--", alpha=0.5)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax1.set_ylabel("Power Ratio (vs unconstrained)")
    ax1.set_title(f"Power Cost ($\\lambda={reference_lambda}$)")
    ax1.grid(True, alpha=0.2)

    ax2.bar(x, coop_scores, color="C1", alpha=0.8)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=30, ha="right", fontsize=9)
    ax2.set_ylabel("Mean |yaw shift| of unconstrained turbines (deg)")
    ax2.set_title(f"Cooperative Adaptation ($\\lambda={reference_lambda}$)")
    ax2.grid(True, alpha=0.2)

    fig.tight_layout()
    path = os.path.join(output_dir, "summary_bars.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def generate_figures(
    results: List[ConfigResult],
    baseline: ConfigResult,
    agent: WindFarmAgent,
    envs: Any,
    actor: nn.Module,
    output_dir: str,
    device: torch.device,
    cli: argparse.Namespace,
) -> None:
    """Generate and save all figures."""
    fig_dir = os.path.join(output_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    constraint_types_used = sorted(
        set(r.config.constraint_type for r in results if r.config.constraint_type != "none")
    )

    # --- Per constraint type: yaw vs lambda, power vs lambda ---
    for ctype in constraint_types_used:
        # Find a representative surrogate for this type
        rep_config = next(
            r.config for r in results if r.config.constraint_type == ctype
        )
        surrogate = create_surrogate_from_config(rep_config, device)
        if surrogate is None:
            continue

        # Extract lambda values used for this type
        lambdas_for_type = sorted(
            set(
                r.config.lambda_val
                for r in results
                if r.config.constraint_type == ctype
            )
        )
        lambdas_for_type = [0.0] + [l for l in lambdas_for_type if l > 0]

        print(f"  Generating yaw_vs_lambda for {ctype}...")
        fig = plot_yaw_vs_lambda(
            agent, envs, surrogate, lambdas_for_type, cli.num_steps, device
        )
        path = os.path.join(fig_dir, f"yaw_vs_lambda_{ctype}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

        print(f"  Generating power_vs_lambda for {ctype}...")
        fig = plot_power_vs_lambda(
            agent, envs, surrogate, lambdas_for_type, cli.num_steps, device
        )
        path = os.path.join(fig_dir, f"power_vs_lambda_{ctype}.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: {path}")

    # --- Yaw trajectory for hero constraint (first constraint type) ---
    if constraint_types_used:
        hero_type = constraint_types_used[0]
        hero_config = next(
            r.config for r in results if r.config.constraint_type == hero_type
        )
        hero_surrogate = create_surrogate_from_config(hero_config, device)
        if hero_surrogate is not None:
            print(f"  Generating yaw trajectory for {hero_type}...")
            traj_lambdas = [0.0, 1.0, 5.0, 10.0]
            fig = plot_yaw_trajectory(
                agent, envs, hero_surrogate, traj_lambdas, cli.num_steps, device
            )
            path = os.path.join(fig_dir, f"yaw_trajectory_{hero_type}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"  Saved: {path}")

    # --- Energy landscape (EBT only) ---
    if hasattr(actor, "compute_energy") and constraint_types_used:
        hero_type = constraint_types_used[0]
        hero_config = next(
            r.config for r in results if r.config.constraint_type == hero_type
        )
        hero_surrogate = create_surrogate_from_config(hero_config, device)
        if hero_surrogate is not None:
            print("  Generating energy landscape...")
            obs, _ = envs.reset()
            batch = agent.batch_preparer.from_envs(envs, obs)

            with torch.no_grad():
                current_act = agent.act(envs, obs)
            current_act_t = torch.tensor(
                current_act, device=device, dtype=torch.float32
            ).unsqueeze(-1)[:1]

            for lam_val in [1.0, 5.0, 10.0]:
                fig = plot_local_energy_landscape(
                    actor,
                    batch.obs,
                    batch.positions,
                    batch.mask,
                    hero_surrogate,
                    lam_val,
                    current_act_t,
                    recep_profile=batch.receptivity,
                    influence_profile=batch.influence,
                    grid_res=cli.grid_res,
                )
                if fig is not None:
                    path = os.path.join(
                        fig_dir, f"energy_landscape_{hero_type}_lam{lam_val}.png"
                    )
                    fig.savefig(path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                    print(f"  Saved: {path}")

    # --- Cooperative adaptation heatmap ---
    for ctype in constraint_types_used:
        print(f"  Generating cooperative heatmap for {ctype}...")
        plot_cooperative_heatmap(results, baseline, ctype, fig_dir)

    # --- Summary bars ---
    print("  Generating summary bars...")
    plot_summary_bars(results, fig_dir)


# =============================================================================
# MAIN
# =============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Post-training constraint composition sweep for EBT actors",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to EBT .pt checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        default="results/constraint_eval",
        help="Output directory for results and figures (default: results/constraint_eval)",
    )
    parser.add_argument(
        "--lambdas",
        type=str,
        default="0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0",
        help="Comma-separated lambda values (default: 0.0,0.1,0.5,1.0,2.0,5.0,10.0,20.0)",
    )
    parser.add_argument(
        "--steepness-values",
        type=str,
        default="6.0",
        help="Comma-separated steepness values (default: 6.0)",
    )
    parser.add_argument(
        "--threshold-values",
        type=str,
        default="5,10,15,20",
        help="Comma-separated threshold degrees (default: 5,10,15,20)",
    )
    parser.add_argument(
        "--constraint-types",
        type=str,
        default="t1_positive_only",
        help="Comma-separated constraint types (default: t1_positive_only)",
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=3,
        help="Episodes per configuration (default: 3)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=100,
        help="Steps per episode (default: 100)",
    )
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device (default: auto-detect)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--yaw-init",
        type=str,
        default="zeros",
        choices=["zeros", "random"],
        help="Yaw initialization: 'zeros' for deterministic, 'random' for stochastic (default: zeros)",
    )
    parser.add_argument(
        "--steady-state-steps",
        type=int,
        default=30,
        help="Number of final steps to average for steady-state yaw (default: 30)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip figure generation",
    )
    parser.add_argument(
        "--grid-res",
        type=int,
        default=80,
        help="Grid resolution for energy landscape heatmaps (default: 80)",
    )
    parser.add_argument(
        "--del-threshold-pct",
        type=float,
        default=0.10,
        help="DEL increase threshold as fraction (0.10 = 10%%) (default: 0.10)",
    )
    parser.add_argument(
        "--del-penalty-type",
        type=str,
        default="exponential",
        choices=["exponential", "quadratic"],
        help="DEL penalty shape (default: exponential)",
    )
    parser.add_argument(
        "--del-context-mode",
        type=str,
        default="frozen",
        choices=["frozen", "precomputed"],
        help=(
            "DEL context: 'frozen' (per-step sector avgs, fast, default) or "
            "'precomputed' (one-off full N-dim DEL grid built at episode "
            "reset, interpolated at inference for a true-surface gradient). "
            "Precomputed cost scales as grid_size**n_turbines."
        ),
    )
    parser.add_argument(
        "--del-grid-size",
        type=int,
        default=21,
        help=(
            "Grid points per turbine for precomputed mode (default: 21). "
            "Total PyWake sims per episode = grid_size**n_turbines."
        ),
    )
    parser.add_argument(
        "--ebt-opt-steps-eval",
        type=int,
        default=None,
        help=(
            "Override checkpoint's eval-time EBT optimization steps. More "
            "steps = better chance to escape the actor's learned basin."
        ),
    )
    parser.add_argument(
        "--ebt-langevin-noise",
        type=float,
        default=None,
        help=(
            "Override checkpoint's eval-time Langevin noise. Higher noise = "
            "more exploration, better at escaping local minima."
        ),
    )
    parser.add_argument(
        "--ebt-num-candidates",
        type=int,
        default=None,
        help=(
            "Override checkpoint's eval-time candidate count. More candidates "
            "= broader sampling of the initial action distribution."
        ),
    )
    return parser.parse_args()


def main():
    cli = parse_args()
    device = torch.device(cli.device)

    # Parse comma-separated CLI args
    lambdas = [float(x) for x in cli.lambdas.split(",")]
    steepness_values = [float(x) for x in cli.steepness_values.split(",")]
    threshold_values = [float(x) for x in cli.threshold_values.split(",")]
    constraint_types = [x.strip() for x in cli.constraint_types.split(",") if x.strip()]

    os.makedirs(cli.output_dir, exist_ok=True)

    # -- Load checkpoint -------------------------------------------------------
    print(f"Loading checkpoint: {cli.checkpoint}")
    print(f"Yaw init: {cli.yaw_init} | Steady-state steps: {cli.steady_state_steps}")
    envs, agent, actor, args, env_info = load_ebt_checkpoint(
        cli.checkpoint, device, yaw_init=cli.yaw_init,
        ebt_opt_steps_eval_override=cli.ebt_opt_steps_eval,
        ebt_langevin_noise_override=cli.ebt_langevin_noise,
        ebt_num_candidates_override=cli.ebt_num_candidates,
    )
    n_turbines = env_info["n_turbines_max"]
    print(f"Actor type: EBT | Turbines: {n_turbines}")
    print(f"EBT opt steps (inference): {args.ebt_opt_steps_eval} | "
          f"Candidates: {args.ebt_num_candidates}")

    # -- Build sweep configs ---------------------------------------------------
    configs = build_sweep_configs(
        constraint_types, lambdas, steepness_values, threshold_values, n_turbines,
        del_threshold_pct=cli.del_threshold_pct,
        del_penalty_type=cli.del_penalty_type,
        del_context_mode=cli.del_context_mode,
        del_grid_size=cli.del_grid_size,
    )
    print(f"\nSweep: {len(configs)} configurations "
          f"({len(constraint_types)} types x {len(lambdas)} lambdas)")

    # -- Set up DEL context updater if any del_* types are in the sweep --------
    has_del_constraints = any(ct.startswith("del_") for ct in constraint_types)
    del_updater: Optional[DELContextUpdater] = None
    if has_del_constraints:
        print("Setting up DEL context updater (probe farm model + Keras surrogate)...")
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        keras_surr = SurrogateLoadModel(
            os.path.join(repo_root, "surrogate/models/ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras"),
            os.path.join(repo_root, "surrogate/scalers/scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl"),
            os.path.join(repo_root, "surrogate/scalers/scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl"),
        )
        del_updater = DELContextUpdater(env_info["wind_turbine"], keras_surr)
        print("  DEL context updater ready.")

    # -- Run sweep -------------------------------------------------------------
    all_results: List[ConfigResult] = []
    baseline: Optional[ConfigResult] = None

    for i, config in enumerate(configs):
        label = config.label
        print(f"\n[{i + 1}/{len(configs)}] {label}")

        surrogate = create_surrogate_from_config(config, device)
        is_stateful = config.constraint_type == "travel_budget"
        use_del_updater = config.constraint_type.startswith("del_")

        episodes = run_constrained_episodes(
            agent,
            envs,
            surrogate,
            config.lambda_val,
            cli.num_episodes,
            cli.num_steps,
            device,
            is_stateful=is_stateful,
            steady_state_steps=cli.steady_state_steps,
            del_context_updater=del_updater if use_del_updater else None,
        )

        result = aggregate_episodes(config, episodes)

        # Store unconstrained baseline
        if config.constraint_type == "none":
            baseline = result
            result.power_ratio = 1.0
            result.cooperative_adaptation_score = 0.0
        elif baseline is not None:
            compute_derived_metrics(result, baseline)

        all_results.append(result)

        # Print inline progress
        yaw_str = ", ".join(f"{y:+.1f}" for y in result.mean_final_yaw)
        print(f"  Yaw: [{yaw_str}] | Power: {result.mean_power:.0f}", end="")
        if result.power_ratio is not None and config.constraint_type != "none":
            print(f" ({(result.power_ratio - 1) * 100:+.1f}%)", end="")
        if result.cooperative_adaptation_score is not None and config.constraint_type != "none":
            print(f" | Coop: {result.cooperative_adaptation_score:.2f}", end="")
        print()

    # -- Summary table ---------------------------------------------------------
    assert baseline is not None, "Unconstrained baseline missing from results"
    print_summary_table(all_results, baseline)

    # -- Save results ----------------------------------------------------------
    print("Saving results...")
    save_results(all_results, cli.output_dir)

    # -- Generate figures ------------------------------------------------------
    if not cli.no_plots:
        print("\nGenerating figures...")
        generate_figures(
            all_results, baseline, agent, envs, actor,
            cli.output_dir, device, cli,
        )

    # -- Cleanup ---------------------------------------------------------------
    envs.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
