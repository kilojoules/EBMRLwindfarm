"""
EBT-SAC for Wind Farm Control.

Trains an explicit Energy-Based Transformer actor with SAC critics on WindGym.
The actor outputs E(s,a) as a scalar energy and generates actions via gradient
descent on this energy landscape. Post-hoc composition uses exact energy addition.

Key differences from diffusion_sac_windfarm.py:
    - No noise schedules, DDPM/DDIM, or denoising chains
    - No BC regularization needed (energy landscape is shaped end-to-end)
    - Actor update backprops through N gradient descent steps (Hessian-vector products)
    - Self-verification at inference: generate M candidates, pick lowest energy

Usage:
    python ebt_sac_windfarm.py --layouts square_1 --total_timesteps 100000 --actor_type ebt

    # With more thinking at inference
    python ebt_sac_windfarm.py --layouts square_1 --ebt_opt_steps_eval 20

    # With load guidance at evaluation
    python ebt_sac_windfarm.py --layouts square_1 --guidance_scale 0.5
"""

import os
import sys
import json
import random
import time
from collections import deque, defaultdict
from typing import Dict, Any, Optional

sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

# WindGym
from WindGym import WindFarmEnv
from WindGym.wrappers import RecordEpisodeVals, PerTurbineObservationWrapper

# Project imports
from config import Args
from replay_buffer import TransformerReplayBuffer
from networks import TransformerCritic, create_profile_encoding
from ebt import TransformerEBTActor
from load_surrogates import create_load_surrogate, YawTravelBudgetSurrogate, NegativeYawBudgetSurrogate
from helpers.agent import WindFarmAgent
from helpers.constraint_viz import (
    plot_yaw_trajectory, plot_local_energy_landscape,
    plot_yaw_vs_lambda, plot_power_vs_lambda,
)
import matplotlib.pyplot as plt
from helpers.eval_utils import PolicyEvaluator
from helpers.multi_layout_env import MultiLayoutEnv, LayoutConfig
from helpers.multi_layout_debug import create_debug_logger
from helpers.layouts import get_layout_positions
from helpers.env_configs import make_env_config
from helpers.receptivity_profiles import compute_layout_profiles
from helpers.helper_funcs import (
    get_env_wind_directions,
    get_env_raw_positions,
    get_env_attention_masks,
    save_checkpoint,
    soft_update,
    EnhancedPerTurbineWrapper,
    get_env_layout_indices,
    get_env_permutations,
)
from helpers.training_utils import clear_gpu_memory


# =============================================================================
# ENVIRONMENT SETUP (identical to diffusion_sac_windfarm.py)
# =============================================================================

def setup_env(args: Args, config_overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Set up WindGym environment, layouts, and profiles."""
    if args.turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif args.turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {args.turbtype}")
    wind_turbine = WT()

    layout_names = [l.strip() for l in args.layouts.split(",")]
    if args.eval_layouts.strip():
        eval_layout_names_list = [l.strip() for l in args.eval_layouts.split(",")]
    else:
        eval_layout_names_list = layout_names

    layouts = []
    for name in layout_names:
        x_pos, y_pos = get_layout_positions(name, wind_turbine)
        layout = LayoutConfig(name=name, x_pos=x_pos, y_pos=y_pos)

        if args.profile_encoding_type is not None:
            if args.profile_source.lower() == "geometric":
                from helpers.geometric_profiles import compute_layout_profiles_vectorized
                D = wind_turbine.diameter()
                receptivity_profiles, influence_profiles = compute_layout_profiles_vectorized(
                    x_pos, y_pos, rotor_diameter=D, k_wake=0.04,
                    n_directions=args.n_profile_directions,
                    sigma_smooth=10.0, scale_factor=15.0,
                )
            else:
                receptivity_profiles, influence_profiles = compute_layout_profiles(
                    x_pos, y_pos, wind_turbine,
                    n_directions=args.n_profile_directions,
                )
            layout.receptivity_profiles = receptivity_profiles
            layout.influence_profiles = influence_profiles
        layouts.append(layout)

    use_profiles = args.profile_encoding_type is not None
    profile_registry = (
        [(l.receptivity_profiles, l.influence_profiles) for l in layouts]
        if use_profiles else None
    )

    config = make_env_config(args.config)
    config["ActionMethod"] = args.action_type
    for mes_type, prefix in {"ws_mes": "ws", "wd_mes": "wd", "yaw_mes": "yaw", "power_mes": "power"}.items():
        config[mes_type][f"{prefix}_history_N"] = args.history_length
        config[mes_type][f"{prefix}_history_length"] = args.history_length

    if config_overrides:
        config.update(config_overrides)

    base_env_kwargs = {
        "turbine": wind_turbine,
        "n_passthrough": args.max_eps,
        "TurbBox": "/work/users/manils/rl_timestep/Boxes/V80env/",
        "config": config,
        "turbtype": args.TI_type,
        "dt_sim": args.dt_sim,
        "dt_env": args.dt_env,
        "yaw_step_sim": args.yaw_step,
        "backend": "pywake",
    }
    if args.wind_timeseries_csv is not None:
        base_env_kwargs["wind_timeseries_csv"] = args.wind_timeseries_csv
        base_env_kwargs["wind_timeseries_random_start"] = args.wind_timeseries_random_start

    def env_factory(x_pos: np.ndarray, y_pos: np.ndarray) -> gym.Env:
        env = WindFarmEnv(x_pos=x_pos, y_pos=y_pos, reset_init=False, **base_env_kwargs)
        env.action_space.seed(args.seed)
        return env

    # Optional DEL-aware reward wrapper — load surrogate ONCE, share across envs
    _del_surrogate = None
    if getattr(args, "del_aware_reward", False):
        from helpers.teodor_surrogate import TeodorDLC12Surrogate
        _del_surrogate = TeodorDLC12Surrogate.from_bundle(
            args.flap_del_bundle,
            outputs=["wrot_Bl1Rad0FlpMnt"])
        _del_surrogate.eval()
        print(f"[reward] DEL-aware reward ON: r_new = r - {args.del_reward_beta} * sum(DEL)/{args.flap_del_ref}")

    def combined_wrapper(env: gym.Env) -> gym.Env:
        env = PerTurbineObservationWrapper(env)
        if args.use_wd_deviation:
            env = EnhancedPerTurbineWrapper(env, wd_scale_range=args.wd_scale_range)
        if _del_surrogate is not None:
            from helpers.del_reward_wrapper import DelRewardWrapper
            env = DelRewardWrapper(env, _del_surrogate,
                                     beta=args.del_reward_beta,
                                     del_ref=args.flap_del_ref)
        return env

    def make_env_fn(seed):
        def _init():
            env = MultiLayoutEnv(
                layouts=layouts, env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,
                seed=seed, shuffle=args.shuffle_turbs,
                max_episode_steps=args.max_episode_steps,
            )
            # Expose 4-sector rotor-disk flow for stateful load surrogates
            # (FlapDELSurrogate, FlapDELBudgetSurrogate). No-op otherwise.
            # Wrap OUTSIDE MultiLayoutEnv so AsyncVectorEnv.call reaches the
            # method directly.
            from helpers.surrogate_hooks import SectorFlowExposer
            env = SectorFlowExposer(env)
            return env
        return _init

    print(f"Creating {args.num_envs} parallel environment(s)...")
    envs = gym.vector.AsyncVectorEnv(
        [make_env_fn(args.seed + i) for i in range(args.num_envs)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    envs = RecordEpisodeVals(envs)

    n_turbines_max = envs.env.get_attr('max_turbines')[0]
    obs_dim_per_turbine = envs.single_observation_space.shape[-1]
    action_dim_per_turbine = 1
    rotor_diameter = envs.env.get_attr('rotor_diameter')[0]

    action_high = envs.single_action_space.high[0]
    action_low = envs.single_action_space.low[0]
    action_scale = float((action_high - action_low) / 2.0)
    action_bias = float((action_high + action_low) / 2.0)

    print(f"Max turbines: {n_turbines_max}")
    print(f"Obs dim per turbine: {obs_dim_per_turbine}")
    print(f"Rotor diameter: {rotor_diameter:.1f} m")
    print(f"Action scale: {action_scale}, bias: {action_bias}")

    return {
        "envs": envs,
        "layouts": layouts,
        "layout_names": layout_names,
        "eval_layout_names": eval_layout_names_list,
        "profile_registry": profile_registry,
        "use_profiles": use_profiles,
        "wind_turbine": wind_turbine,
        "n_turbines_max": n_turbines_max,
        "obs_dim_per_turbine": obs_dim_per_turbine,
        "action_dim_per_turbine": action_dim_per_turbine,
        "rotor_diameter": rotor_diameter,
        "action_scale": action_scale,
        "action_bias": action_bias,
        "env_factory": env_factory,
        "combined_wrapper": combined_wrapper,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    args = tyro.cli(Args)

    if args.exp_name == "transformer_sac_windfarm":
        args.exp_name = "ebt_sac_windfarm"
    if args.actor_type != "ebt":
        args.actor_type = "ebt"

    run_name = args.exp_name
    os.makedirs(f"runs/{run_name}", exist_ok=True)
    os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)

    clear_gpu_memory()

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    # =========================================================================
    # ENVIRONMENT SETUP
    # =========================================================================
    env_info = setup_env(args)
    envs = env_info["envs"]
    layout_names = env_info["layout_names"]
    eval_layout_names = env_info["eval_layout_names"]
    use_profiles = env_info["use_profiles"]
    profile_registry = env_info["profile_registry"]
    n_turbines_max = env_info["n_turbines_max"]
    obs_dim_per_turbine = env_info["obs_dim_per_turbine"]
    action_dim_per_turbine = env_info["action_dim_per_turbine"]
    rotor_diameter = env_info["rotor_diameter"]
    action_scale = env_info["action_scale"]
    action_bias = env_info["action_bias"]

    # =========================================================================
    # LOGGING
    # =========================================================================
    debug_logger = create_debug_logger(layout_names=layout_names, log_every=250000)

    if args.track:
        import wandb
        project = args.wandb_project_name
        if project == "transformer_windfarm":
            project = "ebt_windfarm"
        wandb.init(
            project=project, entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args) | {"debug/n_layouts": len(layout_names), "debug/max_turbines": n_turbines_max},
            name=run_name, monitor_gym=True, save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
        "|param|value|\n|-|-|\n" + "\n".join([f"|{k}|{v}|" for k, v in vars(args).items()]))

    # =========================================================================
    # NETWORK SETUP
    # =========================================================================

    # Shared profile encoders
    shared_recep_encoder = None
    shared_influence_encoder = None
    if use_profiles and args.share_profile_encoder:
        encoder_kwargs = json.loads(args.profile_encoder_kwargs)
        shared_recep_encoder, shared_influence_encoder = create_profile_encoding(
            profile_type=args.profile_encoding_type,
            embed_dim=args.embed_dim,
            hidden_channels=args.profile_encoder_hidden,
            **encoder_kwargs,
        )
        shared_recep_encoder = shared_recep_encoder.to(device)
        shared_influence_encoder = shared_influence_encoder.to(device)

    common_kwargs = {
        "obs_dim_per_turbine": obs_dim_per_turbine,
        "action_dim_per_turbine": action_dim_per_turbine,
        "embed_dim": args.embed_dim,
        "pos_embed_dim": args.pos_embed_dim,
        "num_heads": args.num_heads,
        "num_layers": args.num_layers,
        "mlp_ratio": args.mlp_ratio,
        "dropout": args.dropout,
        "pos_encoding_type": args.pos_encoding_type,
        "rel_pos_hidden_dim": args.rel_pos_hidden_dim,
        "rel_pos_per_head": args.rel_pos_per_head,
        "pos_embedding_mode": args.pos_embedding_mode,
        "profile_encoding": args.profile_encoding_type,
        "profile_encoder_hidden": args.profile_encoder_hidden,
        "n_profile_directions": args.n_profile_directions,
        "profile_fusion_type": args.profile_fusion_type,
        "profile_embed_mode": args.profile_embed_mode,
        "shared_recep_encoder": shared_recep_encoder,
        "shared_influence_encoder": shared_influence_encoder,
        "args": args,
    }

    # EBT actor
    actor = TransformerEBTActor(
        action_scale=action_scale,
        action_bias=action_bias,
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

    # Agent wrapper
    agent = WindFarmAgent(
        actor=actor, device=device, rotor_diameter=rotor_diameter,
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=args.rotate_profiles,
    )

    # Critics (SAC dual-critic)
    critic_kwargs = {**common_kwargs}
    if args.use_droq:
        critic_kwargs["droq_dropout"] = args.droq_dropout
        critic_kwargs["droq_layer_norm"] = args.droq_layer_norm

    qf1 = TransformerCritic(**critic_kwargs).to(device)
    qf2 = TransformerCritic(**critic_kwargs).to(device)
    qf1_target = TransformerCritic(**critic_kwargs).to(device)
    qf2_target = TransformerCritic(**critic_kwargs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Optimizers
    shared_encoder_params = []
    if shared_recep_encoder is not None:
        shared_encoder_params += list(shared_recep_encoder.parameters())
    if shared_influence_encoder is not None:
        shared_encoder_params += list(shared_influence_encoder.parameters())
    shared_param_ids = {id(p) for p in shared_encoder_params}

    def get_params_excluding_shared(network):
        return [p for p in network.parameters() if id(p) not in shared_param_ids]

    qf1_params = get_params_excluding_shared(qf1)
    qf2_params = get_params_excluding_shared(qf2)
    q_optimizer = optim.Adam(qf1_params + qf2_params + shared_encoder_params, lr=args.q_lr)
    actor_optimizer = optim.Adam(
        [p for p in actor.parameters() if id(p) not in shared_param_ids],
        lr=args.policy_lr,
    )

    # LR warmup schedulers
    if args.lr_warmup_steps > 0:
        def warmup_fn(step):
            return min(1.0, step / args.lr_warmup_steps)
        actor_scheduler = optim.lr_scheduler.LambdaLR(actor_optimizer, warmup_fn)
        q_scheduler = optim.lr_scheduler.LambdaLR(q_optimizer, warmup_fn)
    else:
        actor_scheduler = None
        q_scheduler = None

    # Evaluator
    evaluator = PolicyEvaluator(
        agent=agent,
        eval_layouts=eval_layout_names,
        env_factory=env_info["env_factory"],
        combined_wrapper=env_info["combined_wrapper"],
        num_envs=args.num_envs,
        num_eval_steps=args.num_eval_steps,
        num_eval_episodes=args.num_eval_episodes,
        device=device,
        rotor_diameter=rotor_diameter,
        wind_turbine=env_info["wind_turbine"],
        seed=args.eval_seed,
        max_turbines=n_turbines_max,
        deterministic=False,
        use_profiles=use_profiles,
        n_profile_directions=args.n_profile_directions,
        profile_source=args.profile_source,
    )

    # Load surrogates for post-training guidance sweep
    load_surrogate = create_load_surrogate(
        args.load_surrogate_type,
        steepness=args.load_steepness,
        per_turbine_thresholds=args.per_turbine_thresholds,
        neg_yaw_budget_steps=int(args.neg_yaw_budget_hours * 3600 / args.dt_env),
        neg_yaw_horizon_steps=int(args.neg_yaw_horizon_hours * 3600 / args.dt_env),
        neg_yaw_risk_aversion=args.neg_yaw_risk_aversion,
        neg_yaw_threshold_deg=args.neg_yaw_threshold_deg,
        flap_del_bundle=args.flap_del_bundle,
        flap_del_ref=args.flap_del_ref,
        flap_del_yaw_max_deg=args.flap_del_yaw_max_deg,
        flap_del_per_turbine_budgets=args.flap_del_per_turbine_budgets,
        flap_del_horizon_steps=args.flap_del_horizon_steps,
        flap_del_risk_aversion=args.flap_del_risk_aversion,
    )
    print(f"Load surrogate: {args.load_surrogate_type} → {type(load_surrogate).__name__}")

    travel_surrogate = YawTravelBudgetSurrogate(
        budget_deg=args.travel_budget_deg,
        window_steps=args.travel_budget_window,
        yaw_max_deg=30.0,
        steepness=args.travel_budget_steepness,
    )

    print(f"Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in qf1.parameters()):,} (x2)")
    print(f"EBT opt steps: {args.ebt_opt_steps_train} (train), {args.ebt_opt_steps_eval} (inference)")
    print(f"EBT candidates: {args.ebt_num_candidates} (self-verification at inference)")
    print(f"Training layouts: {layout_names}")

    # =========================================================================
    # REPLAY BUFFER
    # =========================================================================
    rb = TransformerReplayBuffer(
        capacity=args.buffer_size,
        device=device,
        rotor_diameter=rotor_diameter,
        max_turbines=n_turbines_max,
        obs_dim=obs_dim_per_turbine,
        action_dim=action_dim_per_turbine,
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=args.rotate_profiles,
        profile_registry=profile_registry,
    )

    # =========================================================================
    # CHECKPOINT RESUME (optional)
    # =========================================================================
    if args.resume_checkpoint is not None:
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        actor.load_state_dict(checkpoint["actor_state_dict"])
        qf1.load_state_dict(checkpoint["qf1_state_dict"])
        qf2.load_state_dict(checkpoint["qf2_state_dict"])
        qf1_target.load_state_dict(checkpoint["qf1_state_dict"])
        qf2_target.load_state_dict(checkpoint["qf2_state_dict"])
        if not args.finetune_reset_actor_optimizer and "actor_optimizer_state_dict" in checkpoint:
            actor_optimizer.load_state_dict(checkpoint["actor_optimizer_state_dict"])
        if not args.finetune_reset_critic_optimizer and "q_optimizer_state_dict" in checkpoint:
            q_optimizer.load_state_dict(checkpoint["q_optimizer_state_dict"])
        print("Checkpoint loaded.")

    if args.save_model:
        save_checkpoint(actor, qf1, qf2, actor_optimizer, q_optimizer, 0, run_name, args)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print(f"\nStarting EBT-SAC training for {args.total_timesteps} steps...")
    obs, info = envs.reset()

    step_reward_window = deque(maxlen=1000)
    loss_accumulator = defaultdict(list)
    total_gradient_steps = 0
    num_gradient_updates = max(1, int(args.num_envs * args.utd_ratio))
    global_step = 0
    start_time = time.time()

    next_save_step = args.save_interval
    next_eval_step = args.eval_interval
    if args.eval_initial:
        next_eval_step = 0

    for global_step in range(1, args.total_timesteps + 1):

        # -----------------------------------------------------------------
        # Data collection
        # -----------------------------------------------------------------
        wind_dirs = get_env_wind_directions(envs)
        raw_positions = get_env_raw_positions(envs)
        current_masks = get_env_attention_masks(envs)

        if use_profiles:
            current_layout_indices = get_env_layout_indices(envs)
            current_permutations = get_env_permutations(envs)
        else:
            current_layout_indices = None
            current_permutations = None

        # Refresh DEL-surrogate flow context BEFORE the agent computes guidance.
        # No-op for surrogates that don't have update_context (yaw-only).
        if hasattr(load_surrogate, "update_context"):
            from helpers.surrogate_hooks import refresh_surrogate_context
            try:
                refresh_surrogate_context(envs, load_surrogate)
            except Exception as _e:
                if global_step <= 1:
                    print(f"  [warn] refresh_surrogate_context failed: {_e}")

        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                actions = agent.act(envs, obs)

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Accumulate realized DEL into the budget tracker.
        if hasattr(load_surrogate, "update"):
            from helpers.surrogate_hooks import update_surrogate_after_step
            try:
                update_surrogate_after_step(envs, load_surrogate, infos)
            except Exception as _e:
                if global_step <= 1:
                    print(f"  [warn] update_surrogate_after_step failed: {_e}")
            # Log DEL accumulator state every 50 steps + at episode end.
            if (global_step % 50 == 0
                    or np.any(terminations) or np.any(truncations)):
                cum = getattr(load_surrogate, "cumulative_del", None)
                budgets = getattr(load_surrogate, "per_turbine_budgets", None)
                if cum is not None and budgets is not None:
                    cum_np = cum.detach().cpu().numpy().flatten()
                    util = cum_np / np.maximum(np.asarray(budgets), 1.0)
                    try:
                        lam = load_surrogate._compute_lambda().detach().cpu().numpy().flatten()
                    except Exception:
                        lam = np.zeros_like(cum_np)
                    for t, (c, u, l) in enumerate(zip(cum_np, util, lam)):
                        writer.add_scalar(f"flap_del/cum_turb{t}", float(c), global_step)
                        writer.add_scalar(f"flap_del/util_turb{t}", float(u), global_step)
                        writer.add_scalar(f"flap_del/lambda_turb{t}", float(l), global_step)
                    writer.add_scalar("flap_del/util_max", float(util.max()),
                                      global_step)
                    writer.add_scalar("flap_del/lambda_max", float(lam.max()),
                                      global_step)
                    if np.any(terminations) or np.any(truncations):
                        viol = util > 1.0
                        n_viol = int(viol.sum())
                        worst = float(util.max())
                        print(f"  [budget] ep end util={util.tolist()} "
                              f"viol={n_viol} worst={worst:.3f}")

            # Reset budget tracker on episode end (autoreset already gave fresh obs).
            if hasattr(load_surrogate, "reset") and (
                np.any(terminations) or np.any(truncations)
            ):
                load_surrogate.reset()

        step_reward_window.extend(np.array(rewards).flatten().tolist())

        if "yaw angles agent" in infos:
            yaw_deg = np.array(infos["yaw angles agent"])
            loss_accumulator['yaw_abs_mean_deg'].append(float(np.abs(yaw_deg).mean()))
            loss_accumulator['yaw_max_deg'].append(float(np.abs(yaw_deg).max()))
            loss_accumulator['yaw_over_20_frac'].append(float(np.mean(np.abs(yaw_deg) > 20.0)))
            for t in range(yaw_deg.shape[-1]):
                loss_accumulator[f'yaw_turb{t}_abs_deg'].append(float(np.abs(yaw_deg[:, t]).mean()))
            env0_yaw = yaw_deg[0] if yaw_deg.ndim > 1 else yaw_deg
            for t in range(len(env0_yaw)):
                loss_accumulator[f'yaw_env0_turb{t}_deg'].append(float(env0_yaw[t]))

        if "final_info" in infos:
            ep_return = np.mean(envs.return_queue)
            ep_power = np.mean(envs.mean_power_queue)
            print(f"Step {global_step}: Episode return={ep_return:.2f}, power={ep_power:.2f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_power", ep_power, global_step)

        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        for i in range(args.num_envs):
            done = terminations[i] or truncations[i]
            action_reshaped = actions[i].reshape(-1, action_dim_per_turbine)

            layout_idx_i = current_layout_indices[i] if current_layout_indices is not None else None
            perm_i = current_permutations[i] if current_permutations is not None else None

            rb.add(
                obs[i], real_next_obs[i], action_reshaped,
                rewards[i], done,
                raw_positions[i], current_masks[i], wind_dirs[i],
                layout_index=layout_idx_i, permutation=perm_i,
            )

        obs = next_obs

        # -----------------------------------------------------------------
        # Gradient updates
        # -----------------------------------------------------------------
        if global_step >= args.learning_starts and rb.size >= args.batch_size:
            for _update in range(num_gradient_updates):
                data = rb.sample(args.batch_size)
                batch_mask = data["attention_mask"]
                batch_receptivity = data.get("receptivity")
                batch_influence = data.get("influence")

                # =============================================================
                # Critic update (standard Bellman, no entropy)
                # =============================================================
                with torch.no_grad():
                    next_actions, _, _, _ = actor.get_action(
                        data["next_observations"], data["positions"], batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )
                    qf1_next = qf1_target(
                        data["next_observations"], next_actions,
                        data["positions"], batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )
                    qf2_next = qf2_target(
                        data["next_observations"], next_actions,
                        data["positions"], batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )
                    min_qf_next = torch.min(qf1_next, qf2_next)
                    target_q = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next

                qf1_value = qf1(
                    data["observations"], data["actions"], data["positions"], batch_mask,
                    recep_profile=batch_receptivity, influence_profile=batch_influence,
                )
                qf2_value = qf2(
                    data["observations"], data["actions"], data["positions"], batch_mask,
                    recep_profile=batch_receptivity, influence_profile=batch_influence,
                )

                qf1_loss = F.mse_loss(qf1_value, target_q)
                qf2_loss = F.mse_loss(qf2_value, target_q)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad(set_to_none=True)
                qf_loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        qf1_params + qf2_params + shared_encoder_params,
                        max_norm=args.grad_clip_max_norm,
                    )
                q_optimizer.step()
                if q_scheduler is not None:
                    q_scheduler.step()

                total_gradient_steps += 1
                loss_accumulator['qf1_loss'].append(qf1_loss.item())
                loss_accumulator['qf2_loss'].append(qf2_loss.item())
                loss_accumulator['qf1_value'].append(qf1_value.mean().item())
                loss_accumulator['qf2_value'].append(qf2_value.mean().item())

                # =============================================================
                # Actor update (EBT: energy optimization + Q-guidance)
                # =============================================================
                if total_gradient_steps % args.policy_frequency == 0:
                    # Encode once, optimize N times
                    turbine_emb = actor.encode(
                        data["observations"], data["positions"], batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )

                    # Differentiable energy optimization (create_graph=True in training)
                    actions_opt, energies = actor.optimize_actions(
                        turbine_emb, batch_mask,
                        num_candidates=1,  # Single candidate during training
                    )
                    actions_opt_scaled = actions_opt * actor.action_scale + actor.action_bias_val

                    # Q-values for optimized actions
                    qf1_pi = qf1(
                        data["observations"], actions_opt_scaled, data["positions"], batch_mask,
                        recep_profile=batch_receptivity, influence_profile=batch_influence,
                    )
                    qf2_pi = qf2(
                        data["observations"], actions_opt_scaled, data["positions"], batch_mask,
                        recep_profile=batch_receptivity, influence_profile=batch_influence,
                    )
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # Actor loss: maximize Q (exploration via Langevin noise in optimize_actions)
                    actor_loss = -min_qf_pi.mean()

                    # Energy magnitude regularization (prevents landscape collapse)
                    if args.ebt_energy_reg > 0:
                        actor_loss = actor_loss + args.ebt_energy_reg * (energies ** 2).mean()

                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            actor.parameters(), max_norm=args.grad_clip_max_norm,
                        )
                    actor_optimizer.step()
                    if actor_scheduler is not None:
                        actor_scheduler.step()

                    loss_accumulator['actor_loss'].append(actor_loss.item())
                    loss_accumulator['q_pi'].append(min_qf_pi.mean().item())
                    loss_accumulator['energy_mean'].append(energies.mean().item())
                    loss_accumulator['action_mean'].append(actions_opt_scaled.mean().item())
                    loss_accumulator['action_std'].append(actions_opt_scaled.std().item())

                # Target network update
                if total_gradient_steps % args.target_network_frequency == 0:
                    soft_update(qf1, qf1_target, args.tau)
                    soft_update(qf2, qf2_target, args.tau)

            # -----------------------------------------------------------------
            # Logging
            # -----------------------------------------------------------------
            if global_step % 20 == 0:
                sps = int(global_step / (time.time() - start_time))
                mean_reward = float(np.mean(step_reward_window)) if step_reward_window else 0.0
                mean_actor = np.mean(loss_accumulator['actor_loss']) if loss_accumulator['actor_loss'] else 0
                mean_qf1 = np.mean(loss_accumulator['qf1_loss']) if loss_accumulator['qf1_loss'] else 0
                mean_qf2 = np.mean(loss_accumulator['qf2_loss']) if loss_accumulator['qf2_loss'] else 0

                mean_qf1_val = np.mean(loss_accumulator['qf1_value']) if loss_accumulator['qf1_value'] else 0
                mean_qf2_val = np.mean(loss_accumulator['qf2_value']) if loss_accumulator['qf2_value'] else 0
                mean_q_pi = np.mean(loss_accumulator['q_pi']) if loss_accumulator['q_pi'] else 0
                mean_act_mean = np.mean(loss_accumulator['action_mean']) if loss_accumulator['action_mean'] else 0
                mean_act_std = np.mean(loss_accumulator['action_std']) if loss_accumulator['action_std'] else 0
                mean_energy = np.mean(loss_accumulator['energy_mean']) if loss_accumulator['energy_mean'] else 0

                writer.add_scalar("losses/actor_loss", mean_actor, global_step)
                writer.add_scalar("losses/qf1_loss", mean_qf1, global_step)
                writer.add_scalar("losses/qf2_loss", mean_qf2, global_step)
                writer.add_scalar("losses/qf1_value", mean_qf1_val, global_step)
                writer.add_scalar("losses/qf2_value", mean_qf2_val, global_step)
                writer.add_scalar("losses/q_policy_actions", mean_q_pi, global_step)
                writer.add_scalar("debug/action_mean", mean_act_mean, global_step)
                writer.add_scalar("debug/action_std", mean_act_std, global_step)
                writer.add_scalar("debug/energy_mean", mean_energy, global_step)

                # Yaw statistics
                if loss_accumulator['yaw_abs_mean_deg']:
                    writer.add_scalar("yaw/abs_mean_deg", np.mean(loss_accumulator['yaw_abs_mean_deg']), global_step)
                    writer.add_scalar("yaw/max_abs_deg", np.mean(loss_accumulator['yaw_max_deg']), global_step)
                    writer.add_scalar("yaw/over_20_frac", np.mean(loss_accumulator['yaw_over_20_frac']), global_step)
                    for t in range(n_turbines_max):
                        key = f'yaw_turb{t}_abs_deg'
                        if loss_accumulator.get(key):
                            writer.add_scalar(f"yaw/turbine_{t}_abs_deg", np.mean(loss_accumulator[key]), global_step)
                    for t in range(n_turbines_max):
                        key = f'yaw_env0_turb{t}_deg'
                        if loss_accumulator.get(key):
                            writer.add_scalar(f"yaw_env0/turbine_{t}_deg", np.mean(loss_accumulator[key]), global_step)

                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/step_reward_mean", mean_reward, global_step)
                writer.add_scalar("debug/total_gradient_steps", total_gradient_steps, global_step)

                print(f"Step {global_step}: SPS={sps}, qf_loss={mean_qf1+mean_qf2:.4f}, "
                      f"actor={mean_actor:.4f}, Q(pi)={mean_q_pi:.4f}, energy={mean_energy:.4f}, "
                      f"reward={mean_reward:.4f}")

                loss_accumulator.clear()

        # -----------------------------------------------------------------
        # Checkpointing
        # -----------------------------------------------------------------
        if args.save_model and global_step >= next_save_step:
            save_checkpoint(actor, qf1, qf2, actor_optimizer, q_optimizer,
                            global_step, run_name, args)
            next_save_step += args.save_interval

        # -----------------------------------------------------------------
        # Periodic evaluation (baseline + guidance sweep)
        # -----------------------------------------------------------------
        if global_step >= next_eval_step:
            print(f"\nRunning evaluation at step {global_step}...")

            eval_metrics = evaluator.evaluate()
            for name, value in eval_metrics.to_dict().items():
                writer.add_scalar(name, value, global_step)
            print(f"  [no guidance] Power ratio={eval_metrics.power_ratio:.4f}, "
                  f"Reward={eval_metrics.mean_reward:.4f}")

            # Guided evaluation sweep
            print(f"  Running guided evaluation sweep...")
            actor.eval()
            assessment_env = evaluator.eval_envs
            guided_steps = args.num_eval_steps
            for lam in [0.0, 1.0, 5.0, 10.0]:
                gfn = load_surrogate if lam > 0 else None
                if hasattr(load_surrogate, "reset"):
                    load_surrogate.reset()
                ep_obs, _ = assessment_env.reset()
                ep_reward = 0.0
                ep_load = 0.0
                ep_yaw_abs = []
                ep_yaw_per_turbine: list[list[float]] = [[] for _ in range(n_turbines_max)]
                ep_power = []

                for _ in range(guided_steps):
                    if hasattr(load_surrogate, "update_context"):
                        from helpers.surrogate_hooks import refresh_surrogate_context
                        try: refresh_surrogate_context(assessment_env, load_surrogate)
                        except Exception: pass
                    with torch.no_grad():
                        act = agent.act(assessment_env, ep_obs, guidance_fn=gfn, guidance_scale=lam)
                    ep_obs, rew, _, _, ep_info = assessment_env.step(act)
                    if hasattr(load_surrogate, "update"):
                        from helpers.surrogate_hooks import update_surrogate_after_step
                        try: update_surrogate_after_step(assessment_env, load_surrogate, ep_info)
                        except Exception: pass
                    ep_reward += float(rew.mean())

                    if "yaw angles agent" in ep_info:
                        yaw = np.array(ep_info["yaw angles agent"])
                        ep_yaw_abs.append(np.abs(yaw).mean())
                        # Per-turbine yaw (mean across envs)
                        yaw_flat = yaw[0] if yaw.ndim > 1 else yaw
                        for t in range(min(len(yaw_flat), n_turbines_max)):
                            ep_yaw_per_turbine[t].append(float(yaw_flat[t]))
                    if "Power agent" in ep_info:
                        ep_power.append(float(np.mean(ep_info["Power agent"])))
                    if gfn is not None:
                        act_t = torch.tensor(act, device=device, dtype=torch.float32).unsqueeze(-1)
                        mask_t = torch.tensor(
                            get_env_attention_masks(assessment_env), device=device, dtype=torch.bool)
                        ep_load += gfn(act_t, mask_t).mean().item()

                lam_str = f"{lam:.1f}".replace(".", "_")
                prefix = f"guidance_{lam_str}"
                mean_reward = ep_reward / max(guided_steps, 1)
                mean_load = ep_load / max(guided_steps, 1)
                writer.add_scalar(f"{prefix}/mean_reward", mean_reward, global_step)
                writer.add_scalar(f"{prefix}/mean_load", mean_load, global_step)
                if ep_yaw_abs:
                    writer.add_scalar(f"{prefix}/mean_abs_yaw_deg", np.mean(ep_yaw_abs), global_step)
                if ep_power:
                    writer.add_scalar(f"{prefix}/mean_power", np.mean(ep_power), global_step)
                # Per-turbine yaw angles
                turb_yaw_strs = []
                for t in range(n_turbines_max):
                    if ep_yaw_per_turbine[t]:
                        mean_yaw_t = np.mean(ep_yaw_per_turbine[t])
                        writer.add_scalar(f"{prefix}/turbine_{t}_yaw_deg", mean_yaw_t, global_step)
                        turb_yaw_strs.append(f"T{t}={mean_yaw_t:.1f}")

                yaw_str = f", AbsYaw={np.mean(ep_yaw_abs):.1f}deg" if ep_yaw_abs else ""
                turb_str = f" [{', '.join(turb_yaw_strs)}]" if turb_yaw_strs else ""
                print(f"  [lambda={lam}] Reward={mean_reward:.2f}, Load={mean_load:.2f}{yaw_str}{turb_str}")

            # --- Visualization figures (every Nth eval) ---
            if args.viz_every_n_evals > 0 and global_step % (args.eval_interval * args.viz_every_n_evals) < args.eval_interval:
                print(f"  Generating visualization figures...")
                viz_lambdas = [0.0, 1.0, 5.0, 10.0, 20.0]

                fig_traj = plot_yaw_trajectory(
                    agent, assessment_env, load_surrogate,
                    viz_lambdas, guided_steps, device,
                )
                writer.add_figure("viz/yaw_trajectory", fig_traj, global_step)
                plt.close(fig_traj)

                # Local energy landscape (EBT only)
                viz_obs, _ = assessment_env.reset()
                viz_batch = agent.batch_preparer.from_envs(assessment_env, viz_obs)
                # Get current action as center for local landscape
                with torch.no_grad():
                    current_act = agent.act(assessment_env, viz_obs)
                current_act_t = torch.tensor(
                    current_act, device=device, dtype=torch.float32
                ).unsqueeze(-1)[:1]  # (1, n_turb, 1)
                fig_local = plot_local_energy_landscape(
                    actor, viz_batch.obs, viz_batch.positions, viz_batch.mask,
                    load_surrogate, 5.0, current_act_t,
                    recep_profile=viz_batch.receptivity,
                    influence_profile=viz_batch.influence,
                )
                if fig_local is not None:
                    writer.add_figure("viz/local_energy", fig_local, global_step)
                    plt.close(fig_local)

                fig_yaw = plot_yaw_vs_lambda(
                    agent, assessment_env, load_surrogate,
                    viz_lambdas, guided_steps, device,
                )
                writer.add_figure("viz/yaw_vs_lambda", fig_yaw, global_step)
                plt.close(fig_yaw)

                fig_pow = plot_power_vs_lambda(
                    agent, assessment_env, load_surrogate,
                    viz_lambdas, guided_steps, device,
                )
                writer.add_figure("viz/power_vs_lambda", fig_pow, global_step)
                plt.close(fig_pow)
                print(f"  Visualizations logged.")

            actor.train()
            print(f"  Evaluation complete.")
            next_eval_step += args.eval_interval

    # =========================================================================
    # FINAL SAVE + GUIDANCE SWEEP
    # =========================================================================
    if args.save_model:
        save_checkpoint(actor, qf1, qf2, actor_optimizer, q_optimizer,
                        global_step, run_name, args)

    print("\n=== Final Evaluation ===")
    eval_metrics = evaluator.evaluate()
    print(f"Power ratio: {eval_metrics.power_ratio:.4f}")

    print("\n=== Final Guidance Scale Sweep ===")
    actor.eval()
    final_env = evaluator.eval_envs
    for lam in [0.0, 1.0, 5.0, 10.0, 20.0]:
        gfn = load_surrogate if lam > 0 else None
        if hasattr(load_surrogate, "reset"):
            load_surrogate.reset()
        test_obs, _ = final_env.reset()
        ep_reward, ep_load = 0.0, 0.0
        ep_yaw_abs = []
        ep_yaw_per_turb: list[list[float]] = [[] for _ in range(n_turbines_max)]
        for _ in range(args.num_eval_steps):
            if hasattr(load_surrogate, "update_context"):
                from helpers.surrogate_hooks import refresh_surrogate_context
                try: refresh_surrogate_context(final_env, load_surrogate)
                except Exception: pass
            with torch.no_grad():
                act = agent.act(final_env, test_obs, guidance_fn=gfn, guidance_scale=lam)
            test_obs, rew, _, _, step_info = final_env.step(act)
            if hasattr(load_surrogate, "update"):
                from helpers.surrogate_hooks import update_surrogate_after_step
                try: update_surrogate_after_step(final_env, load_surrogate, step_info)
                except Exception: pass
            ep_reward += float(rew.mean())
            if "yaw angles agent" in step_info:
                yaw_arr = np.array(step_info["yaw angles agent"])
                ep_yaw_abs.append(float(np.abs(yaw_arr).mean()))
                yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                for t in range(min(len(yaw_flat), n_turbines_max)):
                    ep_yaw_per_turb[t].append(float(yaw_flat[t]))
            if gfn is not None:
                act_t = torch.tensor(act, device=device, dtype=torch.float32).unsqueeze(-1)
                mask_t = torch.tensor(
                    get_env_attention_masks(final_env), device=device, dtype=torch.bool)
                ep_load += gfn(act_t, mask_t).mean().item()
        final_steps = args.num_eval_steps
        mean_reward = ep_reward / max(final_steps, 1)
        mean_load = ep_load / max(final_steps, 1)
        yaw_str = f", AbsYaw={np.mean(ep_yaw_abs):.1f}deg" if ep_yaw_abs else ""
        turb_strs = []
        for t in range(n_turbines_max):
            if ep_yaw_per_turb[t]:
                turb_strs.append(f"T{t}={np.mean(ep_yaw_per_turb[t]):.1f}")
        turb_str = f" [{', '.join(turb_strs)}]" if turb_strs else ""
        print(f"  lambda={lam}: Reward={mean_reward:.2f}, Load={mean_load:.2f}{yaw_str}{turb_str}")

    # === Budget Surrogate Evaluation Sweep ===
    # Test the NegativeYawBudgetSurrogate post-hoc on the trained actor.
    # Sweep over steepness, guidance_scale, risk_aversion, and budget levels.
    # Multiple episodes per config for statistical robustness.
    if args.load_surrogate_type == "neg_yaw_budget":
        print("\n=== Negative Yaw Budget Surrogate Sweep ===")
        horizon_steps = args.num_eval_steps
        n_eval_episodes = 5
        budget_eval_env = evaluator.eval_envs

        # --- Helper: run one episode and collect metrics ---
        def _run_budget_episode(surr, gs_val):
            if surr is not None and hasattr(surr, 'reset'):
                surr.reset()
            obs, _ = budget_eval_env.reset()
            ep_rew, ep_powers, neg_counts = 0.0, [], np.zeros(n_turbines_max)
            for _ in range(horizon_steps):
                with torch.no_grad():
                    act = agent.act(budget_eval_env, obs,
                                    guidance_fn=surr if gs_val > 0 else None,
                                    guidance_scale=gs_val)
                obs, rew, _, _, info = budget_eval_env.step(act)
                ep_rew += float(rew.mean())
                if "yaw angles agent" in info:
                    yaw = np.array(info["yaw angles agent"])
                    yaw_flat = yaw[0] if yaw.ndim > 1 else yaw
                    for ti in range(min(len(yaw_flat), n_turbines_max)):
                        if yaw_flat[ti] < 0:
                            neg_counts[ti] += 1
                    if surr is not None and hasattr(surr, 'update'):
                        surr.update(torch.tensor(yaw_flat, device=device, dtype=torch.float32))
                if "Power agent" in info:
                    ep_powers.append(float(np.mean(info["Power agent"])))
            return ep_rew, np.mean(ep_powers) if ep_powers else 0.0, neg_counts

        # --- Unconstrained baseline (multiple episodes) ---
        print(f"Running unconstrained baseline ({n_eval_episodes} episodes)...")
        uncon_rewards, uncon_powers, uncon_negs = [], [], []
        for _ in range(n_eval_episodes):
            r, p, n = _run_budget_episode(None, 0.0)
            uncon_rewards.append(r)
            uncon_powers.append(p)
            uncon_negs.append(n)
        uncon_mean_power = np.mean(uncon_powers)
        uncon_mean_reward = np.mean(uncon_rewards)
        uncon_mean_neg = np.mean(uncon_negs, axis=0)
        print(f"  Unconstrained: Reward={uncon_mean_reward:.2f} +/- {np.std(uncon_rewards):.2f}, "
              f"Power={uncon_mean_power:.0f}, "
              f"NegYaw={uncon_mean_neg.astype(int).tolist()}")

        # --- Hard-clip baseline: positive-only yaw (no neg yaw ever) ---
        print(f"\nRunning hard-clip baseline (positive-only yaw)...")
        from load_surrogates import PositiveYawT1Surrogate, ExponentialYawSurrogate
        # Use exponential surrogate with threshold=0 to ban all negative yaw
        hard_clip_surr = ExponentialYawSurrogate(threshold_deg=0.0, yaw_max_deg=30.0, steepness=10.0)
        clip_rewards, clip_powers = [], []
        for _ in range(n_eval_episodes):
            r, p, n = _run_budget_episode(hard_clip_surr, 5.0)
            clip_rewards.append(r)
            clip_powers.append(p)
        clip_mean_power = np.mean(clip_powers)
        print(f"  Hard-clip (no neg yaw): Reward={np.mean(clip_rewards):.2f}, "
              f"Power={clip_mean_power:.0f}, "
              f"PowerRatio={clip_mean_power/uncon_mean_power:.4f}")

        # --- Budget sweep: steepness x gs x RA x budget_level ---
        budget_levels = [15, 30, 50, 100]
        steepness_values = [2.0, 3.0, 5.0]
        gs_values = [0.05, 0.1, 0.5, 1.0]
        ra_values = [0.0, 1.0, 2.0, 5.0]

        print(f"\nBudget sweep: {len(budget_levels)} budgets x {len(steepness_values)} steepness "
              f"x {len(gs_values)} gs x {len(ra_values)} RA x {n_eval_episodes} episodes")
        print(f"Horizon: {horizon_steps} steps")
        print(f"{'Budget':>6s} {'k':>4s} {'gs':>5s} {'RA':>4s} | "
              f"{'Reward':>8s} {'Power':>10s} {'PwrRatio':>8s} {'NegYaw':>20s}")
        print("-" * 80)

        for budget_steps in budget_levels:
            for k_val in steepness_values:
                for gs_val in gs_values:
                    for ra_val in ra_values:
                        ep_rewards, ep_powers, ep_negs = [], [], []
                        for _ in range(n_eval_episodes):
                            surr = NegativeYawBudgetSurrogate(
                                budget_steps=budget_steps,
                                horizon_steps=horizon_steps,
                                risk_aversion=ra_val,
                                steepness=k_val,
                                yaw_max_deg=30.0,
                                neg_yaw_threshold_deg=args.neg_yaw_threshold_deg,
                            )
                            r, p, n = _run_budget_episode(surr, gs_val)
                            ep_rewards.append(r)
                            ep_powers.append(p)
                            ep_negs.append(n)

                        mean_pwr = np.mean(ep_powers)
                        pwr_ratio = mean_pwr / uncon_mean_power if uncon_mean_power > 0 else 0
                        mean_neg = np.mean(ep_negs, axis=0).astype(int).tolist()
                        print(f"{budget_steps:6d} {k_val:4.1f} {gs_val:5.2f} {ra_val:4.1f} | "
                              f"{np.mean(ep_rewards):8.2f} {mean_pwr:10.0f} {pwr_ratio:8.4f} "
                              f"{str(mean_neg):>20s}")

                        if args.track:
                            tag = f"budget_eval/B{budget_steps}_k{k_val}_gs{gs_val}_ra{ra_val}"
                            wandb.log({
                                f"{tag}/reward": np.mean(ep_rewards),
                                f"{tag}/power": mean_pwr,
                                f"{tag}/power_ratio": pwr_ratio,
                                **{f"{tag}/neg_yaw_T{ti}": int(np.mean([n[ti] for n in ep_negs]))
                                   for ti in range(n_turbines_max)},
                            })

        # --- Constant penalty ablation (no AC adaptation) ---
        print(f"\nConstant penalty ablation (lambda=const, no time-varying adaptation):")
        print(f"{'Budget':>6s} {'k':>4s} {'gs':>5s} {'lam':>5s} | "
              f"{'Reward':>8s} {'Power':>10s} {'PwrRatio':>8s} {'NegYaw':>20s}")
        print("-" * 80)
        for budget_steps in [15, 50]:
            for k_val in [2.0, 3.0]:
                for gs_val in [0.1, 0.5]:
                    # RA=0 means lambda=1 always (constant penalty, no AC)
                    ep_rewards, ep_powers, ep_negs = [], [], []
                    for _ in range(n_eval_episodes):
                        surr = NegativeYawBudgetSurrogate(
                            budget_steps=budget_steps,
                            horizon_steps=horizon_steps,
                            risk_aversion=0.0,  # constant lambda
                            steepness=k_val,
                            yaw_max_deg=30.0,
                        )
                        r, p, n = _run_budget_episode(surr, gs_val)
                        ep_rewards.append(r)
                        ep_powers.append(p)
                        ep_negs.append(n)
                    mean_pwr = np.mean(ep_powers)
                    pwr_ratio = mean_pwr / uncon_mean_power if uncon_mean_power > 0 else 0
                    mean_neg = np.mean(ep_negs, axis=0).astype(int).tolist()
                    print(f"{budget_steps:6d} {k_val:4.1f} {gs_val:5.2f} {'1.0':>5s} | "
                          f"{np.mean(ep_rewards):8.2f} {mean_pwr:10.0f} {pwr_ratio:8.4f} "
                          f"{str(mean_neg):>20s}")

        # --- Trajectory logging for visualization ---
        # Run a few representative configs with per-step yaw/power/lambda recording
        print("\n=== Logging trajectories for visualization ===")
        import json

        traj_configs = [
            {"ra": 0.0, "gs": 0.0, "k": 3.0, "budget": 15, "label": "unconstrained"},
            {"ra": 0.0, "gs": 0.5, "k": 3.0, "budget": 15, "label": "constant_eta0"},
            {"ra": 2.0, "gs": 0.5, "k": 3.0, "budget": 15, "label": "ac_eta2"},
            {"ra": 5.0, "gs": 0.5, "k": 3.0, "budget": 15, "label": "ac_eta5"},
            {"ra": 2.0, "gs": 0.5, "k": 3.0, "budget": 50, "label": "ac_eta2_b50"},
        ]

        all_trajectories = {}
        for cfg in traj_configs:
            label = cfg["label"]
            is_uncon = cfg["gs"] == 0.0

            surr = None
            if not is_uncon:
                surr = NegativeYawBudgetSurrogate(
                    budget_steps=cfg["budget"],
                    horizon_steps=horizon_steps,
                    risk_aversion=cfg["ra"],
                    steepness=cfg["k"],
                    yaw_max_deg=30.0,
                )
                surr.reset()

            obs, _ = budget_eval_env.reset()
            steps_data = []

            for t_step in range(horizon_steps):
                if surr is not None:
                    lam_raw = surr._compute_lambda()
                    lam_val = float(lam_raw.mean().cpu()) if lam_raw.numel() > 1 else float(lam_raw.cpu())
                else:
                    lam_val = 1.0

                with torch.no_grad():
                    act = agent.act(budget_eval_env, obs,
                                    guidance_fn=surr if not is_uncon else None,
                                    guidance_scale=cfg["gs"])

                obs, rew, _, _, info = budget_eval_env.step(act)

                yaw_flat = np.zeros(n_turbines_max)
                power_val = 0.0
                ws_val, wd_val = 0.0, 0.0

                if "yaw angles agent" in info:
                    yaw_arr = np.array(info["yaw angles agent"])
                    yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                    if surr is not None and hasattr(surr, 'update'):
                        surr.update(torch.tensor(yaw_flat[:n_turbines_max],
                                                  device=device, dtype=torch.float32))

                if "Power agent" in info:
                    power_val = float(np.mean(info["Power agent"]))

                step_record = {
                    "t": t_step,
                    "lambda": float(lam_val) if np.isscalar(lam_val) else float(lam_val),
                    "power": power_val,
                    "reward": float(rew.mean()),
                }
                for ti in range(min(len(yaw_flat), n_turbines_max)):
                    step_record[f"yaw_T{ti}"] = float(yaw_flat[ti])

                steps_data.append(step_record)

            all_trajectories[label] = steps_data
            neg_count = sum(1 for s in steps_data if s.get("yaw_T0", 0) < 0)
            print(f"  {label}: {len(steps_data)} steps, T0 neg_yaw={neg_count}")

        # Save trajectories
        os.makedirs("results", exist_ok=True)
        traj_path = "results/windfarm_trajectories.json"
        with open(traj_path, "w") as f:
            json.dump(all_trajectories, f)
        print(f"  Saved trajectories to {traj_path}")

    writer.close()
    try:
        envs.close()
    except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as _e:
        print(f"  [warn] envs.close() raised {type(_e).__name__}; subprocess "
              f"likely already gone, continuing")
    print("Done.")


if __name__ == "__main__":
    main()
