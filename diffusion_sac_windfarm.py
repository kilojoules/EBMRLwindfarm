"""
Diffusion-SAC for Wind Farm Control.

Trains a diffusion-based EBM actor with SAC critics on the WindGym environment.
After training, demonstrates post-hoc load-safety composition via classifier guidance.

The training loop mirrors transformer_sac_windfarm.py but replaces the Gaussian
actor + entropy tuning with a diffusion denoiser + Q-guidance. Everything else
(env setup, replay buffer, evaluation, checkpointing) is reused from existing helpers.

Usage:
    # Single-layout training
    python diffusion_sac_windfarm.py --layouts square_1 --total_timesteps 100000

    # With BC regularization (recommended for stability)
    python diffusion_sac_windfarm.py --layouts square_1 --diffusion_bc_weight 1.0

    # With load guidance at evaluation
    python diffusion_sac_windfarm.py --layouts square_1 --guidance_scale 0.5
"""

import os
import sys
import json
import random
import time
from collections import deque, defaultdict
from typing import Dict, Any

# Unbuffered stdout so nohup logs appear immediately
sys.stdout.reconfigure(line_buffering=True)

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import gymnasium as gym
import numpy as np
import torch
import math
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
from diffusion import TransformerDiffusionActor
from load_surrogates import create_load_surrogate, YawTravelBudgetSurrogate, NegativeYawBudgetSurrogate
from helpers.agent import WindFarmAgent
from helpers.constraint_viz import plot_yaw_trajectory, plot_yaw_vs_lambda, plot_power_vs_lambda
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
# ENVIRONMENT SETUP
# =============================================================================

def setup_env(args: Args) -> Dict[str, Any]:
    """
    Set up WindGym environment, layouts, and profiles.

    Replicates the env setup from transformer_sac_windfarm.py so both training
    scripts use identical environments.
    """
    # Wind turbine selection
    if args.turbtype == "DTU10MW":
        from py_wake.examples.data.dtu10mw import DTU10MW as WT
    elif args.turbtype == "V80":
        from py_wake.examples.data.hornsrev1 import V80 as WT
    else:
        raise ValueError(f"Unknown turbine type: {args.turbtype}")
    wind_turbine = WT()

    # Parse layouts
    layout_names = [l.strip() for l in args.layouts.split(",")]
    if args.eval_layouts.strip():
        eval_layout_names = [l.strip() for l in args.eval_layouts.split(",")]
    else:
        eval_layout_names = layout_names

    # Create layout configs + profiles
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

    # Environment config
    config = make_env_config(args.config)
    config["ActionMethod"] = args.action_type
    for mes_type, prefix in {"ws_mes": "ws", "wd_mes": "wd", "yaw_mes": "yaw", "power_mes": "power"}.items():
        config[mes_type][f"{prefix}_history_N"] = args.history_length
        config[mes_type][f"{prefix}_history_length"] = args.history_length

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

    def env_factory(x_pos: np.ndarray, y_pos: np.ndarray) -> gym.Env:
        env = WindFarmEnv(x_pos=x_pos, y_pos=y_pos, reset_init=False, **base_env_kwargs)
        env.action_space.seed(args.seed)
        return env

    def combined_wrapper(env: gym.Env) -> gym.Env:
        env = PerTurbineObservationWrapper(env)
        if args.use_wd_deviation:
            env = EnhancedPerTurbineWrapper(env, wd_scale_range=args.wd_scale_range)
        return env

    def make_env_fn(seed):
        def _init():
            return MultiLayoutEnv(
                layouts=layouts, env_factory=env_factory,
                per_turbine_wrapper=combined_wrapper,
                seed=seed, shuffle=args.shuffle_turbs,
                max_episode_steps=args.max_episode_steps,
            )
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
        "eval_layout_names": eval_layout_names,
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
        args.exp_name = "diffusion_sac_windfarm"

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
        # Default to separate project for diffusion experiments
        project = args.wandb_project_name
        if project == "transformer_windfarm":
            project = "diffusion_windfarm"
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

    # Diffusion actor
    actor = TransformerDiffusionActor(
        action_scale=action_scale,
        action_bias=action_bias,
        num_diffusion_steps=args.num_diffusion_steps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        noise_schedule=args.noise_schedule,
        cosine_s=args.cosine_schedule_s,
        timestep_embed_dim=args.timestep_embed_dim,
        denoiser_hidden_dim=args.denoiser_hidden_dim,
        denoiser_num_layers=args.denoiser_num_layers,
        **common_kwargs,
    ).to(device)

    # Agent wrapper
    agent = WindFarmAgent(
        actor=actor, device=device, rotor_diameter=rotor_diameter,
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=args.rotate_profiles,
    )

    # Critics (SAC dual-critic, no TQC for diffusion)
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

    # Optimizers — exclude shared encoder params from actor optimizer
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
    print(f"Diffusion steps: {args.num_diffusion_steps} (train), {args.num_inference_steps} (infer)")
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

    # Save initial checkpoint
    if args.save_model:
        save_checkpoint(actor, qf1, qf2, actor_optimizer, q_optimizer, 0, run_name, args)

    # =========================================================================
    # TRAINING LOOP
    # =========================================================================
    print(f"\nStarting training for {args.total_timesteps} steps...")
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

        # Action selection
        if global_step < args.learning_starts:
            actions = envs.action_space.sample()
        else:
            with torch.no_grad():
                actions = agent.act(envs, obs)

        # Step environment
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # Track rewards
        step_reward_window.extend(np.array(rewards).flatten().tolist())

        # Track yaw angles from environment state (physical degrees, not normalized actions)
        if "yaw angles agent" in infos:
            yaw_deg = np.array(infos["yaw angles agent"])  # (num_envs, n_turbines)
            loss_accumulator['yaw_abs_mean_deg'].append(float(np.abs(yaw_deg).mean()))
            loss_accumulator['yaw_max_deg'].append(float(np.abs(yaw_deg).max()))
            loss_accumulator['yaw_over_20_frac'].append(float(np.mean(np.abs(yaw_deg) > 20.0)))
            # Per-turbine absolute yaw (averaged across envs)
            for t in range(yaw_deg.shape[-1]):
                loss_accumulator[f'yaw_turb{t}_abs_deg'].append(float(np.abs(yaw_deg[:, t]).mean()))
            # Per-turbine raw yaw from env 0 only (time history)
            env0_yaw = yaw_deg[0] if yaw_deg.ndim > 1 else yaw_deg
            for t in range(len(env0_yaw)):
                loss_accumulator[f'yaw_env0_turb{t}_deg'].append(float(env0_yaw[t]))

        # Log episodes via RecordEpisodeVals
        if "final_info" in infos:
            ep_return = np.mean(envs.return_queue)
            ep_power = np.mean(envs.mean_power_queue)
            print(f"Step {global_step}: Episode return={ep_return:.2f}, power={ep_power:.2f}")
            writer.add_scalar("charts/episodic_return", ep_return, global_step)
            writer.add_scalar("charts/episodic_power", ep_power, global_step)

        # Handle final observations for truncated episodes
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_obs"][idx]

        # Store in replay buffer
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
            for update in range(num_gradient_updates):
                data = rb.sample(args.batch_size)
                batch_mask = data["attention_mask"]
                batch_receptivity = data.get("receptivity")
                batch_influence = data.get("influence")

                # =============================================================
                # Critic update
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
                # Actor update (delayed)
                # =============================================================
                if total_gradient_steps % args.policy_frequency == 0:
                    # Encode once, denoise K times
                    turbine_emb = actor.encode(
                        data["observations"], data["positions"], batch_mask,
                        recep_profile=batch_receptivity,
                        influence_profile=batch_influence,
                    )

                    # Differentiable denoising chain
                    actions_pi = actor.denoise_chain(turbine_emb, batch_mask, use_ddim=False)
                    actions_pi_scaled = actions_pi * actor.action_scale + actor.action_bias_val

                    # Q-values for generated actions
                    qf1_pi = qf1(
                        data["observations"], actions_pi_scaled, data["positions"], batch_mask,
                        recep_profile=batch_receptivity, influence_profile=batch_influence,
                    )
                    qf2_pi = qf2(
                        data["observations"], actions_pi_scaled, data["positions"], batch_mask,
                        recep_profile=batch_receptivity, influence_profile=batch_influence,
                    )
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # Actor loss: maximize Q (no entropy term for diffusion)
                    actor_loss = -min_qf_pi.mean()

                    # Action regularization: penalize non-zero delta actions
                    if args.action_reg_weight > 0:
                        action_reg = (actions_pi ** 2).mean()
                        actor_loss = actor_loss + args.action_reg_weight * action_reg

                    # BC regularization with optional annealing
                    if args.bc_weight_start > 0:
                        progress = min(1.0, global_step / max(1, args.bc_anneal_steps))
                        if args.bc_anneal_type == "cosine":
                            bc_weight = args.bc_weight_end + 0.5 * (args.bc_weight_start - args.bc_weight_end) * (1 + math.cos(math.pi * progress))
                        else:
                            bc_weight = args.bc_weight_start + (args.bc_weight_end - args.bc_weight_start) * progress
                    elif args.diffusion_bc_weight > 0:
                        bc_weight = args.diffusion_bc_weight
                    else:
                        bc_weight = 0.0

                    if bc_weight > 0:
                        batch_size_cur = data["actions"].shape[0]
                        t = torch.randint(0, args.num_diffusion_steps, (batch_size_cur,), device=device)
                        noise = torch.randn_like(data["actions"])
                        actions_norm = (data["actions"] - actor.action_bias_val) / actor.action_scale
                        noisy_actions = actor.q_sample(actions_norm, t, noise)
                        eps_pred = actor.predict_noise(turbine_emb.detach(), noisy_actions, t, batch_mask)
                        bc_loss = F.mse_loss(eps_pred, noise)
                        actor_loss = actor_loss + bc_weight * bc_loss

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
                    loss_accumulator['action_mean'].append(actions_pi_scaled.mean().item())
                    loss_accumulator['action_std'].append(actions_pi_scaled.std().item())
                    if bc_weight > 0:
                        loss_accumulator['bc_loss'].append(bc_loss.item())
                        loss_accumulator['bc_weight'].append(bc_weight)
                    if args.action_reg_weight > 0:
                        loss_accumulator['action_reg'].append(action_reg.item())

                # Target network update
                if total_gradient_steps % args.target_network_frequency == 0:
                    soft_update(qf1, qf1_target, args.tau)
                    soft_update(qf2, qf2_target, args.tau)

            # -----------------------------------------------------------------
            # Logging (every 20 outer steps that have gradient updates)
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

                writer.add_scalar("losses/actor_loss", mean_actor, global_step)
                writer.add_scalar("losses/qf1_loss", mean_qf1, global_step)
                writer.add_scalar("losses/qf2_loss", mean_qf2, global_step)
                writer.add_scalar("losses/qf1_value", mean_qf1_val, global_step)
                writer.add_scalar("losses/qf2_value", mean_qf2_val, global_step)
                writer.add_scalar("losses/q_policy_actions", mean_q_pi, global_step)
                writer.add_scalar("debug/action_mean", mean_act_mean, global_step)
                writer.add_scalar("debug/action_std", mean_act_std, global_step)
                if loss_accumulator['bc_loss']:
                    writer.add_scalar("losses/bc_loss",
                                      np.mean(loss_accumulator['bc_loss']), global_step)
                if loss_accumulator['bc_weight']:
                    writer.add_scalar("debug/bc_weight",
                                      np.mean(loss_accumulator['bc_weight']), global_step)
                if loss_accumulator['action_reg']:
                    writer.add_scalar("losses/action_reg",
                                      np.mean(loss_accumulator['action_reg']), global_step)
                # Yaw statistics (from env state, in physical degrees)
                if loss_accumulator['yaw_abs_mean_deg']:
                    writer.add_scalar("yaw/abs_mean_deg", np.mean(loss_accumulator['yaw_abs_mean_deg']), global_step)
                    writer.add_scalar("yaw/max_abs_deg", np.mean(loss_accumulator['yaw_max_deg']), global_step)
                    writer.add_scalar("yaw/over_20_frac", np.mean(loss_accumulator['yaw_over_20_frac']), global_step)
                    # Per-turbine absolute yaw (averaged across all envs)
                    for t in range(n_turbines_max):
                        key = f'yaw_turb{t}_abs_deg'
                        if loss_accumulator.get(key):
                            writer.add_scalar(f"yaw/turbine_{t}_abs_deg", np.mean(loss_accumulator[key]), global_step)
                    # Per-turbine raw yaw from env 0 (time history)
                    for t in range(n_turbines_max):
                        key = f'yaw_env0_turb{t}_deg'
                        if loss_accumulator.get(key):
                            writer.add_scalar(f"yaw_env0/turbine_{t}_deg", np.mean(loss_accumulator[key]), global_step)

                writer.add_scalar("charts/SPS", sps, global_step)
                writer.add_scalar("charts/step_reward_mean", mean_reward, global_step)
                writer.add_scalar("debug/total_gradient_steps", total_gradient_steps, global_step)

                print(f"Step {global_step}: SPS={sps}, qf_loss={mean_qf1+mean_qf2:.4f}, "
                      f"actor={mean_actor:.4f}, Q(pi)={mean_q_pi:.4f}, reward={mean_reward:.4f}")

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

            # 1) Standard evaluation (no guidance)
            eval_metrics = evaluator.evaluate()
            for name, value in eval_metrics.to_dict().items():
                writer.add_scalar(name, value, global_step)
            print(f"  [no guidance] Power ratio={eval_metrics.power_ratio:.4f}, "
                  f"Reward={eval_metrics.mean_reward:.4f}")

            # 2) Guided evaluation sweep — uses evaluator's env (not training env)
            print(f"  Running guided eval sweep...")
            actor.eval()
            eval_env = evaluator.eval_envs  # Reuse the eval env created above
            guided_eval_steps = min(50, args.num_eval_steps)
            for lam in [0.0, 1.0, 5.0, 10.0]:
                gfn = load_surrogate if lam > 0 else None
                ep_obs, _ = eval_env.reset()
                ep_reward = 0.0
                ep_load = 0.0
                ep_yaw_abs = []
                ep_yaw_per_turbine: list[list[float]] = [[] for _ in range(n_turbines_max)]
                ep_power = []

                for _ in range(guided_eval_steps):
                    with torch.no_grad():
                        act = agent.act(eval_env, ep_obs, guidance_fn=gfn, guidance_scale=lam)
                    ep_obs, rew, _, _, ep_info = eval_env.step(act)
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
                            get_env_attention_masks(eval_env), device=device, dtype=torch.bool)
                        ep_load += gfn(act_t, mask_t).mean().item()

                lam_str = f"{lam:.1f}".replace(".", "_")
                prefix = f"guidance_{lam_str}"
                mean_reward = ep_reward / max(guided_eval_steps, 1)
                mean_load = ep_load / max(guided_eval_steps, 1)
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
                    agent, eval_env, load_surrogate,
                    viz_lambdas, guided_eval_steps, device,
                )
                writer.add_figure("viz/yaw_trajectory", fig_traj, global_step)
                plt.close(fig_traj)

                fig_yaw = plot_yaw_vs_lambda(
                    agent, eval_env, load_surrogate,
                    viz_lambdas, guided_eval_steps, device,
                )
                writer.add_figure("viz/yaw_vs_lambda", fig_yaw, global_step)
                plt.close(fig_yaw)

                fig_pow = plot_power_vs_lambda(
                    agent, eval_env, load_surrogate,
                    viz_lambdas, guided_eval_steps, device,
                )
                writer.add_figure("viz/power_vs_lambda", fig_pow, global_step)
                plt.close(fig_pow)
                print(f"  Visualizations logged.")

            actor.train()
            print(f"  Eval complete.")
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
    eval_env = evaluator.eval_envs
    for lam in [0.0, 1.0, 5.0, 10.0, 20.0]:
        gfn = load_surrogate if lam > 0 else None
        test_obs, _ = eval_env.reset()
        ep_reward, ep_load = 0.0, 0.0
        ep_yaw_abs = []
        ep_yaw_per_turb: list[list[float]] = [[] for _ in range(n_turbines_max)]
        for _ in range(min(50, args.num_eval_steps)):
            with torch.no_grad():
                act = agent.act(eval_env, test_obs, guidance_fn=gfn, guidance_scale=lam)
            test_obs, rew, _, _, info = eval_env.step(act)
            ep_reward += float(rew.mean())
            if "yaw angles agent" in info:
                yaw_arr = np.array(info["yaw angles agent"])
                ep_yaw_abs.append(float(np.abs(yaw_arr).mean()))
                yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                for t in range(min(len(yaw_flat), n_turbines_max)):
                    ep_yaw_per_turb[t].append(float(yaw_flat[t]))
            if gfn is not None:
                act_t = torch.tensor(act, device=device, dtype=torch.float32).unsqueeze(-1)
                mask_t = torch.tensor(
                    get_env_attention_masks(eval_env), device=device, dtype=torch.bool)
                ep_load += gfn(act_t, mask_t).mean().item()
        final_steps = min(50, args.num_eval_steps)
        mean_reward = ep_reward / max(final_steps, 1)
        mean_load = ep_load / max(final_steps, 1)
        yaw_str = f", AbsYaw={np.mean(ep_yaw_abs):.1f}deg" if ep_yaw_abs else ""
        turb_strs = []
        for t in range(n_turbines_max):
            if ep_yaw_per_turb[t]:
                turb_strs.append(f"T{t}={np.mean(ep_yaw_per_turb[t]):.1f}")
        turb_str = f" [{', '.join(turb_strs)}]" if turb_strs else ""
        print(f"  lambda={lam}: Reward={mean_reward:.2f}, Load={mean_load:.2f}{yaw_str}{turb_str}")

    writer.close()
    envs.close()
    print("Done.")


if __name__ == "__main__":
    main()
