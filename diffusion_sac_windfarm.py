"""
Diffusion-SAC for Wind Farm Control.

Trains a diffusion-based EBM actor with SAC critics on the toy wind farm env.
After training, demonstrates post-hoc load-safety composition via classifier guidance.

Usage:
    # Train diffusion actor
    python diffusion_sac_windfarm.py --total_timesteps 50000

    # Train and test with load guidance
    python diffusion_sac_windfarm.py --total_timesteps 50000 --guidance_scale 0.5
"""

import os
import random
import time
from collections import deque

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter

import gymnasium as gym

from config import Args
from replay_buffer import TransformerReplayBuffer
from networks import TransformerCritic
from diffusion import TransformerDiffusionActor, ReluLoadSurrogate
from helpers.helper_funcs import soft_update
from toy_wind_farm_env import ToyWindFarmEnv


def make_toy_env(seed: int = 0):
    """Create a toy wind farm env wrapped in a vectorized interface."""
    def _make():
        env = ToyWindFarmEnv()
        env.reset(seed=seed)
        return env
    return _make


def run_assessment(
    actor, env, device, rotor_diameter, n_turbines, label,
    guidance_fn=None, guidance_scale=0.0, num_episodes=5, max_steps=50,
):
    """Run assessment episodes and print results."""
    total_rewards = []
    total_loads = []
    all_final_yaws = []
    load_fn = ReluLoadSurrogate()

    for ep in range(num_episodes):
        obs, info = env.reset()
        ep_reward = 0.0
        ep_load = 0.0

        for step in range(max_steps):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            positions = torch.tensor(
                env.positions / rotor_diameter,
                dtype=torch.float32, device=device,
            ).unsqueeze(0)
            mask = torch.zeros(1, n_turbines, dtype=torch.bool, device=device)

            with torch.no_grad():
                action_tensor, _, _, _ = actor.get_action(
                    obs_tensor, positions, mask,
                    guidance_fn=guidance_fn,
                    guidance_scale=guidance_scale,
                )
            action = action_tensor.squeeze(-1).cpu().numpy()[0]

            # Compute load for this action
            with torch.no_grad():
                load = load_fn(action_tensor, mask).item()
            ep_load += load

            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward

            if terminated or truncated:
                break

        total_rewards.append(ep_reward)
        total_loads.append(ep_load)
        all_final_yaws.append(info["yaw_angles"].copy())

    avg_reward = np.mean(total_rewards)
    avg_load = np.mean(total_loads)
    avg_yaws = np.mean(all_final_yaws, axis=0)

    print(f"  [{label}] Reward: {avg_reward:.4f} | Load: {avg_load:.4f} | "
          f"Final yaws: [{avg_yaws[0]:+.1f}, {avg_yaws[1]:+.1f}, {avg_yaws[2]:+.1f}]")


def main():
    args = tyro.cli(Args)

    # Override defaults for toy env
    if args.exp_name == "transformer_sac_windfarm":
        args.exp_name = "diffusion_sac_toy"

    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    # --- Environment setup ---
    envs = gym.vector.SyncVectorEnv([make_toy_env(args.seed)])
    single_env = ToyWindFarmEnv()

    n_turbines = single_env.n_turbines  # 3
    obs_dim_per_turbine = single_env.observation_space.shape[1]  # 2
    action_dim_per_turbine = 1
    rotor_diameter = single_env.rotor_diameter

    # Action scaling: env actions in [-1, 1], no extra scaling needed
    action_scale = 1.0
    action_bias = 0.0

    # --- TensorBoard ---
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters", str(args))

    # --- Networks ---
    common_kwargs = dict(
        obs_dim_per_turbine=obs_dim_per_turbine,
        action_dim_per_turbine=action_dim_per_turbine,
        embed_dim=args.embed_dim,
        pos_embed_dim=args.pos_embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        pos_encoding_type=args.pos_encoding_type,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
        pos_embedding_mode=args.pos_embedding_mode,
        profile_encoding=args.profile_encoding_type,
        profile_encoder_hidden=args.profile_encoder_hidden,
        n_profile_directions=args.n_profile_directions,
        profile_fusion_type=args.profile_fusion_type,
        profile_embed_mode=args.profile_embed_mode,
        args=args,
    )

    actor = TransformerDiffusionActor(
        action_scale=action_scale,
        action_bias=action_bias,
        num_diffusion_steps=args.num_diffusion_steps,
        num_inference_steps=args.num_inference_steps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
        timestep_embed_dim=args.timestep_embed_dim,
        denoiser_hidden_dim=args.denoiser_hidden_dim,
        denoiser_num_layers=args.denoiser_num_layers,
        **common_kwargs,
    ).to(device)

    qf1 = TransformerCritic(**common_kwargs).to(device)
    qf2 = TransformerCritic(**common_kwargs).to(device)
    qf1_target = TransformerCritic(**common_kwargs).to(device)
    qf2_target = TransformerCritic(**common_kwargs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    # Optimizers
    actor_optimizer = optim.Adam(actor.parameters(), lr=args.policy_lr)
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )

    # Load surrogate for post-training assessment
    load_surrogate = ReluLoadSurrogate()

    # --- Replay buffer ---
    rb = TransformerReplayBuffer(
        capacity=args.buffer_size,
        device=device,
        rotor_diameter=rotor_diameter,
        max_turbines=n_turbines,
        obs_dim=obs_dim_per_turbine,
        action_dim=action_dim_per_turbine,
        use_wind_relative=False,  # Toy env has fixed wind, no need
        use_profiles=False,
    )

    # --- Logging ---
    print(f"Device: {device}")
    print(f"Actor params: {sum(p.numel() for p in actor.parameters()):,}")
    print(f"Critic params: {sum(p.numel() for p in qf1.parameters()):,}")
    print(f"Diffusion steps: {args.num_diffusion_steps} (train), {args.num_inference_steps} (infer)")
    print(f"Starting training for {args.total_timesteps} steps...")

    # --- Training loop ---
    obs, info = envs.reset()
    episode_rewards = deque(maxlen=20)
    episode_reward = 0.0
    episode_count = 0
    total_gradient_steps = 0

    start_time = time.time()

    for global_step in range(1, args.total_timesteps + 1):
        # --- Action selection ---
        if global_step < args.learning_starts:
            # Random exploration
            actions = np.array([envs.single_action_space.sample()])
        else:
            actor.eval()
            with torch.no_grad():
                obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device)
                positions = torch.tensor(
                    single_env.positions / rotor_diameter,
                    dtype=torch.float32, device=device,
                ).unsqueeze(0)
                mask = torch.zeros(1, n_turbines, dtype=torch.bool, device=device)

                action_tensor, _, _, _ = actor.get_action(
                    obs_tensor, positions, mask,
                )
            actions = action_tensor.squeeze(-1).cpu().numpy()
            actor.train()

        # --- Environment step ---
        next_obs, rewards, terminateds, truncateds, infos = envs.step(actions)
        dones = np.logical_or(terminateds, truncateds)

        episode_reward += float(rewards[0])

        # Store transition
        action_to_store = actions[0].reshape(n_turbines, action_dim_per_turbine)
        rb.add(
            obs=obs[0],
            next_obs=next_obs[0],
            action=action_to_store,
            reward=float(rewards[0]),
            done=bool(dones[0]),
            raw_positions=single_env.positions,
            attention_mask=np.zeros(n_turbines, dtype=bool),
            wind_direction=270.0,
        )

        obs = next_obs

        # Episode done
        if dones[0]:
            episode_rewards.append(episode_reward)
            episode_count += 1
            writer.add_scalar("charts/episode_reward", episode_reward, global_step)

            if episode_count % 10 == 0:
                avg_reward = np.mean(episode_rewards) if episode_rewards else 0
                sps = global_step / (time.time() - start_time)
                print(f"Step {global_step:>6d} | Ep {episode_count:>4d} | "
                      f"Avg reward: {avg_reward:.4f} | SPS: {sps:.0f}")

            episode_reward = 0.0
            obs, info = envs.reset()

        # --- Training updates ---
        if global_step >= args.learning_starts:
            for _ in range(int(args.utd_ratio)):
                data = rb.sample(args.batch_size)
                batch_mask = data["attention_mask"]

                # =============================================================
                # Critic update
                # =============================================================
                with torch.no_grad():
                    next_actions, _, _, _ = actor.get_action(
                        data["next_observations"],
                        data["positions"],
                        batch_mask,
                    )
                    qf1_next = qf1_target(
                        data["next_observations"], next_actions,
                        data["positions"], batch_mask,
                    )
                    qf2_next = qf2_target(
                        data["next_observations"], next_actions,
                        data["positions"], batch_mask,
                    )
                    # No entropy term: just min Q
                    min_qf_next = torch.min(qf1_next, qf2_next)
                    target_q = data["rewards"] + (1 - data["dones"]) * args.gamma * min_qf_next

                qf1_value = qf1(data["observations"], data["actions"],
                                data["positions"], batch_mask)
                qf2_value = qf2(data["observations"], data["actions"],
                                data["positions"], batch_mask)

                qf1_loss = F.mse_loss(qf1_value, target_q)
                qf2_loss = F.mse_loss(qf2_value, target_q)
                qf_loss = qf1_loss + qf2_loss

                q_optimizer.zero_grad(set_to_none=True)
                qf_loss.backward()
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(
                        list(qf1.parameters()) + list(qf2.parameters()),
                        max_norm=args.grad_clip_max_norm,
                    )
                q_optimizer.step()

                total_gradient_steps += 1

                # =============================================================
                # Actor update (delayed)
                # =============================================================
                if total_gradient_steps % args.policy_frequency == 0:
                    # Encode once, denoise K times
                    turbine_emb = actor.encode(
                        data["observations"], data["positions"], batch_mask,
                    )

                    # Generate actions via full denoising chain (differentiable)
                    actions_pi = actor.denoise_chain(
                        turbine_emb, batch_mask, use_ddim=False,
                    )

                    # Scale to action space for critic
                    actions_pi_scaled = actions_pi * actor.action_scale + actor.action_bias_val

                    # Q-values for generated actions
                    qf1_pi = qf1(data["observations"], actions_pi_scaled,
                                 data["positions"], batch_mask)
                    qf2_pi = qf2(data["observations"], actions_pi_scaled,
                                 data["positions"], batch_mask)
                    min_qf_pi = torch.min(qf1_pi, qf2_pi)

                    # Actor loss: maximize Q-value (no entropy term)
                    actor_loss = -min_qf_pi.mean()

                    # Optional: diffusion BC loss for stability
                    if args.diffusion_bc_weight > 0:
                        batch_size_cur = data["actions"].shape[0]
                        t = torch.randint(
                            0, args.num_diffusion_steps,
                            (batch_size_cur,), device=device,
                        )
                        noise = torch.randn_like(data["actions"])
                        # Normalize stored actions to [-1, 1] for diffusion
                        actions_norm = (data["actions"] - actor.action_bias_val) / actor.action_scale
                        noisy_actions = actor.q_sample(actions_norm, t, noise)
                        eps_pred = actor.predict_noise(
                            turbine_emb.detach(), noisy_actions, t, batch_mask,
                        )
                        bc_loss = F.mse_loss(eps_pred, noise)
                        actor_loss = actor_loss + args.diffusion_bc_weight * bc_loss

                    actor_optimizer.zero_grad(set_to_none=True)
                    actor_loss.backward()
                    if args.grad_clip:
                        torch.nn.utils.clip_grad_norm_(
                            actor.parameters(),
                            max_norm=args.grad_clip_max_norm,
                        )
                    actor_optimizer.step()

                    # Log
                    if total_gradient_steps % 100 == 0:
                        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                        writer.add_scalar("losses/qf_loss", qf_loss.item(), global_step)
                        writer.add_scalar("losses/qf1_value", qf1_value.mean().item(), global_step)

                # =============================================================
                # Target network update
                # =============================================================
                if total_gradient_steps % args.target_network_frequency == 0:
                    soft_update(qf1, qf1_target, args.tau)
                    soft_update(qf2, qf2_target, args.tau)

    # --- Final assessment ---
    print("\n=== Final Assessment ===")
    actor.eval()
    run_assessment(actor, single_env, device, rotor_diameter, n_turbines, "No guidance")

    if args.guidance_scale > 0:
        print(f"\n=== Assessment with Load Guidance (lambda={args.guidance_scale}) ===")
        run_assessment(
            actor, single_env, device, rotor_diameter, n_turbines,
            f"Guidance lambda={args.guidance_scale}",
            guidance_fn=load_surrogate,
            guidance_scale=args.guidance_scale,
        )

    # Sweep guidance scales
    print("\n=== Guidance Scale Sweep ===")
    for lam in [0.0, 0.1, 0.5, 1.0, 2.0]:
        gfn = load_surrogate if lam > 0 else None
        run_assessment(
            actor, single_env, device, rotor_diameter, n_turbines,
            f"lambda={lam}",
            guidance_fn=gfn,
            guidance_scale=lam,
        )

    # Save checkpoint
    if args.save_model:
        os.makedirs(f"runs/{run_name}/checkpoints", exist_ok=True)
        checkpoint_path = f"runs/{run_name}/checkpoints/final.pt"
        torch.save({
            "actor": actor.state_dict(),
            "qf1": qf1.state_dict(),
            "qf2": qf2.state_dict(),
            "args": args,
        }, checkpoint_path)
        print(f"\nCheckpoint saved to {checkpoint_path}")

    writer.close()
    envs.close()
    print("Done.")


if __name__ == "__main__":
    main()
