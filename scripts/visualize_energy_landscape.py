#!/usr/bin/env python3
"""
Visualize energy landscape and constraint composition for trained checkpoints.

Usage:
    python scripts/visualize_energy_landscape.py \
        --checkpoint runs/<run>/checkpoints/step_100000.pt \
        --surrogate t1_positive_only \
        --lambda-val 10.0 \
        --output energy_landscape.png
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Args
from load_surrogates import create_load_surrogate
from helpers.agent import WindFarmAgent
from helpers.constraint_viz import (
    plot_energy_landscape,
    plot_optimization_trajectories,
    plot_yaw_vs_lambda,
    plot_power_vs_lambda,
)


def load_checkpoint_and_setup(path: str, device: torch.device):
    """Load checkpoint, detect actor type, reconstruct env + actor + agent."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    args = Args(**{k: v for k, v in args_dict.items() if hasattr(Args, k)})

    is_ebt = "ebt_opt_steps_train" in args_dict

    if is_ebt:
        from ebt_sac_windfarm import setup_env
        from ebt import TransformerEBTActor as ActorClass
    else:
        from diffusion_sac_windfarm import setup_env
        from diffusion import TransformerDiffusionActor as ActorClass

    env_info = setup_env(args)

    # Profile encoders
    from networks import create_profile_encoding
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

    if is_ebt:
        actor = ActorClass(
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
    else:
        actor = ActorClass(
            action_scale=env_info["action_scale"],
            action_bias=env_info["action_bias"],
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

    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    agent = WindFarmAgent(
        actor=actor, device=device,
        rotor_diameter=env_info["rotor_diameter"],
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=getattr(args, "rotate_profiles", False),
    )

    envs = env_info["envs"]
    return envs, agent, actor, args, is_ebt


def main():
    parser = argparse.ArgumentParser(description="Visualize energy landscape and constraint composition")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--surrogate", default="t1_positive_only", help="Surrogate type")
    parser.add_argument("--lambda-val", type=float, default=1.0, help="Lambda for energy landscape panels")
    parser.add_argument("--grid-res", type=int, default=80, help="Grid resolution for heatmaps")
    parser.add_argument("--eval-steps", type=int, default=50, help="Eval steps per lambda")
    parser.add_argument("--opt-steps", type=int, default=20, help="Optimization steps for trajectory plot")
    parser.add_argument("--num-candidates", type=int, default=8, help="Candidates for trajectory plot")
    parser.add_argument("--output", default=None, help="Output path (default: show interactively)")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli = parser.parse_args()

    device = torch.device(cli.device)
    print(f"Loading checkpoint: {cli.checkpoint}")
    envs, agent, actor, args, is_ebt = load_checkpoint_and_setup(cli.checkpoint, device)

    surrogate = create_load_surrogate(cli.surrogate, steepness=args.load_steepness)
    surrogate = surrogate.to(device)
    print(f"Surrogate: {cli.surrogate} (lambda={cli.lambda_val})")
    print(f"Actor type: {'EBT' if is_ebt else 'Diffusion'}")

    # Get a single observation for energy landscape
    obs, _ = envs.reset()
    batch = agent.batch_preparer.from_envs(envs, obs)

    lambdas = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    figures = {}

    # Panel 1-2: Energy landscape (EBT only)
    print("Generating energy landscape...")
    fig = plot_energy_landscape(
        actor, batch.obs, batch.positions, batch.mask,
        surrogate, cli.lambda_val,
        recep_profile=batch.receptivity, influence_profile=batch.influence,
        grid_res=cli.grid_res,
    )
    if fig is not None:
        figures["energy_landscape"] = fig
        print("  Done.")
    else:
        print("  Skipped (diffusion actor, no energy function).")

    # Panel 3: Optimization trajectories (EBT only)
    print("Generating optimization trajectories...")
    fig = plot_optimization_trajectories(
        actor, batch.obs, batch.positions, batch.mask,
        surrogate, cli.lambda_val,
        num_candidates=cli.num_candidates, num_steps=cli.opt_steps,
        recep_profile=batch.receptivity, influence_profile=batch.influence,
        grid_res=cli.grid_res,
    )
    if fig is not None:
        figures["optimization_trajectories"] = fig
        print("  Done.")
    else:
        print("  Skipped (diffusion actor).")

    # Panel 4: Yaw vs lambda
    print("Generating yaw vs lambda...")
    figures["yaw_vs_lambda"] = plot_yaw_vs_lambda(
        agent, envs, surrogate, lambdas, cli.eval_steps, device,
    )
    print("  Done.")

    # Panel 5: Power vs lambda
    print("Generating power vs lambda...")
    figures["power_vs_lambda"] = plot_power_vs_lambda(
        agent, envs, surrogate, lambdas, cli.eval_steps, device,
    )
    print("  Done.")

    # Save or show
    if cli.output:
        base, ext = os.path.splitext(cli.output)
        for name, fig in figures.items():
            path = f"{base}_{name}{ext}"
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"Saved: {path}")
            plt.close(fig)
    else:
        matplotlib.use("TkAgg")
        plt.show()

    envs.close()
    print("Done.")


if __name__ == "__main__":
    main()
