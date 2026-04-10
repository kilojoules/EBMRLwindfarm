#!/usr/bin/env python3
"""
Demo: per-turbine constraints via energy composition.

Loads a trained diffusion checkpoint and runs evaluation with:
  1. No constraint (baseline power)
  2. Uniform yaw limit (all turbines <= threshold)
  3. Heterogeneous limits (turbine 1 constrained, others free)
  4. Yaw travel budget (limit cumulative yaw travel over a window)

Shows that constraining one turbine forces the system to find a
completely different optimum -- no retraining needed.

Usage:
    python scripts/demo_per_turbine_constraints.py --checkpoint runs/<run>/checkpoints/step_10000.pt
    python scripts/demo_per_turbine_constraints.py --checkpoint runs/<run>/checkpoints/step_10000.pt --steps 200
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import numpy as np
import torch

from config import Args
from diffusion import TransformerDiffusionActor
from load_surrogates import (
    ExponentialYawSurrogate,
    PerTurbineYawSurrogate,
    YawTravelBudgetSurrogate,
    NegativeYawBudgetSurrogate,
)
from helpers.agent import WindFarmAgent
from helpers.helper_funcs import get_env_attention_masks


def load_checkpoint_and_setup(path: str, device: torch.device):
    """Load checkpoint, reconstruct args, set up env and agent."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    args = Args(**{k: v for k, v in args_dict.items() if hasattr(Args, k)})

    # Use the existing setup_env from the training script
    from diffusion_sac_windfarm import setup_env
    env_info = setup_env(args)

    # Build actor from checkpoint
    from networks import create_profile_encoding
    use_profiles = env_info["use_profiles"]

    shared_recep_encoder, shared_influence_encoder = None, None
    if use_profiles:
        encoder_kwargs = {}
        shared_recep_encoder, shared_influence_encoder = create_profile_encoding(
            profile_type=args.profile_encoding_type,
            embed_dim=args.embed_dim,
            hidden_channels=args.profile_encoder_hidden,
            **encoder_kwargs,
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

    actor = TransformerDiffusionActor(
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
        rotate_profiles=args.rotate_profiles if hasattr(args, 'rotate_profiles') else False,
    )

    envs = env_info["envs"]
    n_turbines = env_info["n_turbines_max"]

    return envs, agent, actor, n_turbines, args


def run_episode(envs, agent, guidance_fn, guidance_scale, num_steps, device,
                stateful_surrogate=None):
    """Run one evaluation episode and return metrics.

    Args:
        stateful_surrogate: Any surrogate with reset()/update() methods
            (e.g. YawTravelBudgetSurrogate, NegativeYawBudgetSurrogate).
            When provided, it is used as the guidance_fn and updated each step.
    """
    obs, _ = envs.reset()
    if stateful_surrogate is not None:
        stateful_surrogate.reset()

    total_reward = 0.0
    yaw_trajectory = []
    powers = []

    for step in range(num_steps):
        gfn = stateful_surrogate if stateful_surrogate is not None else guidance_fn

        with torch.no_grad():
            act = agent.act(envs, obs, guidance_fn=gfn, guidance_scale=guidance_scale)

        obs, rew, _, _, info = envs.step(act)
        total_reward += float(np.mean(rew))

        if "yaw angles agent" in info:
            yaw = np.array(info["yaw angles agent"])
            yaw_trajectory.append(yaw[0] if yaw.ndim > 1 else yaw)

            if stateful_surrogate is not None:
                yaw_t = torch.tensor(yaw[0] if yaw.ndim > 1 else yaw,
                                     device=device, dtype=torch.float32)
                stateful_surrogate.update(yaw_t)

        if "Power agent" in info:
            powers.append(float(np.mean(info["Power agent"])))

    yaw_arr = np.array(yaw_trajectory) if yaw_trajectory else np.array([])
    return {
        "reward": total_reward,
        "mean_power": np.mean(powers) if powers else 0.0,
        "mean_abs_yaw": np.mean(np.abs(yaw_arr)) if yaw_arr.size else 0.0,
        "final_yaw": yaw_arr[-1] if yaw_arr.size else np.array([]),
        "yaw_trajectory": yaw_arr,
    }


def main():
    parser = argparse.ArgumentParser(description="Demo per-turbine constraints")
    parser.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--steps", type=int, default=100, help="Steps per episode")
    parser.add_argument("--guidance-scale", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    cli_args = parser.parse_args()

    device = torch.device(cli_args.device)
    print(f"Loading checkpoint: {cli_args.checkpoint}")
    envs, agent, actor, n_turbines, args = load_checkpoint_and_setup(
        cli_args.checkpoint, device)

    lam = cli_args.guidance_scale
    num_steps = cli_args.steps

    # Budget of 25% of episode steps at negative yaw, over the full episode
    budget_steps = max(num_steps // 4, 1)

    scenarios = [
        ("No constraint", None, None),
        ("Uniform +/-15 deg", ExponentialYawSurrogate(15.0, 30.0, 10.0), None),
        ("Uniform +/-10 deg", ExponentialYawSurrogate(10.0, 30.0, 10.0), None),
        ("T1 +/-5, others +/-20",
         PerTurbineYawSurrogate([5.0] + [20.0] * (n_turbines - 1), 30.0, 10.0), None),
        ("T1 +/-5, others free",
         PerTurbineYawSurrogate([5.0] + [30.0] * (n_turbines - 1), 30.0, 10.0), None),
        ("Travel 50 deg/100 steps", None,
         YawTravelBudgetSurrogate(50.0, 100, 30.0, 5.0)),
        ("Travel 20 deg/100 steps", None,
         YawTravelBudgetSurrogate(20.0, 100, 30.0, 5.0)),
        (f"NegYaw budget (risk=0)", None,
         NegativeYawBudgetSurrogate(budget_steps, num_steps, 0.0, 10.0)),
        (f"NegYaw budget (risk=1)", None,
         NegativeYawBudgetSurrogate(budget_steps, num_steps, 1.0, 10.0)),
        (f"NegYaw budget (risk=5)", None,
         NegativeYawBudgetSurrogate(budget_steps, num_steps, 5.0, 10.0)),
    ]

    print(f"\n{'='*75}")
    print(f"  Per-Turbine Constraint Demo | {n_turbines} turbines | lam={lam} | {num_steps} steps")
    print(f"{'='*75}\n")

    results = []
    for name, gfn, stateful_surr in scenarios:
        try:
            result = run_episode(envs, agent, gfn, lam, num_steps, device,
                                 stateful_surrogate=stateful_surr)
        except Exception as e:
            print(f"  {name:30s} | FAILED: {e}")
            # Recreate env after crash
            envs.close()
            envs, _, _, _, _ = load_checkpoint_and_setup(cli_args.checkpoint, device)
            result = {"reward": 0, "mean_power": 0, "mean_abs_yaw": 0,
                      "final_yaw": np.array([]), "yaw_trajectory": np.array([])}
        results.append((name, result))

        final_yaw_str = ""
        if result["final_yaw"].size:
            angles = [f"{y:+.1f}" for y in result["final_yaw"]]
            final_yaw_str = f"  Final yaw: [{', '.join(angles)}]"

        print(f"  {name:30s} | Reward={result['reward']:8.2f} | "
              f"Power={result['mean_power']:12.0f} | "
              f"MeanAbsYaw={result['mean_abs_yaw']:5.1f} deg"
              f"{final_yaw_str}")

    baseline_reward = results[0][1]["reward"]
    print(f"\n{'_'*75}")
    print(f"  Power cost of constraints (vs unconstrained):")
    for name, result in results[1:]:
        delta = result["reward"] - baseline_reward
        pct = 100 * delta / abs(baseline_reward) if baseline_reward != 0 else 0
        print(f"    {name:30s}: {delta:+.2f} reward ({pct:+.1f}%)")

    envs.close()
    print(f"\nDone.")


if __name__ == "__main__":
    main()
