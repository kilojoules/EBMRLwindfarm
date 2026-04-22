#!/usr/bin/env python3
"""
Per-step trajectory logging for paper figures.

Runs evaluation episodes with the trained EBT agent and budget surrogate,
recording per-step data: lambda, wind conditions, yaw actions, power,
cumulative budget usage. Outputs NPZ files for figure generation.

Usage:
    python scripts/log_trajectories.py --domain windfarm \
        --checkpoint runs/ebt_sac_windfarm/checkpoints/step_100000.pt \
        --output results/trajectories_windfarm.npz

    python scripts/log_trajectories.py --domain cheetah \
        --checkpoint checkpoints/sac_cheetah.pt \
        --output results/trajectories_cheetah.npz
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch


def log_cheetah_trajectories(checkpoint, n_episodes=10, horizon=1000,
                              budget_frac=0.25, output_path="results/traj_cheetah.npz"):
    """Log per-step trajectories for HalfCheetah."""
    import gymnasium as gym
    from scripts.safety_gym_budget import (
        GaussianActor, TimeVaryingRewardWrapper, VelocityBudgetSurrogate,
    )

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    env = gym.make("HalfCheetah-v5")
    env = TimeVaryingRewardWrapper(env, amplitude=0.5, period=200)

    budget = int(horizon * budget_frac)

    # Auto velocity threshold
    velocities = []
    for _ in range(3):
        obs, _ = env.reset()
        for _ in range(horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            obs, _, t, tr, _ = env.step(a.squeeze(0).numpy() * act_limit)
            velocities.append(abs(obs[8]) if len(obs) > 8 else abs(obs[0]))
            if t or tr:
                break
    v_threshold = np.percentile(velocities, 50)
    print(f"Velocity threshold: {v_threshold:.2f}")

    all_episodes = []
    for ra in [0.0, 2.0, 5.0]:
        for ep in range(n_episodes):
            surr = VelocityBudgetSurrogate(
                budget_steps=budget, horizon_steps=horizon,
                risk_aversion=ra, steepness=3.0,
            )
            surr.reset()

            obs, _ = env.reset()
            steps = []

            for t in range(horizon):
                lam = surr._compute_lambda()

                with torch.no_grad():
                    a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = a.squeeze(0).numpy() * act_limit

                # Apply penalty
                if surr.cumulative_violations < budget:
                    penalty = surr.penalize_action(action / act_limit)
                    scale = 1.0 / (1.0 + 0.1 * penalty)
                    action = action * scale

                obs, reward, term, trunc, info = env.step(action)
                velocity = abs(obs[8]) if len(obs) > 8 else abs(obs[0])
                above_thresh = velocity > v_threshold
                surr.update(velocity, v_threshold)

                multiplier = info.get("reward_multiplier", 1.0)

                steps.append({
                    "t": t,
                    "lambda": lam,
                    "velocity": velocity,
                    "above_threshold": above_thresh,
                    "reward": reward,
                    "multiplier": multiplier,
                    "cumulative_violations": surr.cumulative_violations,
                    "ra": ra,
                    "episode": ep,
                })

                if term or trunc:
                    break

            all_episodes.extend(steps)

    # Convert to arrays
    data = {k: np.array([s[k] for s in all_episodes]) for k in all_episodes[0]}
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez(output_path, **data, v_threshold=v_threshold, budget=budget)
    print(f"Saved {len(all_episodes)} steps to {output_path}")
    env.close()


def log_windfarm_trajectories(checkpoint, n_episodes=10, horizon=200,
                               budget=15, output_path="results/traj_windfarm.npz"):
    """Log per-step trajectories for wind farm."""
    from config import Args
    from load_surrogates import NegativeYawBudgetSurrogate

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Reuse the evaluate.py checkpoint loading infrastructure
    from evaluate import load_checkpoint_for_eval
    envs, agent, actor, n_turb_loaded, args = load_checkpoint_for_eval(
        checkpoint, device
    )

    n_turb = env_info["n_turbines_max"]
    all_steps = []

    for ra in [0.0, 2.0, 5.0]:
        for gs in [0.1, 0.5]:
            surr = NegativeYawBudgetSurrogate(
                budget_steps=budget, horizon_steps=horizon,
                risk_aversion=ra, steepness=3.0,
            )

            for ep in range(n_episodes):
                surr.reset()
                obs, _ = envs.reset()

                for t in range(horizon):
                    lam = surr._compute_lambda().to(device)
                    lam_val = lam.squeeze().cpu().numpy() if lam.numel() > 1 else float(lam)

                    with torch.no_grad():
                        gfn = surr if gs > 0 else None
                        act = agent.act(envs, obs, guidance_fn=gfn, guidance_scale=gs)

                    obs, rew, _, _, info = envs.step(act)

                    yaw_flat = np.zeros(n_turb)
                    ws_val, wd_val = 0.0, 0.0
                    power_val = 0.0

                    if "yaw angles agent" in info:
                        yaw_arr = np.array(info["yaw angles agent"])
                        yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                        surr.update(torch.tensor(yaw_flat[:n_turb], device=device,
                                                  dtype=torch.float32))

                    if "Power agent" in info:
                        power_val = float(np.mean(info["Power agent"]))

                    # Try to get wind conditions
                    if hasattr(envs, 'envs') and hasattr(envs.envs[0], 'ws'):
                        ws_val = float(envs.envs[0].ws)
                        wd_val = float(envs.envs[0].wd)

                    cum_neg = surr.cumulative_neg_steps
                    cum_neg_arr = cum_neg.squeeze().cpu().numpy() if cum_neg is not None else np.zeros(n_turb)

                    all_steps.append({
                        "t": t, "episode": ep, "ra": ra, "gs": gs,
                        "lambda": float(lam_val) if np.isscalar(lam_val) else float(lam_val[0]),
                        "ws": ws_val, "wd": wd_val,
                        "power": power_val, "reward": float(rew.mean()),
                        **{f"yaw_T{i}": float(yaw_flat[i]) for i in range(min(len(yaw_flat), n_turb))},
                        **{f"cum_neg_T{i}": float(cum_neg_arr[i]) if i < len(cum_neg_arr) else 0
                           for i in range(n_turb)},
                    })

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert list of dicts to dict of arrays
    keys = all_steps[0].keys()
    data = {k: np.array([s[k] for s in all_steps]) for k in keys}
    np.savez(output_path, **data, budget=budget, horizon=horizon, n_turb=n_turb)
    print(f"Saved {len(all_steps)} steps to {output_path}")
    envs.close()


def main():
    parser = argparse.ArgumentParser(description="Log per-step trajectories")
    parser.add_argument("--domain", required=True, choices=["windfarm", "cheetah"])
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="results/trajectories.npz")
    parser.add_argument("--n-episodes", type=int, default=10)
    cli = parser.parse_args()

    if cli.domain == "cheetah":
        log_cheetah_trajectories(cli.checkpoint, cli.n_episodes,
                                  output_path=cli.output)
    else:
        log_windfarm_trajectories(cli.checkpoint, cli.n_episodes,
                                   output_path=cli.output)


if __name__ == "__main__":
    main()
