#!/usr/bin/env python3
"""
Comprehensive wind farm budget constraint sweep.

Sweeps budget levels × η × k × gs with many episodes per config
for tight confidence intervals. Outputs structured JSON for paper figures.

Usage:
    python scripts/windfarm_sweep.py \
        --checkpoint runs/ebt_sac_windfarm/checkpoints/step_100000.pt \
        --n-episodes 50 --output results/windfarm_sweep.json
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from config import Args
from load_surrogates import NegativeYawBudgetSurrogate
from helpers.agent import WindFarmAgent


def load_windfarm(checkpoint_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    args = Args(**{k: v for k, v in args_dict.items() if hasattr(Args, k)})

    from ebt_sac_windfarm import setup_env
    env_info = setup_env(args)
    envs = env_info["envs"]
    n_turb = env_info["n_turbines_max"]

    from ebt import TransformerEBTActor
    from networks import create_profile_encoding

    use_profiles = env_info["use_profiles"]
    shared_recep, shared_inf = None, None
    if use_profiles:
        shared_recep, shared_inf = create_profile_encoding(
            profile_type=args.profile_encoding_type,
            embed_dim=args.embed_dim,
            hidden_channels=args.profile_encoder_hidden,
        )

    actor = TransformerEBTActor(
        obs_dim_per_turbine=env_info["obs_dim_per_turbine"],
        action_dim_per_turbine=1,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        mlp_ratio=args.mlp_ratio,
        dropout=args.dropout,
        pos_encoding_type=args.pos_encoding_type,
        pos_embed_dim=args.pos_embed_dim,
        pos_embedding_mode=args.pos_embedding_mode,
        rel_pos_hidden_dim=args.rel_pos_hidden_dim,
        rel_pos_per_head=args.rel_pos_per_head,
        profile_encoding=args.profile_encoding_type,
        shared_recep_encoder=shared_recep,
        shared_influence_encoder=shared_inf,
        action_scale=env_info["action_scale"],
        action_bias=env_info["action_bias"],
        opt_steps_train=args.ebt_opt_steps_train,
        opt_steps_eval=args.ebt_opt_steps_eval,
        opt_lr=args.ebt_opt_lr,
        num_candidates=args.ebt_num_candidates,
        args=args,
    ).to(device)

    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    agent = WindFarmAgent(
        actor=actor, device=device,
        rotor_diameter=env_info["rotor_diameter"],
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=getattr(args, 'rotate_profiles', False),
    )

    return agent, envs, n_turb, device


def run_episode(agent, envs, n_turb, device, surr, gs, horizon):
    if surr is not None:
        surr.reset()
    obs, _ = envs.reset()
    ep_power = 0.0
    neg_counts = np.zeros(n_turb)

    for t in range(horizon):
        with torch.no_grad():
            if surr is not None and gs > 0:
                act = agent.act(envs, obs, guidance_fn=surr, guidance_scale=gs)
            else:
                act = agent.act(envs, obs)

        obs, rew, _, _, info = envs.step(act)

        if "yaw angles agent" in info:
            yaw_arr = np.array(info["yaw angles agent"])
            yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
            for ti in range(min(len(yaw_flat), n_turb)):
                if yaw_flat[ti] < 0:
                    neg_counts[ti] += 1
            if surr is not None and hasattr(surr, 'update'):
                surr.update(torch.tensor(
                    yaw_flat[:n_turb], device=device, dtype=torch.float32,
                ))

        if "Power agent" in info:
            ep_power += float(np.mean(info["Power agent"]))

    return ep_power / horizon, neg_counts


def run_config(agent, envs, n_turb, device, budget_steps, horizon, ra, k, gs,
               n_episodes, schedule_type="exp"):
    powers, negs = [], []
    for _ in range(n_episodes):
        surr = NegativeYawBudgetSurrogate(
            budget_steps=budget_steps, horizon_steps=horizon,
            risk_aversion=ra, steepness=k, yaw_max_deg=30.0,
            schedule_type=schedule_type,
        )
        p, n = run_episode(agent, envs, n_turb, device, surr, gs, horizon)
        powers.append(p)
        negs.append(n)
    return {
        "power_mean": float(np.mean(powers)),
        "power_std": float(np.std(powers)),
        "power_se": float(np.std(powers) / np.sqrt(len(powers))),
        "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
        "neg_yaw_std": np.std(negs, axis=0).tolist(),
        "neg_yaw_max_mean": float(np.mean(np.max(negs, axis=1))),
        "budget_satisfied": bool(np.all(np.mean(negs, axis=0) <= budget_steps)),
    }


def main():
    parser = argparse.ArgumentParser(description="Wind farm budget sweep")
    parser.add_argument("--checkpoint",
                        default="runs/ebt_sac_windfarm/checkpoints/step_100000.pt")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--budgets", type=int, nargs="+",
                        default=None, help="Budget levels to sweep (default: all)")
    parser.add_argument("--output", default="results/windfarm_sweep.json")
    cli = parser.parse_args()

    if not os.path.exists(cli.checkpoint):
        import glob
        ckpts = glob.glob("runs/*/checkpoints/step_*.pt")
        if ckpts:
            cli.checkpoint = sorted(ckpts)[-1]
        else:
            print("No checkpoint found.")
            return

    agent, envs, n_turb, device = load_windfarm(cli.checkpoint)
    horizon = cli.horizon
    n_ep = cli.n_episodes
    print(f"Loaded {n_turb}-turbine wind farm, {n_ep} episodes per config")

    # --- Unconstrained baseline ---
    print("Running unconstrained baseline...")
    powers, negs = [], []
    for _ in range(n_ep):
        obs, _ = envs.reset()
        ep_power = 0.0
        neg_counts = np.zeros(n_turb)
        for t in range(horizon):
            with torch.no_grad():
                act = agent.act(envs, obs)
            obs, _, _, _, info = envs.step(act)
            if "yaw angles agent" in info:
                yaw_arr = np.array(info["yaw angles agent"])
                yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                for ti in range(min(len(yaw_flat), n_turb)):
                    if yaw_flat[ti] < 0:
                        neg_counts[ti] += 1
            if "Power agent" in info:
                ep_power += float(np.mean(info["Power agent"]))
        powers.append(ep_power / horizon)
        negs.append(neg_counts)

    uncon = {
        "power_mean": float(np.mean(powers)),
        "power_std": float(np.std(powers)),
        "power_se": float(np.std(powers) / np.sqrt(len(powers))),
        "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
    }
    print(f"  Unconstrained: Power={uncon['power_mean']:.0f}±{uncon['power_se']:.0f} (SE), "
          f"NegYaw={[int(x) for x in uncon['neg_yaw_mean']]}")

    results = {"unconstrained": uncon, "configs": [], "meta": {
        "n_episodes": n_ep, "horizon": horizon, "n_turbines": n_turb,
        "checkpoint": cli.checkpoint,
    }}

    # --- Main sweep ---
    budget_levels = cli.budgets or [5, 10, 15, 20, 25, 30, 50, 75, 100]
    eta_values = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0]
    k_values = [2.0, 3.0, 5.0]
    gs_values = [0.05, 0.1, 0.2, 0.5]

    total = len(budget_levels) * len(eta_values) * len(k_values) * len(gs_values)
    print(f"\nSweep: {len(budget_levels)} budgets × {len(eta_values)} η × "
          f"{len(k_values)} k × {len(gs_values)} gs = {total} configs × {n_ep} episodes")

    header = (f"{'Budget':>6s} {'η':>5s} {'k':>4s} {'gs':>5s} | "
              f"{'Power':>12s} {'SE':>6s} {'%Uncon':>7s} {'NegYaw':>20s} {'OK':>3s}")
    print(header)
    print("-" * len(header))

    done = 0
    t0 = time.time()
    for budget_steps in budget_levels:
        for ra in eta_values:
            for k_val in k_values:
                for gs_val in gs_values:
                    res = run_config(agent, envs, n_turb, device,
                                     budget_steps, horizon, ra, k_val, gs_val,
                                     n_ep)
                    pct = 100 * res["power_mean"] / uncon["power_mean"]
                    neg_str = str([int(x) for x in res["neg_yaw_mean"]])
                    ok = "✓" if res["budget_satisfied"] else "✗"
                    print(f"  {budget_steps:4d} {ra:5.1f} {k_val:4.1f} {gs_val:5.2f} | "
                          f"{res['power_mean']:12.0f} {res['power_se']:6.0f} "
                          f"{pct:6.1f}% {neg_str:>20s} {ok:>3s}")

                    results["configs"].append({
                        "budget": budget_steps, "eta": ra, "k": k_val, "gs": gs_val,
                        **res, "pct_unconstrained": pct,
                    })

                    done += 1
                    if done % 20 == 0:
                        elapsed = time.time() - t0
                        rate = elapsed / done
                        remaining = rate * (total - done)
                        print(f"  --- {done}/{total} done, "
                              f"{elapsed/60:.1f}min elapsed, "
                              f"~{remaining/60:.1f}min remaining ---")

    # --- 1/u schedule comparison (subset) ---
    print("\n--- 1/u schedule comparison ---")
    for budget_steps in [15, 25, 50]:
        for ra in [0.5, 1.0, 2.0, 5.0]:
            res = run_config(agent, envs, n_turb, device,
                             budget_steps, horizon, ra, 2.0, 0.1,
                             n_ep, schedule_type="inverse")
            pct = 100 * res["power_mean"] / uncon["power_mean"]
            neg_str = str([int(x) for x in res["neg_yaw_mean"]])
            ok = "✓" if res["budget_satisfied"] else "✗"
            print(f"  1/u  B={budget_steps:3d} η={ra:4.1f} | "
                  f"{res['power_mean']:12.0f} {res['power_se']:6.0f} "
                  f"{pct:6.1f}% {neg_str:>20s} {ok:>3s}")
            results["configs"].append({
                "budget": budget_steps, "eta": ra, "k": 2.0, "gs": 0.1,
                "schedule": "inverse", **res, "pct_unconstrained": pct,
            })

    os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
    with open(cli.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n{len(results['configs'])} configs saved to {cli.output}")

    envs.close()
    print("Done.")


if __name__ == "__main__":
    main()
