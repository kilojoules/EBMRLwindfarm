"""Minimal AC schedule eval at single config for multi-seed aggregation.

No bisection. No baselines. Just: load checkpoint → run AC eta=5, gs=0.1
at budget=15 for N episodes → save power + neg-yaw stats.
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, ".")

from config import Args
from helpers.agent import WindFarmAgent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--eta", type=float, default=5.0)
    p.add_argument("--gs", type=float, default=0.1)
    p.add_argument("--k", type=float, default=2.0)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tr_args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})

    from ebt_sac_windfarm import setup_env
    env_info = setup_env(tr_args)
    envs = env_info["envs"]
    n_turb = env_info["n_turbines_max"]

    from ebt import TransformerEBTActor
    from networks import create_profile_encoding
    use_profiles = env_info["use_profiles"]
    sr, si = None, None
    if use_profiles:
        sr, si = create_profile_encoding(
            profile_type=tr_args.profile_encoding_type,
            embed_dim=tr_args.embed_dim,
            hidden_channels=tr_args.profile_encoder_hidden)
    actor = TransformerEBTActor(
        obs_dim_per_turbine=env_info["obs_dim_per_turbine"],
        action_dim_per_turbine=1,
        embed_dim=tr_args.embed_dim, num_heads=tr_args.num_heads,
        num_layers=tr_args.num_layers, mlp_ratio=tr_args.mlp_ratio,
        dropout=tr_args.dropout,
        pos_encoding_type=tr_args.pos_encoding_type,
        pos_embed_dim=tr_args.pos_embed_dim,
        pos_embedding_mode=tr_args.pos_embedding_mode,
        rel_pos_hidden_dim=tr_args.rel_pos_hidden_dim,
        rel_pos_per_head=tr_args.rel_pos_per_head,
        profile_encoding=tr_args.profile_encoding_type,
        shared_recep_encoder=sr, shared_influence_encoder=si,
        action_scale=env_info["action_scale"],
        action_bias=env_info["action_bias"],
        opt_steps_train=tr_args.ebt_opt_steps_train,
        opt_steps_eval=tr_args.ebt_opt_steps_eval,
        opt_lr=tr_args.ebt_opt_lr,
        num_candidates=tr_args.ebt_num_candidates,
        args=tr_args).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    agent = WindFarmAgent(
        actor=actor, device=device,
        rotor_diameter=env_info["rotor_diameter"],
        use_wind_relative=tr_args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=getattr(tr_args, "rotate_profiles", False))

    from load_surrogates import NegativeYawBudgetSurrogate

    # Unconstrained baseline
    print("1. Unconstrained...")
    powers, negs = [], []
    for ep in range(args.n_episodes):
        obs, _ = envs.reset()
        p_ep, nc = 0.0, np.zeros(n_turb)
        for t in range(args.horizon):
            with torch.no_grad():
                act = agent.act(envs, obs)
            obs, _, _, _, info = envs.step(act)
            if "yaw angles agent" in info:
                ya = np.array(info["yaw angles agent"])
                yf = ya[0] if ya.ndim > 1 else ya
                for ti in range(min(len(yf), n_turb)):
                    if yf[ti] < 0:
                        nc[ti] += 1
            if "Power agent" in info:
                p_ep += float(np.mean(info["Power agent"]))
        powers.append(p_ep / args.horizon)
        negs.append(nc)
    uncon = {
        "power_mean": float(np.mean(powers)),
        "power_se": float(np.std(powers) / np.sqrt(len(powers))),
        "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
    }
    print(f"  power={uncon['power_mean']:.0f}±{uncon['power_se']:.0f}, "
          f"neg={[int(x) for x in uncon['neg_yaw_mean']]}")

    # AC schedule
    print(f"\n2. AC η={args.eta} gs={args.gs} k={args.k} d={args.budget}...")
    powers, negs = [], []
    for ep in range(args.n_episodes):
        surr = NegativeYawBudgetSurrogate(
            budget_steps=args.budget, horizon_steps=args.horizon,
            risk_aversion=args.eta, steepness=args.k, yaw_max_deg=30.0)
        surr.reset()
        obs, _ = envs.reset()
        p_ep, nc = 0.0, np.zeros(n_turb)
        for t in range(args.horizon):
            with torch.no_grad():
                act = agent.act(envs, obs, guidance_fn=surr, guidance_scale=args.gs)
            obs, _, _, _, info = envs.step(act)
            if "yaw angles agent" in info:
                ya = np.array(info["yaw angles agent"])
                yf = ya[0] if ya.ndim > 1 else ya
                for ti in range(min(len(yf), n_turb)):
                    if yf[ti] < 0:
                        nc[ti] += 1
                surr.update(torch.tensor(yf[:n_turb], device=device,
                                          dtype=torch.float32))
            if "Power agent" in info:
                p_ep += float(np.mean(info["Power agent"]))
        powers.append(p_ep / args.horizon)
        negs.append(nc)
    ac = {
        "power_mean": float(np.mean(powers)),
        "power_se": float(np.std(powers) / np.sqrt(len(powers))),
        "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
        "budget_satisfied": bool(np.all(np.mean(negs, axis=0) <= args.budget)),
    }
    pct = 100 * ac["power_mean"] / uncon["power_mean"]
    ok = "✓" if ac["budget_satisfied"] else "✗"
    print(f"  power={ac['power_mean']:.0f}±{ac['power_se']:.0f} "
          f"({pct:.1f}%), neg={[int(x) for x in ac['neg_yaw_mean']]} {ok}")

    out = {"unconstrained": uncon, "ac": ac, "pct_uncon": pct,
           "meta": {"budget": args.budget, "eta": args.eta,
                    "gs": args.gs, "k": args.k,
                    "n_episodes": args.n_episodes,
                    "checkpoint": args.checkpoint}}
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
