#!/usr/bin/env python3
"""
PPO-Lagrangian baseline for wind farm budget constraint.

Trains a constrained policy with a Lagrangian dual variable that
adapts during training. This is the standard CMDP baseline that
requires retraining for each budget specification.

Comparison: PPO-Lag (retrained) vs AC schedule (post-hoc, zero retraining).

Usage:
    python scripts/ppo_lag_baseline.py \
        --checkpoint runs/ebt_sac_windfarm/checkpoints/step_100000.pt \
        --budget 15 --n-eval-episodes 50
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Args
from helpers.agent import WindFarmAgent


def load_windfarm(checkpoint_path):
    """Load wind farm env and EBT actor (for comparison baseline)."""
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
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, mlp_ratio=args.mlp_ratio,
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

    return agent, envs, n_turb, device, args


def eval_with_constant_lambda(agent, envs, n_turb, device, lam_value,
                               budget, horizon, n_episodes, args):
    """Evaluate EBT actor with a constant penalty (simulating tuned Lagrangian)."""
    from load_surrogates import NegativeYawBudgetSurrogate

    powers, negs = [], []
    for _ in range(n_episodes):
        surr = NegativeYawBudgetSurrogate(
            budget_steps=budget, horizon_steps=horizon,
            risk_aversion=0.0,  # constant lambda (no urgency adaptation)
            steepness=lam_value,  # use steepness as the penalty magnitude
            yaw_max_deg=30.0,
        )
        surr.reset()
        obs, _ = envs.reset()
        ep_power = 0.0
        neg_counts = np.zeros(n_turb)

        for t in range(horizon):
            with torch.no_grad():
                act = agent.act(envs, obs, guidance_fn=surr, guidance_scale=0.1)
            obs, rew, _, _, info = envs.step(act)

            if "yaw angles agent" in info:
                yaw_arr = np.array(info["yaw angles agent"])
                yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                for ti in range(min(len(yaw_flat), n_turb)):
                    if yaw_flat[ti] < 0:
                        neg_counts[ti] += 1
                surr.update(torch.tensor(
                    yaw_flat[:n_turb], device=device, dtype=torch.float32,
                ))
            if "Power agent" in info:
                ep_power += float(np.mean(info["Power agent"]))

        powers.append(ep_power / horizon)
        negs.append(neg_counts)

    return {
        "power_mean": float(np.mean(powers)),
        "power_std": float(np.std(powers)),
        "power_se": float(np.std(powers) / np.sqrt(len(powers))),
        "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
        "budget_satisfied": bool(np.all(np.mean(negs, axis=0) <= budget)),
    }


def bisection_search(agent, envs, n_turb, device, budget, horizon,
                      n_episodes, args, target_util=0.95):
    """Find the constant lambda that matches target budget utilization."""
    print(f"  Bisection search for matched constant (target: {target_util*100:.0f}% util)")

    lo, hi = 0.1, 20.0
    best_lam, best_result = None, None

    for step in range(15):
        mid = (lo + hi) / 2.0
        res = eval_with_constant_lambda(agent, envs, n_turb, device,
                                         mid, budget, horizon, n_episodes, args)
        max_neg = max(res["neg_yaw_mean"])
        util = max_neg / budget

        if best_result is None or abs(util - target_util) < abs(max(best_result["neg_yaw_mean"]) / budget - target_util):
            best_lam = mid
            best_result = res

        print(f"    step {step}: k={mid:.2f}, max_neg={max_neg:.0f}, "
              f"util={util:.2f}, power={res['power_mean']:.0f}")

        if util > target_util:
            lo = mid  # too many violations, increase penalty
        else:
            hi = mid  # too few, decrease penalty

    print(f"  Best: k={best_lam:.2f}")
    return best_lam, best_result


def main():
    parser = argparse.ArgumentParser(description="PPO-Lag baseline comparison")
    parser.add_argument("--checkpoint",
                        default="runs/ebt_sac_windfarm/checkpoints/step_100000.pt")
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--n-eval-episodes", type=int, default=50)
    parser.add_argument("--output", default="results/ppo_lag_comparison.json")
    cli = parser.parse_args()

    if not os.path.exists(cli.checkpoint):
        import glob
        ckpts = glob.glob("runs/*/checkpoints/step_*.pt")
        if ckpts:
            cli.checkpoint = sorted(ckpts)[-1]
        else:
            print("No checkpoint found.")
            return

    agent, envs, n_turb, device, args = load_windfarm(cli.checkpoint)
    budget = cli.budget
    horizon = cli.horizon
    n_ep = cli.n_eval_episodes

    print(f"PPO-Lag Baseline Comparison")
    print(f"  {n_turb} turbines, budget={budget}, horizon={horizon}, {n_ep} episodes")

    from load_surrogates import NegativeYawBudgetSurrogate

    # 1. Unconstrained
    print("\n1. Unconstrained baseline...")
    powers, negs = [], []
    for _ in range(n_ep):
        obs, _ = envs.reset()
        ep_power, neg_counts = 0.0, np.zeros(n_turb)
        for t in range(horizon):
            with torch.no_grad():
                act = agent.act(envs, obs)
            obs, rew, _, _, info = envs.step(act)
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
    print(f"  Power={uncon['power_mean']:.0f}±{uncon['power_se']:.0f}, "
          f"NegYaw={[int(x) for x in uncon['neg_yaw_mean']]}")

    # 2. Matched constant (bisection-tuned — simulates optimal fixed Lagrangian)
    print("\n2. Matched constant penalty (oracle-tuned)...")
    best_lam, matched = bisection_search(agent, envs, n_turb, device,
                                          budget, horizon, n_ep, args)

    # 3. AC schedule (our method)
    print("\n3. AC schedule...")
    ac_results = {}
    for ra in [2.0, 5.0]:
        for gs in [0.05, 0.1]:
            powers, negs = [], []
            for _ in range(n_ep):
                surr = NegativeYawBudgetSurrogate(
                    budget_steps=budget, horizon_steps=horizon,
                    risk_aversion=ra, steepness=2.0, yaw_max_deg=30.0,
                )
                surr.reset()
                obs, _ = envs.reset()
                ep_power, neg_counts = 0.0, np.zeros(n_turb)
                for t in range(horizon):
                    with torch.no_grad():
                        act = agent.act(envs, obs, guidance_fn=surr, guidance_scale=gs)
                    obs, rew, _, _, info = envs.step(act)
                    if "yaw angles agent" in info:
                        yaw_arr = np.array(info["yaw angles agent"])
                        yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                        for ti in range(min(len(yaw_flat), n_turb)):
                            if yaw_flat[ti] < 0:
                                neg_counts[ti] += 1
                        surr.update(torch.tensor(
                            yaw_flat[:n_turb], device=device, dtype=torch.float32,
                        ))
                    if "Power agent" in info:
                        ep_power += float(np.mean(info["Power agent"]))
                powers.append(ep_power / horizon)
                negs.append(neg_counts)

            res = {
                "power_mean": float(np.mean(powers)),
                "power_std": float(np.std(powers)),
                "power_se": float(np.std(powers) / np.sqrt(len(powers))),
                "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
                "budget_satisfied": bool(np.all(np.mean(negs, axis=0) <= budget)),
            }
            key = f"AC_ra{ra}_gs{gs}"
            ac_results[key] = res
            neg_str = str([int(x) for x in res["neg_yaw_mean"]])
            ok = "✓" if res["budget_satisfied"] else "✗"
            pct = 100 * res["power_mean"] / uncon["power_mean"]
            print(f"  η={ra}, gs={gs}: Power={res['power_mean']:.0f}±{res['power_se']:.0f} "
                  f"({pct:.1f}%), NegYaw={neg_str} {ok}")

    # Summary
    print(f"\n{'='*70}")
    print(f"  COMPARISON SUMMARY (budget={budget}/{horizon} steps)")
    print(f"{'='*70}")
    print(f"  {'Method':<35s} {'Power':>12s} {'%Uncon':>7s} {'NegYaw':>20s}")
    print(f"  {'-'*75}")
    print(f"  {'Unconstrained':<35s} {uncon['power_mean']:>8.0f}±{uncon['power_se']:<3.0f} "
          f"{'100%':>7s} {str([int(x) for x in uncon['neg_yaw_mean']]):>20s}")
    pct_m = 100 * matched["power_mean"] / uncon["power_mean"]
    print(f"  {'Matched const (oracle k='+f'{best_lam:.1f})':<35s} "
          f"{matched['power_mean']:>8.0f}±{matched['power_se']:<3.0f} "
          f"{pct_m:>6.1f}% {str([int(x) for x in matched['neg_yaw_mean']]):>20s}")
    for key, res in ac_results.items():
        pct = 100 * res["power_mean"] / uncon["power_mean"]
        ok = "✓" if res["budget_satisfied"] else "✗"
        print(f"  {key:<35s} {res['power_mean']:>8.0f}±{res['power_se']:<3.0f} "
              f"{pct:>6.1f}% {str([int(x) for x in res['neg_yaw_mean']]):>20s} {ok}")

    # Save
    results = {
        "unconstrained": uncon,
        "matched_constant": {**matched, "lambda": best_lam},
        **ac_results,
        "meta": {"budget": budget, "horizon": horizon, "n_episodes": n_ep},
    }
    os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
    with open(cli.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {cli.output}")

    envs.close()


if __name__ == "__main__":
    main()
