#!/usr/bin/env python3
"""
Wind farm ablation experiments for the budget constraint paper.

Runs ablations on a trained EBT-SAC checkpoint:
1. Hard guard: AC+guard vs AC-only vs guard-only vs unconstrained
2. Schedule comparison: 1/u^eta vs exp(eta*(1/u-1))
3. Budget flexibility: one policy at multiple budget levels

Usage:
    python scripts/windfarm_ablations.py \
        --checkpoint runs/ebt_sac_windfarm/checkpoints/step_100000.pt \
        --ablation all --n-episodes 10
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
    """Load trained EBT actor and wind farm environment."""
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

    return agent, envs, n_turb, device, args


def run_episode(agent, envs, n_turb, device, surr, gs, horizon,
                use_hard_guard=True):
    """Run one wind farm episode with optional budget surrogate."""
    if surr is not None:
        surr.reset()

    obs, _ = envs.reset()
    ep_power = 0.0
    neg_counts = np.zeros(n_turb)

    for t in range(horizon):
        with torch.no_grad():
            if surr is not None and gs > 0:
                can_spend = True
                if use_hard_guard and surr.cumulative_neg_steps is not None:
                    budget_total = surr._get_budget_tensor()
                    if (surr.cumulative_neg_steps >= budget_total).all():
                        can_spend = False
                act = agent.act(
                    envs, obs,
                    guidance_fn=surr if (gs > 0 and can_spend) else None,
                    guidance_scale=gs,
                )
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
               n_episodes, use_hard_guard=True, schedule_type="exp"):
    """Run multiple episodes for one configuration."""
    powers, negs = [], []
    for _ in range(n_episodes):
        surr = NegativeYawBudgetSurrogate(
            budget_steps=budget_steps,
            horizon_steps=horizon,
            risk_aversion=ra,
            steepness=k,
            yaw_max_deg=30.0,
            schedule_type=schedule_type,
        )
        p, n = run_episode(agent, envs, n_turb, device, surr, gs, horizon,
                           use_hard_guard)
        powers.append(p)
        negs.append(n)
    return {
        "power_mean": float(np.mean(powers)),
        "power_std": float(np.std(powers)),
        "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
        "neg_yaw_std": np.std(negs, axis=0).tolist(),
    }


def run_unconstrained(agent, envs, n_turb, horizon, n_episodes):
    """Run unconstrained baseline."""
    powers, negs = [], []
    for _ in range(n_episodes):
        obs, _ = envs.reset()
        ep_power = 0.0
        neg_counts = np.zeros(n_turb)
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
    return {
        "power_mean": float(np.mean(powers)),
        "power_std": float(np.std(powers)),
        "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
    }


# =============================================================================
# ABLATIONS
# =============================================================================

def ablation_hard_guard(agent, envs, n_turb, device, n_ep, horizon=200):
    """Ablation 1: Hard guard vs AC schedule vs both."""
    print("\n" + "=" * 70)
    print("  ABLATION 1: Hard Guard vs AC Schedule vs Both")
    print("=" * 70)
    budget_steps, ra, k, gs = 15, 2.0, 2.0, 0.1

    configs = [
        ("Both (AC + guard)", ra, True),
        ("AC only (no guard)", ra, False),
        ("Guard only (η=0)", 0.0, True),
    ]

    uncon = run_unconstrained(agent, envs, n_turb, horizon, n_ep)
    print(f"Unconstrained: Power={uncon['power_mean']:.0f}±{uncon['power_std']:.0f}, "
          f"NegYaw={[int(x) for x in uncon['neg_yaw_mean']]}")

    print(f"\nBudget: {budget_steps}/{horizon} steps, η={ra}, k={k}, gs={gs}")
    print(f"{'Config':<30s} | {'Power':>12s} | {'NegYaw':>20s}")
    print("-" * 70)

    results = {"unconstrained": uncon, "configs": {}}
    for name, ra_val, guard in configs:
        res = run_config(agent, envs, n_turb, device, budget_steps, horizon,
                         ra_val, k, gs, n_ep, guard)
        neg_str = str([int(x) for x in res["neg_yaw_mean"]])
        print(f"  {name:<28s} | {res['power_mean']:8.0f}±{res['power_std']:<3.0f} "
              f"| {neg_str:>20s}")
        results["configs"][name] = res

    return results


def ablation_schedule_comparison(agent, envs, n_turb, device, n_ep, horizon=200):
    """Ablation 2: 1/u^eta (optimal) vs exp(eta*(1/u-1)) (practical)."""
    print("\n" + "=" * 70)
    print("  ABLATION 2: Schedule Comparison (1/u vs exp)")
    print("=" * 70)
    budget_steps, k, gs = 15, 2.0, 0.1

    print(f"{'Schedule':<10s} {'η':>4s} | {'Power':>12s} | {'NegYaw':>20s}")
    print("-" * 60)

    results = {}
    for sched in ["exp", "inverse"]:
        for ra in [0.5, 1.0, 2.0, 5.0]:
            res = run_config(agent, envs, n_turb, device, budget_steps, horizon,
                             ra, k, gs, n_ep, True, sched)
            neg_str = str([int(x) for x in res["neg_yaw_mean"]])
            label = "exp" if sched == "exp" else "1/u"
            print(f"  {label:<8s} {ra:4.1f} | {res['power_mean']:8.0f}±{res['power_std']:<3.0f} "
                  f"| {neg_str:>20s}")
            results[f"{sched}_eta{ra}"] = res

    return results


def ablation_budget_flexibility(agent, envs, n_turb, device, n_ep, horizon=200):
    """Ablation 3: Same policy, different budget levels."""
    print("\n" + "=" * 70)
    print("  ABLATION 3: Budget Flexibility (same policy, varying budgets)")
    print("=" * 70)
    ra, k, gs = 2.0, 2.0, 0.1

    uncon = run_unconstrained(agent, envs, n_turb, horizon, n_ep)
    uncon_power = uncon["power_mean"]
    print(f"Unconstrained: Power={uncon_power:.0f}±{uncon['power_std']:.0f}")

    print(f"\n{'Budget':>8s} | {'Power':>12s} | {'%Uncon':>7s} | {'NegYaw':>20s}")
    print("-" * 60)

    results = {"unconstrained": uncon, "budgets": {}}
    for budget_steps in [5, 10, 15, 25, 50, 75, 100]:
        res = run_config(agent, envs, n_turb, device, budget_steps, horizon,
                         ra, k, gs, n_ep)
        pct = 100 * res["power_mean"] / uncon_power if uncon_power > 0 else 0
        neg_str = str([int(x) for x in res["neg_yaw_mean"]])
        print(f"  {budget_steps:3d}/200 | {res['power_mean']:8.0f}±{res['power_std']:<3.0f} "
              f"| {pct:6.1f}% | {neg_str:>20s}")
        results["budgets"][str(budget_steps)] = {**res, "pct_unconstrained": pct}

    return results


def measure_overhead(agent, envs, n_turb, device, horizon=200, n_episodes=5):
    """Measure wall-clock overhead of lambda computation."""
    print("\n" + "=" * 70)
    print("  TIMING: Computational Overhead")
    print("=" * 70)

    # Baseline: unconstrained step timing
    times_baseline = []
    obs, _ = envs.reset()
    for _ in range(horizon * n_episodes):
        t0 = time.perf_counter()
        with torch.no_grad():
            act = agent.act(envs, obs)
        obs, _, _, _, _ = envs.step(act)
        times_baseline.append(time.perf_counter() - t0)

    # With AC schedule
    surr = NegativeYawBudgetSurrogate(
        budget_steps=15, horizon_steps=horizon,
        risk_aversion=2.0, steepness=2.0,
    )
    times_ac = []
    obs, _ = envs.reset()
    surr.reset()
    for t in range(horizon * n_episodes):
        t0 = time.perf_counter()
        with torch.no_grad():
            act = agent.act(envs, obs, guidance_fn=surr, guidance_scale=0.1)
        obs, _, _, _, info = envs.step(act)
        times_ac.append(time.perf_counter() - t0)
        if "yaw angles agent" in info:
            yaw_arr = np.array(info["yaw angles agent"])
            yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
            surr.update(torch.tensor(yaw_flat[:n_turb], device=device,
                                     dtype=torch.float32))

    base_ms = 1000 * np.mean(times_baseline)
    ac_ms = 1000 * np.mean(times_ac)
    overhead_ms = ac_ms - base_ms
    overhead_pct = 100 * overhead_ms / base_ms if base_ms > 0 else 0

    print(f"  Baseline step: {base_ms:.3f} ms")
    print(f"  AC step:       {ac_ms:.3f} ms")
    print(f"  Overhead:      {overhead_ms:.3f} ms ({overhead_pct:.1f}%)")

    return {
        "baseline_ms": base_ms, "ac_ms": ac_ms,
        "overhead_ms": overhead_ms, "overhead_pct": overhead_pct,
    }


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Wind farm budget ablations")
    parser.add_argument("--checkpoint",
                        default="runs/ebt_sac_windfarm/checkpoints/step_100000.pt")
    parser.add_argument("--ablation", nargs="+", default=["all"],
                        choices=["all", "hard_guard", "schedule", "flexibility",
                                 "timing"])
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--output", default="results/windfarm_ablations.json")
    cli = parser.parse_args()

    if not os.path.exists(cli.checkpoint):
        import glob
        ckpts = glob.glob("runs/*/checkpoints/step_*.pt")
        if ckpts:
            cli.checkpoint = sorted(ckpts)[-1]
            print(f"Using checkpoint: {cli.checkpoint}")
        else:
            print("No checkpoint found. Run training first.")
            return

    agent, envs, n_turb, device, args = load_windfarm(cli.checkpoint)
    print(f"Loaded {n_turb}-turbine wind farm from {cli.checkpoint}")

    ablations = set(cli.ablation)
    if "all" in ablations:
        ablations = {"hard_guard", "schedule", "flexibility", "timing"}

    results = {}
    n_ep = cli.n_episodes

    if "hard_guard" in ablations:
        results["hard_guard"] = ablation_hard_guard(
            agent, envs, n_turb, device, n_ep, cli.horizon)

    if "schedule" in ablations:
        results["schedule"] = ablation_schedule_comparison(
            agent, envs, n_turb, device, n_ep, cli.horizon)

    if "flexibility" in ablations:
        results["flexibility"] = ablation_budget_flexibility(
            agent, envs, n_turb, device, n_ep, cli.horizon)

    if "timing" in ablations:
        results["timing"] = measure_overhead(
            agent, envs, n_turb, device, cli.horizon)

    os.makedirs(os.path.dirname(cli.output) or ".", exist_ok=True)
    with open(cli.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {cli.output}")

    envs.close()
    print("Done.")


if __name__ == "__main__":
    main()
