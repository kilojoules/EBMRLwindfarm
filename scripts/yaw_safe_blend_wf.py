"""
Safety-controller blending for WindGym (parallel to SG APF blend).

At each step:
    a_exec = (1 − σ(λ(u))) · π_RL(s)  +  σ(λ(u)) · π_safe(s)

- π_RL    : frozen EBT/SAC actor trained for aggressive wake steering
- π_safe  : calibrated "no wake-steering" setpoint. Under our per-turbine
            fatigue surrogate, the load-minimizing setpoint is yaw = 0
            (normalized action = 0). In real deployment, π_safe would be
            the OEM yaw controller or a WakeAdapt-style lookup table.
- σ(u)    : urgency-driven blend weight. On pace → 0 (pure RL). Deficit → 1
            (pure safe). Sharpness tunable.

Budget: total neg-yaw timesteps per turbine ≤ d (same as main paper).

Usage:
  python scripts/yaw_safe_blend_wf.py --checkpoint <ckpt> --budget 15 \
      --horizon 200 --sharpness 3.0 --n-episodes 50
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


def urgency_sigma(t, T, C_per_turb, budget, sharpness=3.0, sigma_max=1.0,
                   mode="urgency", const_sigma=0.0, switch_thresh=0.5):
    """Per-turbine σ ∈ [0, σ_max]. C_per_turb is array (n_turb,)."""
    n = len(C_per_turb)
    out = np.zeros(n, dtype=float)
    if mode == "pure_safe":
        return np.full(n, sigma_max)
    if mode == "const":
        return np.full(n, const_sigma)
    tau = max((T - t) / T, 1e-6)
    for i, c in enumerate(C_per_turb):
        rho = (budget - c) / max(budget, 1e-9)
        if rho <= 0:
            out[i] = sigma_max
            continue
        u = rho / tau
        if mode == "switch":
            out[i] = sigma_max if u < switch_thresh else 0.0
            continue
        if mode == "projected":
            cost_rate = c / max(t, 1)
            proj = cost_rate * T
            out[i] = sigma_max if proj > budget else 0.0
            continue
        # urgency (default)
        if u >= 1:
            out[i] = 0.0
        else:
            out[i] = float(np.clip(
                sigma_max * (1.0 - np.exp(-sharpness * (1.0 / max(u, 1e-6) - 1.0))),
                0.0, sigma_max))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--sharpness", type=float, default=3.0)
    p.add_argument("--sigma-max", type=float, default=1.0)
    p.add_argument("--mode", choices=["urgency", "const", "switch", "pure_safe", "projected"],
                   default="urgency")
    p.add_argument("--const-sigma", type=float, default=0.0)
    p.add_argument("--switch-thresh", type=float, default=0.5)
    p.add_argument("--n-episodes", type=int, default=50)
    p.add_argument("--output", required=True)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    tr_args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})

    from ebt_sac_windfarm import setup_env
    from helpers.agent import WindFarmAgent
    from ebt import TransformerEBTActor
    from networks import create_profile_encoding

    env_info = setup_env(tr_args)
    envs = env_info["envs"]
    n_turb = env_info["n_turbines_max"]

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

    # Unconstrained reference
    print("1. Unconstrained...")
    powers_u, negs_u = [], []
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
        powers_u.append(p_ep / args.horizon)
        negs_u.append(nc)
    uncon = {
        "power_mean": float(np.mean(powers_u)),
        "power_se": float(np.std(powers_u) / np.sqrt(len(powers_u))),
        "neg_yaw_mean": np.mean(negs_u, axis=0).tolist(),
    }
    print(f"  power={uncon['power_mean']:.0f}±{uncon['power_se']:.0f} "
          f"neg={[int(x) for x in uncon['neg_yaw_mean']]}")

    # Safety-controller blend
    print(f"\n2. Yaw-safe blend (sharpness={args.sharpness}, σ_max={args.sigma_max}, "
          f"d={args.budget})...")
    powers, negs, sigma_mean = [], [], []
    for ep in range(args.n_episodes):
        obs, _ = envs.reset()
        p_ep, nc = 0.0, np.zeros(n_turb)
        sigmas_ep = []
        for t in range(args.horizon):
            with torch.no_grad():
                act_actor = agent.act(envs, obs)
            act_actor = np.asarray(act_actor, dtype=np.float32)
            act_safe = np.zeros_like(act_actor)
            sigma = urgency_sigma(t, args.horizon, nc, args.budget,
                                   sharpness=args.sharpness,
                                   sigma_max=args.sigma_max,
                                   mode=args.mode,
                                   const_sigma=args.const_sigma,
                                   switch_thresh=args.switch_thresh).astype(np.float32)
            # Align σ (n_turb,) with act_actor's turbine axis.
            # WindGym vector env: act shape is (num_envs=1, n_turb, [act_dim])
            if act_actor.ndim == 2:
                sigma_b = sigma[None, :]                  # (1, n_turb)
            elif act_actor.ndim == 3:
                sigma_b = sigma[None, :, None]            # (1, n_turb, 1)
            else:
                sigma_b = sigma.reshape((1,) * (act_actor.ndim - 1) + (-1,))
            act_exec = (1.0 - sigma_b) * act_actor + sigma_b * act_safe
            sigmas_ep.append(float(np.mean(sigma)))
            obs, _, _, _, info = envs.step(act_exec)
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
        sigma_mean.append(float(np.mean(sigmas_ep)))
    blend = {
        "power_mean": float(np.mean(powers)),
        "power_se": float(np.std(powers) / np.sqrt(len(powers))),
        "neg_yaw_mean": np.mean(negs, axis=0).tolist(),
        "sigma_mean": float(np.mean(sigma_mean)),
        "budget_satisfied": bool(np.all(np.mean(negs, axis=0) <= args.budget)),
    }
    pct = 100 * blend["power_mean"] / uncon["power_mean"]
    ok = "✓" if blend["budget_satisfied"] else "✗"
    print(f"  power={blend['power_mean']:.0f}±{blend['power_se']:.0f} "
          f"({pct:.1f}%), neg={[int(x) for x in blend['neg_yaw_mean']]} "
          f"σ̄={blend['sigma_mean']:.2f} {ok}")

    out = {"unconstrained": uncon, "blend": blend, "pct_uncon": pct,
           "meta": {"budget": args.budget, "sharpness": args.sharpness,
                    "sigma_max": args.sigma_max,
                    "n_episodes": args.n_episodes,
                    "checkpoint": args.checkpoint}}
    Path(args.output).parent.mkdir(exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()
