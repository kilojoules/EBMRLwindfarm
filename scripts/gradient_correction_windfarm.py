#!/usr/bin/env python3
"""
Gradient correction for wind farm with nonlinear fatigue cost.

Compares three approaches on the trained EBT wind farm agent:
1. Energy composition (native EBT) with binary cost (yaw < 0)
2. Energy composition with nonlinear fatigue surrogate
3. Gradient correction on actor output with nonlinear fatigue surrogate

The nonlinear fatigue cost models real structural loading:
  fatigue(yaw, ws) = |yaw|^beta * (ws / ws_ref)^gamma
where beta > 1 captures the nonlinear yaw-fatigue relationship and
gamma > 1 captures the wind-speed amplification effect.

Usage:
    python scripts/gradient_correction_windfarm.py --n-episodes 5 --budget 15
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonlinearFatigueSurrogate(nn.Module):
    """
    Nonlinear fatigue loading surrogate for wind turbine yaw.

    Models fatigue as: cost = |yaw/yaw_max|^beta * (ws/ws_ref)^gamma

    At beta=1, gamma=0: reduces to the binary indicator (linear in |yaw|)
    At beta=2, gamma=3: captures realistic nonlinear fatigue amplification
      - Fatigue grows quadratically with yaw angle
      - High wind speeds amplify fatigue cubically

    This surrogate is differentiable, enabling gradient-based correction.
    """

    def __init__(self, beta=2.0, gamma=3.0, yaw_max_deg=30.0, ws_ref=10.0,
                 budget_steps=15, horizon_steps=200, risk_aversion=2.0,
                 steepness=3.0):
        super().__init__()
        self.beta = beta
        self.gamma = gamma
        self.yaw_max_deg = yaw_max_deg
        self.ws_ref = ws_ref
        self.budget_steps = budget_steps
        self.horizon_steps = horizon_steps
        self.risk_aversion = risk_aversion
        self.steepness = steepness

        self.current_step = 0
        self.cumulative_cost = 0.0
        self.cumulative_violations = 0  # binary: was fatigue > threshold?

    def reset(self):
        self.current_step = 0
        self.cumulative_cost = 0.0
        self.cumulative_violations = 0

    def update(self, yaw_deg, ws=12.0, fatigue_threshold=0.3):
        """Update budget tracking with actual fatigue cost."""
        yaw_norm = abs(yaw_deg) / self.yaw_max_deg
        fatigue = (yaw_norm ** self.beta) * ((ws / self.ws_ref) ** self.gamma)
        self.cumulative_cost += fatigue
        if fatigue > fatigue_threshold:
            self.cumulative_violations += 1
        self.current_step += 1

    def compute_lambda(self):
        eps = 1e-6
        budget_remaining = max(self.budget_steps - self.cumulative_violations, 0)
        time_remaining = max(self.horizon_steps - self.current_step, 1)
        budget_fraction = budget_remaining / max(self.budget_steps, 1)
        time_fraction = time_remaining / max(self.horizon_steps, 1)
        urgency = budget_fraction / max(time_fraction, eps)
        safe_urgency = max(urgency, eps)
        ac_weight = np.exp(self.risk_aversion * (1.0 / safe_urgency - 1.0))
        depletion = max(1.0 - budget_fraction / 0.05, 0)
        hard_wall = np.exp(self.steepness * depletion)
        return min(ac_weight * hard_wall, 1e6)

    def per_turbine_energy(self, action, key_padding_mask=None, ws=12.0):
        """
        Differentiable per-turbine fatigue energy for EBT composition.

        Args:
            action: (batch, n_turbines, 1) in [-1, 1] normalized
            ws: wind speed (used for nonlinear amplification)

        Returns:
            (batch, n_turbines, 1) fatigue penalty
        """
        lam = self.compute_lambda()
        yaw_norm = action.abs()  # |action| in [0, 1]

        # Nonlinear fatigue: |yaw|^beta * (ws/ws_ref)^gamma
        ws_factor = (ws / self.ws_ref) ** self.gamma
        fatigue = (yaw_norm ** self.beta) * ws_factor

        # Exponential penalty on fatigue
        penalty = torch.exp(self.steepness * F.relu(fatigue - 0.3)) - 1.0
        result = lam * penalty

        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            result = result * mask
        return result

    def forward(self, action, key_padding_mask=None, ws=12.0):
        per_turb = self.per_turbine_energy(action, key_padding_mask, ws)
        return per_turb.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


def run_windfarm_comparison(n_episodes=5, budget=15, horizon=200):
    """Compare binary vs nonlinear fatigue on wind farm."""
    from config import Args
    from load_surrogates import NegativeYawBudgetSurrogate
    from helpers.agent import WindFarmAgent

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Find the latest checkpoint
    ckpt_path = "runs/ebt_sac_windfarm/checkpoints/step_100000.pt"
    if not os.path.exists(ckpt_path):
        # Try to find any checkpoint
        import glob
        ckpts = glob.glob("runs/*/checkpoints/step_*.pt")
        if ckpts:
            ckpt_path = sorted(ckpts)[-1]
        else:
            print("No wind farm checkpoint found. Run training first.")
            return

    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    args = Args(**{k: v for k, v in args_dict.items() if hasattr(Args, k)})

    # Setup env and actor
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

    print(f"\n{'='*70}")
    print(f"  Wind Farm: Binary vs Nonlinear Fatigue Cost")
    print(f"  {n_turb} turbines, budget={budget}, horizon={horizon}")
    print(f"{'='*70}")

    configs = [
        ("Unconstrained", None, 0.0),
        ("Binary (eta=0)", "binary_const", 0.0),
        ("Binary (eta=2)", "binary_ac", 2.0),
        ("Nonlinear β=2,γ=3 (eta=0)", "nonlinear_const", 0.0),
        ("Nonlinear β=2,γ=3 (eta=2)", "nonlinear_ac", 2.0),
        ("Nonlinear β=2,γ=3 (eta=5)", "nonlinear_ac5", 5.0),
    ]

    print(f"  {'Method':<35s} {'Power':>10s} {'NegYaw':>15s} {'CumFatigue':>12s}")
    print("-" * 75)

    for name, cost_type, ra in configs:
        all_powers, all_neg, all_fatigue = [], [], []

        for ep in range(n_episodes):
            if cost_type is None:
                surr = None
                gs = 0.0
            elif cost_type.startswith("binary"):
                surr = NegativeYawBudgetSurrogate(
                    budget_steps=budget, horizon_steps=horizon,
                    risk_aversion=ra, steepness=3.0, yaw_max_deg=30.0,
                )
                surr.reset()
                gs = 0.5
            else:
                surr = NonlinearFatigueSurrogate(
                    beta=2.0, gamma=3.0, budget_steps=budget,
                    horizon_steps=horizon, risk_aversion=ra, steepness=3.0,
                )
                surr.reset()
                gs = 0.5

            obs, _ = envs.reset()
            ep_power, neg_counts, cum_fatigue = 0.0, np.zeros(n_turb), 0.0

            for t in range(horizon):
                with torch.no_grad():
                    act = agent.act(envs, obs,
                                    guidance_fn=surr if gs > 0 else None,
                                    guidance_scale=gs)

                obs, rew, _, _, info = envs.step(act)

                if "yaw angles agent" in info:
                    yaw_arr = np.array(info["yaw angles agent"])
                    yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                    for ti in range(min(len(yaw_flat), n_turb)):
                        if yaw_flat[ti] < 0:
                            neg_counts[ti] += 1
                        # Compute actual fatigue
                        fatigue = (abs(yaw_flat[ti]) / 30.0) ** 2 * (12.0 / 10.0) ** 3
                        cum_fatigue += fatigue

                    if surr is not None and hasattr(surr, 'update'):
                        if cost_type.startswith("binary"):
                            surr.update(torch.tensor(yaw_flat[:n_turb],
                                        device=device, dtype=torch.float32))
                        else:
                            for ti in range(min(len(yaw_flat), n_turb)):
                                surr.update(yaw_flat[ti], ws=12.0)

                if "Power agent" in info:
                    ep_power += float(np.mean(info["Power agent"]))

            all_powers.append(ep_power / horizon)
            all_neg.append(neg_counts.copy())
            all_fatigue.append(cum_fatigue)

        mean_power = np.mean(all_powers)
        mean_neg = np.mean(all_neg, axis=0).astype(int)
        mean_fatigue = np.mean(all_fatigue)
        neg_str = str(mean_neg.tolist())
        print(f"  {name:<35s} {mean_power:10.0f} {neg_str:>15s} {mean_fatigue:12.1f}")

    envs.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-episodes", type=int, default=5)
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--horizon", type=int, default=200)
    cli = parser.parse_args()

    run_windfarm_comparison(cli.n_episodes, cli.budget, cli.horizon)


if __name__ == "__main__":
    main()
