#!/usr/bin/env python3
"""
Cost critic Q_c for wind farm fatigue budget constraints.

Trains Q_c(obs, action) to predict cumulative future fatigue cost, then
composes with the EBT actor via per-turbine energy addition. The urgency
schedule λ(t) modulates the Q_c energy weight.

This bridges the action-to-cost gap for state-dependent fatigue costs
(e.g., partial wake interactions where the cost at a downstream turbine
depends on the upstream turbine's past yaw through wake propagation).

Usage:
    python scripts/windfarm_cost_critic.py \
        --checkpoint runs/ebt_sac_windfarm/checkpoints/step_100000.pt \
        --collect --train --compare
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# COST CRITIC FOR WIND FARM
# =============================================================================

class WindFarmCostCritic(nn.Module):
    """
    Q_c(obs, action) for per-turbine fatigue cost prediction.

    Implements the guidance_fn interface for EBT energy composition:
    - set_obs(obs_flat): store current observation before action optimization
    - per_turbine_energy(action, mask): return λ(t) * Q_c per turbine

    The critic predicts total future fatigue cost from the current
    (obs, action) pair. During EBT optimization, ∇_a Q_c provides
    per-turbine gradients that steer yaw away from high-fatigue actions.
    """

    def __init__(self, obs_dim_total, act_dim_total, n_turbines, hidden=256):
        super().__init__()
        self.n_turbines = n_turbines
        self.obs_dim_total = obs_dim_total
        self.act_dim_total = act_dim_total

        # Twin critics
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim_total + act_dim_total, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_turbines),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim_total + act_dim_total, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_turbines),
        )

        # State for guidance_fn interface
        self._obs_context = None
        self._lambda_val = 1.0

    def forward(self, obs_flat, action_flat):
        """Forward pass: (batch, obs+act) → (batch, n_turbines) per-turbine Q_c."""
        x = torch.cat([obs_flat, action_flat], dim=-1)
        return self.q1(x), self.q2(x)

    def predict(self, obs_flat, action_flat):
        q1, q2 = self.forward(obs_flat, action_flat)
        return torch.max(q1, q2)

    def set_obs(self, obs_flat):
        """Store current observation for per_turbine_energy calls."""
        self._obs_context = obs_flat

    def set_lambda(self, lam):
        self._lambda_val = lam

    def per_turbine_energy(self, action, key_padding_mask=None):
        """
        EBT guidance interface: returns (batch, n_turbines, 1) cost energy.

        action: (batch, n_turbines, 1) in degrees (physical units)
        """
        batch = action.shape[0]
        action_flat = action.squeeze(-1)  # (batch, n_turbines)

        if self._obs_context is None:
            return torch.zeros(batch, self.n_turbines, 1, device=action.device)

        obs = self._obs_context
        if obs.dim() == 1:
            obs = obs.unsqueeze(0).expand(batch, -1)

        # Normalize action to [-1, 1] for the critic
        action_norm = action_flat / 30.0  # yaw_max_deg

        q_per_turb = self.predict(obs, action_norm)  # (batch, n_turbines)

        energy = self._lambda_val * q_per_turb.unsqueeze(-1)  # (batch, n_turb, 1)

        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            energy = energy * mask

        return energy


# =============================================================================
# DATA COLLECTION
# =============================================================================

def collect_windfarm_data(checkpoint_path, n_episodes, horizon,
                          save_path, fatigue_beta=2.0, fatigue_gamma=3.0,
                          ws_ref=10.0, yaw_max=30.0):
    """Roll out EBT policy, collect (obs_flat, action_norm, fatigue_cost, next_obs_flat, done)."""
    from config import Args
    from helpers.agent import WindFarmAgent
    from ebt import TransformerEBTActor
    from networks import create_profile_encoding
    from ebt_sac_windfarm import setup_env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args_dict = ckpt["args"]
    args = Args(**{k: v for k, v in args_dict.items() if hasattr(Args, k)})

    env_info = setup_env(args)
    envs = env_info["envs"]
    n_turb = env_info["n_turbines_max"]

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

    # Determine observation size from first reset
    obs, _ = envs.reset()
    obs_flat = np.array(obs).flatten() if isinstance(obs, dict) else np.array(obs).flatten()
    obs_dim = len(obs_flat)

    all_obs, all_act, all_cost, all_next_obs, all_done = [], [], [], [], []
    total_fatigue = 0

    for ep in range(n_episodes):
        obs, _ = envs.reset()
        prev_obs_flat = np.array(obs).flatten()[:obs_dim]

        for t in range(horizon):
            with torch.no_grad():
                act = agent.act(envs, obs)

            next_obs, rew, term, trunc, info = envs.step(act)

            # Compute per-turbine fatigue cost
            per_turb_cost = np.zeros(n_turb)
            if "yaw angles agent" in info:
                yaw_arr = np.array(info["yaw angles agent"])
                yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                ws = 12.0  # approximate; could extract from info
                for ti in range(min(len(yaw_flat), n_turb)):
                    yaw_norm = abs(yaw_flat[ti]) / yaw_max
                    fatigue = (yaw_norm ** fatigue_beta) * ((ws / ws_ref) ** fatigue_gamma)
                    per_turb_cost[ti] = fatigue

            # Normalize action to [-1, 1]
            action_norm = np.array(act).flatten()[:n_turb]
            if np.max(np.abs(action_norm)) > 1.5:
                action_norm = action_norm / yaw_max

            next_obs_flat = np.array(next_obs).flatten()[:obs_dim]
            done = bool(term) if isinstance(term, (bool, np.bool_)) else bool(np.any(term))

            all_obs.append(prev_obs_flat.copy())
            all_act.append(action_norm[:n_turb])
            all_cost.append(per_turb_cost)
            all_next_obs.append(next_obs_flat.copy())
            all_done.append(float(done))

            total_fatigue += per_turb_cost.sum()
            prev_obs_flat = next_obs_flat
            obs = next_obs

            if done:
                break

        if (ep + 1) % 10 == 0:
            print(f"  Ep {ep+1}/{n_episodes}: {len(all_obs)} transitions, "
                  f"avg fatigue/ep={total_fatigue/(ep+1):.1f}")

    envs.close()

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    np.savez(save_path,
             obs=np.array(all_obs, dtype=np.float32),
             act=np.array(all_act, dtype=np.float32),
             cost=np.array(all_cost, dtype=np.float32),
             next_obs=np.array(all_next_obs, dtype=np.float32),
             done=np.array(all_done, dtype=np.float32))
    print(f"Saved {len(all_obs)} transitions to {save_path}")
    print(f"  Per-turbine cost: mean={np.mean(all_cost):.4f}, "
          f"max={np.max(all_cost):.4f}")

    return n_turb, obs_dim


# =============================================================================
# TRAINING Q_c
# =============================================================================

def train_windfarm_qc(data_path, actor_checkpoint, save_path,
                      gamma_c=0.99, epochs=200, batch_size=256,
                      lr=3e-4, hidden=256, tau=0.005):
    """Train per-turbine Q_c via offline Bellman updates."""
    data = np.load(data_path)
    obs = torch.FloatTensor(data["obs"])
    act = torch.FloatTensor(data["act"])
    cost = torch.FloatTensor(data["cost"])  # (N, n_turbines)
    next_obs = torch.FloatTensor(data["next_obs"])
    done = torch.FloatTensor(data["done"]).unsqueeze(1)

    n = len(obs)
    obs_dim = obs.shape[1]
    n_turb = act.shape[1]
    act_dim = n_turb

    print(f"  Training data: {n} transitions, obs_dim={obs_dim}, "
          f"n_turbines={n_turb}")
    print(f"  Cost stats: mean={cost.mean():.4f}, "
          f"nonzero_frac={(cost > 0.01).float().mean():.3f}")

    qc = WindFarmCostCritic(obs_dim, act_dim, n_turb, hidden)
    qc_target = WindFarmCostCritic(obs_dim, act_dim, n_turb, hidden)
    qc_target.load_state_dict(qc.state_dict())

    # We need the actor to compute a' ~ π(s') for the Bellman target.
    # For wind farm, the EBT actor is complex. Instead, use the stored
    # next actions from consecutive transitions as an approximation.
    # This is valid when episodes are stored sequentially.

    optimizer = torch.optim.Adam(qc.parameters(), lr=lr)
    steps_per_epoch = max(n // batch_size, 50)

    best_loss = float('inf')
    best_state = None

    for epoch in range(epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            idx = np.random.randint(0, n, batch_size)

            o = obs[idx]
            a = act[idx]
            c = cost[idx]  # (batch, n_turb)
            no = next_obs[idx]
            d = done[idx]  # (batch, 1)

            # For next action, use a small perturbation of current action
            # (offline approximation when we can't query the EBT actor)
            na = a + 0.01 * torch.randn_like(a)
            na = na.clamp(-1, 1)

            with torch.no_grad():
                tq1, tq2 = qc_target(no, na)
                target_q = torch.min(tq1, tq2)  # (batch, n_turb)
                target = c + gamma_c * (1 - d) * target_q

            q1, q2 = qc(o, a)
            loss = F.mse_loss(q1, target) + F.mse_loss(q2, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            for p, pt in zip(qc.parameters(), qc_target.parameters()):
                pt.data.lerp_(p.data, tau)

        avg_loss = epoch_loss / steps_per_epoch
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.clone() for k, v in qc.state_dict().items()}

        if (epoch + 1) % 25 == 0:
            with torch.no_grad():
                idx = np.random.randint(0, n, 1000)
                q_pred = qc.predict(obs[idx], act[idx])
                print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, "
                      f"Q_c mean={q_pred.mean():.3f}, max={q_pred.max():.3f}")

    qc.load_state_dict(best_state)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save({
        "model": qc.state_dict(),
        "obs_dim": obs_dim, "act_dim": act_dim, "n_turbines": n_turb,
        "hidden": hidden, "gamma_c": gamma_c,
    }, save_path)
    print(f"Saved Q_c to {save_path} (loss={best_loss:.4f})")


# =============================================================================
# EVALUATION
# =============================================================================

def eval_windfarm_with_qc(checkpoint_path, qc_path, budget_steps, horizon,
                          risk_aversion, n_episodes, guidance_scale=0.1,
                          yaw_max=30.0, fatigue_beta=2.0, fatigue_gamma=3.0,
                          ws_ref=10.0):
    """Evaluate wind farm with Q_c-composed EBT actor."""
    from config import Args
    from helpers.agent import WindFarmAgent
    from ebt import TransformerEBTActor
    from networks import create_profile_encoding
    from ebt_sac_windfarm import setup_env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})

    env_info = setup_env(args)
    envs = env_info["envs"]
    n_turb = env_info["n_turbines_max"]

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

    # Load Q_c
    qc_ckpt = torch.load(qc_path, map_location=device, weights_only=False)
    qc = WindFarmCostCritic(
        qc_ckpt["obs_dim"], qc_ckpt["act_dim"],
        qc_ckpt["n_turbines"], qc_ckpt["hidden"]).to(device)
    qc.load_state_dict(qc_ckpt["model"])
    qc.eval()

    powers, neg_counts_all, fatigue_all = [], [], []

    for ep in range(n_episodes):
        obs, _ = envs.reset()
        ep_power = 0.0
        neg_counts = np.zeros(n_turb)
        cum_fatigue = np.zeros(n_turb)
        cum_violations = 0

        # Budget tracking for urgency
        for t in range(horizon):
            # Compute urgency-based lambda
            eps = 1e-6
            budget_remaining = max(budget_steps - cum_violations, 0)
            time_remaining = max(horizon - t, 1)
            bf = budget_remaining / max(budget_steps, 1)
            tf = time_remaining / max(horizon, 1)
            u = bf / max(tf, eps)
            lam = min(np.exp(risk_aversion * (1.0 / max(u, eps) - 1.0)), 1e4)

            # Set Q_c context
            obs_flat = np.array(obs).flatten()
            qc.set_obs(torch.FloatTensor(obs_flat).to(device))
            qc.set_lambda(lam)

            with torch.no_grad():
                if guidance_scale > 0 and risk_aversion > 0:
                    act = agent.act(envs, obs, guidance_fn=qc,
                                    guidance_scale=guidance_scale)
                else:
                    act = agent.act(envs, obs)

            obs, rew, _, _, info = envs.step(act)

            if "yaw angles agent" in info:
                yaw_arr = np.array(info["yaw angles agent"])
                yaw_flat = yaw_arr[0] if yaw_arr.ndim > 1 else yaw_arr
                ws = 12.0
                for ti in range(min(len(yaw_flat), n_turb)):
                    if yaw_flat[ti] < 0:
                        neg_counts[ti] += 1
                    fatigue = (abs(yaw_flat[ti]) / yaw_max) ** fatigue_beta * \
                              (ws / ws_ref) ** fatigue_gamma
                    cum_fatigue[ti] += fatigue
                    if fatigue > 0.3:
                        cum_violations += 1

            if "Power agent" in info:
                ep_power += float(np.mean(info["Power agent"]))

        powers.append(ep_power / horizon)
        neg_counts_all.append(neg_counts)
        fatigue_all.append(cum_fatigue.sum())

    envs.close()
    return {
        "power_mean": float(np.mean(powers)),
        "power_std": float(np.std(powers)),
        "neg_yaw_mean": np.mean(neg_counts_all, axis=0).tolist(),
        "fatigue_mean": float(np.mean(fatigue_all)),
        "fatigue_std": float(np.std(fatigue_all)),
    }


def compare_windfarm(checkpoint_path, qc_path, n_episodes=10, horizon=200,
                     output_json=None):
    """Compare unconstrained vs Q_c-guided on wind farm."""
    print(f"\n{'='*70}")
    print(f"  Wind Farm Cost Critic Q_c Comparison")
    print(f"{'='*70}")

    # Unconstrained
    uncon = eval_windfarm_with_qc(
        checkpoint_path, qc_path, budget_steps=200, horizon=horizon,
        risk_aversion=0.0, n_episodes=n_episodes, guidance_scale=0.0)
    print(f"\n  Unconstrained: Power={uncon['power_mean']:.0f}±{uncon['power_std']:.0f}, "
          f"Fatigue={uncon['fatigue_mean']:.1f}")

    results = {"unconstrained": uncon}

    print(f"\n  {'Method':<35s} {'Power':>12s} {'NegYaw':>15s} {'Fatigue':>10s}")
    print(f"  {'-'*75}")

    for budget in [15, 30, 50]:
        for ra in [0.0, 2.0, 5.0]:
            for gs in [0.05, 0.1, 0.5]:
                res = eval_windfarm_with_qc(
                    checkpoint_path, qc_path, budget_steps=budget,
                    horizon=horizon, risk_aversion=ra,
                    n_episodes=n_episodes, guidance_scale=gs)
                neg_str = str([int(x) for x in res["neg_yaw_mean"]])
                pct = 100 * res["power_mean"] / uncon["power_mean"]
                label = f"B={budget} η={ra} gs={gs}"
                print(f"  {label:<35s} {res['power_mean']:8.0f} ({pct:5.1f}%) "
                      f"{neg_str:>15s} {res['fatigue_mean']:10.1f}")
                results[f"B{budget}_ra{ra}_gs{gs}"] = res

    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Results saved to {output_json}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Wind farm cost critic Q_c")
    parser.add_argument("--checkpoint",
                        default="runs/ebt_sac_windfarm/checkpoints/step_100000.pt")
    parser.add_argument("--collect", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--compare", action="store_true")
    parser.add_argument("--n-collect-episodes", type=int, default=100)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=200)
    parser.add_argument("--gamma-c", type=float, default=0.99)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--data-path", default="data/windfarm_cost_data.npz")
    parser.add_argument("--qc-path", default="checkpoints/windfarm_qc.pt")
    parser.add_argument("--output-json", default=None)
    cli = parser.parse_args()

    torch.manual_seed(cli.seed)
    np.random.seed(cli.seed)

    if cli.collect:
        print(f"Collecting {cli.n_collect_episodes} episodes...")
        collect_windfarm_data(cli.checkpoint, cli.n_collect_episodes,
                              cli.horizon, cli.data_path)

    if cli.train:
        print(f"\nTraining Q_c (γ={cli.gamma_c})...")
        train_windfarm_qc(cli.data_path, cli.checkpoint, cli.qc_path,
                           gamma_c=cli.gamma_c, epochs=cli.epochs)

    if cli.compare:
        print(f"\nComparing methods...")
        compare_windfarm(cli.checkpoint, cli.qc_path,
                          cli.n_eval_episodes, cli.horizon, cli.output_json)

    if not any([cli.collect, cli.train, cli.compare]):
        print("Specify --collect, --train, --compare, or all three")


if __name__ == "__main__":
    main()
