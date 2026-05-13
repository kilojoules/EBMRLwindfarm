"""Sweep blend modes + budgets, produce Pareto data for paper figure.

Modes:
  perf            — pure pi_perf (sigma=0)
  safe            — pure pi_safe setpoint controller
  per_turb_blend  — per-turbine AC schedule on B_i
  uniform_blend   — scalar AC schedule on aggregate B_total

Outputs JSON: per-mode per-budget {power_mean, power_std, del_per_turb, util}.
For paper power-vs-aggregate-DEL Pareto + showing per-turb beats uniform.

Usage:
  python scripts/sweep_blend_budgets.py \\
    --checkpoint runs/del_aware_b1p0/checkpoints/step_200000.pt \\
    --bundle checkpoints/teodor_dlc12_torch.pt \\
    --layout multi_modal --safe-yaw-deg 7.5,7.5,7.5 \\
    --horizon 200 --n-episodes 3 \\
    --t2-budgets 19500,20500,21000,21300,21500,99999 \\
    --aggregate-budgets 60000,62000,63000,64000,200000 \\
    --eta 0.5 \\
    --out-json results/blend_sweep_b1p0.json
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import importlib.util
def _load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); sys.modules[name] = m
    return m
ts = _load("helpers.teodor_surrogate", ROOT / "helpers/teodor_surrogate.py")
rdf = _load("helpers.rotor_disk_flow", ROOT / "helpers/rotor_disk_flow.py")


def sigma_of_u(u, eta):
    if u >= 1.0: return 0.0
    return 1.0 - np.exp(-eta * (1.0 / max(u, 1e-6) - 1.0))


def rollout(envs, agent, surr, n_turb, horizon, sensor,
            mode, safe_target, B_per_turb=None, B_total=None, eta=0.5):
    """One episode. mode in {perf, safe, per_turb, uniform}."""
    obs, _ = envs.reset()
    pset = np.full(n_turb, 0.93, dtype=np.float32)
    cum_del = np.zeros(n_turb, dtype=np.float64)
    cum_power = 0.0
    sigma_traj = []
    for t in range(horizon):
        feats_pre = envs.env.call("get_sector_features")[0]
        cur_yaw = (np.asarray(feats_pre["yaw_deg"], dtype=np.float32)[:n_turb]
                    if isinstance(feats_pre, dict) and "yaw_deg" in feats_pre
                    else np.zeros(n_turb, dtype=np.float32))
        with torch.no_grad():
            act_perf = agent.act(envs, obs)
        act_perf = np.asarray(act_perf, dtype=np.float32)
        a_safe_1d = np.clip((safe_target - cur_yaw) / 0.5, -1.0, 1.0)
        if act_perf.ndim == 2:
            act_safe = np.broadcast_to(a_safe_1d[None, :], act_perf.shape).astype(np.float32)
        else:
            act_safe = np.broadcast_to(a_safe_1d[None, :, None], act_perf.shape).astype(np.float32)

        if mode == "perf":
            sigma = np.zeros(n_turb)
        elif mode == "safe":
            sigma = np.ones(n_turb)
        elif mode == "per_turb":
            rho = np.maximum(B_per_turb - cum_del, 0.0) / np.maximum(B_per_turb, 1.0)
            tau = max((horizon - t) / horizon, 1e-6)
            u_per = rho / tau
            sigma = np.array([sigma_of_u(u, eta) for u in u_per])
        elif mode == "uniform":
            cum_total = cum_del.sum()
            rho = max(B_total - cum_total, 0.0) / max(B_total, 1.0)
            tau = max((horizon - t) / horizon, 1e-6)
            u = rho / tau
            sigma = np.full(n_turb, sigma_of_u(u, eta))
        else:
            raise ValueError(mode)
        sigma_traj.append(sigma.copy())

        if act_perf.ndim == 2:
            sb = sigma[None, :]
        else:
            sb = sigma[None, :, None]
        act_exec = (1.0 - sb) * act_perf + sb * act_safe

        # DEL at current yaw via Teodor surrogate (pre-step yaw)
        feats = envs.env.call("get_sector_features")[0]
        saws = feats.get("saws", np.full((n_turb, 4), 9.0, dtype=np.float32))
        sati = feats.get("sati", np.full((n_turb, 4), 0.07, dtype=np.float32))
        yaw_now = np.asarray(feats.get("yaw_deg", np.zeros(n_turb)), dtype=np.float32)
        per = [dict(saws_left=saws[i,0], saws_right=saws[i,1],
                     saws_up=saws[i,2], saws_down=saws[i,3],
                     sati_left=sati[i,0], sati_right=sati[i,1],
                     sati_up=sati[i,2], sati_down=sati[i,3]) for i in range(n_turb)]
        x_in = rdf.features_to_surrogate_input(
            per, pset.tolist(), yaw_now.tolist())
        del_step = surr.predict_one(sensor, torch.from_numpy(x_in)).flatten().numpy()
        cum_del += del_step

        ret = envs.step(act_exec)
        if len(ret) == 5:
            obs, _, term, trunc, info = ret
        else:
            obs, _, done, info = ret
            term, trunc = done, False
        if "Power agent" in info:
            cum_power += float(np.mean(info["Power agent"]))
        if np.any(term) or np.any(trunc): break
    return {
        "cum_del": cum_del.tolist(),
        "cum_power": cum_power,
        "sigma_mean": np.mean(sigma_traj, axis=0).tolist(),
        "sigma_final": sigma_traj[-1].tolist() if sigma_traj else [0]*n_turb,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--layout", default="multi_modal")
    p.add_argument("--safe-yaw-deg", default="7.5,7.5,7.5")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n-episodes", type=int, default=3)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--sensor", default="wrot_Bl1Rad0FlpMnt")
    p.add_argument("--t2-budgets", default="20500,21000,21300,21500,99999",
                    help="CSV B_T2 values (T0,T1 always loose)")
    p.add_argument("--aggregate-budgets", default="61000,62000,63000,64000,200000",
                    help="CSV aggregate B_total for uniform-blend mode")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    surr = ts.TeodorDLC12Surrogate.from_bundle(args.bundle, outputs=[args.sensor])
    surr.eval()

    print(f"loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from config import Args
    tr_args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})
    tr_args.layouts = args.layout
    if args.layout == "multi_modal":
        tr_args.config = "multi_modal"
    tr_args.num_envs = 1

    from ebt_sac_windfarm import setup_env
    env_info = setup_env(tr_args)
    envs = env_info["envs"]
    n_turb = env_info["n_turbines_max"]

    use_profiles = env_info["use_profiles"]
    sr, si = None, None
    if use_profiles:
        from networks import create_profile_encoding
        sr, si = create_profile_encoding(
            profile_type=tr_args.profile_encoding_type,
            embed_dim=tr_args.embed_dim,
            hidden_channels=tr_args.profile_encoder_hidden)
    from ebt import TransformerEBTActor
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
        num_candidates=tr_args.ebt_num_candidates, args=tr_args)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()
    from helpers.agent import WindFarmAgent
    agent = WindFarmAgent(actor=actor, device=torch.device("cpu"),
                            rotor_diameter=env_info["rotor_diameter"],
                            use_wind_relative=tr_args.use_wind_relative_pos,
                            use_profiles=use_profiles,
                            rotate_profiles=getattr(tr_args, "rotate_profiles", False))

    safe_target = np.array([float(x) for x in args.safe_yaw_deg.split(",")],
                            dtype=np.float32)
    if safe_target.shape[0] < n_turb:
        safe_target = np.pad(safe_target, (0, n_turb - safe_target.shape[0]))

    t2_budgets = [float(x) for x in args.t2_budgets.split(",")]
    agg_budgets = [float(x) for x in args.aggregate_budgets.split(",")]

    def aggregate_ep(rows):
        del_per = np.array([r["cum_del"] for r in rows])
        return {
            "power_mean": float(np.mean([r["cum_power"] for r in rows])),
            "power_std": float(np.std([r["cum_power"] for r in rows])),
            "del_per_turb": del_per.mean(axis=0).tolist(),
            "del_per_turb_std": del_per.std(axis=0).tolist(),
            "del_total_mean": float(del_per.sum(axis=1).mean()),
            "sigma_mean": np.mean([r["sigma_mean"] for r in rows], axis=0).tolist(),
        }

    results = []

    # perf-only
    print("\n=== perf-only ===")
    rows = [rollout(envs, agent, surr, n_turb, args.horizon, args.sensor,
                     "perf", safe_target) for _ in range(args.n_episodes)]
    agg = aggregate_ep(rows)
    print(f"  power={agg['power_mean']:.0f} DEL/turb={agg['del_per_turb']} DEL_tot={agg['del_total_mean']:.0f}")
    results.append({"mode": "perf", "B_t2": None, "B_total": None, **agg})

    # safe-only
    print("\n=== safe-only ===")
    rows = [rollout(envs, agent, surr, n_turb, args.horizon, args.sensor,
                     "safe", safe_target) for _ in range(args.n_episodes)]
    agg = aggregate_ep(rows)
    print(f"  power={agg['power_mean']:.0f} DEL/turb={agg['del_per_turb']} DEL_tot={agg['del_total_mean']:.0f}")
    results.append({"mode": "safe", "B_t2": None, "B_total": None, **agg})

    # per-turb blend
    for B_t2 in t2_budgets:
        B = np.array([1e9, 1e9, B_t2], dtype=np.float64)
        print(f"\n=== per_turb B_T2={B_t2:.0f} ===")
        rows = [rollout(envs, agent, surr, n_turb, args.horizon, args.sensor,
                         "per_turb", safe_target, B_per_turb=B, eta=args.eta)
                 for _ in range(args.n_episodes)]
        agg = aggregate_ep(rows)
        print(f"  power={agg['power_mean']:.0f} DEL/turb={agg['del_per_turb']} sigma={agg['sigma_mean']}")
        results.append({"mode": "per_turb", "B_t2": B_t2, "B_total": None, **agg})

    # uniform blend
    for B_tot in agg_budgets:
        print(f"\n=== uniform B_total={B_tot:.0f} ===")
        rows = [rollout(envs, agent, surr, n_turb, args.horizon, args.sensor,
                         "uniform", safe_target, B_total=B_tot, eta=args.eta)
                 for _ in range(args.n_episodes)]
        agg = aggregate_ep(rows)
        print(f"  power={agg['power_mean']:.0f} DEL/turb={agg['del_per_turb']} sigma={agg['sigma_mean']}")
        results.append({"mode": "uniform", "B_t2": None, "B_total": B_tot, **agg})

    out = {
        "layout": args.layout, "horizon": args.horizon,
        "n_episodes": args.n_episodes, "eta": args.eta,
        "checkpoint": args.checkpoint, "safe_yaw_deg": safe_target.tolist(),
        "results": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
