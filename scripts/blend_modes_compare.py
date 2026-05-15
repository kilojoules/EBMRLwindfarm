"""Compare linear vs rejection blend at fixed sigma, sweeping pi_safe direction.

Linear (baseline):     act_exec = (1-sigma) * act_perf + sigma * act_safe
Rejection blend (B3):  per-turbine, engage sigma ONLY if 1-step DEL probe shows
                       safe move LOWERS DEL. Otherwise sigma_eff=0 (reject).

Hazard claim: linear blend has anti-Pareto bulge at yaw~0; rejection recovers
endpoints (no worse than pi_perf). Sweep pi_safe direction at fixed sigma=0.7,
both modes side-by-side.

Output JSON: per-mode per-pi_safe DEL/power/yaw.
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


YAW_STEP = 0.5  # degrees per unit action


def predict_del(surr, sensor, per_features, pset, yaws):
    """Query Teodor surrogate at given yaws (degrees) under per-turbine
    sector-flow context. Returns per-turbine DEL array."""
    x_in = rdf.features_to_surrogate_input(per_features, pset.tolist(),
                                              yaws.tolist())
    return surr.predict_one(sensor, torch.from_numpy(x_in)).flatten().numpy()


def rollout(envs, agent, surr, n_turb, horizon, sensor,
            safe_target, sigma_fixed, mode):
    """One episode. mode in {linear, rejection}."""
    obs, _ = envs.reset()
    pset = np.full(n_turb, 0.93, dtype=np.float32)
    cum_del = np.zeros(n_turb, dtype=np.float64)
    cum_power = 0.0
    yaws_seen = []
    sigma_eff_seen = []
    for t in range(horizon):
        feats = envs.env.call("get_sector_features")[0]
        cur_yaw = np.asarray(feats.get("yaw_deg", np.zeros(n_turb)),
                              dtype=np.float32)[:n_turb]
        saws = feats.get("saws", np.full((n_turb, 4), 9.0, dtype=np.float32))
        sati = feats.get("sati", np.full((n_turb, 4), 0.07, dtype=np.float32))
        per = [dict(saws_left=saws[i,0], saws_right=saws[i,1],
                     saws_up=saws[i,2], saws_down=saws[i,3],
                     sati_left=sati[i,0], sati_right=sati[i,1],
                     sati_up=sati[i,2], sati_down=sati[i,3])
                 for i in range(n_turb)]

        with torch.no_grad():
            act_perf = agent.act(envs, obs)
        act_perf = np.asarray(act_perf, dtype=np.float32)
        a_safe_1d = np.clip((safe_target - cur_yaw) / YAW_STEP, -1.0, 1.0)

        # 1-step yaw projections
        a_perf_1d = act_perf.reshape(-1)[:n_turb]
        next_yaw_perf = cur_yaw + a_perf_1d * YAW_STEP
        next_yaw_safe = cur_yaw + a_safe_1d * YAW_STEP

        if mode == "linear":
            sigma_eff = np.full(n_turb, sigma_fixed)
        elif mode == "rejection":
            # Probe 1-step DEL under each candidate
            del_perf = predict_del(surr, sensor, per, pset, next_yaw_perf)
            del_safe = predict_del(surr, sensor, per, pset, next_yaw_safe)
            # Engage sigma ONLY where safe lowers DEL
            engage = (del_safe < del_perf).astype(np.float32)
            sigma_eff = sigma_fixed * engage
        else:
            raise ValueError(mode)

        # Score actual DEL at current (pre-step) yaw
        del_step = predict_del(surr, sensor, per, pset, cur_yaw)
        cum_del += del_step

        # Build act_exec
        if act_perf.ndim == 2:
            sb = sigma_eff[None, :]
            act_safe = np.broadcast_to(a_safe_1d[None, :], act_perf.shape).astype(np.float32)
        else:
            sb = sigma_eff[None, :, None]
            act_safe = np.broadcast_to(a_safe_1d[None, :, None], act_perf.shape).astype(np.float32)
        act_exec = (1.0 - sb) * act_perf + sb * act_safe

        yaws_seen.append(cur_yaw.copy())
        sigma_eff_seen.append(sigma_eff.copy())

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
        "yaw_mean": np.mean(yaws_seen, axis=0).tolist(),
        "sigma_eff_mean": np.mean(sigma_eff_seen, axis=0).tolist(),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--layout", default="multi_modal")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.7)
    p.add_argument("--safe-yaw-grid", default="-25,-15,-7.5,0,7.5,15,25")
    p.add_argument("--modes", default="linear,rejection")
    p.add_argument("--sensor", default="wrot_Bl1Rad0FlpMnt")
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

    safe_targets = [float(x) for x in args.safe_yaw_grid.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]
    results = []
    print(f"\nsigma={args.sigma}, modes={modes}, targets={safe_targets}")

    for mode in modes:
        for tgt in safe_targets:
            safe_target = np.full(n_turb, tgt, dtype=np.float32)
            rows = [rollout(envs, agent, surr, n_turb, args.horizon, args.sensor,
                             safe_target, args.sigma, mode)
                     for _ in range(args.n_episodes)]
            del_per = np.array([r["cum_del"] for r in rows])
            agg = {
                "mode": mode,
                "safe_target": tgt,
                "power_mean": float(np.mean([r["cum_power"] for r in rows])),
                "power_std": float(np.std([r["cum_power"] for r in rows])),
                "del_per_turb_mean": del_per.mean(axis=0).tolist(),
                "del_per_turb_std": del_per.std(axis=0).tolist(),
                "del_total_mean": float(del_per.sum(axis=1).mean()),
                "yaw_mean_per_turb": np.array([r["yaw_mean"] for r in rows]).mean(axis=0).tolist(),
                "sigma_eff_mean": np.array([r["sigma_eff_mean"] for r in rows]).mean(axis=0).tolist(),
            }
            print(f"  {mode} target={tgt:+.1f}: pwr={agg['power_mean']:.0f} "
                  f"DEL={agg['del_per_turb_mean']} sigma_eff={agg['sigma_eff_mean']}")
            results.append(agg)

    out = {
        "layout": args.layout, "horizon": args.horizon,
        "n_episodes": args.n_episodes, "sigma_fixed": args.sigma,
        "checkpoint": args.checkpoint,
        "results": results,
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
