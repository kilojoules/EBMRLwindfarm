"""Cost-vs-sigma sweep: the experiment that matches the Proposition.

Fix pi_perf (DEL-aware MM actor settling at ~-15°), fix pi_safe at +15°
setpoint. Sweep sigma in [0, 1] in fine grid. Linear blend should pass through
yaw ~0 somewhere in sigma in {0.4, 0.5, 0.6} (depending on actor's actual
yaw and pi_safe's reach). If DEL has a non-convex ridge at yaw=0, the cum DEL
curve should hump.

Output JSON: per-sigma cum_del per turbine, cum_power, mean yaw, std across episodes.
Plot DEL_tot and Pwr vs sigma -- this is the figure that matches the theorem.
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


def rollout(envs, agent, surr, n_turb, horizon, sensor, safe_target, sigma_fixed):
    obs, _ = envs.reset()
    pset = np.full(n_turb, 0.93, dtype=np.float32)
    cum_del = np.zeros(n_turb, dtype=np.float64)
    cum_power = 0.0
    yaws_seen = []
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

        act_exec = (1.0 - sigma_fixed) * act_perf + sigma_fixed * act_safe

        # Score DEL at pre-step yaw via Teodor surrogate
        saws = feats_pre.get("saws", np.full((n_turb, 4), 9.0, dtype=np.float32))
        sati = feats_pre.get("sati", np.full((n_turb, 4), 0.07, dtype=np.float32))
        per = [dict(saws_left=saws[i,0], saws_right=saws[i,1],
                     saws_up=saws[i,2], saws_down=saws[i,3],
                     sati_left=sati[i,0], sati_right=sati[i,1],
                     sati_up=sati[i,2], sati_down=sati[i,3])
                 for i in range(n_turb)]
        x_in = rdf.features_to_surrogate_input(per, pset.tolist(), cur_yaw.tolist())
        del_step = surr.predict_one(sensor, torch.from_numpy(x_in)).flatten().numpy()
        cum_del += del_step
        yaws_seen.append(cur_yaw.copy())

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
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--layout", default="multi_modal")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--safe-yaw-deg", required=True,
                    help="CSV per-turbine pi_safe target yaw (e.g. '15,15,15')")
    p.add_argument("--sigma-grid", default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="CSV sigma values (sweep dimension)")
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

    safe_target = np.array([float(x) for x in args.safe_yaw_deg.split(",")],
                            dtype=np.float32)
    if safe_target.shape[0] < n_turb:
        safe_target = np.pad(safe_target, (0, n_turb - safe_target.shape[0]))
    sigmas = [float(x) for x in args.sigma_grid.split(",")]
    print(f"pi_safe target: {safe_target.tolist()}; sigmas: {sigmas}")

    results = []
    for sigma in sigmas:
        rows = [rollout(envs, agent, surr, n_turb, args.horizon, args.sensor,
                         safe_target, sigma) for _ in range(args.n_episodes)]
        del_per = np.array([r["cum_del"] for r in rows])
        yaw_mean = np.array([r["yaw_mean"] for r in rows]).mean(axis=0)
        agg = {
            "sigma": sigma,
            "power_mean": float(np.mean([r["cum_power"] for r in rows])),
            "power_std": float(np.std([r["cum_power"] for r in rows])),
            "del_per_turb_mean": del_per.mean(axis=0).tolist(),
            "del_per_turb_std": del_per.std(axis=0).tolist(),
            "del_total_mean": float(del_per.sum(axis=1).mean()),
            "del_total_std": float(del_per.sum(axis=1).std()),
            "yaw_mean_per_turb": yaw_mean.tolist(),
        }
        print(f"  sigma={sigma:.2f}: pwr={agg['power_mean']:.0f}+-{agg['power_std']:.0f} "
              f"DEL_tot={agg['del_total_mean']:.0f}+-{agg['del_total_std']:.0f} "
              f"yaw_mean={agg['yaw_mean_per_turb']}")
        results.append(agg)

    out = {"layout": args.layout, "horizon": args.horizon,
            "n_episodes": args.n_episodes,
            "safe_yaw_deg": safe_target.tolist(),
            "checkpoint": args.checkpoint, "results": results}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
