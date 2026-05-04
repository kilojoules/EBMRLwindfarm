"""Both-halves comparison: pi_perf (loaded checkpoint) vs pi_safe (zero yaw)
on power AND blade-flap DEL.

Returns per-turbine power and DEL for both policies; reports whether
pi_perf beats pi_safe on power (policy-advantage check) and whether
DEL_perf > DEL_safe (budget-tension check). Both required for the
"both halves" experiment.

Usage:
  pixi run python scripts/eval_pi_perf_vs_safe.py \\
      --checkpoint runs/pi_perf_stag4_5d_300k/checkpoints/step_300000.pt \\
      --layout stag4_5d --horizon 200 --n-episodes 5 \\
      --out-json results/pi_perf_vs_safe_stag4_5d.json
"""
import argparse
import json
import sys
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
ec = _load("helpers.env_configs", ROOT / "helpers/env_configs.py")


def rollout(envs, actor_fn, surr, n_turb, horizon, sensor):
    """One episode via vector env. envs.env.call('get_sector_features') gives
    serializable per-turbine sector dict from inside the subprocess."""
    obs, _ = envs.reset()
    cum_del = np.zeros(n_turb, dtype=np.float64)
    powers = []
    pset = np.full(n_turb, 0.93, dtype=np.float32)
    for t in range(horizon):
        action = actor_fn(obs, envs)
        try:
            feats_list = envs.env.call("get_sector_features")
        except Exception:
            feats_list = None
        if feats_list and isinstance(feats_list[0], dict) and "err" not in feats_list[0]:
            f = feats_list[0]
            saws = f["saws"]
            sati = f["sati"]
            n = saws.shape[0]
        else:
            saws = np.full((n_turb, 4), 9.0, dtype=np.float32)
            sati = np.full((n_turb, 4), 0.07, dtype=np.float32)
            n = n_turb
        # Yaw from action; vector env action shape (1, n_turb)
        a_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        yaw_deg_world = a_arr[:n] * 30.0
        per = [
            dict(saws_left=saws[i,0], saws_right=saws[i,1],
                  saws_up=saws[i,2], saws_down=saws[i,3],
                  sati_left=sati[i,0], sati_right=sati[i,1],
                  sati_up=sati[i,2], sati_down=sati[i,3])
            for i in range(n)
        ]
        x = rdf.features_to_surrogate_input(per, pset[:n].tolist(),
                                              yaw_deg_world.tolist())
        out = surr.predict_one(sensor, torch.from_numpy(x)).flatten().numpy()
        cum_del[:n] += out
        try:
            ret = envs.step(action)
            obs, _, term, trunc, info = (ret if len(ret) == 5
                                          else (ret[0], ret[1], ret[2], ret[3], ret[4]))
            if "Power agent" in info:
                p = info["Power agent"]
                powers.append(float(np.mean(p)))
            if np.any(term) or np.any(trunc):
                break
        except Exception as e:
            print(f"  step err: {e}")
            break
    return cum_del, np.mean(powers) if powers else 0.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--layout", default="stag4_5d")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--n-episodes", type=int, default=5)
    p.add_argument("--sensor", default="wrot_Bl1Rad0FlpMnt")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    surr = ts.TeodorDLC12Surrogate.from_bundle(args.bundle, outputs=[args.sensor])
    surr.eval()

    import warnings; warnings.filterwarnings("ignore")
    from py_wake.examples.data.iea37 import IEA37_WindTurbines
    turbine = IEA37_WindTurbines()

    # Load actor + tr_args from checkpoint
    print(f"loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from config import Args
    tr_args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})
    # Override layout for eval
    tr_args.layouts = args.layout
    if args.layout == "multi_modal":
        tr_args.config = "multi_modal"

    # Use the project's setup_env to get a properly wrapped vector env
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

    def actor_fn(obs, env):
        with torch.no_grad():
            a = agent.act(env, obs)
        return np.asarray(a, dtype=np.float32)  # vector env shape

    def safe_fn(obs, env):
        return np.zeros((1, n_turb), dtype=np.float32)

    cums_perf = []; pwrs_perf = []
    cums_safe = []; pwrs_safe = []
    for ep in range(args.n_episodes):
        c, p = rollout(envs, actor_fn, surr, n_turb, args.horizon, args.sensor)
        cums_perf.append(c); pwrs_perf.append(p)
        c, p = rollout(envs, safe_fn, surr, n_turb, args.horizon, args.sensor)
        cums_safe.append(c); pwrs_safe.append(p)
        print(f"ep{ep}: perf=p:{pwrs_perf[-1]:.0f} d:{cums_perf[-1]} | "
              f"safe=p:{pwrs_safe[-1]:.0f} d:{cums_safe[-1]}")

    cums_perf = np.asarray(cums_perf); cums_safe = np.asarray(cums_safe)
    pwrs_perf = np.asarray(pwrs_perf); pwrs_safe = np.asarray(pwrs_safe)
    print(f"\n=== {args.layout} ===")
    print(f"  pi_perf:  power={pwrs_perf.mean():.0f}±{pwrs_perf.std():.0f} W,"
          f" DEL/turb={cums_perf.mean(axis=0)}")
    print(f"  pi_safe:  power={pwrs_safe.mean():.0f}±{pwrs_safe.std():.0f} W,"
          f" DEL/turb={cums_safe.mean(axis=0)}")
    pwr_advantage = pwrs_perf.mean() / max(pwrs_safe.mean(), 1.0) - 1.0
    print(f"  pi_perf power advantage: {100*pwr_advantage:+.2f}% vs zero-yaw")
    del_diff = cums_perf.mean(axis=0) - cums_safe.mean(axis=0)
    print(f"  DEL diff (perf-safe per turb): {del_diff}")
    print(f"  any turbine DEL_perf > DEL_safe? {bool(np.any(del_diff > 0))}")

    out = {
        "layout": args.layout,
        "horizon": args.horizon,
        "n_episodes": args.n_episodes,
        "checkpoint": args.checkpoint,
        "perf": {"power_mean": float(pwrs_perf.mean()),
                  "power_std": float(pwrs_perf.std()),
                  "del_per_turb": cums_perf.mean(axis=0).tolist(),
                  "del_per_turb_std": cums_perf.std(axis=0).tolist()},
        "safe": {"power_mean": float(pwrs_safe.mean()),
                  "power_std": float(pwrs_safe.std()),
                  "del_per_turb": cums_safe.mean(axis=0).tolist(),
                  "del_per_turb_std": cums_safe.std(axis=0).tolist()},
        "pwr_advantage_pct": float(100 * pwr_advantage),
        "del_diff_per_turb": del_diff.tolist(),
        "both_halves_feasible": bool(pwr_advantage > 0
                                      and np.any(del_diff > 0)),
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
