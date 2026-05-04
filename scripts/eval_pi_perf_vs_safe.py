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


def rollout(env, actor_fn, surr, n_turb, horizon, sensor):
    """Run one episode. actor_fn(obs, env) -> action_array (n_turb,)."""
    obs, _ = env.reset()
    cum_del = np.zeros(n_turb, dtype=np.float64)
    powers = []
    pset = np.full(n_turb, 0.93, dtype=np.float32)
    for t in range(horizon):
        action = actor_fn(obs, env)
        # Read disk features, score DEL
        per = []
        for i in range(n_turb):
            try:
                feats = rdf.disk_features_for_env(env, turbine_idx=i,
                                                    yaw_deg=0.0)
            except Exception:
                feats = dict(saws_left=9.0, saws_right=9.0, saws_up=9.0,
                              saws_down=9.0, sati_left=0.07, sati_right=0.07,
                              sati_up=0.07, sati_down=0.07)
            per.append(feats)
        # Use action as yaw_deg
        yaw_deg_world = (np.asarray(action, dtype=np.float32).flatten()
                          * 30.0)  # action_scale ≈ 30deg per surrogate yaw_max
        x = rdf.features_to_surrogate_input(per, pset.tolist(),
                                              yaw_deg_world.tolist())
        out = surr.predict_one(sensor, torch.from_numpy(x)).flatten().numpy()
        cum_del += out
        # Step env with action (zero yaw or actor)
        try:
            ret = env.step(action)
            obs, _, term, trunc, info = ret if len(ret) == 5 else (
                ret[0], ret[1], ret[2], ret[3], ret[4])
            if "Power agent" in info:
                powers.append(float(np.mean(info["Power agent"])))
            if term or trunc: break
        except Exception:
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
    from WindGym import WindFarmEnv
    from py_wake.examples.data.iea37 import IEA37_WindTurbines
    layouts_mod = _load("helpers.layouts", ROOT / "helpers/layouts.py")
    turbine = IEA37_WindTurbines()
    x_arr, y_arr = layouts_mod.get_layout_positions(args.layout, turbine)
    xs, ys = list(map(float, x_arr)), list(map(float, y_arr))
    n_turb = len(xs)
    cfg_name = "multi_modal" if args.layout == "multi_modal" else "default"
    cfg = ec.make_env_config(cfg_name)

    # Load actor
    print(f"loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from config import Args
    tr_args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})
    from ebt import TransformerEBTActor
    from networks import create_profile_encoding
    sr, si = None, None
    if getattr(tr_args, "use_profiles", False):
        sr, si = create_profile_encoding(
            profile_type=tr_args.profile_encoding_type,
            embed_dim=tr_args.embed_dim,
            hidden_channels=tr_args.profile_encoder_hidden)
    actor = TransformerEBTActor(
        obs_dim_per_turbine=4, action_dim_per_turbine=1,
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
        action_scale=1.0, action_bias=0.0,
        opt_steps_train=tr_args.ebt_opt_steps_train,
        opt_steps_eval=tr_args.ebt_opt_steps_eval,
        opt_lr=tr_args.ebt_opt_lr,
        num_candidates=tr_args.ebt_num_candidates, args=tr_args)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    from helpers.agent import WindFarmAgent
    agent = WindFarmAgent(actor=actor, device=torch.device("cpu"),
                            rotor_diameter=float(turbine.diameter()),
                            use_wind_relative=tr_args.use_wind_relative_pos,
                            use_profiles=getattr(tr_args, "use_profiles", False),
                            rotate_profiles=getattr(tr_args, "rotate_profiles", False))

    def actor_fn(obs, env):
        with torch.no_grad():
            a = agent.act(env, obs)
        return np.asarray(a, dtype=np.float32).flatten()

    def safe_fn(obs, env):
        return np.zeros(n_turb, dtype=np.float32)

    cums_perf = []; pwrs_perf = []
    cums_safe = []; pwrs_safe = []
    for ep in range(args.n_episodes):
        env = WindFarmEnv(turbine=turbine, x_pos=xs, y_pos=ys,
                          config=cfg, backend="dynamiks", seed=ep + 1)
        c, p = rollout(env, actor_fn, surr, n_turb, args.horizon, args.sensor)
        env.close()
        cums_perf.append(c); pwrs_perf.append(p)
        env = WindFarmEnv(turbine=turbine, x_pos=xs, y_pos=ys,
                          config=cfg, backend="dynamiks", seed=ep + 1)
        c, p = rollout(env, safe_fn, surr, n_turb, args.horizon, args.sensor)
        env.close()
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
