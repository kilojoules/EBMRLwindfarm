"""Safety Gym hazard test: does alignment hazard appear in 2D navigation?

Setup analogous to yaw sweep: sweep pi_safe direction relative to pi_perf,
fix sigma=0.7, measure cost+reward under linear vs rejection blend.

pi_safe rotation: APF direction rotated in world frame by angle theta.
- theta=0:    APF points toward goal, away from hazards (genuine safety)
- theta=pi/2: APF points 90 deg off goal direction (no safety, no goal)
- theta=pi:   APF points away from goal, toward hazards (anti-safety)

Linear blend at sigma=0.7 averages pi_perf (toward goal) and rotated APF.
At theta != 0, averaged action points along blend direction -> may cross
hazard rather than going around.

Rejection blend: probe 1-step cost (hazard indicator) under each candidate,
engage sigma only when safe lowers cost.

Output JSON for plotting Pareto: reward vs cost across theta, per mode.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# Shim removed-in-gymnasium-1.x module used by safety-gymnasium 1.0.0
import types
try:
    import gymnasium.wrappers.compatibility  # noqa: F401
except ImportError:
    _shim = types.ModuleType("gymnasium.wrappers.compatibility")
    class _EC: pass
    _shim.EnvCompatibility = _EC
    sys.modules["gymnasium.wrappers.compatibility"] = _shim

# Reuse apf_blend_sg's helpers
from scripts.apf_blend_sg import (
    apf_world_direction, world_to_action, agent_state,
    goal_pos, hazards, DEVICE,
)


def rotate(v, theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


def in_hazard(pos, haz_pos, r_haz):
    """Cheap 1-step cost probe: is current pos inside any hazard?"""
    for h in haz_pos:
        if np.linalg.norm(np.asarray(pos) - np.asarray(h)) < r_haz:
            return 1.0
    return 0.0


def project_next_pos(pos, vel, heading, action, dt=0.002, mass_inv=1.0):
    """Rough 1-step pos projection from action (thrust, turn).
    Used by rejection probe to estimate which action keeps agent outside hazard.
    Crude — but enough for tie-breaking near hazard edges."""
    thrust, turn = action
    new_heading = heading + 0.5 * turn  # rough
    fwd = np.array([np.cos(new_heading), np.sin(new_heading)])
    new_vel = np.asarray(vel) + thrust * fwd * mass_inv * 0.1
    new_pos = np.asarray(pos) + new_vel * 0.05
    return new_pos


def run_episode(env, actor, theta, sigma_fixed, mode,
                budget=25.0, horizon=1000, alpha_apf=0.5, k_rep=1.0,
                r_repel=0.6, act_limit=1.0, hazard_proximity_pad=0.0):
    obs, _ = env.reset(seed=hash((theta, mode)) % (2**31))
    cum_r = 0.0; cum_c = 0.0
    for t in range(horizon):
        pos, vel, heading = agent_state(env)
        goal = goal_pos(env)
        haz_pos, r_haz = hazards(env)

        with torch.no_grad():
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            a, _ = actor.sample(s)
        a_perf = a.squeeze(0).cpu().numpy() * act_limit

        dir_w = apf_world_direction(pos, goal, haz_pos, r_haz,
                                      r_repel=r_repel, alpha=alpha_apf, k_rep=k_rep)
        dir_w_rot = rotate(dir_w, theta)
        a_safe = world_to_action(dir_w_rot, heading, act_limit=act_limit)

        if mode == "linear":
            sigma_eff = sigma_fixed
        elif mode == "rejection":
            # Probe 1-step pos under each candidate; engage sigma only if safe avoids hazard
            next_p_perf = project_next_pos(pos, vel, heading, a_perf)
            next_p_safe = project_next_pos(pos, vel, heading, a_safe)
            c_perf = in_hazard(next_p_perf, haz_pos, r_haz + hazard_proximity_pad)
            c_safe = in_hazard(next_p_safe, haz_pos, r_haz + hazard_proximity_pad)
            sigma_eff = sigma_fixed if c_safe < c_perf else 0.0
        elif mode == "argmin":
            next_p_perf = project_next_pos(pos, vel, heading, a_perf)
            next_p_safe = project_next_pos(pos, vel, heading, a_safe)
            c_perf = in_hazard(next_p_perf, haz_pos, r_haz + hazard_proximity_pad)
            c_safe = in_hazard(next_p_safe, haz_pos, r_haz + hazard_proximity_pad)
            sigma_eff = 1.0 if c_safe < c_perf else 0.0
        else:
            raise ValueError(mode)

        a_exec = (1.0 - sigma_eff) * a_perf + sigma_eff * a_safe
        a_exec = np.clip(a_exec, -act_limit, act_limit)

        ret = env.step(a_exec)
        if len(ret) == 6:
            obs, r, c, term, trunc, info = ret
        else:
            obs, r, term, trunc, info = ret
            c = info.get("cost", 0.0)
        cum_r += float(r); cum_c += float(c)
        if term or trunc: break
    return dict(reward=cum_r, cost=cum_c, steps=t + 1)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="SG SAC actor checkpoint")
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--n-episodes", type=int, default=10)
    p.add_argument("--sigma", type=float, default=0.7)
    p.add_argument("--theta-grid", default="0,0.785,1.571,2.356,3.141",
                    help="CSV theta in radians (0, pi/4, pi/2, 3pi/4, pi)")
    p.add_argument("--modes", default="linear,rejection,argmin")
    p.add_argument("--out-json", required=True)
    args = p.parse_args()

    import safety_gymnasium

    print(f"loading {args.checkpoint}")
    ac = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    from uncertainty_gated_qc import SafetyGymActor
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()
    env = safety_gymnasium.make(ac["env_name"])

    thetas = [float(x) for x in args.theta_grid.split(",")]
    modes = [m.strip() for m in args.modes.split(",")]

    results = []
    for mode in modes:
        for theta in thetas:
            rows = [run_episode(env, actor, theta, args.sigma, mode,
                                  horizon=args.horizon)
                     for _ in range(args.n_episodes)]
            r_mean = float(np.mean([r["reward"] for r in rows]))
            c_mean = float(np.mean([r["cost"] for r in rows]))
            r_std = float(np.std([r["reward"] for r in rows]))
            c_std = float(np.std([r["cost"] for r in rows]))
            print(f"  {mode} theta={theta:+.3f} rad: R={r_mean:.2f}±{r_std:.2f} "
                  f"C={c_mean:.2f}±{c_std:.2f}")
            results.append(dict(mode=mode, theta=theta,
                                  reward_mean=r_mean, reward_std=r_std,
                                  cost_mean=c_mean, cost_std=c_std))

    out = {"env_name": ac["env_name"], "horizon": args.horizon,
            "n_episodes": args.n_episodes, "sigma": args.sigma,
            "checkpoint": args.checkpoint, "results": results}
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
