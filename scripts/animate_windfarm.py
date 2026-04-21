"""
Birds-eye animation of 3-turbine wind farm episode.

- Left: x-y plane, turbines (triangles oriented by yaw) + wake trails
        (cost indicator shaded behind each turbine when active)
- Right top: cumulative neg-yaw steps per turbine vs budget
- Right bottom: cumulative power

Usage: python scripts/animate_windfarm.py --budget 15 --seed 1 --out results/wf_ep.mp4
"""
import argparse
import sys
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import RegularPolygon
from pathlib import Path

sys.path.insert(0, ".")

from config import Args
from helpers.agent import WindFarmAgent


def run_episode(checkpoint, budget, seed, horizon=200, eta=2.0, gs=0.1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})

    from ebt_sac_windfarm import setup_env
    env_info = setup_env(args)
    envs = env_info["envs"]
    n_turb = env_info["n_turbines_max"]

    from ebt import TransformerEBTActor
    from networks import create_profile_encoding
    use_profiles = env_info["use_profiles"]
    sr, si = None, None
    if use_profiles:
        sr, si = create_profile_encoding(
            profile_type=args.profile_encoding_type, embed_dim=args.embed_dim,
            hidden_channels=args.profile_encoder_hidden)
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
        shared_recep_encoder=sr, shared_influence_encoder=si,
        action_scale=env_info["action_scale"],
        action_bias=env_info["action_bias"],
        opt_steps_train=args.ebt_opt_steps_train,
        opt_steps_eval=args.ebt_opt_steps_eval,
        opt_lr=args.ebt_opt_lr, num_candidates=args.ebt_num_candidates,
        args=args).to(device)
    actor.load_state_dict(ckpt["actor_state_dict"])
    actor.eval()

    agent = WindFarmAgent(actor=actor, device=device,
                          rotor_diameter=env_info["rotor_diameter"],
                          use_wind_relative=args.use_wind_relative_pos,
                          use_profiles=use_profiles,
                          rotate_profiles=getattr(args, "rotate_profiles", False))

    from load_surrogates import NegativeYawBudgetSurrogate
    surr = NegativeYawBudgetSurrogate(
        budget_steps=budget, horizon_steps=horizon,
        risk_aversion=eta, steepness=2.0, yaw_max_deg=30.0)
    surr.reset()
    obs, _ = envs.reset(seed=seed)

    # Trajectory containers
    yaws = np.zeros((horizon, n_turb))
    powers = np.zeros((horizon, n_turb))
    cum_neg = np.zeros((horizon, n_turb))
    wind_speed = np.zeros(horizon)
    wind_dir = np.zeros(horizon)
    lambdas = np.zeros(horizon)

    turb_xy = None

    for t in range(horizon):
        with torch.no_grad():
            act = agent.act(envs, obs, guidance_fn=surr, guidance_scale=gs)
        obs, rew, _, _, info = envs.step(act)

        if "yaw angles agent" in info:
            ya = np.array(info["yaw angles agent"])
            yaw_flat = ya[0] if ya.ndim > 1 else ya
            for ti in range(min(len(yaw_flat), n_turb)):
                yaws[t, ti] = yaw_flat[ti]
                if yaw_flat[ti] < 0:
                    cum_neg[t:, ti] = cum_neg[max(t - 1, 0), ti] + 1
                else:
                    cum_neg[t, ti] = cum_neg[max(t - 1, 0), ti]
            surr.update(torch.tensor(yaw_flat[:n_turb], device=device,
                                     dtype=torch.float32))
        if "Power agent" in info:
            pa = np.array(info["Power agent"])
            pa = pa[0] if pa.ndim > 1 else pa
            for ti in range(min(len(pa), n_turb)):
                powers[t, ti] = pa[ti]
        if "wind_speed" in info:
            wind_speed[t] = float(np.atleast_1d(info["wind_speed"])[0])
        if "wind_direction" in info:
            wind_dir[t] = float(np.atleast_1d(info["wind_direction"])[0])
        lambdas[t] = float(surr.compute_lambda()) if hasattr(surr, "compute_lambda") else 1.0
        if turb_xy is None and "turbine_positions" in info:
            tp = np.array(info["turbine_positions"])
            turb_xy = tp[0] if tp.ndim == 3 else tp

    if turb_xy is None:
        # fallback layout from helpers.layouts
        from helpers.layouts import get_layout
        xs, ys = get_layout("3turb")
        turb_xy = np.stack([xs, ys], axis=-1)

    envs.close()
    return dict(yaws=yaws, powers=powers, cum_neg=cum_neg,
                wind_speed=wind_speed, wind_dir=wind_dir,
                lambdas=lambdas, turb_xy=turb_xy, budget=budget,
                rotor_d=env_info["rotor_diameter"])


def make_animation(data, out_path):
    yaws = data["yaws"]; powers = data["powers"]; cum_neg = data["cum_neg"]
    ws = data["wind_speed"]; wd = data["wind_dir"]; lam = data["lambdas"]
    xy = np.array(data["turb_xy"]); D = data["rotor_d"]; B = data["budget"]
    T, N = yaws.shape

    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.4, 1], hspace=0.3, wspace=0.28)
    ax_xy = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    # Birds-eye layout
    ax_xy.set_aspect("equal"); ax_xy.grid(alpha=0.3)
    xmin, xmax = xy[:, 0].min() - 3 * D, xy[:, 0].max() + 3 * D
    ymin, ymax = xy[:, 1].min() - 3 * D, xy[:, 1].max() + 3 * D
    ax_xy.set_xlim(xmin, xmax); ax_xy.set_ylim(ymin, ymax)
    ax_xy.set_xlabel("x [m]"); ax_xy.set_ylabel("y [m]")

    turb_patches = []
    wake_lines = []
    for i, (x, y) in enumerate(xy):
        rotor, = ax_xy.plot([x, x], [y - D / 2, y + D / 2], "k-", lw=3)
        blade, = ax_xy.plot([x - D / 4, x + D / 4], [y, y], "C0-", lw=4)
        wake, = ax_xy.plot([], [], "C3-", lw=2, alpha=0.5)
        ax_xy.text(x + D * 0.15, y + D * 0.6, f"T{i}", fontsize=10, weight="bold")
        turb_patches.append((rotor, blade))
        wake_lines.append(wake)

    wind_arrow = ax_xy.annotate("", xy=(xmax - D, ymax - D * 0.7),
                                 xytext=(xmax - 3 * D, ymax - D * 0.7),
                                 arrowprops=dict(arrowstyle="->", color="C2", lw=2.5))
    title = ax_xy.set_title("t=0")

    # Cumulative neg-yaw per turbine
    colors = ["C0", "C1", "C2"]
    lines_c = [ax_c.plot(cum_neg[:, i], colors[i] + "-", lw=1.5, alpha=0.4,
                          label=f"T{i}")[0] for i in range(N)]
    pts_c = [ax_c.plot([0], [cum_neg[0, i]], colors[i] + "o", ms=6)[0]
             for i in range(N)]
    ax_c.axhline(B, color="k", ls="--", lw=1, label=f"budget d={B}")
    ax_c.set_ylabel("cum. neg-yaw steps"); ax_c.set_xlim(0, T)
    ax_c.legend(fontsize=8, loc="upper left"); ax_c.grid(alpha=0.3)

    # Cumulative power (sum of turbines)
    cum_pow = np.cumsum(powers.sum(axis=1)) / 1e6
    ax_p.plot(cum_pow, "C4-", lw=1.5, alpha=0.4)
    pt_p, = ax_p.plot([0], [cum_pow[0]], "C4o", ms=6)
    ax_p.set_xlabel("step"); ax_p.set_ylabel("cum. power [MW·step]")
    ax_p.set_xlim(0, T); ax_p.grid(alpha=0.3)

    def yaw_rotor_coords(x, y, yaw_deg):
        theta = np.deg2rad(yaw_deg)
        cx, sy = np.cos(theta), np.sin(theta)
        dx, dy = -sy * D / 2, cx * D / 2
        return [x - dx, x + dx], [y - dy, y + dy]

    def update(t):
        for i, (x, y) in enumerate(xy):
            rx, ry = yaw_rotor_coords(x, y, yaws[t, i])
            rotor, blade = turb_patches[i]
            rotor.set_data(rx, ry)
            # blade perpendicular to rotor
            theta = np.deg2rad(yaws[t, i])
            blade.set_data([x - np.cos(theta) * D / 4, x + np.cos(theta) * D / 4],
                            [y - np.sin(theta) * D / 4, y + np.sin(theta) * D / 4])
            if yaws[t, i] < 0:
                rotor.set_color("C3")
            else:
                rotor.set_color("k")
        # Wind arrow direction (wind goes toward +x at 270°, flip for wd)
        theta_w = np.deg2rad(wd[t] if wd[t] != 0 else 270.0)
        awx, awy = xmax - D, ymax - D * 0.7
        dx = np.sin(theta_w) * 2 * D
        dy = -np.cos(theta_w) * 2 * D
        wind_arrow.xy = (awx, awy)
        wind_arrow.set_position((awx - dx, awy - dy))

        title.set_text(f"t={t}  ws={ws[t]:.1f} m/s  λ={lam[t]:.2f}  "
                        f"yaws=[{yaws[t,0]:+.0f},{yaws[t,1]:+.0f},{yaws[t,2]:+.0f}]°")
        for i, l in enumerate(pts_c):
            l.set_data([t], [cum_neg[t, i]])
        pt_p.set_data([t], [cum_pow[t]])
        return [*sum(turb_patches, ()), *pts_c, pt_p, title]

    ani = animation.FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    ani.save(out_path, writer="ffmpeg", fps=10, dpi=100, bitrate=2000)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/ebt_sac_windfarm/checkpoints/step_100000.pt")
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--eta", type=float, default=2.0)
    p.add_argument("--gs", type=float, default=0.1)
    p.add_argument("--out", default="results/windfarm_ep.mp4")
    args = p.parse_args()

    import glob
    if not os.path.exists(args.checkpoint):
        cands = glob.glob("runs/*/checkpoints/step_*.pt")
        if cands:
            args.checkpoint = sorted(cands)[-1]
            print(f"Using checkpoint {args.checkpoint}")

    data = run_episode(args.checkpoint, args.budget, args.seed,
                       args.horizon, args.eta, args.gs)
    Path(args.out).parent.mkdir(exist_ok=True)
    make_animation(data, args.out)


if __name__ == "__main__":
    main()
