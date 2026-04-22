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
        # fallback: 3-turbine row, 5D spacing
        D = env_info["rotor_diameter"]
        turb_xy = np.array([[0.0, 0.0], [5 * D, 0.0], [10 * D, 0.0]])

    envs.close()
    return dict(yaws=yaws, powers=powers, cum_neg=cum_neg,
                wind_speed=wind_speed, wind_dir=wind_dir,
                lambdas=lambdas, turb_xy=turb_xy, budget=budget,
                rotor_d=env_info["rotor_diameter"])


def jensen_field(xy, yaws, ws, wd_deg, D, grid_x, grid_y, k=0.05):
    """Jensen wake on 2D grid. Wind direction wd_deg follows meteorological
    convention (0=N, 90=E). Project onto wind vector: u_rel along wind."""
    # Wind unit vector (flow direction, from where wind blows toward)
    theta = np.deg2rad(270.0 - wd_deg)  # convert to standard x-axis
    u_dir = np.array([np.cos(theta), np.sin(theta)])
    deficit = np.zeros(grid_x.shape)
    for (tx, ty), yaw in zip(xy, yaws):
        # downstream distance from turbine
        dx = (grid_x - tx) * u_dir[0] + (grid_y - ty) * u_dir[1]
        # crosswind distance
        dy = (grid_x - tx) * (-u_dir[1]) + (grid_y - ty) * u_dir[0]
        R = D / 2.0
        wake_r = R + k * dx
        # Ct reduced by yaw (cos^2)
        ct = 0.8 * np.cos(np.deg2rad(yaw)) ** 2
        # Yaw deflection: shifts wake center crosswind
        deflect = np.tan(np.deg2rad(yaw)) * dx * 0.3
        in_wake = (dx > 0) & (np.abs(dy - deflect) < wake_r)
        local = np.where(in_wake,
                         ct * (R / np.maximum(wake_r, 1e-3)) ** 2,
                         0.0)
        deficit += local
    return ws * (1.0 - np.clip(deficit, 0, 0.9))


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
    ax_xy.set_aspect("equal")
    xmin, xmax = xy[:, 0].min() - 3 * D, xy[:, 0].max() + 6 * D
    ymin, ymax = xy[:, 1].min() - 3 * D, xy[:, 1].max() + 3 * D
    ax_xy.set_xlim(xmin, xmax); ax_xy.set_ylim(ymin, ymax)
    ax_xy.set_xlabel("x [m]"); ax_xy.set_ylabel("y [m]")

    # Flow field grid
    gx, gy = np.meshgrid(np.linspace(xmin, xmax, 120),
                         np.linspace(ymin, ymax, 80))
    base_ws = float(ws[0]) if ws[0] > 0 else 10.0
    base_wd = float(wd[0]) if wd[0] != 0 else 270.0
    field0 = jensen_field(xy, yaws[0], base_ws, base_wd, D, gx, gy)
    flow_im = ax_xy.imshow(field0, extent=[xmin, xmax, ymin, ymax],
                           origin="lower", cmap="RdYlBu_r",
                           vmin=0.55 * base_ws, vmax=1.0 * base_ws,
                           alpha=0.65, zorder=0, aspect="auto")
    cbar = plt.colorbar(flow_im, ax=ax_xy, shrink=0.7, pad=0.02)
    cbar.set_label("u [m/s]", fontsize=9)

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
        cur_ws = float(ws[t]) if ws[t] > 0 else base_ws
        cur_wd = float(wd[t]) if wd[t] != 0 else base_wd
        field = jensen_field(xy, yaws[t], cur_ws, cur_wd, D, gx, gy)
        flow_im.set_data(field)
        for i, (x, y) in enumerate(xy):
            rx, ry = yaw_rotor_coords(x, y, yaws[t, i])
            rotor, blade = turb_patches[i]
            rotor.set_data(rx, ry)
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
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        ani.save(out_path, writer="ffmpeg", fps=10, dpi=100, bitrate=2000)
    except Exception as e:
        print(f"ffmpeg failed ({e}); saving as gif")
        gif_path = str(out_path).replace(".mp4", ".gif")
        ani.save(gif_path, writer="pillow", fps=10, dpi=80)
        out_path = gif_path
    print(f"wrote {out_path}")

    # Final-frame snapshot for 2-panel κ-diagnostic figure (Option 8)
    fig2, ax2 = plt.subplots(figsize=(6, 6))
    field_f = jensen_field(xy, yaws[T-1], base_ws, base_wd, D, gx, gy)
    im = ax2.imshow(field_f, extent=[xmin, xmax, ymin, ymax], origin="lower",
                    cmap="RdYlBu_r", vmin=0.55 * base_ws, vmax=1.0 * base_ws,
                    alpha=0.7)
    for i, (x, y) in enumerate(xy):
        yang = yaws[T-1, i]
        th = np.deg2rad(yang)
        dxr = (D / 2) * np.sin(th); dyr = (D / 2) * np.cos(th)
        ax2.plot([x - dxr, x + dxr], [y - dyr, y + dyr], "k-", lw=3)
        ax2.text(x + D * 0.15, y + D * 0.6, f"T{i}\nψ={yang:.0f}°",
                 fontsize=9, weight="bold")
    plt.colorbar(im, ax=ax2, shrink=0.7).set_label("u [m/s]", fontsize=9)
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
    ax2.set_aspect("equal"); ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    final_neg = cum_neg[T-1]
    ax2.set_title(
        f"Wind farm  κ=0.72  (strong coupling)\n"
        f"neg-yaw=[{int(final_neg[0])},{int(final_neg[1])},"
        f"{int(final_neg[2])}]/{B}", fontsize=10)
    snap_path = str(out_path).replace(".mp4", "_snap.png").replace(".gif", "_snap.png")
    fig2.tight_layout(); fig2.savefig(snap_path, dpi=140)
    print(f"wrote {snap_path}")


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
