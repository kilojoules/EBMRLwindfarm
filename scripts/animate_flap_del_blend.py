"""Animation: blade-flap DEL budget pacing in action.

Records a 200-step rollout with the AC-blend FlapDELBudgetSurrogate, capturing
per step:
  - flow field (hub-height u-magnitude on a 2D xy grid)
  - turbine positions + yaw markers
  - per-turbine cumulative DEL vs budget bars
  - per-turbine lambda schedule weight
  - sigma blend weight (pi_perf vs pi_safe)

Renders 5-panel animation as mp4. Designed to run on LUMI; mp4 is portable.

Usage:
  pixi run python scripts/animate_flap_del_blend.py \\
      --checkpoint runs/pi_perf_mm_300k/checkpoints/step_290000.pt \\
      --layout multi_modal \\
      --budgets 51738.4,57909.3,30484.6 \\
      --horizon 200 --eta 3.0 \\
      --out results/flap_del_blend_animation.mp4
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from matplotlib.patches import Rectangle, FancyArrowPatch

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
ls_mod = _load("helpers.layouts", ROOT / "helpers/layouts.py")


def sigma_of_u(u, eta=3.0):
    if u >= 1.0:
        return 0.0
    return 1.0 - np.exp(-eta * (1.0 / max(u, 1e-6) - 1.0))


def flow_grid(envs, x_extent, y_extent, hub_h, nx=48, ny=24):
    """Sample flow grid via a single subprocess call (avoids pipe hammering)."""
    res = envs.env.call("get_flow_grid",
                         float(x_extent[0]), float(x_extent[1]), int(nx),
                         float(y_extent[0]), float(y_extent[1]), int(ny),
                         float(hub_h))
    d = res[0]
    return d["xs"], d["ys"], d["U"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--layout", default="multi_modal")
    p.add_argument("--budgets", required=True, help="CSV per-turbine B_i [kNm-step]")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--eta", type=float, default=3.0)
    p.add_argument("--grid-nx", type=int, default=48)
    p.add_argument("--grid-ny", type=int, default=24)
    p.add_argument("--frame-stride", type=int, default=2,
                   help="render every Nth step (smaller mp4)")
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    budgets = np.array([float(x) for x in args.budgets.split(",")],
                       dtype=np.float64)
    sensor = "wrot_Bl1Rad0FlpMnt"

    # Surrogate
    surr = ts.TeodorDLC12Surrogate.from_bundle(args.bundle, outputs=[sensor])
    surr.eval()

    # Actor
    print(f"loading {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    from config import Args
    tr_args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})
    tr_args.layouts = args.layout
    if args.layout == "multi_modal":
        tr_args.config = "multi_modal"

    import warnings; warnings.filterwarnings("ignore")
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

    # Rollout — collect per-step data
    obs, _ = envs.reset()
    pset = np.full(n_turb, 0.93, dtype=np.float32)
    cum_del = np.zeros(n_turb, dtype=np.float64)

    # Record positions / extent via subprocess call
    info_list = envs.env.call("get_layout_info")
    info = info_list[0]
    pos_x = np.asarray(info["pos_x"], dtype=float)
    pos_y = np.asarray(info["pos_y"], dtype=float)
    hub_h = float(info["hub_height"])
    rd = float(info["rotor_diameter"])
    print(f"layout pos x={pos_x} y={pos_y} D={rd:.0f}m hub={hub_h:.0f}m")

    # Domain
    x_pad = 5 * rd
    y_pad = 4 * rd
    xx_extent = (pos_x.min() - x_pad, pos_x.max() + x_pad)
    yy_extent = (pos_y.min() - y_pad, pos_y.max() + y_pad)

    frames = []
    for t in range(args.horizon):
        # AC-blend action: pi_perf from actor, pi_safe = zero, sigma per turbine.
        with torch.no_grad():
            act_perf = agent.act(envs, obs)
        act_perf = np.asarray(act_perf, dtype=np.float32)
        # Per-turbine sigma from urgency
        rho = np.maximum(budgets - cum_del, 0.0) / np.maximum(budgets, 1.0)
        tau = max((args.horizon - t) / args.horizon, 1e-6)
        u_per = rho / max(tau, 1e-6)
        sigma_per = np.array([sigma_of_u(u, args.eta) for u in u_per])
        # Blend
        if act_perf.ndim == 2:
            sb = sigma_per[None, :]
        else:
            sb = sigma_per[None, :, None]
        act_safe = np.zeros_like(act_perf)
        act_exec = (1.0 - sb) * act_perf + sb * act_safe

        # Score DEL at the executed action (real flow context)
        feats_list = envs.env.call("get_sector_features")
        if isinstance(feats_list[0], dict) and "err" not in feats_list[0]:
            saws = feats_list[0]["saws"]; sati = feats_list[0]["sati"]
            yaw_now_deg = feats_list[0]["yaw_deg"]
        else:
            saws = np.full((n_turb, 4), 9.0, dtype=np.float32)
            sati = np.full((n_turb, 4), 0.07, dtype=np.float32)
            yaw_now_deg = np.zeros(n_turb)
        per = [
            dict(saws_left=saws[i,0], saws_right=saws[i,1],
                  saws_up=saws[i,2], saws_down=saws[i,3],
                  sati_left=sati[i,0], sati_right=sati[i,1],
                  sati_up=sati[i,2], sati_down=sati[i,3])
            for i in range(n_turb)
        ]
        x_in = rdf.features_to_surrogate_input(
            per, pset.tolist(), yaw_now_deg.tolist())
        del_step = surr.predict_one(sensor,
                                     torch.from_numpy(x_in)).flatten().numpy()
        cum_del += del_step

        # Snapshot flow grid (only on rendered frames to save time)
        if t % args.frame_stride == 0:
            xs, ys, U = flow_grid(envs, xx_extent, yy_extent,
                                    hub_h, nx=args.grid_nx, ny=args.grid_ny)
            frames.append({
                "t": t,
                "xs": xs, "ys": ys, "U": U,
                "yaw_deg": yaw_now_deg.copy(),
                "cum_del": cum_del.copy(),
                "del_step": del_step.copy(),
                "sigma": sigma_per.copy(),
                "u": u_per.copy(),
                "rho": rho.copy(),
                "tau": tau,
            })
            print(f"  t={t:3d}  σ={[f'{s:.2f}' for s in sigma_per]}  "
                  f"util={[f'{u:.2f}' for u in cum_del / np.maximum(budgets, 1)]}")

        # Step env
        ret = envs.step(act_exec)
        obs, _, term, trunc, info = (ret if len(ret) == 5
                                      else (ret[0], ret[1], ret[2], ret[3], ret[4]))
        if np.any(term) or np.any(trunc): break

    # Close env now — ffmpeg subprocess + AsyncVectorEnv subprocess fight
    # over signals during mp4 render and break the env pipe.
    try:
        envs.close()
    except Exception as _e:
        print(f"  envs.close warn: {_e}")

    # ---------------- Render animation ----------------
    print(f"\nrecorded {len(frames)} frames; rendering mp4...")

    fig = plt.figure(figsize=(13, 7), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1.0],
                           width_ratios=[1.0, 1.0, 1.1])
    ax_flow = fig.add_subplot(gs[0, :2])
    ax_del  = fig.add_subplot(gs[0, 2])
    ax_sig  = fig.add_subplot(gs[1, 0])
    ax_u    = fig.add_subplot(gs[1, 1])
    ax_rate = fig.add_subplot(gs[1, 2])

    # Flow init
    ax_flow.set_aspect("equal")
    ax_flow.set_xlim(*xx_extent); ax_flow.set_ylim(*yy_extent)
    ax_flow.set_xlabel("x [m]"); ax_flow.set_ylabel("y [m]")
    ax_flow.set_title("Hub-height wind magnitude (with wakes) + turbine yaw")

    flow_im = ax_flow.imshow(frames[0]["U"], origin="lower",
                              extent=[*xx_extent, *yy_extent],
                              cmap="viridis", vmin=4, vmax=11, aspect="equal",
                              interpolation="bilinear", zorder=1)
    cbar = fig.colorbar(flow_im, ax=ax_flow, fraction=0.025, pad=0.01)
    cbar.set_label("u [m/s]")

    rotor_lines = []
    yaw_arrows = []
    for i in range(n_turb):
        line, = ax_flow.plot([], [], "-", color="black", lw=4, zorder=4)
        rotor_lines.append(line)
        ax_flow.scatter(pos_x[i], pos_y[i], marker="o", s=18, c="black", zorder=5)
        ax_flow.text(pos_x[i] + 30, pos_y[i] + 30,
                     f"T{i}", fontsize=10, color="black", zorder=6)
    title_text = ax_flow.text(0.02, 0.97, "", transform=ax_flow.transAxes,
                                va="top", ha="left", color="white",
                                fontsize=11, family="monospace",
                                bbox=dict(boxstyle="round,pad=0.3",
                                            fc="black", alpha=0.55))

    # DEL bars vs budget
    bar_x = np.arange(n_turb)
    util0 = frames[0]["cum_del"] / np.maximum(budgets, 1)
    bars = ax_del.bar(bar_x, util0, color="#1f77b4", edgecolor="black")
    ax_del.axhline(1.0, color="crimson", lw=1.5, ls="--",
                   label="budget B$_i$")
    ax_del.set_ylim(0, 1.3)
    ax_del.set_xticks(bar_x); ax_del.set_xticklabels([f"T{i}" for i in range(n_turb)])
    ax_del.set_ylabel("util = $C_t/B_i$")
    ax_del.set_title("Per-turbine DEL utilisation")
    ax_del.legend(loc="upper right", fontsize=8)
    ax_del.grid(alpha=0.25, axis="y")

    # Sigma trace (per turbine over time)
    sig_lines = []
    for i in range(n_turb):
        l, = ax_sig.plot([], [], lw=1.6, label=f"T{i}")
        sig_lines.append(l)
    ax_sig.set_xlim(0, args.horizon); ax_sig.set_ylim(-0.02, 1.05)
    ax_sig.set_xlabel("step"); ax_sig.set_ylabel("$\\sigma_i(u_i)$")
    ax_sig.set_title("blend weight (0=$\\pi_{\\rm perf}$, 1=$\\pi_{\\rm safe}$)")
    ax_sig.grid(alpha=0.25); ax_sig.legend(fontsize=7, loc="upper left")

    # Urgency u over time
    u_lines = []
    for i in range(n_turb):
        l, = ax_u.plot([], [], lw=1.6)
        u_lines.append(l)
    ax_u.set_xlim(0, args.horizon); ax_u.set_ylim(0, 4.0)
    ax_u.axhline(1.0, color="crimson", lw=1.0, ls="--", label="$u=1$ pace")
    ax_u.set_xlabel("step"); ax_u.set_ylabel("urgency $u_i=\\rho_i/\\tau$")
    ax_u.set_title("urgency ratio per turbine"); ax_u.grid(alpha=0.25)
    ax_u.legend(fontsize=7, loc="upper right")

    # DEL rate (per-step DEL over time)
    rate_lines = []
    for i in range(n_turb):
        l, = ax_rate.plot([], [], lw=1.4)
        rate_lines.append(l)
    ax_rate.set_xlim(0, args.horizon)
    ax_rate.set_ylim(0, 1500)
    ax_rate.set_xlabel("step"); ax_rate.set_ylabel("DEL$_t$ [kNm/step]")
    ax_rate.set_title("instantaneous DEL per step")
    ax_rate.grid(alpha=0.25)

    # History buffers
    ts_hist = []
    sig_hist = [[] for _ in range(n_turb)]
    u_hist = [[] for _ in range(n_turb)]
    rate_hist = [[] for _ in range(n_turb)]

    def update(idx):
        f = frames[idx]
        flow_im.set_data(f["U"])
        # Rotor markers — line oriented perpendicular to the yaw direction
        for i in range(n_turb):
            yaw_rad = np.deg2rad(f["yaw_deg"][i])
            # rotor disk is perpendicular to (cos yaw, sin yaw) — lies along (-sin, cos)
            ex = -np.sin(yaw_rad); ey = np.cos(yaw_rad)
            half = rd / 2
            x0, y0 = pos_x[i] - ex * half, pos_y[i] - ey * half
            x1, y1 = pos_x[i] + ex * half, pos_y[i] + ey * half
            rotor_lines[i].set_data([x0, x1], [y0, y1])
        title_text.set_text(
            f"t={f['t']:3d}/{args.horizon}  τ={f['tau']:.2f}  "
            f"yaw={[f'{y:+.0f}°' for y in f['yaw_deg']]}")

        util = f["cum_del"] / np.maximum(budgets, 1)
        for b, u in zip(bars, util):
            b.set_height(u)
            b.set_color("crimson" if u > 1.0 else "#1f77b4")

        ts_hist.append(f["t"])
        for i in range(n_turb):
            sig_hist[i].append(f["sigma"][i])
            u_hist[i].append(f["u"][i])
            rate_hist[i].append(f["del_step"][i])
            sig_lines[i].set_data(ts_hist, sig_hist[i])
            u_lines[i].set_data(ts_hist, u_hist[i])
            rate_lines[i].set_data(ts_hist, rate_hist[i])

        return (flow_im, *rotor_lines, *bars, *sig_lines, *u_lines,
                *rate_lines, title_text)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // args.fps, blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=2400,
                           codec="libx264",
                           extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out), writer=writer, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
