"""Explicit Q_r + Q_c Lagrangian deployment + animation.

Departs from blend-only animation (animate_flap_del_blend.py): replaces the
trained EBT actor with a model-based planner that uses the same Teodor DLC12
surrogate for both critics:
  Q_c(yaw) = wrot_Bl1Rad0FlpMnt        (cost: blade-flap DEL)
  Q_r(yaw) = wtur_W                    (reward: turbine power)

At each step:
  1. read 4-sector flow + pset per turbine
  2. for each candidate yaw vector a (grid sample), score
        J(a) = sum_i [ Q_r_i / P_ref - lambda_i(t) * Q_c_i / del_ref ]
     where lambda_i(t) = AC schedule from per-turbine urgency u_i = rho_i / tau
  3. pick argmax J and execute
  4. update cumulative DEL tracker

Visualises the same 5-panel layout as the blend animation but the action
selection is now explicit Lagrangian, not EBT-actor blend.
"""
import argparse
import sys
import itertools
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
try:
    import imageio_ffmpeg
    matplotlib.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    pass
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

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


def sigma_of_u(u, eta=3.0):
    if u >= 1.0:
        return 0.0
    return 1.0 - np.exp(-eta * (1.0 / max(u, 1e-6) - 1.0))


def lambda_of_u(u, eta=3.0):
    """AC schedule lambda — Lagrangian dual variable (>= 0).

    Same shape as sigma but unbounded above for hard binding.
    """
    if u >= 1.0:
        return 0.0
    return float(np.clip(np.exp(eta * (1.0 / max(u, 1e-6) - 1.0)) - 1.0,
                          0.0, 1e4))


def build_yaw_candidates(n_turb, n_levels=5, yaw_max=30.0):
    """Cartesian grid of yaw angles per turbine."""
    levels = np.linspace(-yaw_max, yaw_max, n_levels)
    return np.array(list(itertools.product(levels, repeat=n_turb)),
                     dtype=np.float32)


def score_candidates(surr, sensor_c, sensor_r, saws, sati, pset, yaws_cand,
                      lambdas, del_ref, p_ref):
    """Score K candidate yaw vectors. Returns J array (K,) and per-turbine P/C."""
    K = yaws_cand.shape[0]
    n_turb = saws.shape[0]
    # Build (K * n_turb, 10) inputs
    saws_b = np.broadcast_to(saws, (K, n_turb, 4))
    sati_b = np.broadcast_to(sati, (K, n_turb, 4))
    pset_b = np.broadcast_to(pset, (K, n_turb))[..., None]
    yaw_b = yaws_cand[..., None]
    x = np.concatenate([saws_b, sati_b, pset_b, yaw_b], axis=-1)
    x_t = torch.from_numpy(x.reshape(-1, 10).astype(np.float32))
    with torch.no_grad():
        c = surr.predict_one(sensor_c, x_t).numpy().reshape(K, n_turb)
        r = surr.predict_one(sensor_r, x_t).numpy().reshape(K, n_turb)
    # J = sum_i [ r_i / p_ref - lambda_i * c_i / del_ref ]
    J = (r / p_ref).sum(axis=1) \
        - (lambdas[None, :] * c / del_ref).sum(axis=1)
    return J, r, c


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bundle", default="checkpoints/teodor_dlc12_torch.pt")
    p.add_argument("--layout", default="multi_modal")
    p.add_argument("--budgets", required=True,
                   help="CSV per-turbine B_i [kNm-step]")
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--eta", type=float, default=3.0)
    p.add_argument("--del-ref", type=float, default=648.6)
    p.add_argument("--p-ref", type=float, default=1500.0,
                   help="Power reference [kW] for Lagrangian normalisation")
    p.add_argument("--n-yaw-levels", type=int, default=5)
    p.add_argument("--yaw-max", type=float, default=30.0)
    p.add_argument("--grid-nx", type=int, default=56)
    p.add_argument("--grid-ny", type=int, default=28)
    p.add_argument("--frame-stride", type=int, default=2)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    budgets = np.array([float(x) for x in args.budgets.split(",")],
                       dtype=np.float64)
    sensor_c = "wrot_Bl1Rad0FlpMnt"
    sensor_r = "wtur_W"

    surr = ts.TeodorDLC12Surrogate.from_bundle(
        args.bundle, outputs=[sensor_c, sensor_r])
    surr.eval()

    import warnings; warnings.filterwarnings("ignore")
    import gymnasium as gym
    from WindGym import WindFarmEnv
    from WindGym.wrappers import PerTurbineObservationWrapper, RecordEpisodeVals
    from py_wake.examples.data.iea37 import IEA37_WindTurbines
    from helpers.surrogate_hooks import SectorFlowExposer
    turbine = IEA37_WindTurbines()
    layouts_mod = _load("helpers.layouts", ROOT / "helpers/layouts.py")
    x_arr, y_arr = layouts_mod.get_layout_positions(args.layout, turbine)
    cfg_name = "multi_modal" if args.layout == "multi_modal" else "default"
    cfg = ec.make_env_config(cfg_name)
    for mes_type, prefix in {"ws_mes": "ws", "wd_mes": "wd",
                               "yaw_mes": "yaw", "power_mes": "power"}.items():
        cfg[mes_type][f"{prefix}_history_N"] = 1
        cfg[mes_type][f"{prefix}_history_length"] = 1

    n_turb = len(x_arr)
    turb_pos = np.stack([np.asarray(x_arr, dtype=np.float32),
                          np.asarray(y_arr, dtype=np.float32)], axis=-1)
    pos_x = turb_pos[:, 0]; pos_y = turb_pos[:, 1]
    rd = float(turbine.diameter())
    hub_h = 119.0  # IEA37 default; refined via subprocess if needed
    att_mask = np.zeros(n_turb, dtype=bool)

    class AnimSectorWrapper(SectorFlowExposer):
        @property
        def wd(self):
            raw = self.env.unwrapped if hasattr(self.env, "unwrapped") else self.env
            return float(getattr(raw, "wd", 270.0))
        @property
        def turbine_positions(self):
            return turb_pos
        @property
        def attention_mask(self):
            return att_mask

    def _env_init():
        e = WindFarmEnv(turbine=turbine,
                        x_pos=list(map(float, x_arr)),
                        y_pos=list(map(float, y_arr)),
                        config=cfg, backend="dynamiks", seed=1)
        e = PerTurbineObservationWrapper(e)
        e = AnimSectorWrapper(e)
        return e

    sync = gym.vector.SyncVectorEnv([_env_init])
    envs = RecordEpisodeVals(sync)

    obs, _ = envs.reset()
    info = envs.env.call("get_layout_info")[0]
    pos_x = np.asarray(info["pos_x"], dtype=float)
    pos_y = np.asarray(info["pos_y"], dtype=float)
    hub_h = float(info["hub_height"])
    rd = float(info["rotor_diameter"])

    yaw_cands = build_yaw_candidates(n_turb, args.n_yaw_levels, args.yaw_max)
    print(f"K={len(yaw_cands)} candidate yaw vectors over {n_turb} turbines")

    cum_del = np.zeros(n_turb, dtype=np.float64)
    pset = np.full(n_turb, 0.93, dtype=np.float32)

    x_pad = 5 * rd; y_pad = 4 * rd
    xx_extent = (pos_x.min() - x_pad, pos_x.max() + x_pad)
    yy_extent = (pos_y.min() - y_pad, pos_y.max() + y_pad)

    frames = []
    for t in range(args.horizon):
        # Read 4-sector flow + current yaw
        feats = envs.env.call("get_sector_features")[0]
        if "err" in feats:
            saws = np.full((n_turb, 4), 9.0, dtype=np.float32)
            sati = np.full((n_turb, 4), 0.07, dtype=np.float32)
        else:
            saws = feats["saws"]; sati = feats["sati"]

        # AC schedule lambda per turbine
        rho = np.maximum(budgets - cum_del, 0.0) / np.maximum(budgets, 1.0)
        tau = max((args.horizon - t) / args.horizon, 1e-6)
        u_per = rho / max(tau, 1e-6)
        sigma_per = np.array([sigma_of_u(u, args.eta) for u in u_per])
        lambda_per = np.array([lambda_of_u(u, args.eta) for u in u_per])

        # Score K candidates with Lagrangian objective
        J, r_per, c_per = score_candidates(
            surr, sensor_c, sensor_r,
            saws, sati, pset, yaw_cands,
            lambdas=lambda_per, del_ref=args.del_ref, p_ref=args.p_ref)
        best_idx = int(np.argmax(J))
        yaw_chosen = yaw_cands[best_idx]                    # (n_turb,) deg

        # DEL at chosen yaw → accumulate
        del_step = c_per[best_idx]
        cum_del += del_step
        p_step = r_per[best_idx]                            # (n_turb,) kW

        # Convert yaw to normalised action a in [-1, 1]
        act_exec = (yaw_chosen / args.yaw_max).reshape(1, n_turb).astype(np.float32)

        if t % args.frame_stride == 0:
            grid = envs.env.call("get_flow_grid",
                                   float(xx_extent[0]), float(xx_extent[1]),
                                   int(args.grid_nx),
                                   float(yy_extent[0]), float(yy_extent[1]),
                                   int(args.grid_ny),
                                   float(hub_h))[0]
            X, Y = np.meshgrid(grid["xs"], grid["ys"])
            frames.append({
                "t": t,
                "X": X, "Y": Y, "U": grid["U"],
                "yaw_deg": yaw_chosen.copy(),
                "cum_del": cum_del.copy(),
                "del_step": del_step.copy(),
                "p_step": p_step.copy(),
                "sigma": sigma_per.copy(),
                "lambda": lambda_per.copy(),
                "u": u_per.copy(),
                "tau": tau,
                "J_best": float(J[best_idx]),
            })
            print(f"  t={t:3d}  yaw={[f'{y:+.0f}' for y in yaw_chosen]}  "
                  f"P={p_step.sum():.0f}  DEL={del_step.sum():.0f}  "
                  f"util_max={(cum_del/np.maximum(budgets,1)).max():.2f}  "
                  f"σ̄={sigma_per.mean():.2f}")

        ret = envs.step(act_exec)
        obs = ret[0]
        if np.any(ret[2]) or np.any(ret[3]):
            break

    try:
        envs.close()
    except Exception:
        pass

    print(f"\nrendering {len(frames)} frames...")

    fig = plt.figure(figsize=(14, 7.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 3, height_ratios=[2.4, 1.0],
                           width_ratios=[1.0, 1.0, 1.1])
    ax_flow = fig.add_subplot(gs[0, :2])
    ax_cum  = fig.add_subplot(gs[0, 2])
    ax_sig  = fig.add_subplot(gs[1, 0])
    ax_lam  = fig.add_subplot(gs[1, 1])
    ax_pwr  = fig.add_subplot(gs[1, 2])

    ff0 = frames[0]
    X0, Y0 = ff0["X"], ff0["Y"]
    U0 = np.asarray(ff0["U"])
    finite_U = U0[np.isfinite(U0)]
    vmin = max(0.0, float(np.percentile(finite_U, 2))) if finite_U.size else 4.0
    vmax = float(np.percentile(finite_U, 99)) + 0.5 if finite_U.size else 11.0
    print(f"flow colormap: {vmin:.1f} → {vmax:.1f}")

    ax_flow.set_aspect("equal")
    ax_flow.set_xlim(*xx_extent); ax_flow.set_ylim(*yy_extent)
    ax_flow.set_xlabel("x [m]"); ax_flow.set_ylabel("y [m]")
    ax_flow.set_title("Q$_r$ + Q$_c$ Lagrangian planner: a* = argmax [Q$_r$ − λ Q$_c$]")
    flow_mesh = ax_flow.pcolormesh(X0, Y0, U0, cmap="viridis",
                                     vmin=vmin, vmax=vmax,
                                     shading="auto", zorder=1)
    fig.colorbar(flow_mesh, ax=ax_flow, fraction=0.025, pad=0.01,
                 label="u [m/s]")

    rotor_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "rotor_lam", [(0.0, "#0b3d91"), (0.5, "#888888"), (1.0, "#d32f2f")])
    rotor_lines = []
    for i in range(n_turb):
        line, = ax_flow.plot([], [], "-", lw=5, zorder=5)
        rotor_lines.append(line)
        ax_flow.scatter(pos_x[i], pos_y[i], marker="o", s=28, c="black", zorder=6)
        ax_flow.text(pos_x[i] + 30, pos_y[i] + 30,
                     f"T{i}", fontsize=11, fontweight="bold", zorder=7)
    title_text = ax_flow.text(0.02, 0.97, "", transform=ax_flow.transAxes,
                                va="top", ha="left", color="white",
                                fontsize=11, family="monospace",
                                bbox=dict(boxstyle="round,pad=0.3",
                                            fc="black", alpha=0.6))
    viol_text = ax_flow.text(0.5, 0.04, "",
                              transform=ax_flow.transAxes,
                              va="bottom", ha="center", color="white",
                              fontsize=11, fontweight="bold",
                              bbox=dict(boxstyle="round,pad=0.4",
                                          fc="#d32f2f", alpha=0.0))

    cmap_t = plt.get_cmap("tab10")
    colors_t = [cmap_t(i) for i in range(n_turb)]
    cum_lines = []
    for i in range(n_turb):
        l, = ax_cum.plot([], [], lw=2.0, color=colors_t[i],
                          label=f"T{i}")
        cum_lines.append(l)
        ax_cum.axhline(budgets[i], color=colors_t[i], lw=1.0, ls="--",
                       alpha=0.6)
        ax_cum.plot([0, args.horizon], [0, budgets[i]],
                     color=colors_t[i], lw=0.8, ls=":", alpha=0.5)
    cum_max = max(float(np.max(f["cum_del"])) for f in frames)
    ax_cum.set_xlim(0, args.horizon)
    ax_cum.set_ylim(0, max(cum_max * 1.1, max(budgets) * 1.3))
    ax_cum.set_xlabel("step"); ax_cum.set_ylabel("$C_t^i$ [kNm·step]")
    ax_cum.set_title("cumulative DEL vs $B_i$ (dotted = uniform pace)")
    ax_cum.legend(fontsize=8, loc="upper left"); ax_cum.grid(alpha=0.25)

    sig_lines = []; lam_lines = []; pwr_lines = []
    for i in range(n_turb):
        ls, = ax_sig.plot([], [], lw=1.6, color=colors_t[i], label=f"T{i}")
        sig_lines.append(ls)
        ll, = ax_lam.plot([], [], lw=1.6, color=colors_t[i])
        lam_lines.append(ll)
        lp, = ax_pwr.plot([], [], lw=1.4, color=colors_t[i])
        pwr_lines.append(lp)
    ax_sig.set_xlim(0, args.horizon); ax_sig.set_ylim(-0.02, 1.05)
    ax_sig.set_xlabel("step"); ax_sig.set_ylabel("σ$_i$"); ax_sig.set_title("blend weight (visual)"); ax_sig.legend(fontsize=7); ax_sig.grid(alpha=0.25)
    ax_lam.set_xlim(0, args.horizon); ax_lam.set_yscale("log")
    ax_lam.set_xlabel("step"); ax_lam.set_ylabel("λ$_i$"); ax_lam.set_title("Lagrangian dual (active multiplier)"); ax_lam.grid(alpha=0.25, which="both")
    p_max = max(float(np.max(f["p_step"].sum())) for f in frames) * 1.1
    ax_pwr.set_xlim(0, args.horizon); ax_pwr.set_ylim(0, p_max)
    ax_pwr.set_xlabel("step"); ax_pwr.set_ylabel("P [kW]"); ax_pwr.set_title("instantaneous power per turbine"); ax_pwr.grid(alpha=0.25)

    ts_hist = []
    sig_hist = [[] for _ in range(n_turb)]
    lam_hist = [[] for _ in range(n_turb)]
    cum_hist = [[] for _ in range(n_turb)]
    pwr_hist = [[] for _ in range(n_turb)]

    def update(idx):
        f = frames[idx]
        try:
            flow_mesh.set_array(np.asarray(f["U"]).ravel())
        except Exception:
            pass
        for i in range(n_turb):
            yaw_rad = np.deg2rad(f["yaw_deg"][i])
            ex = -np.sin(yaw_rad); ey = np.cos(yaw_rad)
            half = rd / 2
            x0, y0 = pos_x[i] - ex * half, pos_y[i] - ey * half
            x1, y1 = pos_x[i] + ex * half, pos_y[i] + ey * half
            rotor_lines[i].set_data([x0, x1], [y0, y1])
            # Color rotor by sigma (visual cue) — high sigma = pulled toward safe
            rotor_lines[i].set_color(rotor_cmap(float(f["sigma"][i])))
        title_text.set_text(
            f"t={f['t']:3d}/{args.horizon}  τ={f['tau']:.2f}  "
            f"yaw={[f'{y:+.0f}°' for y in f['yaw_deg']]}  "
            f"P_total={f['p_step'].sum():.0f}kW")

        util = f["cum_del"] / np.maximum(budgets, 1)
        n_viol = int(np.sum(util > 1.0))
        if n_viol > 0:
            wi = int(np.argmax(util))
            viol_text.set_text(
                f"⚠ T{wi} budget exceeded ({100*util[wi]:.0f}% of $B_i$)")
            viol_text.get_bbox_patch().set_alpha(0.85)
        else:
            viol_text.set_text(""); viol_text.get_bbox_patch().set_alpha(0.0)

        ts_hist.append(f["t"])
        for i in range(n_turb):
            sig_hist[i].append(f["sigma"][i])
            lam_hist[i].append(max(f["lambda"][i], 1e-3))
            cum_hist[i].append(f["cum_del"][i])
            pwr_hist[i].append(f["p_step"][i])
            sig_lines[i].set_data(ts_hist, sig_hist[i])
            lam_lines[i].set_data(ts_hist, lam_hist[i])
            cum_lines[i].set_data(ts_hist, cum_hist[i])
            pwr_lines[i].set_data(ts_hist, pwr_hist[i])

        return (flow_mesh, *rotor_lines, *cum_lines, *sig_lines, *lam_lines,
                *pwr_lines, title_text, viol_text)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // args.fps, blit=False)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=args.fps, bitrate=2400, codec="libx264",
                           extra_args=["-pix_fmt", "yuv420p"])
    anim.save(str(out), writer=writer, dpi=120)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
