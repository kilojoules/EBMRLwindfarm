"""Native WindGym render: env.render_mode='rgb_array'.

Uses WindGym's built-in renderer (flow field + turbines) per step.
"""
import argparse
import os
import sys
import glob
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pathlib import Path

sys.path.insert(0, ".")


def run_episode(checkpoint, budget, horizon, eta, gs, seed=1):
    from config import Args
    from helpers.agent import WindFarmAgent
    from ebt import TransformerEBTActor
    from networks import create_profile_encoding
    from load_surrogates import NegativeYawBudgetSurrogate
    from ebt_sac_windfarm import setup_env

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    args = Args(**{k: v for k, v in ckpt["args"].items() if hasattr(Args, k)})
    # Inject render_mode into WindFarmEnv.__init__ + force SyncVectorEnv
    import ebt_sac_windfarm as esw
    import gymnasium as gym
    _orig_cls = esw.WindFarmEnv
    class RenderingWindFarmEnv(_orig_cls):
        def __init__(self, *a, **kw):
            kw.setdefault("render_mode", "rgb_array")
            super().__init__(*a, **kw)
    esw.WindFarmEnv = RenderingWindFarmEnv
    # Force sync vector so render_mode propagates in-process
    gym.vector.AsyncVectorEnv = gym.vector.SyncVectorEnv

    args.num_envs = 1
    env_info = setup_env(args)
    envs = env_info["envs"]
    n_turb = env_info["n_turbines_max"]

    use_profiles = env_info["use_profiles"]
    sr, si = None, None
    if use_profiles:
        sr, si = create_profile_encoding(
            profile_type=args.profile_encoding_type,
            embed_dim=args.embed_dim, hidden_channels=args.profile_encoder_hidden)

    actor = TransformerEBTActor(
        obs_dim_per_turbine=env_info["obs_dim_per_turbine"],
        action_dim_per_turbine=1,
        embed_dim=args.embed_dim, num_heads=args.num_heads,
        num_layers=args.num_layers, mlp_ratio=args.mlp_ratio,
        dropout=args.dropout, pos_encoding_type=args.pos_encoding_type,
        pos_embed_dim=args.pos_embed_dim, pos_embedding_mode=args.pos_embedding_mode,
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

    agent = WindFarmAgent(
        actor=actor, device=device,
        rotor_diameter=env_info["rotor_diameter"],
        use_wind_relative=args.use_wind_relative_pos,
        use_profiles=use_profiles,
        rotate_profiles=getattr(args, "rotate_profiles", False))

    surr = NegativeYawBudgetSurrogate(
        budget_steps=budget, horizon_steps=horizon,
        risk_aversion=eta, steepness=2.0, yaw_max_deg=30.0)
    surr.reset()
    obs, _ = envs.reset(seed=seed)

    frames, powers, neg_cum, lams = [], [], [], []
    nc = np.zeros(n_turb)
    for t in range(horizon):
        with torch.no_grad():
            act = agent.act(envs, obs, guidance_fn=surr, guidance_scale=gs)
        obs, rew, _, _, info = envs.step(act)
        if "yaw angles agent" in info:
            ya = np.array(info["yaw angles agent"])
            yf = ya[0] if ya.ndim > 1 else ya
            for ti in range(min(len(yf), n_turb)):
                if yf[ti] < 0: nc[ti] += 1
            surr.update(torch.tensor(yf[:n_turb], device=device, dtype=torch.float32))
        p = 0.0
        if "Power agent" in info:
            pa = np.array(info["Power agent"])
            p = float(np.sum(pa.flatten()))
        # Vector env render — call on underlying
        frame = None
        try:
            r = envs.call("render")
            if r is not None and len(r) > 0:
                frame = r[0]
        except Exception:
            pass
        if frame is None and hasattr(envs, 'envs'):
            try:
                frame = envs.envs[0].render()
            except Exception:
                pass
        if frame is not None:
            frame = np.asarray(frame)
            if frame.dtype == object:
                # Might be list of stacks — take first
                frame = np.asarray(frame.item() if frame.ndim == 0 else frame[0])
        if t == 0:
            print(f"first frame: type={type(frame)} shape={getattr(frame, 'shape', None)} dtype={getattr(frame, 'dtype', None)}")
        frames.append(frame)
        powers.append(p)
        neg_cum.append(nc.copy())
        lams.append(float(surr.compute_lambda()) if hasattr(surr, 'compute_lambda') else 1.0)
    envs.close()
    return {"frames": frames, "powers": np.array(powers),
            "neg_cum": np.array(neg_cum), "lams": np.array(lams), "budget": budget}


def make_animation(data, out_path):
    fr = data["frames"]; pw = data["powers"]; nc = data["neg_cum"]
    lam = data["lams"]; B = data["budget"]
    T = len(fr)
    fig = plt.figure(figsize=(11, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.6, 1], hspace=0.3, wspace=0.25)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_n = fig.add_subplot(gs[0, 1])
    ax_p = fig.add_subplot(gs[1, 1])

    ax_img.axis("off")
    img = ax_img.imshow(fr[0])
    title = ax_img.set_title("t=0")

    ts = np.arange(T)
    colors = ["C0", "C1", "C2"]
    for i, c in enumerate(colors):
        ax_n.plot(ts, nc[:, i], c + "-", lw=1.5, alpha=0.4, label=f"T{i}")
    pts_n = [ax_n.plot([0], [nc[0, i]], c + "o", ms=6)[0] for i, c in enumerate(colors)]
    ax_n.axhline(B, color="k", ls="--", lw=1, label=f"d={B}")
    ax_n.set_ylabel("cum. neg-yaw"); ax_n.set_xlim(0, T)
    ax_n.legend(fontsize=8, loc="upper left"); ax_n.grid(alpha=0.3)

    cum_p = np.cumsum(pw) / 1e6
    ax_p.plot(ts, cum_p, "C4-", lw=1.5, alpha=0.4)
    pt_p, = ax_p.plot([0], [cum_p[0]], "C4o", ms=6)
    ax_p.set_xlabel("step"); ax_p.set_ylabel("cum. power [MW·step]")
    ax_p.set_xlim(0, T); ax_p.grid(alpha=0.3)

    def update(t):
        img.set_data(fr[t])
        title.set_text(f"t={t}  λ={lam[t]:.2f}  neg=[{int(nc[t,0])},{int(nc[t,1])},{int(nc[t,2])}]/{B}")
        for i, p in enumerate(pts_n):
            p.set_data([t], [nc[t, i]])
        pt_p.set_data([t], [cum_p[t]])
        return img, *pts_n, pt_p, title

    ani = animation.FuncAnimation(fig, update, frames=T, interval=100, blit=False)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        ani.save(out_path, writer="ffmpeg", fps=10, dpi=100, bitrate=2000)
    except Exception as e:
        print(f"ffmpeg failed ({e}); gif fallback")
        gif_path = str(out_path).replace(".mp4", ".gif")
        ani.save(gif_path, writer="pillow", fps=10, dpi=80)
        out_path = gif_path
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="runs/ebt_sac_windfarm/checkpoints/step_100000.pt")
    p.add_argument("--budget", type=int, default=15)
    p.add_argument("--horizon", type=int, default=200)
    p.add_argument("--eta", type=float, default=5.0)
    p.add_argument("--gs", type=float, default=0.1)
    p.add_argument("--out", default="results/windfarm_ep_native.mp4")
    args = p.parse_args()
    if not os.path.exists(args.checkpoint):
        c = glob.glob("runs/*/checkpoints/step_*.pt")
        if c: args.checkpoint = sorted(c)[-1]
    data = run_episode(args.checkpoint, args.budget, args.horizon,
                       args.eta, args.gs)
    Path(args.out).parent.mkdir(exist_ok=True)
    make_animation(data, args.out)


if __name__ == "__main__":
    main()
