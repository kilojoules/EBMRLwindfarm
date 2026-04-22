"""Native Safety Gym render via MuJoCo rgb_array. Uses env.render().

Requires working EGL (LUMI: LD_LIBRARY_PATH=/usr/lib64, MUJOCO_GL=egl).
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import safety_gymnasium
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import (
    SafetyGymActor, CostCritic, urgency_lambda,
    CORRECTION_STEPS, CORRECTION_LR, MAX_STEP, pess_q,
)

CKPT = Path("checkpoints")


def run(budget, seed, horizon=500, kappa=0.0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    actor_ck = torch.load(CKPT / f"sac_safety_point_seed{seed}.pt",
                           map_location=device, weights_only=False)
    actor = SafetyGymActor(actor_ck["obs_dim"], actor_ck["act_dim"]).to(device)
    actor.load_state_dict(actor_ck["actor"])
    actor.eval()

    ckpt = torch.load(CKPT / f"cost_critic_seed{seed}.pt",
                      map_location=device, weights_only=False)
    qc = CostCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"]).to(device)
    qc.load_state_dict(ckpt["model"])
    qc.eval()

    env = safety_gymnasium.make(actor_ck["env_name"], render_mode="rgb_array",
                                 width=400, height=400, camera_name="fixedfar")
    obs, _ = env.reset(seed=seed * 13 + 7)
    frames, costs, rewards, lambdas = [], [], [], []
    C, R = 0.0, 0.0
    for t in range(horizon):
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a_raw, _ = actor.sample(s)
        lam = urgency_lambda(t, horizon, C, budget)
        a = a_raw.clone().detach()
        for _ in range(CORRECTION_STEPS):
            a.requires_grad_(True)
            q = pess_q([qc], s, a, kappa)
            g = torch.autograd.grad(q.sum(), a)[0]
            step = lam * CORRECTION_LR * g
            sn = step.norm()
            if sn > MAX_STEP:
                step = step * (MAX_STEP / sn)
            a = (a.detach() - step).clamp(-1.0, 1.0)
        a_np = a.squeeze(0).cpu().numpy() * actor_ck["act_limit"]
        step_ret = env.step(a_np)
        if len(step_ret) == 6:
            obs, r, c, term, trunc, info = step_ret
        else:
            obs, r, term, trunc, info = step_ret
            c = info.get("cost", 0.0)
        R += float(r); C += float(c)
        frames.append(env.render())
        rewards.append(R); costs.append(C); lambdas.append(lam)
        if term or trunc:
            break
    env.close()
    return frames, np.array(costs), np.array(rewards), np.array(lambdas)


def make_animation(frames, costs, rewards, lambdas, budget, out_path):
    fig = plt.figure(figsize=(10, 5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.3, 1], hspace=0.35, wspace=0.25)
    ax_img = fig.add_subplot(gs[:, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_r = fig.add_subplot(gs[1, 1])
    ax_img.axis("off")
    img = ax_img.imshow(frames[0])
    title = ax_img.set_title("t=0")

    T = len(frames); ts = np.arange(T)
    ax_c.plot(ts, costs, "C3-", lw=1.5, alpha=0.4)
    ax_c.axhline(budget, color="k", ls="--", lw=1, label=f"budget d={budget}")
    pt_c, = ax_c.plot([0], [costs[0]], "C3o", ms=7)
    ax_c.set_ylabel("cumulative cost"); ax_c.set_xlim(0, T)
    ax_c.legend(fontsize=8, loc="upper left"); ax_c.grid(alpha=0.3)

    ax_r.plot(ts, rewards, "C0-", lw=1.5, alpha=0.4)
    pt_r, = ax_r.plot([0], [rewards[0]], "C0o", ms=7)
    ax_r.set_xlabel("step"); ax_r.set_ylabel("cumulative reward")
    ax_r.set_xlim(0, T); ax_r.grid(alpha=0.3)

    def update(t):
        img.set_data(frames[t])
        title.set_text(f"t={t}  λ={lambdas[t]:.1f}  C={costs[t]:.1f}/{budget}  R={rewards[t]:.1f}")
        pt_c.set_data([t], [costs[t]])
        pt_r.set_data([t], [rewards[t]])
        return img, pt_c, pt_r, title

    ani = animation.FuncAnimation(fig, update, frames=T, interval=33, blit=False)
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
        ani.save(out_path, writer="ffmpeg", fps=30, dpi=100, bitrate=2000)
    except Exception as e:
        print(f"ffmpeg failed ({e}); gif fallback")
        gif_path = str(out_path).replace(".mp4", ".gif")
        ani.save(gif_path, writer="pillow", fps=15, dpi=80)
        out_path = gif_path
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--budget", type=int, default=40)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--horizon", type=int, default=500)
    p.add_argument("--kappa", type=float, default=0.0)
    p.add_argument("--out", default="results/safety_gym_ep_native.mp4")
    args = p.parse_args()

    frames, costs, rewards, lambdas = run(args.budget, args.seed,
                                           args.horizon, args.kappa)
    print(f"episode len={len(frames)}, final C={costs[-1]:.1f}, R={rewards[-1]:.1f}")
    Path(args.out).parent.mkdir(exist_ok=True)
    make_animation(frames, costs, rewards, lambdas, args.budget, args.out)


if __name__ == "__main__":
    main()
