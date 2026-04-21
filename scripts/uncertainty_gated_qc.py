"""
Uncertainty-gated Q_c via ensemble pessimism.

Q_c_pess(s,a) = mean_i Q_c^i(s,a) + kappa * std_i Q_c^i(s,a)

Deploy:
  a ← a - lam_gs * lam(t) * alpha * grad_a Q_c_pess

Motivation: discriminator (job 17672836) showed ensemble std predicts
Q_c failure 3x better than ||Delta a||. Penalizing uncertainty steers
away from OOD regions.

Sweep kappa in {0, 0.5, 1, 2, 4} x budget in {10, 25, 40}.
Success: single kappa* transfers across budgets.
"""
import json
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torch.distributions import Normal
import safety_gymnasium

CKPT_DIR = Path("checkpoints")
RESULTS = Path("results") / "uncertainty_gated_qc.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPISODES = 20
HORIZON = 1000
CORRECTION_STEPS = 3
CORRECTION_LR = 0.05
MAX_STEP = 0.15  # per-iter L2 clamp (matches cost_critic.py)
ETA = 5.0  # risk aversion
LAM_GS = 0.1


def make_q(obs_dim, act_dim, hidden):
    return nn.Sequential(
        nn.Linear(obs_dim + act_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def load_ensemble():
    members = []
    for s in [1, 2, 3, 4, 5]:
        c = torch.load(CKPT_DIR / f"cost_critic_seed{s}.pt",
                       map_location=DEVICE, weights_only=False)
        q = make_q(c["obs_dim"], c["act_dim"], c["hidden"]).to(DEVICE)
        sd = {k[3:]: v for k, v in c["model"].items() if k.startswith("q1.")}
        q.load_state_dict(sd)
        q.eval()
        members.append(q)
    return members


class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim, act_limit, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.act_limit = act_limit

    def forward(self, s, deterministic=True):
        h = self.net(s)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)
        if deterministic:
            return torch.tanh(mu) * self.act_limit
        std = log_std.exp()
        z = Normal(mu, std).rsample()
        return torch.tanh(z) * self.act_limit


def load_actor(seed=1):
    c = torch.load(CKPT_DIR / f"sac_safety_point_seed{seed}.pt",
                   map_location=DEVICE, weights_only=False)
    a = SACActor(c["obs_dim"], c["act_dim"], c["act_limit"]).to(DEVICE)
    a.load_state_dict(c["actor"])
    a.eval()
    return a, c["act_limit"], c["env_name"]


def urgency(t, T, C, B):
    rho = max(1e-8, (B - C) / B)
    tau = max(1e-8, (T - t) / T)
    u = rho / tau
    return min(np.exp(ETA * (1.0 / u - 1.0)), 1e6)


def pess_q(qs, s, a, kappa):
    """Pessimistic Q: mean + kappa * std."""
    x = torch.cat([s, a], dim=-1)
    preds = torch.stack([q(x).squeeze(-1) for q in qs])
    return preds.mean(dim=0) + kappa * preds.std(dim=0)


def grad_correct_pess(qs, s, a_base, act_limit, kappa, lam):
    a = a_base.clone().detach()
    for _ in range(CORRECTION_STEPS):
        a.requires_grad_(True)
        q_val = pess_q(qs, s, a, kappa)
        g = torch.autograd.grad(q_val.sum(), a)[0]
        step = lam * CORRECTION_LR * g
        sn = step.norm()
        if sn > MAX_STEP:
            step = step * (MAX_STEP / sn)
        a = (a.detach() - step).clamp(-act_limit, act_limit)
    return a.detach()


def run_config(qs, actor, act_limit, env_name, kappa, budget):
    env = safety_gymnasium.make(env_name)
    rewards, costs, reached = [], [], []
    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=2000 + ep)
        ep_r, C = 0.0, 0.0
        for t in range(HORIZON):
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a_base = actor(s, deterministic=True)
            lam = urgency(t, HORIZON, C, budget)
            a_corr = grad_correct_pess(qs, s, a_base, act_limit, kappa, lam)
            a_np = a_corr.squeeze(0).cpu().numpy()
            step_ret = env.step(a_np)
            if len(step_ret) == 6:
                obs, r, c, term, trunc, info = step_ret
            else:
                obs, r, term, trunc, info = step_ret
                c = info.get("cost", 0.0)
            ep_r += r
            C += float(c)
            if term or trunc:
                break
        rewards.append(ep_r)
        costs.append(C)
        reached.append(float(info.get("goal_met", False)) if isinstance(info, dict) else 0.0)
    env.close()
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "cost_sat": bool(np.mean(costs) <= budget),
    }


def main():
    qs = load_ensemble()
    actor, act_limit, env_name = load_actor(seed=1)
    print(f"Ensemble Q_c (5 seeds), SAC actor, env={env_name}")

    kappas = [0.0, 0.5, 1.0, 2.0, 4.0]
    budgets = [10, 25, 40]
    out = {"meta": {"n_episodes": N_EPISODES, "eta": ETA, "lam_gs": LAM_GS}}

    for B in budgets:
        for k in kappas:
            key = f"B{B}_k{k}"
            print(f"  {key} ...", flush=True)
            out[key] = run_config(qs, actor, act_limit, env_name, k, B)
            o = out[key]
            ok = "OK" if o["cost_sat"] else "--"
            print(f"    reward={o['reward_mean']:.1f}+-{o['reward_std']:.1f}"
                  f" cost={o['cost_mean']:.1f}+-{o['cost_std']:.1f} {ok}")

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS}")

    # Summary
    print("\n=== Summary ===")
    print(f"{'Budget':>8s} " + " ".join(f"kappa={k:<4.1f}" for k in kappas))
    for B in budgets:
        row = [f"B={B:>3d}  "]
        for k in kappas:
            o = out[f"B{B}_k{k}"]
            ok = "v" if o["cost_sat"] else "x"
            row.append(f"r{o['reward_mean']:.0f}/c{o['cost_mean']:.0f}{ok}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
