"""
Discriminator experiment: does ensemble std(Q_c) predict failure better than ||Δa||?

Loads 5 Q_c seeds + 1 SAC actor (Safety Gym). Rolls out policy with grad correction.
Per step, logs (ensemble_std, ‖Δa‖, cost_over_next_H). Computes Spearman rank
correlation of each predictor to realized cost.

If ensemble_std wins → pivot to uncertainty-gated Q_c.
If ‖Δa‖ wins → pursue L2 proximal cap.

Usage: python scripts/discriminator_qc.py
"""
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import safety_gymnasium

CKPT_DIR = Path(__file__).parent.parent / "checkpoints"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPISODES = 20
HORIZON = 1000
BUDGET = 10  # failure regime
LOOKAHEAD_H = 20  # cost over next H steps as outcome
CORRECTION_STEPS = 3
CORRECTION_LR = 0.05
LAM_DEPLOY = 5.0  # fixed correction strength for this diagnostic


def make_cost_critic(obs_dim, act_dim, hidden):
    return nn.Sequential(
        nn.Linear(obs_dim + act_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, 1),
    )


def load_q_ensemble():
    members = []
    for seed in [1, 2, 3, 4, 5]:
        ckpt = torch.load(CKPT_DIR / f"cost_critic_seed{seed}.pt",
                          map_location=DEVICE, weights_only=False)
        q = make_cost_critic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"]).to(DEVICE)
        # Use q1 branch only (simpler; twin structure not critical for discriminator)
        sd = {k[3:]: v for k, v in ckpt["model"].items() if k.startswith("q1.")}
        q.load_state_dict(sd)
        q.eval()
        members.append(q)
    return members


def q_ensemble_stats(qs, s_t, a_t):
    """Return mean, std of ensemble predictions at (s,a)."""
    x = torch.cat([s_t, a_t], dim=-1)
    preds = torch.stack([q(x).squeeze(-1) for q in qs])
    return preds.mean(dim=0), preds.std(dim=0)


def load_sac_actor(seed=1):
    from torch.distributions import Normal
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
                a = torch.tanh(mu) * self.act_limit
            else:
                std = log_std.exp()
                z = Normal(mu, std).rsample()
                a = torch.tanh(z) * self.act_limit
            return a

    ckpt = torch.load(CKPT_DIR / f"sac_safety_point_seed{seed}.pt",
                      map_location=DEVICE, weights_only=False)
    actor = SACActor(ckpt["obs_dim"], ckpt["act_dim"], ckpt["act_limit"]).to(DEVICE)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    return actor, ckpt["act_limit"], ckpt["env_name"]


def grad_correct(qs, s_t, a_base, act_limit, steps=CORRECTION_STEPS, lr=CORRECTION_LR,
                 lam=LAM_DEPLOY):
    """Standard Q_c gradient correction using ensemble mean."""
    a = a_base.clone().detach()
    for _ in range(steps):
        a.requires_grad_(True)
        x = torch.cat([s_t, a], dim=-1)
        q_mean = torch.stack([q(x).squeeze(-1) for q in qs]).mean(dim=0)
        g = torch.autograd.grad(q_mean.sum(), a)[0]
        a = (a.detach() - lam * lr * g).clamp(-act_limit, act_limit)
    return a.detach()


def main():
    qs = load_q_ensemble()
    actor, act_limit, env_name = load_sac_actor(seed=1)
    print(f"Loaded 5 Q_c ensemble + SAC actor. env={env_name}, act_limit={act_limit}")

    env = safety_gymnasium.make(env_name)
    records = []

    for ep in range(N_EPISODES):
        obs, _ = env.reset(seed=1000 + ep)
        ep_costs = []
        ep_features = []
        for t in range(HORIZON):
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a_base = actor(s_t, deterministic=True)
            a_corr = grad_correct(qs, s_t, a_base, act_limit)

            with torch.no_grad():
                _, ens_std = q_ensemble_stats(qs, s_t, a_base)
            delta_norm = (a_corr - a_base).norm().item()

            a_np = a_corr.squeeze(0).cpu().numpy()
            step_ret = env.step(a_np)
            if len(step_ret) == 6:
                obs, rew, cost, term, trunc, info = step_ret
            else:
                obs, rew, term, trunc, info = step_ret
                cost = info.get("cost", 0.0)

            ep_costs.append(float(cost))
            ep_features.append({
                "t": t, "ep": ep,
                "ens_std": float(ens_std.item()),
                "delta_norm": float(delta_norm),
            })
            if term or trunc:
                break

        # attach lookahead-H cost to each feature
        for i, feat in enumerate(ep_features):
            j = min(i + LOOKAHEAD_H, len(ep_costs))
            feat["cost_ahead"] = float(sum(ep_costs[i:j]))
            records.append(feat)
        print(f"ep {ep:2d}: len={len(ep_costs)}, total_cost={sum(ep_costs):.0f}")

    arr = np.array([(r["ens_std"], r["delta_norm"], r["cost_ahead"]) for r in records])
    print(f"\n{len(records)} state-level records")
    print(f"ens_std   range: [{arr[:,0].min():.3f}, {arr[:,0].max():.3f}] mean={arr[:,0].mean():.3f}")
    print(f"delta_norm range: [{arr[:,1].min():.3f}, {arr[:,1].max():.3f}] mean={arr[:,1].mean():.3f}")
    print(f"cost_ahead range: [{arr[:,2].min():.0f}, {arr[:,2].max():.0f}] mean={arr[:,2].mean():.3f}")

    rho_ens, p_ens = spearmanr(arr[:, 0], arr[:, 2])
    rho_del, p_del = spearmanr(arr[:, 1], arr[:, 2])
    print(f"\nSpearman rank corr vs cost_ahead:")
    print(f"  ensemble std:  rho={rho_ens:+.3f}  p={p_ens:.2e}")
    print(f"  ‖Δa‖:          rho={rho_del:+.3f}  p={p_del:.2e}")
    print(f"\nWinner: {'ensemble std' if abs(rho_ens) > abs(rho_del) else 'L2 ‖Δa‖'}")

    np.savez("results/discriminator_qc.npz",
             ens_std=arr[:, 0], delta_norm=arr[:, 1], cost_ahead=arr[:, 2])
    print("Saved results/discriminator_qc.npz")


if __name__ == "__main__":
    main()
