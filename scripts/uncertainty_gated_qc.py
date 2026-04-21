"""
Uncertainty-gated Q_c on Safety Gym. Matches baseline exactly to original
cost_critic.py wiring (stochastic actor.sample, act_limit scaling, 5-step
correction, lambda clamp 1e4, hard guard at budget exhaustion).

Pessimism via ensemble of Q_c networks:
  Q_pess(s,a) = mean_i Q_i(s,a) + kappa * std_i Q_i(s,a)
where i runs over 10 critics (5 seeds x {q1, q2} twin heads).

Sweep kappa in {0, 0.5, 1, 2, 4} x budget in {10, 25, 40} with 50 eps.
"""
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
import safety_gymnasium

CKPT_DIR = Path("checkpoints")
RESULTS = Path("results") / "uncertainty_gated_qc.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

N_EPISODES = 50
HORIZON = 1000
CORRECTION_STEPS = 5
CORRECTION_LR = 0.05
MAX_STEP = 0.15
ETA = 5.0
LAM_CLAMP = 1e4
LAM_HARD_GUARD = 100.0


class CostCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, s, a):
        x = torch.cat([s, a], dim=-1)
        return self.q1(x), self.q2(x)


class SafetyGymActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, s):
        h = self.net(s)
        return self.mu(h), self.log_std(h).clamp(-20, 2)

    def sample(self, s):
        mu, log_std = self(s)
        dist = torch.distributions.Normal(mu, log_std.exp())
        return torch.tanh(dist.rsample()), mu


def load_ensemble():
    """Load 5 Q_c seeds, collect 10 critic heads (5 seeds x {q1, q2})."""
    critics = []
    for s in [1, 2, 3, 4, 5]:
        c = torch.load(CKPT_DIR / f"cost_critic_seed{s}.pt",
                       map_location=DEVICE, weights_only=False)
        qc = CostCritic(c["obs_dim"], c["act_dim"], c["hidden"]).to(DEVICE)
        qc.load_state_dict(c["model"])
        qc.eval()
        critics.append(qc)
    obs_dim = c["obs_dim"]
    act_dim = c["act_dim"]
    return critics, obs_dim, act_dim


def load_actor():
    c = torch.load(CKPT_DIR / "sac_safety_point_seed1.pt",
                   map_location=DEVICE, weights_only=False)
    a = SafetyGymActor(c["obs_dim"], c["act_dim"]).to(DEVICE)
    a.load_state_dict(c["actor"])
    a.eval()
    return a, c["act_limit"], c["env_name"]


def pess_q(critics, s, a, kappa):
    """Pessimistic Q.

    kappa=0: single-critic twin-max (matches original paper wiring exactly).
    kappa>0: mean + kappa*std over 10-head ensemble (5 seeds x twin).
    """
    if kappa == 0.0:
        # Match original: single critic, max(q1, q2) pessimistic twin
        q1, q2 = critics[0](s, a)
        return torch.max(q1, q2).squeeze(-1)
    preds = []
    for qc in critics:
        q1, q2 = qc(s, a)
        preds.append(q1.squeeze(-1))
        preds.append(q2.squeeze(-1))
    stack = torch.stack(preds)
    return stack.mean(dim=0) + kappa * stack.std(dim=0)


def urgency_lambda(t, T, C, B):
    eps = 1e-6
    bf = max(B - C, 0) / max(B, 1)
    tf = max(T - t, 1) / max(T, 1)
    u = bf / max(tf, eps)
    lam = min(np.exp(ETA * (1.0 / max(u, eps) - 1.0)), LAM_CLAMP)
    if C >= B:
        lam = LAM_HARD_GUARD
    return lam


def grad_correct_pess(critics, s_t, a_base, act_limit, kappa, lam):
    a = a_base.clone().detach()
    for _ in range(CORRECTION_STEPS):
        a.requires_grad_(True)
        q_val = pess_q(critics, s_t, a, kappa)
        g = torch.autograd.grad(q_val.sum(), a)[0]
        step = lam * CORRECTION_LR * g
        sn = step.norm()
        if sn > MAX_STEP:
            step = step * (MAX_STEP / sn)
        a = (a.detach() - step).clamp(-1.0, 1.0)
    return a.detach()


def run_config(critics, actor, act_limit, env_name, kappa, budget):
    env = safety_gymnasium.make(env_name)
    rewards, costs = [], []
    for ep in range(N_EPISODES):
        torch.manual_seed(2000 + ep)
        np.random.seed(2000 + ep)
        obs, _ = env.reset(seed=2000 + ep)
        ep_r, C = 0.0, 0.0
        for t in range(HORIZON):
            s_t = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a_raw, _ = actor.sample(s_t)  # in [-1, 1], stochastic
            lam = urgency_lambda(t, HORIZON, C, budget)
            a_corr = grad_correct_pess(critics, s_t, a_raw, act_limit, kappa, lam)
            action_np = a_corr.squeeze(0).cpu().numpy() * act_limit
            step_ret = env.step(action_np)
            if len(step_ret) == 6:
                obs, r, c, term, trunc, info = step_ret
            else:
                obs, r, term, trunc, info = step_ret
                c = info.get("cost", 0.0)
            ep_r += float(r)
            C += float(c)
            if term or trunc:
                break
        rewards.append(ep_r)
        costs.append(C)
    env.close()
    return {
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "reward_se": float(np.std(rewards) / np.sqrt(len(rewards))),
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "cost_se": float(np.std(costs) / np.sqrt(len(costs))),
        "cost_sat": bool(np.mean(costs) <= budget),
    }


def main():
    critics, obs_dim, act_dim = load_ensemble()
    actor, act_limit, env_name = load_actor()
    print(f"Ensemble: 5 seeds x 2 heads = 10 critics")
    print(f"env={env_name}, obs_dim={obs_dim}, act_dim={act_dim}, act_limit={act_limit}")

    kappas = [0.0, 1.0, 4.0]
    budgets = [10, 25, 40]
    out = {"meta": {"n_episodes": N_EPISODES, "eta": ETA,
                    "correction_steps": CORRECTION_STEPS,
                    "n_critics": 10}}

    for B in budgets:
        for k in kappas:
            key = f"B{B}_k{k}"
            print(f"  {key} ...", flush=True)
            out[key] = run_config(critics, actor, act_limit, env_name, k, B)
            o = out[key]
            ok = "OK" if o["cost_sat"] else "--"
            print(f"    reward={o['reward_mean']:.1f}+-{o['reward_se']:.2f}"
                  f" cost={o['cost_mean']:.1f}+-{o['cost_se']:.2f} {ok}", flush=True)

    RESULTS.parent.mkdir(exist_ok=True)
    with open(RESULTS, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved {RESULTS}")

    print("\n=== Cost (mean +- SE) ===")
    print(f"{'Budget':>8s} " + "  ".join(f"k={k:<4.1f}" for k in kappas))
    for B in budgets:
        row = [f"B={B:>2d}    "]
        for k in kappas:
            o = out[f"B{B}_k{k}"]
            ok = "v" if o["cost_sat"] else "x"
            row.append(f"{o['cost_mean']:4.1f}+-{o['cost_se']:.1f}{ok}")
        print(" ".join(row))


if __name__ == "__main__":
    main()
