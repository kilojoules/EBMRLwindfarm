"""
Measure coupling strength κ = E[||∇_a Q_c||] / E[||∇_s Q_c||]

Strong coupling: action strongly affects cost (wind farm yaw → fatigue).
Weak coupling: cost mostly state-driven (Safety Gym position → hazard).

Domain selected by --domain {safety_gym, wind_farm}.
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path


def measure_coupling(qc, obs_batch, act_batch, device):
    """Compute ||grad_a Q|| and ||grad_s Q|| per sample, return arrays."""
    obs = obs_batch.clone().detach().to(device).requires_grad_(True)
    act = act_batch.clone().detach().to(device).requires_grad_(True)
    q = qc(obs, act)
    if isinstance(q, tuple):
        q = q[0]  # twin critic: take q1 (or max(q1,q2))
    q = q.squeeze(-1) if q.dim() > 1 else q
    g_a = torch.autograd.grad(q.sum(), act, retain_graph=True)[0]
    g_s = torch.autograd.grad(q.sum(), obs)[0]
    na = g_a.norm(dim=-1).detach().cpu().numpy()
    ns = g_s.norm(dim=-1).detach().cpu().numpy()
    return na, ns


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


def run_safety_gym(n_samples=2000, seed=1):
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import safety_gymnasium
    from uncertainty_gated_qc import SafetyGymActor

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    CKPT = Path("checkpoints")

    ckpt = torch.load(CKPT / f"cost_critic_seed{seed}.pt",
                      map_location=device, weights_only=False)
    qc = CostCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"]).to(device)
    qc.load_state_dict(ckpt["model"])
    qc.eval()

    actor_ckpt = torch.load(CKPT / f"sac_safety_point_seed{seed}.pt",
                            map_location=device, weights_only=False)
    actor = SafetyGymActor(actor_ckpt["obs_dim"], actor_ckpt["act_dim"]).to(device)
    actor.load_state_dict(actor_ckpt["actor"])
    actor.eval()

    env = safety_gymnasium.make(actor_ckpt["env_name"])
    obs_list, act_list = [], []
    obs, _ = env.reset(seed=1)
    while len(obs_list) < n_samples:
        s = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            a, _ = actor.sample(s)
        obs_list.append(s.squeeze(0))
        act_list.append(a.squeeze(0))
        step = env.step(a.squeeze(0).cpu().numpy() * actor_ckpt["act_limit"])
        if len(step) == 6:
            obs = step[0]
            if step[3] or step[4]:
                obs, _ = env.reset()
        else:
            obs = step[0]
            if step[2] or step[3]:
                obs, _ = env.reset()
    env.close()

    obs_batch = torch.stack(obs_list)
    act_batch = torch.stack(act_list)
    na, ns = measure_coupling(qc, obs_batch, act_batch, device)
    return na, ns


def run_wind_farm(n_samples=2000, seed=1):
    import sys
    sys.path.insert(0, ".")
    # Uses windfarm_cost_critic checkpoints — need to find format
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    qc_path = Path("checkpoints/windfarm_qc.pt")
    if not qc_path.exists():
        raise FileNotFoundError(f"{qc_path} missing — sync from LUMI first")
    ckpt = torch.load(qc_path, map_location=device, weights_only=False)
    qc = CostCritic(ckpt["obs_dim"], ckpt["act_dim"], ckpt["hidden"]).to(device)
    qc.load_state_dict(ckpt["model"])
    qc.eval()

    # Load transitions from cost data npz if available
    data_path = Path("data/windfarm_cost_data.npz")
    if data_path.exists():
        data = np.load(data_path)
        obs_batch = torch.tensor(data["obs"][:n_samples], dtype=torch.float32, device=device)
        act_batch = torch.tensor(data["act"][:n_samples], dtype=torch.float32, device=device)
    else:
        raise FileNotFoundError(f"{data_path} missing")

    na, ns = measure_coupling(qc, obs_batch, act_batch, device)
    return na, ns


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=["safety_gym", "wind_farm"], required=True)
    parser.add_argument("--n-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--output", default="results/coupling_metric.json")
    args = parser.parse_args()

    if args.domain == "safety_gym":
        na, ns = run_safety_gym(args.n_samples, args.seed)
    else:
        na, ns = run_wind_farm(args.n_samples, args.seed)

    ma, sa = float(np.mean(na)), float(np.std(na))
    msv, ss = float(np.mean(ns)), float(np.std(ns))
    kappa = ma / msv if msv > 0 else float("inf")

    print(f"\n[{args.domain}] n={len(na)}")
    print(f"  ||grad_a Q_c|| mean={ma:.4f} std={sa:.4f}")
    print(f"  ||grad_s Q_c|| mean={msv:.4f} std={ss:.4f}")
    print(f"  kappa = ||grad_a|| / ||grad_s|| = {kappa:.4f}")

    Path(args.output).parent.mkdir(exist_ok=True)
    out = {}
    if Path(args.output).exists():
        with open(args.output) as f:
            out = json.load(f)
    out[args.domain] = {
        "grad_a_mean": ma, "grad_a_std": sa,
        "grad_s_mean": msv, "grad_s_std": ss,
        "kappa": kappa, "n_samples": len(na),
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
