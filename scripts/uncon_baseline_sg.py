"""
Per-episode unconstrained baseline on Safety Gym, 5 seeds x 20 eps.
Measures budget-satisfaction rate vs MPC — sanity check that MPC adds
something beyond variance.
"""
import json, argparse, numpy as np, torch
from pathlib import Path
import safety_gymnasium
import sys
sys.path.insert(0, str(Path(__file__).parent))
from uncertainty_gated_qc import SafetyGymActor

CKPT = Path("checkpoints")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-eps", type=int, default=20)
    p.add_argument("--horizon", type=int, default=1000)
    p.add_argument("--out", default="results/uncon_baseline_sg.json")
    args = p.parse_args()

    ac = torch.load(CKPT / f"sac_safety_point_seed{args.seed}.pt",
                    map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(ac["obs_dim"], ac["act_dim"]).to(DEVICE)
    actor.load_state_dict(ac["actor"]); actor.eval()

    env = safety_gymnasium.make(ac["env_name"])
    per_ep = []
    for ep in range(args.n_eps):
        obs, _ = env.reset(seed=ep * 101 + 7)  # SAME seeding as MPC script
        C = 0.0; R = 0.0
        for t in range(args.horizon):
            s = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                a, _ = actor.sample(s)
            a_np = a.squeeze(0).cpu().numpy() * ac["act_limit"]
            step = env.step(a_np)
            if len(step) == 6:
                obs, r, c, term, trunc, info = step
            else:
                obs, r, term, trunc, info = step
                c = info.get("cost", 0.0)
            R += float(r); C += float(c)
            if term or trunc: break
        per_ep.append(dict(reward=R, cost=C))
        print(f"  ep {ep}: R={R:.1f} C={C:.1f}")
    env.close()
    costs = [e["cost"] for e in per_ep]
    summary = {
        "seed": args.seed, "n_eps": args.n_eps,
        "cost_mean": float(np.mean(costs)),
        "cost_std": float(np.std(costs)),
        "sat_d10": float(np.mean([c <= 10 for c in costs])),
        "sat_d25": float(np.mean([c <= 25 for c in costs])),
        "sat_d40": float(np.mean([c <= 40 for c in costs])),
        "per_ep": per_ep,
    }
    Path(args.out).parent.mkdir(exist_ok=True)
    data = {}
    if Path(args.out).exists():
        try: data = json.load(open(args.out))
        except Exception: pass
    data[f"seed{args.seed}"] = summary
    with open(args.out, "w") as f:
        json.dump(data, f, indent=2)
    print(f"SEED {args.seed}: cost={summary['cost_mean']:.1f} "
          f"sat_d10={summary['sat_d10']*100:.0f}% "
          f"sat_d25={summary['sat_d25']*100:.0f}% "
          f"sat_d40={summary['sat_d40']*100:.0f}%")


if __name__ == "__main__":
    main()
