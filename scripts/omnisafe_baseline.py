"""OmniSafe baseline: train PPOLag on SafetyPointGoal1-v0 for apples-to-apples.

Uses the canonical CMDP library. Saves eval stats for comparison plot.

Usage:
  python scripts/omnisafe_baseline.py --algo PPOLag --budget 25 --steps 2000000
"""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--algo", default="PPOLag",
                   choices=["PPOLag", "CPO", "TRPOLag", "PPOEarlyTerminated"])
    p.add_argument("--env", default="SafetyPointGoal1-v0")
    p.add_argument("--budget", type=int, default=25)
    p.add_argument("--steps", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", default="runs/omnisafe")
    p.add_argument("--out", default="results/omnisafe_eval.json")
    args = p.parse_args()

    import omnisafe
    custom_cfgs = {
        "train_cfgs": {
            "total_steps": args.steps,
            "torch_threads": 4,
            "vector_env_nums": 1,
            "parallel": 1,
        },
        "algo_cfgs": {
            "cost_limit": float(args.budget),
        },
        "logger_cfgs": {
            "log_dir": args.log_dir,
        },
        "seed": args.seed,
    }
    agent = omnisafe.Agent(args.algo, args.env, custom_cfgs=custom_cfgs)
    agent.learn()

    # Evaluate
    print(f"\nEvaluating {args.algo} at d={args.budget}...")
    eval_results = agent.evaluate(num_episodes=20)
    print(f"eval results: {eval_results}")

    # Save
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out_data = {}
    if Path(args.out).exists():
        try: out_data = json.load(open(args.out))
        except Exception: pass
    key = f"{args.algo}_B{args.budget}_s{args.seed}"
    out_data[key] = {
        "algo": args.algo,
        "budget": args.budget,
        "total_steps": args.steps,
        "seed": args.seed,
        "eval": eval_results if isinstance(eval_results, dict) else str(eval_results),
    }
    with open(args.out, "w") as f:
        json.dump(out_data, f, indent=2, default=str)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
