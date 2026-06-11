#!/usr/bin/env python3
"""Safety Gym test-time constrained policy improvement with Q_r/Q_c critics."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent))
from budgeted_qc_core import (  # noqa: E402
    DEVICE,
    ScalarCritic,
    TransitionArrays,
    TwinCostCritic,
    monte_carlo_returns,
    select_budgeted_action,
    set_seed,
    stack_episode_metrics,
    train_scalar_critic,
    write_json,
)


class SafetyGymActor(nn.Module):
    """Gaussian squashed actor matching existing Safety Gym checkpoints."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

    def forward(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = self.net(observations)
        return self.mu(hidden), self.log_std(hidden).clamp(-20, 2)

    def sample(self, observations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self(observations)
        distribution = torch.distributions.Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        return torch.tanh(raw_action), torch.tanh(mean)


def import_safety_gymnasium():
    try:
        import safety_gymnasium  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "safety_gymnasium is required for this script. "
            "Install it or run wake_proxy_qc_experiment.py for a Safety-Gym-free test."
        ) from exc
    return safety_gymnasium


def load_actor_checkpoint(path: str | Path) -> Tuple[SafetyGymActor, Dict]:
    checkpoint = torch.load(path, map_location=DEVICE, weights_only=False)
    actor = SafetyGymActor(checkpoint["obs_dim"], checkpoint["act_dim"]).to(DEVICE)
    actor.load_state_dict(checkpoint["actor"])
    actor.eval()
    return actor, checkpoint


def step_safety_env(env, action: np.ndarray) -> Tuple[np.ndarray, float, float, bool]:
    result = env.step(action)
    if len(result) == 6:
        next_obs, reward, cost, terminated, truncated, _ = result
    else:
        next_obs, reward, terminated, truncated, info = result
        cost = info.get("cost", 0.0)
    return next_obs, float(reward), float(cost), bool(terminated or truncated)


def collect_dataset(
    checkpoint_path: str | Path,
    episodes: int,
    horizon: int,
    seed: int,
) -> Tuple[TransitionArrays, Dict]:
    safety_gymnasium = import_safety_gymnasium()
    actor, checkpoint = load_actor_checkpoint(checkpoint_path)
    env_name = checkpoint.get("env_name", "SafetyPointGoal1-v0")
    act_limit = float(checkpoint["act_limit"])
    env = safety_gymnasium.make(env_name)

    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    costs: List[float] = []
    next_observations: List[np.ndarray] = []
    done_flags: List[float] = []

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed * 10_000 + episode_idx)
        for _ in range(horizon):
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                action_norm, _ = actor.sample(obs_tensor)
            action_np = action_norm.squeeze(0).cpu().numpy()
            next_obs, reward, cost, done = step_safety_env(env, action_np * act_limit)
            observations.append(obs)
            actions.append(action_np)
            rewards.append(reward)
            costs.append(cost)
            next_observations.append(next_obs)
            done_flags.append(float(done))
            obs = next_obs
            if done:
                break
        print(f"  collected episode {episode_idx + 1}/{episodes}")
    env.close()

    arrays = TransitionArrays(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        costs=np.asarray(costs, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        done_flags=np.asarray(done_flags, dtype=np.float32),
    )
    return arrays, checkpoint


def sample_actor_candidates(
    actor: SafetyGymActor,
    obs: np.ndarray,
    n_candidates: int,
    noise_scale: float,
) -> np.ndarray:
    obs_tensor = torch.tensor(obs, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    obs_batch = obs_tensor.repeat(n_candidates, 1)
    with torch.no_grad():
        sampled_actions, mean_action = actor.sample(obs_batch)
    candidates = sampled_actions.cpu().numpy().astype(np.float32)
    candidates[0] = mean_action[0].cpu().numpy().astype(np.float32)
    if noise_scale > 0:
        noise = np.random.normal(0.0, noise_scale, size=candidates.shape).astype(np.float32)
        candidates = np.clip(candidates + noise, -1.0, 1.0)
    return candidates


def evaluate(
    mode: str,
    checkpoint_path: str | Path,
    reward_critic: ScalarCritic,
    cost_critic: TwinCostCritic,
    budget: float,
    episodes: int,
    horizon: int,
    n_candidates: int,
    eta: float,
    seed: int,
) -> Dict[str, float]:
    safety_gymnasium = import_safety_gymnasium()
    actor, checkpoint = load_actor_checkpoint(checkpoint_path)
    env = safety_gymnasium.make(checkpoint.get("env_name", "SafetyPointGoal1-v0"))
    act_limit = float(checkpoint["act_limit"])
    per_episode: List[Dict[str, float]] = []

    for episode_idx in range(episodes):
        obs, _ = env.reset(seed=seed * 20_000 + episode_idx)
        total_reward = 0.0
        total_cost = 0.0
        feasible_fraction_sum = 0.0
        budgeted_steps = 0
        for step_idx in range(horizon):
            candidates = sample_actor_candidates(actor, obs, n_candidates, noise_scale=0.04)
            if mode == "unconstrained":
                action_norm = candidates[0]
            elif mode == "reward_only_qr":
                obs_batch = np.repeat(obs[None, :], candidates.shape[0], axis=0)
                with torch.no_grad():
                    values = reward_critic(
                        torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE),
                        torch.tensor(candidates, dtype=torch.float32, device=DEVICE),
                    ).squeeze(-1)
                action_norm = candidates[int(torch.argmax(values).item())]
            elif mode == "budgeted_qc":
                action_norm, diagnostics = select_budgeted_action(
                    obs,
                    candidates,
                    reward_critic,
                    cost_critic,
                    cumulative_cost=total_cost,
                    budget=budget,
                    step_idx=step_idx,
                    horizon=horizon,
                    eta=eta,
                )
                feasible_fraction_sum += diagnostics["feasible_fraction"]
                budgeted_steps += 1
            else:
                raise ValueError(f"Unknown mode: {mode}")

            obs, reward, cost, done = step_safety_env(env, action_norm * act_limit)
            total_reward += reward
            total_cost += cost
            if done:
                break
        per_episode.append(
            {
                "reward": total_reward,
                "cost": total_cost,
                "satisfied": float(total_cost <= budget),
                "utilization": float(total_cost / max(budget, 1.0e-6)),
                "feasible_fraction": float(feasible_fraction_sum / max(budgeted_steps, 1)),
            }
        )
        print(f"  {mode} episode {episode_idx + 1}/{episodes}: reward={total_reward:.1f} cost={total_cost:.1f}")
    env.close()
    return stack_episode_metrics(per_episode)


def run(args: argparse.Namespace) -> Dict:
    if args.quick:
        args.collect_episodes = min(args.collect_episodes, 5)
        args.train_steps = min(args.train_steps, 80)
        args.eval_episodes = min(args.eval_episodes, 3)
        args.n_candidates = min(args.n_candidates, 32)
    set_seed(args.seed)
    dataset, checkpoint = collect_dataset(args.checkpoint, args.collect_episodes, args.horizon, args.seed)
    reward_returns = monte_carlo_returns(dataset.rewards, dataset.done_flags, args.gamma_reward)
    cost_returns = monte_carlo_returns(dataset.costs, dataset.done_flags, args.gamma_cost)

    reward_critic = ScalarCritic(dataset.obs_dim, dataset.act_dim, args.hidden_dim)
    cost_critic = TwinCostCritic(dataset.obs_dim, dataset.act_dim, args.hidden_dim)
    reward_stats = train_scalar_critic(
        "SafetyGym Q_r",
        reward_critic,
        dataset.observations,
        dataset.actions,
        reward_returns,
        args.train_steps,
        args.batch_size,
        args.learning_rate,
        twin_cost=False,
    )
    cost_stats = train_scalar_critic(
        "SafetyGym Q_c",
        cost_critic,
        dataset.observations,
        dataset.actions,
        cost_returns,
        args.train_steps,
        args.batch_size,
        args.learning_rate,
        twin_cost=True,
    )

    summaries = {}
    for mode in ["unconstrained", "reward_only_qr", "budgeted_qc"]:
        summaries[mode] = evaluate(
            mode,
            args.checkpoint,
            reward_critic,
            cost_critic,
            args.budget,
            args.eval_episodes,
            args.horizon,
            args.n_candidates,
            args.eta,
            args.seed,
        )

    result = {
        "config": vars(args),
        "checkpoint_env": checkpoint.get("env_name", "SafetyPointGoal1-v0"),
        "dataset": {
            "transitions": int(dataset.observations.shape[0]),
            "reward_return_mean": float(np.mean(reward_returns)),
            "cost_return_mean": float(np.mean(cost_returns)),
        },
        "train": {
            "reward_critic": reward_stats,
            "cost_critic": cost_stats,
        },
        "eval": summaries,
    }
    write_json(args.out, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--collect-episodes", type=int, default=50)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma-reward", type=float, default=0.99)
    parser.add_argument("--gamma-cost", type=float, default=0.99)
    parser.add_argument("--budget", type=float, default=25.0)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--n-candidates", type=int, default=128)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results/budgeted_lhc/safety_gym_budgeted_qc.json")
    return parser.parse_args()


if __name__ == "__main__":
    output = run(parse_args())
    for method_name, summary in output["eval"].items():
        print(
            f"{method_name:>16}: "
            f"reward={summary['reward_mean']:.2f} "
            f"cost={summary['cost_mean']:.2f} "
            f"sat={summary['satisfied_mean']:.2f} "
            f"util={summary['utilization_mean']:.2f}"
        )
