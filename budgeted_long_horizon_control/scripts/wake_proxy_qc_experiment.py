#!/usr/bin/env python3
"""Self-contained wake-convection proxy for budgeted long-horizon load control."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch

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
    urgency_lambda,
    write_json,
)


@dataclass
class WakeProxyConfig:
    n_turbines: int = 3
    horizon: int = 160
    max_delay: int = 8
    yaw_rate_weight: float = 0.025
    self_load_weight: float = 0.015
    convective_load_weight: float = 0.34
    yaw_loss_weight: float = 0.18
    steering_gain: float = 0.38
    seed: int = 1


class WakeConvectionProxyEnv:
    """Cheap delayed-load environment.

    Negative upstream yaw produces reward immediately during high-value wake
    events, but it inserts downstream load packets into delay queues. The load
    is realized only after wake convection, so one-step action penalties miss
    the important cost.
    """

    def __init__(self, config: WakeProxyConfig):
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.step_idx = 0
        self.prev_action = np.zeros(config.n_turbines, dtype=np.float32)
        self.pending_load = np.zeros((config.n_turbines, config.max_delay), dtype=np.float32)
        self.wind_speed = 1.0

    @property
    def obs_dim(self) -> int:
        return int(4 + self.config.n_turbines + self.config.n_turbines * self.config.max_delay)

    @property
    def act_dim(self) -> int:
        return int(self.config.n_turbines)

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.step_idx = 0
        self.prev_action = np.zeros(self.config.n_turbines, dtype=np.float32)
        self.pending_load.fill(0.0)
        self.wind_speed = self._wind_speed()
        return self._observe()

    def _wind_speed(self) -> float:
        periodic = 1.0 + 0.22 * np.sin(2.0 * np.pi * self.step_idx / 53.0)
        gust = 0.07 * np.sin(2.0 * np.pi * (self.step_idx + 11.0) / 17.0)
        return float(np.clip(periodic + gust, 0.65, 1.35))

    def _opportunity(self) -> float:
        return float(1.0 + 0.75 * np.sin(2.0 * np.pi * (self.step_idx - 9.0) / 47.0))

    def _observe(self) -> np.ndarray:
        time_angle = 2.0 * np.pi * self.step_idx / self.config.horizon
        pending_flat = self.pending_load.reshape(-1)
        obs = np.concatenate(
            [
                np.asarray(
                    [
                        np.sin(time_angle),
                        np.cos(time_angle),
                        self.wind_speed,
                        self._opportunity(),
                    ],
                    dtype=np.float32,
                ),
                self.prev_action.astype(np.float32),
                pending_flat.astype(np.float32),
            ]
        )
        return obs

    def _delay_for_pair(self, upstream_idx: int, downstream_idx: int) -> int:
        spacing = 4.0 * (downstream_idx - upstream_idx)
        delay = int(np.clip(round(spacing / max(self.wind_speed, 0.1)), 1, self.config.max_delay))
        return delay

    def _push_wake_loads(self, action: np.ndarray) -> None:
        negative_yaw = np.maximum(-action, 0.0)
        for upstream_idx in range(self.config.n_turbines - 1):
            for downstream_idx in range(upstream_idx + 1, self.config.n_turbines):
                delay = self._delay_for_pair(upstream_idx, downstream_idx)
                distance_decay = np.exp(-0.38 * (downstream_idx - upstream_idx - 1))
                packet = (
                    distance_decay
                    * self.wind_speed**3
                    * negative_yaw[upstream_idx] ** 2
                )
                self.pending_load[downstream_idx, delay - 1] += float(packet)

    def _pop_arrivals(self) -> np.ndarray:
        arrivals = self.pending_load[:, 0].copy()
        self.pending_load[:, :-1] = self.pending_load[:, 1:]
        self.pending_load[:, -1] = 0.0
        return arrivals

    def instant_cost_proxy(self, action: np.ndarray) -> float:
        yaw_rate = action - self.prev_action
        return float(
            self.config.yaw_rate_weight * np.sum(yaw_rate**2)
            + self.config.self_load_weight * self.wind_speed**3 * np.sum(action**2)
        )

    def reward_proxy(self, action: np.ndarray) -> float:
        negative_yaw = np.maximum(-action[:-1], 0.0)
        yaw_loss = self.config.yaw_loss_weight * np.sum(action**2)
        steering_value = self.config.steering_gain * self._opportunity() * self.wind_speed**3 * np.sum(negative_yaw)
        base_power = self.config.n_turbines * self.wind_speed**3
        return float(base_power + steering_value - yaw_loss)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, float, bool, Dict[str, float]]:
        clipped_action = np.clip(action, -1.0, 1.0).astype(np.float32)
        arrivals = self._pop_arrivals()
        immediate_cost = self.instant_cost_proxy(clipped_action)
        convective_cost = self.config.convective_load_weight * float(np.sum(arrivals))
        cost = float(immediate_cost + convective_cost)
        reward = self.reward_proxy(clipped_action)
        self._push_wake_loads(clipped_action)
        self.prev_action = clipped_action
        self.step_idx += 1
        self.wind_speed = self._wind_speed()
        done = self.step_idx >= self.config.horizon
        info = {
            "immediate_cost": float(immediate_cost),
            "convective_cost": float(convective_cost),
            "opportunity": float(self._opportunity()),
        }
        return self._observe(), reward, cost, done, info


def behavior_policy(env: WakeConvectionProxyEnv, exploration_scale: float) -> np.ndarray:
    opportunity = env._opportunity()
    risky_level = np.clip((opportunity - 0.75) / 1.5, 0.0, 1.0)
    action = np.zeros(env.act_dim, dtype=np.float32)
    action[:-1] = -0.85 * risky_level
    action += env.rng.normal(0.0, exploration_scale, size=env.act_dim).astype(np.float32)
    return np.clip(action, -1.0, 1.0)


def collect_dataset(config: WakeProxyConfig, episodes: int, exploration_scale: float) -> TransitionArrays:
    env = WakeConvectionProxyEnv(config)
    observations: List[np.ndarray] = []
    actions: List[np.ndarray] = []
    rewards: List[float] = []
    costs: List[float] = []
    next_observations: List[np.ndarray] = []
    done_flags: List[float] = []
    for episode_idx in range(episodes):
        obs = env.reset(seed=config.seed * 1000 + episode_idx)
        for _ in range(config.horizon):
            action = behavior_policy(env, exploration_scale)
            next_obs, reward, cost, done, _ = env.step(action)
            observations.append(obs)
            actions.append(action)
            rewards.append(reward)
            costs.append(cost)
            next_observations.append(next_obs)
            done_flags.append(float(done))
            obs = next_obs
            if done:
                break
    return TransitionArrays(
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        costs=np.asarray(costs, dtype=np.float32),
        next_observations=np.asarray(next_observations, dtype=np.float32),
        done_flags=np.asarray(done_flags, dtype=np.float32),
    )


def candidate_actions(env: WakeConvectionProxyEnv, n_candidates: int) -> np.ndarray:
    candidates = env.rng.uniform(-1.0, 1.0, size=(n_candidates, env.act_dim)).astype(np.float32)
    heuristic = behavior_policy(env, exploration_scale=0.05)
    candidates[0] = heuristic
    candidates[1] = np.zeros(env.act_dim, dtype=np.float32)
    candidates[2] = env.prev_action
    if env.act_dim > 1:
        candidates[3, :-1] = -0.85
        candidates[3, -1] = 0.0
    return np.clip(candidates, -1.0, 1.0)


def select_instant_penalty_action(
    env: WakeConvectionProxyEnv,
    candidates: np.ndarray,
    cumulative_cost: float,
    budget: float,
    eta: float,
) -> np.ndarray:
    shadow_price = urgency_lambda(env.step_idx, env.config.horizon, cumulative_cost, budget, eta=eta)
    scores = []
    for candidate in candidates:
        scores.append(env.reward_proxy(candidate) - shadow_price * env.instant_cost_proxy(candidate))
    return candidates[int(np.argmax(scores))].copy()


def evaluate_mode(
    mode: str,
    config: WakeProxyConfig,
    reward_critic: ScalarCritic,
    cost_critic: TwinCostCritic,
    episodes: int,
    budget: float,
    n_candidates: int,
    eta: float,
) -> Dict[str, float]:
    env = WakeConvectionProxyEnv(config)
    per_episode: List[Dict[str, float]] = []
    for episode_idx in range(episodes):
        obs = env.reset(seed=900_000 + config.seed * 100 + episode_idx)
        total_reward = 0.0
        total_cost = 0.0
        budgeted_steps = 0
        feasible_fraction_sum = 0.0
        for _ in range(config.horizon):
            candidates = candidate_actions(env, n_candidates)
            if mode == "zero_yaw":
                action = np.zeros(env.act_dim, dtype=np.float32)
            elif mode == "instant_penalty":
                action = select_instant_penalty_action(env, candidates, total_cost, budget, eta)
            elif mode == "reward_only_qr":
                obs_batch = np.repeat(obs[None, :], candidates.shape[0], axis=0)
                with torch.no_grad():
                    values = reward_critic(
                        torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE),
                        torch.tensor(candidates, dtype=torch.float32, device=DEVICE),
                    ).squeeze(-1)
                action = candidates[int(torch.argmax(values).item())].copy()
            elif mode == "budgeted_qc":
                action, diagnostics = select_budgeted_action(
                    obs,
                    candidates,
                    reward_critic,
                    cost_critic,
                    cumulative_cost=total_cost,
                    budget=budget,
                    step_idx=env.step_idx,
                    horizon=config.horizon,
                    eta=eta,
                )
                feasible_fraction_sum += diagnostics["feasible_fraction"]
                budgeted_steps += 1
            else:
                raise ValueError(f"Unknown mode: {mode}")

            obs, reward, cost, done, _ = env.step(action)
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
    return stack_episode_metrics(per_episode)


def run(args: argparse.Namespace) -> Dict:
    if args.quick:
        args.collect_episodes = min(args.collect_episodes, 12)
        args.train_steps = min(args.train_steps, 120)
        args.eval_episodes = min(args.eval_episodes, 4)
        args.n_candidates = min(args.n_candidates, 48)
    set_seed(args.seed)
    config = WakeProxyConfig(seed=args.seed, horizon=args.horizon)
    dataset = collect_dataset(config, args.collect_episodes, args.exploration_scale)
    reward_returns = monte_carlo_returns(dataset.rewards, dataset.done_flags, args.gamma_reward)
    cost_returns = monte_carlo_returns(dataset.costs, dataset.done_flags, args.gamma_cost)

    reward_critic = ScalarCritic(dataset.obs_dim, dataset.act_dim, args.hidden_dim)
    cost_critic = TwinCostCritic(dataset.obs_dim, dataset.act_dim, args.hidden_dim)
    reward_stats = train_scalar_critic(
        "Q_r",
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
        "Q_c",
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
    for mode in ["zero_yaw", "instant_penalty", "reward_only_qr", "budgeted_qc"]:
        print(f"Evaluating {mode}")
        summaries[mode] = evaluate_mode(
            mode,
            config,
            reward_critic,
            cost_critic,
            args.eval_episodes,
            args.budget,
            args.n_candidates,
            args.eta,
        )

    result = {
        "config": vars(args),
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
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=160)
    parser.add_argument("--collect-episodes", type=int, default=80)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--train-steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma-reward", type=float, default=0.995)
    parser.add_argument("--gamma-cost", type=float, default=0.995)
    parser.add_argument("--budget", type=float, default=10.0)
    parser.add_argument("--eta", type=float, default=3.0)
    parser.add_argument("--n-candidates", type=int, default=128)
    parser.add_argument("--exploration-scale", type=float, default=0.28)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--out", default="results/budgeted_lhc/wake_proxy_qc.json")
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

