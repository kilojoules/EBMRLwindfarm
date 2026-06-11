#!/usr/bin/env python3
"""Shared utilities for budgeted long-horizon cost-to-go experiments."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def write_json(path: str | Path, payload: Dict) -> None:
    ensure_parent(path)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def urgency_lambda(
    step_idx: int,
    horizon: int,
    cumulative_cost: float,
    budget: float,
    eta: float = 3.0,
    clamp: float = 1.0e4,
) -> float:
    """Closed-form budget shadow price used only as a controller parameter."""
    epsilon = 1.0e-6
    if cumulative_cost >= budget:
        return float(clamp)
    budget_fraction = max(budget - cumulative_cost, 0.0) / max(budget, epsilon)
    time_fraction = max(horizon - step_idx, 1) / max(horizon, 1)
    urgency = budget_fraction / max(time_fraction, epsilon)
    exponent = eta * (1.0 / max(urgency, epsilon) - 1.0)
    if exponent >= math.log(clamp):
        return float(clamp)
    shadow_price = math.exp(exponent)
    return float(min(shadow_price, clamp))


def monte_carlo_returns(values: np.ndarray, done_flags: np.ndarray, gamma: float) -> np.ndarray:
    """Episode-resetting discounted returns for flat transition arrays."""
    returns = np.zeros_like(values, dtype=np.float32)
    running_return = 0.0
    for transition_idx in reversed(range(len(values))):
        if done_flags[transition_idx] > 0.5:
            running_return = 0.0
        running_return = float(values[transition_idx]) + gamma * running_return
        returns[transition_idx] = running_return
    return returns


@dataclass
class TransitionArrays:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    costs: np.ndarray
    next_observations: np.ndarray
    done_flags: np.ndarray

    @property
    def obs_dim(self) -> int:
        return int(self.observations.shape[-1])

    @property
    def act_dim(self) -> int:
        return int(self.actions.shape[-1])

    def as_dict(self) -> Dict[str, np.ndarray]:
        return {
            "observations": self.observations,
            "actions": self.actions,
            "rewards": self.rewards,
            "costs": self.costs,
            "next_observations": self.next_observations,
            "done_flags": self.done_flags,
        }


class ScalarCritic(nn.Module):
    """Small scalar critic for Q_r or Q_c."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([observations, actions], dim=-1))


class TwinCostCritic(nn.Module):
    """Conservative cost critic using max aggregation over twin heads."""

    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.first = ScalarCritic(obs_dim, act_dim, hidden_dim)
        self.second = ScalarCritic(obs_dim, act_dim, hidden_dim)

    def forward(self, observations: torch.Tensor, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.first(observations, actions), self.second(observations, actions)

    def conservative(self, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        first_values, second_values = self.forward(observations, actions)
        return torch.maximum(first_values, second_values)


def train_scalar_critic(
    name: str,
    critic: nn.Module,
    observations: np.ndarray,
    actions: np.ndarray,
    targets: np.ndarray,
    train_steps: int,
    batch_size: int,
    learning_rate: float,
    twin_cost: bool = False,
    verbose: bool = True,
) -> Dict[str, float]:
    critic.to(DEVICE)
    critic.train()
    optimizer = torch.optim.Adam(critic.parameters(), lr=learning_rate)
    obs_tensor = torch.tensor(observations, dtype=torch.float32, device=DEVICE)
    action_tensor = torch.tensor(actions, dtype=torch.float32, device=DEVICE)
    target_tensor = torch.tensor(targets, dtype=torch.float32, device=DEVICE).unsqueeze(-1)
    dataset_size = int(observations.shape[0])
    final_loss = float("nan")

    for train_idx in range(train_steps):
        batch_indices = torch.randint(0, dataset_size, (batch_size,), device=DEVICE)
        batch_obs = obs_tensor[batch_indices]
        batch_actions = action_tensor[batch_indices]
        batch_targets = target_tensor[batch_indices]
        if twin_cost:
            first_pred, second_pred = critic(batch_obs, batch_actions)
            loss = F.mse_loss(first_pred, batch_targets) + F.mse_loss(second_pred, batch_targets)
        else:
            prediction = critic(batch_obs, batch_actions)
            loss = F.mse_loss(prediction, batch_targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        if verbose and (train_idx == 0 or (train_idx + 1) % max(train_steps // 4, 1) == 0):
            print(f"  {name} step {train_idx + 1:>5}/{train_steps}: loss={final_loss:.4e}")

    critic.eval()
    return {"final_loss": final_loss}


def critic_predict(critic: nn.Module, observations: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
    if isinstance(critic, TwinCostCritic):
        return critic.conservative(observations, actions)
    return critic(observations, actions)


def select_budgeted_action(
    observations: np.ndarray,
    candidates: np.ndarray,
    reward_critic: nn.Module,
    cost_critic: nn.Module,
    cumulative_cost: float,
    budget: float,
    step_idx: int,
    horizon: int,
    eta: float,
    feasibility_margin: float = 0.0,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Select a candidate by constrained Q_r/Q_c scoring."""
    obs_batch = np.repeat(observations[None, :], candidates.shape[0], axis=0)
    obs_tensor = torch.tensor(obs_batch, dtype=torch.float32, device=DEVICE)
    action_tensor = torch.tensor(candidates, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        reward_values = critic_predict(reward_critic, obs_tensor, action_tensor).squeeze(-1)
        cost_values = critic_predict(cost_critic, obs_tensor, action_tensor).squeeze(-1).clamp(min=0.0)

    remaining_budget = max(budget - cumulative_cost, 0.0)
    shadow_price = urgency_lambda(step_idx, horizon, cumulative_cost, budget, eta=eta)
    violations = torch.relu(cost_values - remaining_budget + feasibility_margin)
    scores = reward_values - shadow_price * violations.square() - 0.01 * shadow_price * cost_values
    best_idx = int(torch.argmax(scores).item())
    diagnostics = {
        "selected_qr": float(reward_values[best_idx].item()),
        "selected_qc": float(cost_values[best_idx].item()),
        "selected_score": float(scores[best_idx].item()),
        "lambda": float(shadow_price),
        "remaining_budget": float(remaining_budget),
        "feasible_fraction": float((cost_values <= remaining_budget).float().mean().item()),
    }
    return candidates[best_idx].copy(), diagnostics


def stack_episode_metrics(metrics: Iterable[Dict[str, float]]) -> Dict[str, float]:
    rows = list(metrics)
    if not rows:
        return {}
    keys = sorted(rows[0].keys())
    summary: Dict[str, float] = {"episodes": float(len(rows))}
    for key in keys:
        values = np.asarray([row[key] for row in rows], dtype=np.float64)
        summary[f"{key}_mean"] = float(np.mean(values))
        summary[f"{key}_std"] = float(np.std(values))
    return summary
