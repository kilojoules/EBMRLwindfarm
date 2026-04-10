#!/usr/bin/env python3
"""
Safety-Gymnasium velocity budget experiment.

Demonstrates the AC-inspired budget surrogate on a standard ML benchmark:
SafetyHalfCheetahVelocity where the agent has a limited budget of
"speeding" timesteps (velocity above threshold).

A time-varying reward multiplier creates temporal structure so the AC
schedule can concentrate speeding into high-value phases.

Usage:
    python scripts/safety_gym_budget.py --train --total-timesteps 100000
    python scripts/safety_gym_budget.py --eval --checkpoint checkpoints/sac_cheetah.pt
    python scripts/safety_gym_budget.py --train --eval --total-timesteps 50000
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import gymnasium as gym

from load_surrogates import NegativeYawBudgetSurrogate


# =============================================================================
# TIME-VARYING REWARD WRAPPER
# =============================================================================

class TimeVaryingRewardWrapper(gym.Wrapper):
    """
    Multiplies reward by a sinusoidal schedule to create temporal structure.

    During high-multiplier phases, forward progress is worth more.
    The AC schedule should concentrate risky actions (speeding) into
    these phases.

    reward_out = reward_in * (1 + amplitude * sin(2*pi*t / period))
    """

    def __init__(self, env, amplitude: float = 0.5, period: int = 200):
        super().__init__(env)
        self.amplitude = amplitude
        self.period = period
        self._step_count = 0

    def reset(self, **kwargs):
        self._step_count = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, term, trunc, info = self.env.step(action)
        multiplier = 1.0 + self.amplitude * np.sin(
            2 * np.pi * self._step_count / self.period
        )
        info["reward_multiplier"] = multiplier
        info["raw_reward"] = reward
        reward = reward * multiplier
        self._step_count += 1
        return obs, reward, term, trunc, info


# =============================================================================
# VELOCITY BUDGET SURROGATE
# =============================================================================

class VelocityBudgetSurrogate(nn.Module):
    """
    AC-inspired budget surrogate for velocity constraints.

    Penalizes actions that lead to high velocity, with a time-varying
    weight based on remaining velocity-budget and remaining time.

    Since we can't directly penalize velocity (it's a state, not an action),
    we penalize large action magnitudes as a proxy — larger actions produce
    faster movement. The actual velocity tracking happens in the update()
    method using observed velocities.

    This is the same math as NegativeYawBudgetSurrogate:
        lambda(t) = exp(eta * (1/urgency - 1)) * hard_wall
        penalty = lambda(t) * (exp(k * relu(action_mag - threshold)) - 1)
    """

    def __init__(
        self,
        budget_steps: int,
        horizon_steps: int,
        risk_aversion: float = 2.0,
        steepness: float = 3.0,
        action_threshold: float = 0.5,
    ):
        super().__init__()
        self.budget_steps = budget_steps
        self.horizon_steps = horizon_steps
        self.risk_aversion = risk_aversion
        self.steepness = steepness
        self.action_threshold = action_threshold

        self.current_step = 0
        self.cumulative_violations = 0

    def reset(self):
        self.current_step = 0
        self.cumulative_violations = 0

    def update(self, velocity: float, v_threshold: float):
        """Update budget tracking after observing actual velocity."""
        if velocity > v_threshold:
            self.cumulative_violations += 1
        self.current_step += 1

    def _compute_lambda(self) -> float:
        eps = 1e-6
        budget_remaining = max(self.budget_steps - self.cumulative_violations, 0)
        time_remaining = max(self.horizon_steps - self.current_step, 1)

        budget_fraction = budget_remaining / max(self.budget_steps, 1)
        time_fraction = time_remaining / max(self.horizon_steps, 1)

        urgency = budget_fraction / max(time_fraction, eps)
        safe_urgency = max(urgency, eps)
        ac_weight = np.exp(self.risk_aversion * (1.0 / safe_urgency - 1.0))

        depletion = max(1.0 - budget_fraction / 0.05, 0)
        hard_wall = np.exp(self.steepness * depletion)

        return min(ac_weight * hard_wall, 1e6)

    def penalize_action(self, action: np.ndarray) -> float:
        """Compute scalar penalty for an action (used to modify reward)."""
        lam = self._compute_lambda()
        action_mag = np.abs(action).mean()
        excess = max(action_mag - self.action_threshold, 0)
        base_penalty = np.exp(self.steepness * excess) - 1.0
        return lam * base_penalty


# =============================================================================
# SIMPLE SAC AGENT (lightweight, no external deps)
# =============================================================================

class GaussianActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(-20, 2)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self(obs)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, mu


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, obs, action):
        return self.net(torch.cat([obs, action], -1))


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity=100000):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)
        self.ptr, self.size, self.cap = 0, 0, capacity

    def add(self, obs, act, rew, next_obs, done):
        self.obs[self.ptr] = obs
        self.act[self.ptr] = act
        self.rew[self.ptr] = rew
        self.next_obs[self.ptr] = next_obs
        self.done[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size):
        idx = np.random.randint(0, self.size, batch_size)
        return (
            torch.FloatTensor(self.obs[idx]),
            torch.FloatTensor(self.act[idx]),
            torch.FloatTensor(self.rew[idx]).unsqueeze(1),
            torch.FloatTensor(self.next_obs[idx]),
            torch.FloatTensor(self.done[idx]).unsqueeze(1),
        )


def train_sac(env_name, total_timesteps=100000, save_path="checkpoints/sac_cheetah.pt",
              reward_amplitude=0.5, reward_period=200):
    """Train a standard SAC agent (unconstrained) on the velocity task."""
    import safety_gymnasium  # noqa: F401

    env = gym.make(env_name)
    env = TimeVaryingRewardWrapper(env, amplitude=reward_amplitude, period=reward_period)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = float(env.action_space.high[0])

    actor = GaussianActor(obs_dim, act_dim)
    qf1 = QNetwork(obs_dim, act_dim)
    qf2 = QNetwork(obs_dim, act_dim)
    qf1_target = QNetwork(obs_dim, act_dim)
    qf2_target = QNetwork(obs_dim, act_dim)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())

    actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
    q_opt = torch.optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=3e-4)

    log_alpha = torch.zeros(1, requires_grad=True)
    alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)
    target_entropy = -act_dim

    buf = ReplayBuffer(obs_dim, act_dim)
    batch_size = 256
    gamma = 0.99
    tau = 0.005
    learning_starts = 5000

    obs, _ = env.reset()
    ep_ret, ep_len, ep_count = 0.0, 0, 0

    for step in range(1, total_timesteps + 1):
        if step < learning_starts:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = a.squeeze(0).numpy() * act_limit

        next_obs, reward, term, trunc, info = env.step(action)
        done = term or trunc
        buf.add(obs, action / act_limit, reward, next_obs, float(term))
        obs = next_obs
        ep_ret += reward
        ep_len += 1

        if done:
            ep_count += 1
            if ep_count % 10 == 0:
                print(f"Step {step}: ep_return={ep_ret:.1f}, ep_len={ep_len}")
            obs, _ = env.reset()
            ep_ret, ep_len = 0.0, 0

        if step >= learning_starts and buf.size >= batch_size:
            o, a, r, no, d = buf.sample(batch_size)
            alpha = log_alpha.exp().detach()

            with torch.no_grad():
                na, nlp, _ = actor.sample(no)
                q1t = qf1_target(no, na)
                q2t = qf2_target(no, na)
                qt = torch.min(q1t, q2t) - alpha * nlp
                target = r + gamma * (1 - d) * qt

            q1_loss = F.mse_loss(qf1(o, a), target)
            q2_loss = F.mse_loss(qf2(o, a), target)
            q_opt.zero_grad()
            (q1_loss + q2_loss).backward()
            q_opt.step()

            sa, slp, _ = actor.sample(o)
            q1a = qf1(o, sa)
            q2a = qf2(o, sa)
            qa = torch.min(q1a, q2a)
            actor_loss = (alpha * slp - qa).mean()
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()

            alpha_loss = -(log_alpha * (slp.detach() + target_entropy)).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()

            for p, pt in zip(qf1.parameters(), qf1_target.parameters()):
                pt.data.lerp_(p.data, tau)
            for p, pt in zip(qf2.parameters(), qf2_target.parameters()):
                pt.data.lerp_(p.data, tau)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        "actor": actor.state_dict(),
        "obs_dim": obs_dim,
        "act_dim": act_dim,
        "act_limit": act_limit,
    }, save_path)
    print(f"Saved checkpoint to {save_path}")
    env.close()
    return save_path


# =============================================================================
# BUDGET-CONSTRAINED EVALUATION
# =============================================================================

def eval_with_budget(
    env_name, checkpoint_path, budget_frac=0.25, risk_aversion=2.0,
    steepness=3.0, n_episodes=10, horizon=1000,
    reward_amplitude=0.5, reward_period=200,
    v_threshold=None,
):
    """Evaluate a trained agent with velocity budget constraint."""
    import safety_gymnasium  # noqa: F401

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    actor = GaussianActor(ckpt["obs_dim"], ckpt["act_dim"])
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    act_limit = ckpt["act_limit"]

    env = gym.make(env_name)
    env = TimeVaryingRewardWrapper(env, amplitude=reward_amplitude, period=reward_period)

    budget_steps = int(horizon * budget_frac)

    # Auto-detect velocity threshold from a few unconstrained episodes
    if v_threshold is None:
        velocities = []
        for _ in range(3):
            obs, _ = env.reset()
            for _ in range(horizon):
                with torch.no_grad():
                    a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = a.squeeze(0).numpy() * act_limit
                obs, _, term, trunc, info = env.step(action)
                # x-velocity is typically obs[8] for HalfCheetah
                velocities.append(abs(obs[8]) if len(obs) > 8 else abs(obs[0]))
                if term or trunc:
                    break
        v_threshold = np.percentile(velocities, 50)
        print(f"Auto velocity threshold (50th pct): {v_threshold:.2f}")

    results = []
    for ra in [0.0, 1.0, 2.0, 5.0]:
        ep_rewards, ep_violations, ep_budgets_used = [], [], []

        for ep in range(n_episodes):
            surr = VelocityBudgetSurrogate(
                budget_steps=budget_steps,
                horizon_steps=horizon,
                risk_aversion=ra,
                steepness=steepness,
            )
            surr.reset()

            obs, _ = env.reset()
            ep_rew = 0.0

            for t in range(horizon):
                with torch.no_grad():
                    a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
                action = a.squeeze(0).numpy() * act_limit

                # Apply budget penalty to action selection
                if ra >= 0 and surr.cumulative_violations < budget_steps:
                    penalty = surr.penalize_action(action / act_limit)
                    # Scale down action magnitude when penalty is high
                    scale = 1.0 / (1.0 + 0.1 * penalty)
                    action = action * scale

                obs, reward, term, trunc, info = env.step(action)
                velocity = abs(obs[8]) if len(obs) > 8 else abs(obs[0])
                surr.update(velocity, v_threshold)
                ep_rew += reward

                if term or trunc:
                    break

            ep_rewards.append(ep_rew)
            ep_violations.append(surr.cumulative_violations)
            ep_budgets_used.append(surr.cumulative_violations / budget_steps * 100)

        mean_rew = np.mean(ep_rewards)
        mean_viol = np.mean(ep_violations)
        mean_util = np.mean(ep_budgets_used)
        results.append({
            "ra": ra, "reward": mean_rew, "violations": mean_viol,
            "utilization": mean_util, "budget": budget_steps,
        })
        print(f"  RA={ra:.1f}: Reward={mean_rew:.1f} +/- {np.std(ep_rewards):.1f}, "
              f"Violations={mean_viol:.0f}/{budget_steps} ({mean_util:.0f}%)")

    # Unconstrained baseline
    uncon_rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_rew = 0.0
        for _ in range(horizon):
            with torch.no_grad():
                a, _, _ = actor.sample(torch.FloatTensor(obs).unsqueeze(0))
            action = a.squeeze(0).numpy() * act_limit
            obs, reward, term, trunc, _ = env.step(action)
            ep_rew += reward
            if term or trunc:
                break
        uncon_rewards.append(ep_rew)
    print(f"  Unconstrained: Reward={np.mean(uncon_rewards):.1f} +/- {np.std(uncon_rewards):.1f}")

    env.close()
    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Safety-Gym velocity budget experiment")
    parser.add_argument("--env", default="SafetyHalfCheetahVelocity-v1")
    parser.add_argument("--train", action="store_true", help="Train unconstrained SAC")
    parser.add_argument("--eval", action="store_true", help="Evaluate with budget")
    parser.add_argument("--checkpoint", default="checkpoints/sac_cheetah.pt")
    parser.add_argument("--total-timesteps", type=int, default=100000)
    parser.add_argument("--budget-frac", type=float, default=0.25,
                        help="Fraction of episode allowed above velocity threshold")
    parser.add_argument("--n-episodes", type=int, default=10)
    parser.add_argument("--horizon", type=int, default=1000)
    parser.add_argument("--reward-amplitude", type=float, default=0.5)
    parser.add_argument("--reward-period", type=int, default=200)
    cli = parser.parse_args()

    if cli.train:
        print(f"Training SAC on {cli.env} for {cli.total_timesteps} steps...")
        train_sac(cli.env, cli.total_timesteps, cli.checkpoint,
                  cli.reward_amplitude, cli.reward_period)

    if cli.eval:
        print(f"\nEvaluating with velocity budget (budget_frac={cli.budget_frac})...")
        eval_with_budget(
            cli.env, cli.checkpoint, cli.budget_frac,
            n_episodes=cli.n_episodes, horizon=cli.horizon,
            reward_amplitude=cli.reward_amplitude,
            reward_period=cli.reward_period,
        )

    if not cli.train and not cli.eval:
        print("Specify --train, --eval, or both. Run --help for options.")


if __name__ == "__main__":
    main()
