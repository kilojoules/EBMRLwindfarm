"""
Toy 3-turbine inline wind farm environment for fast RL prototyping.

Simple analytical wake model with two symmetric optimal yaw setpoints:
  - Front two turbines at +20° (or -20°), back turbine at 0°
  - The sign symmetry creates a multimodal optimum, useful for testing
    EBM-based policies that must represent multiple modes.

No heavy dependencies — pure numpy + gymnasium.

Usage:
    python toy_wind_farm_env.py          # Run sanity check
    env = ToyWindFarmEnv()               # Use in your training code
    obs, info = env.reset()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import gymnasium
import numpy as np
from gymnasium import spaces


class ToyWindFarmEnv(gymnasium.Env):
    """
    3 turbines in a line at x=[0, 500, 1000], y=[0, 0, 0].
    Wind from 270° (left to right, aligned with turbine row).

    Observations (per turbine): [yaw_normalized, effective_wind_speed_normalized]
    Actions: yaw delta per turbine in [-1, 1], scaled by yaw_step.
    Reward: total farm power normalized by greedy maximum (3 * U₀³).

    Wake model:
        - Power: P_i = U_eff_i³ · cos³(γ_i)
        - Deficit from upstream yaw γ: d_base · exp(-3 · sin²(γ))
        - Yawing deflects the wake sideways, reducing deficit on downstream turbines
        - Multiple wakes combined via root-sum-of-squares
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        wind_speed: float = 9.0,
        max_yaw: float = 30.0,
        yaw_step: float = 5.0,
        max_steps: int = 50,
    ):
        super().__init__()

        self.n_turbines = 3
        self.positions = np.array([[0, 0], [500, 0], [1000, 0]], dtype=np.float32)
        self.wind_speed = wind_speed
        self.max_yaw = max_yaw
        self.yaw_step = yaw_step
        self.max_steps = max_steps

        # Wake model parameters
        self.base_deficit = 0.3  # 30% velocity deficit in full wake at 500m
        self.deficit_decay_far = 0.5  # T1→T3 deficit is 50% of T1→T2 (double distance)
        self.deflection_sharpness = 3.0  # Controls how fast deficit drops with yaw

        # Normalization constant: max possible power (3 turbines, no wake, no yaw)
        self.power_normalization = 3.0 * wind_speed**3

        # Action: yaw delta per turbine, in [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.n_turbines,), dtype=np.float32
        )

        # Obs per turbine: [yaw_normalized, effective_wind_speed_normalized]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(self.n_turbines, 2),
            dtype=np.float32,
        )

        # State
        self.yaw_angles = np.zeros(self.n_turbines, dtype=np.float32)
        self.step_count = 0

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.yaw_angles = np.zeros(self.n_turbines, dtype=np.float32)
        self.step_count = 0
        obs = self._get_obs()
        return obs, self._get_info()

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        delta = action * self.yaw_step
        self.yaw_angles = np.clip(
            self.yaw_angles + delta, -self.max_yaw, self.max_yaw
        )
        self.step_count += 1

        total_power, _, _ = self._compute_power()
        reward = total_power / self.power_normalization

        terminated = False
        truncated = self.step_count >= self.max_steps

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _compute_power(self):
        """
        Analytical wake model for 3 inline turbines.

        Returns:
            total_power: scalar, sum of all turbine powers
            per_turbine_power: array of shape (3,)
            effective_wind_speeds: array of shape (3,)
        """
        gamma = np.radians(self.yaw_angles)
        cos_factor = np.cos(gamma) ** 3
        sin2 = np.sin(gamma) ** 2

        # Deficit from T1 on T2 (500m downstream)
        d12 = self.base_deficit * np.exp(-self.deflection_sharpness * sin2[0])

        # Deficit from T1 on T3 (1000m downstream, decayed)
        d13 = self.deficit_decay_far * self.base_deficit * np.exp(
            -self.deflection_sharpness * sin2[0]
        )

        # Deficit from T2 on T3 (500m downstream)
        d23 = self.base_deficit * np.exp(-self.deflection_sharpness * sin2[1])

        # Combined deficit on T3 (root-sum-of-squares)
        d3 = np.sqrt(d13**2 + d23**2)

        # Effective wind speeds
        u_eff = np.array([
            self.wind_speed,
            self.wind_speed * (1 - d12),
            self.wind_speed * (1 - d3),
        ], dtype=np.float32)

        per_turbine_power = u_eff**3 * cos_factor
        total_power = per_turbine_power.sum()

        return float(total_power), per_turbine_power, u_eff

    def _get_obs(self):
        _, _, u_eff = self._compute_power()
        obs = np.stack([
            self.yaw_angles / self.max_yaw,        # normalized yaw in [-1, 1]
            u_eff / self.wind_speed,                # normalized wind speed in [0, 1]
        ], axis=-1).astype(np.float32)
        return obs

    def _get_info(self):
        total_power, per_turbine_power, u_eff = self._compute_power()
        return {
            "yaw_angles": self.yaw_angles.copy(),
            "per_turbine_power": per_turbine_power,
            "effective_wind_speeds": u_eff,
            "total_power": total_power,
            "positions": self.positions.copy(),
        }


if __name__ == "__main__":
    env = ToyWindFarmEnv()

    # --- Sanity check: compare baseline vs optimal setpoints ---
    print("=== Toy Wind Farm Env — Sanity Check ===\n")

    env.reset()
    baseline_power, baseline_per, baseline_u = env._compute_power()
    print(f"Baseline (0, 0, 0)°:")
    print(f"  Total power: {baseline_power:.1f}  (normalized: {baseline_power / env.power_normalization:.4f})")
    print(f"  Per-turbine:  {baseline_per}")
    print(f"  Wind speeds:  {baseline_u}\n")

    # Test optimal setpoints
    for yaw_config, label in [
        ([20, 20, 0], "+20° optimum"),
        ([-20, -20, 0], "-20° optimum"),
    ]:
        env.reset()
        env.yaw_angles = np.array(yaw_config, dtype=np.float32)
        power, per, u = env._compute_power()
        gain = (power - baseline_power) / baseline_power * 100
        print(f"{label} ({yaw_config})°:")
        print(f"  Total power: {power:.1f}  (normalized: {power / env.power_normalization:.4f})")
        print(f"  Per-turbine:  {per}")
        print(f"  Wind speeds:  {u}")
        print(f"  Gain over baseline: {gain:+.1f}%\n")

    # --- Sweep to find actual optimum ---
    print("=== Yaw Sweep (T1=T2=γ, T3=0) ===")
    best_yaw, best_power = 0, 0
    for yaw_deg in range(0, 31):
        env.reset()
        env.yaw_angles = np.array([yaw_deg, yaw_deg, 0], dtype=np.float32)
        power, _, _ = env._compute_power()
        marker = ""
        if power > best_power:
            best_power = power
            best_yaw = yaw_deg
            marker = " ←"
        print(f"  γ={yaw_deg:2d}°  power={power:.1f}{marker}")

    print(f"\nOptimal yaw: ±{best_yaw}°  power={best_power:.1f}")
    print(f"Gain over baseline: {(best_power - baseline_power) / baseline_power * 100:+.1f}%")

    # --- Quick episode rollout ---
    print("\n=== Random Episode Rollout (5 steps) ===")
    obs, info = env.reset()
    print(f"Reset obs shape: {obs.shape}, obs:\n{obs}")
    for i in range(5):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Step {i+1}: action={action}, reward={reward:.4f}, yaw={info['yaw_angles']}")
