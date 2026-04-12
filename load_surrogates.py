"""
Differentiable load surrogates for post-hoc constraint composition.

These surrogates are added to the actor's energy landscape at inference time,
steering actions toward safe regions with zero retraining. They work with both
the EBT actor (per-turbine energy composition) and the diffusion actor
(classifier guidance during denoising).

Interface:
    per_turbine_energy(action, mask) -> (batch, n_turbines, 1)  # for EBT
    forward(action, mask) -> (batch, 1)                         # for diffusion
"""

from collections import deque
from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# LOAD SURROGATES
# =============================================================================

class ReluLoadSurrogate(nn.Module):
    """
    Simple differentiable load surrogate for proof-of-concept.

    Load model: positive yaw = larger loads, negative yaw = no load increase.
    load(action) = sum(relu(action_i)) over unmasked turbines.

    This is intentionally trivial — the point is to demonstrate energy
    composition, not to model loads accurately. Swap in a real surrogate later.
    """

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action: (batch, n_turbines, action_dim) in scaled action space
            key_padding_mask: (batch, n_turbines) True = padding

        Returns:
            (batch, 1) scalar load estimate per batch element
        """
        load_per_turbine = F.relu(action)
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            load_per_turbine = load_per_turbine * mask
        return load_per_turbine.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class YawThresholdLoadSurrogate(nn.Module):
    """
    Load surrogate that penalizes yaw angles exceeding a threshold.

    Penalizes |yaw| > threshold with a quadratic ramp:
        load_i = relu(|action_i| - threshold)^2

    Actions are in [-1, 1], mapping to [-yaw_max, +yaw_max] degrees.
    Default threshold of 20° with yaw_max=30° → threshold_normalized = 20/30 ≈ 0.667.
    """

    def __init__(self, threshold_deg: float = 20.0, yaw_max_deg: float = 30.0):
        super().__init__()
        self.threshold = threshold_deg / yaw_max_deg  # Normalize to action space

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            action: (batch, n_turbines, action_dim) in [-1, 1] action space
            key_padding_mask: (batch, n_turbines) True = padding

        Returns:
            (batch, 1) scalar load estimate per batch element
        """
        excess = F.relu(action.abs() - self.threshold)
        load_per_turbine = excess ** 2
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            load_per_turbine = load_per_turbine * mask
        return load_per_turbine.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class ExponentialYawSurrogate(nn.Module):
    """
    Exponential wall penalty for yaw angles exceeding a threshold.

    Creates a steep, near-hard-cap at the threshold:
        penalty_i = exp(k * relu(|action_i| - threshold)) - 1

    Returns per-turbine energies for composition with the EBT actor,
    or a scalar sum for backward compatibility with the diffusion actor.
    """

    def __init__(self, threshold_deg: float = 15.0, yaw_max_deg: float = 30.0, steepness: float = 10.0):
        super().__init__()
        self.threshold = threshold_deg / yaw_max_deg  # Normalize to action space
        self.steepness = steepness

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-turbine exponential penalty.

        Args:
            action: (batch, n_turbines, action_dim) in [-1, 1] action space
            key_padding_mask: (batch, n_turbines) True = padding

        Returns:
            (batch, n_turbines, 1) per-turbine penalty (0 below threshold, steep wall above)
        """
        excess = F.relu(action.abs() - self.threshold)
        penalty = torch.exp(self.steepness * excess) - 1.0
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            penalty = penalty * mask
        return penalty

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar version for backward compat. Returns (batch, 1)."""
        per_turb = self.per_turbine_energy(action, key_padding_mask)
        return per_turb.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class PerTurbineYawSurrogate(nn.Module):
    """
    Exponential wall with per-turbine thresholds.

    Enables heterogeneous constraints: e.g. turbine 1 at ±20° while
    turbine 3 (worn bearing) is limited to ±5°.

    Args:
        thresholds_deg: Per-turbine yaw limits in degrees, e.g. [20.0, 20.0, 5.0]
        yaw_max_deg: Maximum yaw angle (for normalization to [-1, 1] action space)
        steepness: Exponential wall steepness (higher = harder cap)
    """

    def __init__(self, thresholds_deg: List[float], yaw_max_deg: float = 30.0,
                 steepness: float = 10.0):
        super().__init__()
        thresholds = torch.tensor(thresholds_deg, dtype=torch.float32) / yaw_max_deg
        self.register_buffer("thresholds", thresholds)
        self.steepness = steepness

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-turbine penalty with heterogeneous thresholds.

        Args:
            action: (batch, n_turbines, action_dim) in [-1, 1]
            key_padding_mask: (batch, n_turbines) True = padding

        Returns:
            (batch, n_turbines, 1) per-turbine penalty
        """
        # self.thresholds: (n_turbines,) → (1, n_turbines, 1) for broadcasting
        thresh = self.thresholds.unsqueeze(0).unsqueeze(-1)
        excess = F.relu(action.abs() - thresh)
        penalty = torch.exp(self.steepness * excess) - 1.0
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            penalty = penalty * mask
        return penalty

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar version for diffusion guidance. Returns (batch, 1)."""
        per_turb = self.per_turbine_energy(action, key_padding_mask)
        return per_turb.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class YawTravelBudgetSurrogate(nn.Module):
    """
    Per-turbine yaw travel budget over a rolling window.

    Tracks cumulative |Δyaw| over the last `window_steps` steps per turbine.
    Penalty ramps exponentially as travel approaches the budget. Does NOT
    penalize the rate of change — a turbine can move fast to a new optimum,
    but it can't keep oscillating.

    Usage:
        surrogate = YawTravelBudgetSurrogate(budget_deg=100, window_steps=100)
        # At episode start:
        surrogate.reset()
        # During episode loop:
        action = agent.act(..., guidance_fn=surrogate, guidance_scale=1.0)
        next_obs, reward, ... = env.step(action)
        surrogate.update(yaw_angles_deg)  # call AFTER env step
    """

    def __init__(self, budget_deg: float = 100.0, window_steps: int = 100,
                 yaw_max_deg: float = 30.0, steepness: float = 5.0):
        super().__init__()
        self.budget = budget_deg / yaw_max_deg  # in normalized [-1, 1] space
        self.window_steps = window_steps
        self.steepness = steepness
        self.yaw_max_deg = yaw_max_deg
        # State (managed by reset/update, not nn parameters)
        self.prev_action: Optional[torch.Tensor] = None
        self.travel_window: Optional[deque] = None
        self.cumulative_travel: Optional[torch.Tensor] = None

    def reset(self):
        """Reset at episode boundaries."""
        self.prev_action = None
        self.travel_window = None
        self.cumulative_travel = None

    def update(self, action_deg: torch.Tensor):
        """
        Update travel tracking after an environment step.

        Args:
            action_deg: (n_turbines,) or (n_turbines, 1) yaw angles in degrees
        """
        if action_deg.dim() == 1:
            action_deg = action_deg.unsqueeze(-1)
        action_norm = action_deg / self.yaw_max_deg

        if self.prev_action is None:
            self.prev_action = action_norm.clone()
            self.travel_window = deque(maxlen=self.window_steps)
            self.cumulative_travel = torch.zeros_like(action_norm)
            return

        delta = (action_norm - self.prev_action).abs()

        # Window is at capacity — subtract the oldest before adding new
        if len(self.travel_window) == self.window_steps:
            oldest = self.travel_window[0]
            self.cumulative_travel = self.cumulative_travel - oldest

        self.travel_window.append(delta)
        self.cumulative_travel = self.cumulative_travel + delta
        self.prev_action = action_norm.clone()

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-turbine penalty based on proposed travel vs remaining budget.

        Args:
            action: (batch, n_turbines, action_dim) in [-1, 1] normalized
            key_padding_mask: (batch, n_turbines) True = padding

        Returns:
            (batch, n_turbines, 1) penalty (0 if under budget, steep wall above)
        """
        if self.prev_action is None or self.cumulative_travel is None:
            return torch.zeros_like(action)

        # How much NEW travel would this action add?
        prev = self.prev_action.unsqueeze(0)  # (1, n_turbines, action_dim)
        proposed_delta = (action - prev).abs()
        proposed_total = self.cumulative_travel.unsqueeze(0) + proposed_delta

        # Exponential wall as travel/budget ratio exceeds 1
        ratio = proposed_total / self.budget
        excess = F.relu(ratio - 1.0)
        penalty = torch.exp(self.steepness * excess) - 1.0

        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            penalty = penalty * mask
        return penalty

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar version for diffusion guidance. Returns (batch, 1)."""
        per_turb = self.per_turbine_energy(action, key_padding_mask)
        return per_turb.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class PositiveYawT1Surrogate(nn.Module):
    """
    Constrains the first turbine (index 0) to positive yaw angles only.
    All other turbines are unconstrained.

    T1 penalty: exp(k * relu(-action_T1)) - 1
        - Negative yaw on T1 → steep exponential penalty
        - Positive yaw on T1 → zero penalty
    Other turbines: always zero penalty.

    Args:
        steepness: Exponential wall steepness (higher = harder cap)
    """

    def __init__(self, steepness: float = 10.0):
        super().__init__()
        self.steepness = steepness

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-turbine penalty (only T1 is penalized for negative yaw).

        Args:
            action: (batch, n_turbines, action_dim) in [-1, 1] action space
            key_padding_mask: (batch, n_turbines) True = padding

        Returns:
            (batch, n_turbines, 1) per-turbine penalty
        """
        penalty = torch.zeros_like(action)
        # Penalize T1 for negative yaw: relu(-action) > 0 when action < 0
        penalty[:, 0, :] = torch.exp(self.steepness * F.relu(-action[:, 0, :])) - 1.0
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            penalty = penalty * mask
        return penalty

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar version for diffusion guidance. Returns (batch, 1)."""
        per_turb = self.per_turbine_energy(action, key_padding_mask)
        return per_turb.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class QuadraticPositiveYawT1Surrogate(nn.Module):
    """
    Constrains T1 (index 0) to positive yaw using a quadratic penalty.

    Softer than PositiveYawT1Surrogate: the gradient is zero at the boundary
    (a=0), so the constraint doesn't overwhelm the actor's energy near the
    decision point. This lets the actor's learned landscape determine where
    T1 settles in the positive region.

    T1 penalty: scale * relu(-action_T1)^2

    Args:
        scale: Penalty scaling factor (higher = stronger constraint)
    """

    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.scale = scale

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        penalty = torch.zeros_like(action)
        penalty[:, 0, :] = self.scale * F.relu(-action[:, 0, :]) ** 2
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            penalty = penalty * mask
        return penalty

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_turb = self.per_turbine_energy(action, key_padding_mask)
        return per_turb.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class LinearPositiveYawT1Surrogate(nn.Module):
    """
    Constrains T1 (index 0) to positive yaw using a linear penalty.

    Intermediate between exponential (gradient explodes) and quadratic
    (gradient starts at zero). Provides a constant gradient when T1 is
    negative, proportional to the scale parameter.

    T1 penalty: scale * relu(-action_T1)

    Args:
        scale: Penalty scaling factor (higher = stronger constraint)
    """

    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.scale = scale

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        penalty = torch.zeros_like(action)
        penalty[:, 0, :] = self.scale * F.relu(-action[:, 0, :])
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            penalty = penalty * mask
        return penalty

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_turb = self.per_turbine_energy(action, key_padding_mask)
        return per_turb.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


# =============================================================================
# FACTORY
# =============================================================================

VALID_LOAD_SURROGATE_TYPES = [
    "exponential",           # ExponentialYawSurrogate — uniform |yaw| threshold
    "threshold",             # YawThresholdLoadSurrogate — quadratic penalty
    "per_turbine",           # PerTurbineYawSurrogate — heterogeneous thresholds
    "t1_positive_only",      # PositiveYawT1Surrogate — T1 positive yaw (exponential)
    "t1_positive_quadratic", # QuadraticPositiveYawT1Surrogate — T1 positive (quadratic)
    "t1_positive_linear",    # LinearPositiveYawT1Surrogate — T1 positive (linear)
    "relu",                  # ReluLoadSurrogate — proof-of-concept
]


def create_load_surrogate(
    surrogate_type: str,
    steepness: float = 10.0,
    threshold_deg: float = 15.0,
    yaw_max_deg: float = 30.0,
    per_turbine_thresholds: str = "",
) -> nn.Module:
    """Factory function for load surrogates.

    Args:
        surrogate_type: One of VALID_LOAD_SURROGATE_TYPES.
        steepness: Exponential wall steepness or penalty scale factor.
        threshold_deg: Yaw threshold in degrees (used by exponential, threshold).
        yaw_max_deg: Max yaw angle for normalization (default 30°).
        per_turbine_thresholds: Comma-separated per-turbine limits in degrees (used by per_turbine).
    """
    if surrogate_type not in VALID_LOAD_SURROGATE_TYPES:
        raise ValueError(
            f"Unknown load_surrogate_type={surrogate_type!r}. "
            f"Valid: {VALID_LOAD_SURROGATE_TYPES}"
        )
    if surrogate_type == "exponential":
        return ExponentialYawSurrogate(threshold_deg, yaw_max_deg, steepness)
    elif surrogate_type == "threshold":
        return YawThresholdLoadSurrogate(threshold_deg, yaw_max_deg)
    elif surrogate_type == "per_turbine":
        thresholds = [float(x) for x in per_turbine_thresholds.split(",")]
        return PerTurbineYawSurrogate(thresholds, yaw_max_deg, steepness)
    elif surrogate_type == "t1_positive_only":
        return PositiveYawT1Surrogate(steepness)
    elif surrogate_type == "t1_positive_quadratic":
        return QuadraticPositiveYawT1Surrogate(steepness)
    elif surrogate_type == "t1_positive_linear":
        return LinearPositiveYawT1Surrogate(steepness)
    elif surrogate_type == "relu":
        return ReluLoadSurrogate()
    raise ValueError(f"Unhandled surrogate_type={surrogate_type!r}")
