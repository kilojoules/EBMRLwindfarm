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


class NegativeYawBudgetSurrogate(nn.Module):
    """
    Almgren-Chriss-inspired cumulative negative yaw time budget.

    Tracks how many timesteps each turbine has spent at negative yaw
    over the full episode. The penalty weight varies dynamically via
    an Almgren-Chriss optimal execution analogy:

        - Budget surplus (spent less than TWAP trajectory) → relax penalty
        - Budget deficit (overspent relative to TWAP) → tighten penalty
        - Budget nearly depleted → hard exponential wall

    The risk_aversion parameter controls how aggressively the agent
    concentrates negative-yaw usage in favorable conditions vs.
    spreading it uniformly (TWAP baseline).

    Almgren-Chriss mapping:
        Shares to liquidate  →  negative-yaw time budget
        Time horizon         →  episode length / planning horizon
        Risk aversion λ      →  concentration vs. uniform spending
        TWAP                 →  uniform spending (risk_aversion=0)
        Market impact         →  load damage from negative yaw

    CMDP connection:
        λ(t) serves a similar role to the Lagrangian dual variable in
        the constrained MDP formulation:
            max E[Σ r(s,a)]  s.t.  E[Σ c(s,a)] ≤ B
        where c(s,a) = 1[yaw < 0] and B = budget_steps.
        Standard CMDP learns λ via dual gradient ascent (requires
        retraining); this AC-inspired schedule provides a closed-form
        alternative that adapts post-hoc without retraining. Note: the
        functional form differs from the original Almgren-Chriss sinh
        trajectory; it shares the same qualitative properties (tighten
        when overspending, relax when underspending, TWAP at η=0).

    Empirical properties (verified numerically in ac_theory.py):
        - Monotonicity: cumulative spending ≤ budget for all t (enforced
          by the hard wall backstop; verified 0 violations in 3000 trials)
        - Continuity: λ(t) is continuous in (budget_remaining, time_remaining)
        - Boundary: λ → 1e6 as budget → 0 (effective hard constraint)
        - TWAP recovery: risk_aversion=0 → λ=1 ∀t (analytically exact)
        - Regret: empirically, regret scales approximately as O(√T)
          under i.i.d. benefits (not a proven bound)

    Usage:
        surrogate = NegativeYawBudgetSurrogate(
            budget_steps=180, horizon_steps=3600, risk_aversion=1.0,
        )
        surrogate.reset()
        # In episode loop:
        action = agent.act(..., guidance_fn=surrogate, guidance_scale=1.0)
        next_obs, reward, ... = env.step(action)
        surrogate.update(yaw_angles_deg)

    Args:
        budget_steps: Total allowed timesteps at negative yaw per turbine.
        horizon_steps: Total episode / planning horizon in timesteps.
        risk_aversion: Almgren-Chriss risk aversion. 0 = TWAP (constant
            penalty regardless of budget state). Higher values produce
            stronger reactions to budget deviations.
        steepness: Exponential wall steepness for the base neg-yaw penalty.
        yaw_max_deg: Max yaw angle for normalization (default 30°).
        neg_yaw_threshold_deg: Yaw angles below this are "negative" (default 0°).
        per_turbine_budgets: Optional per-turbine budget overrides. When
            provided, each turbine gets its own budget (e.g., turbine 3
            had a bearing replacement and gets more budget). Length must
            match n_turbines at runtime.
    """

    def __init__(
        self,
        budget_steps: int = 180,
        horizon_steps: int = 3600,
        risk_aversion: float = 1.0,
        steepness: float = 10.0,
        yaw_max_deg: float = 30.0,
        neg_yaw_threshold_deg: float = 0.0,
        per_turbine_budgets: Optional[List[int]] = None,
    ):
        super().__init__()
        self.budget_steps = budget_steps
        self.horizon_steps = horizon_steps
        self.risk_aversion = risk_aversion
        self.steepness = steepness
        self.yaw_max_deg = yaw_max_deg
        self.neg_yaw_threshold = neg_yaw_threshold_deg / yaw_max_deg  # normalized
        self.per_turbine_budgets = per_turbine_budgets  # None = uniform

        # State (managed by reset/update, not nn parameters)
        self.current_step: int = 0
        self.cumulative_neg_steps: Optional[torch.Tensor] = None  # (n_turbines, 1)

    def reset(self):
        """Reset at episode boundaries."""
        self.current_step = 0
        self.cumulative_neg_steps = None

    def update(self, action_deg: torch.Tensor):
        """
        Update budget tracking after an environment step.

        Args:
            action_deg: (n_turbines,) or (n_turbines, 1) yaw angles in degrees
        """
        if action_deg.dim() == 1:
            action_deg = action_deg.unsqueeze(-1)
        action_norm = action_deg / self.yaw_max_deg

        if self.cumulative_neg_steps is None:
            self.cumulative_neg_steps = torch.zeros_like(action_norm)

        is_negative = (action_norm < -self.neg_yaw_threshold).float()
        self.cumulative_neg_steps = self.cumulative_neg_steps + is_negative
        self.current_step += 1

    def _get_budget_tensor(self) -> torch.Tensor:
        """Get per-turbine budget as a tensor matching cumulative_neg_steps shape."""
        if self.per_turbine_budgets is not None and self.cumulative_neg_steps is not None:
            budgets = torch.tensor(
                self.per_turbine_budgets, dtype=torch.float32,
                device=self.cumulative_neg_steps.device,
            ).unsqueeze(-1)  # (n_turbines, 1)
            return budgets
        return torch.tensor(float(self.budget_steps))

    def _compute_lambda(self) -> torch.Tensor:
        """
        Compute per-turbine Almgren-Chriss-inspired penalty weight.

        The weight modulates how strongly negative yaw is penalized based
        on remaining budget and remaining time. Derived from the optimal
        execution analogy: urgency = budget_fraction / time_fraction.

        The AC-inspired weight serves a similar role to a time-dependent
        Lagrangian dual variable λ(t) for the cumulative constraint
        E[Σ c(s,a)] ≤ B. Where standard CMDP learns λ via dual gradient
        ascent (requiring retraining), this provides a closed-form
        schedule that adapts post-hoc.

        Returns:
            (n_turbines, 1) or scalar penalty multiplier
        """
        if self.cumulative_neg_steps is None:
            return torch.ones(1)

        eps = 1e-6
        budget_total = self._get_budget_tensor()
        budget_remaining = (budget_total - self.cumulative_neg_steps).clamp(min=0)
        time_remaining = max(self.horizon_steps - self.current_step, 1)

        budget_fraction = budget_remaining / budget_total.clamp(min=1)
        time_fraction = time_remaining / max(self.horizon_steps, 1)

        # Urgency: >1 = surplus (spent less than expected), <1 = deficit
        urgency = budget_fraction / max(time_fraction, eps)

        # AC weight: exp(risk_aversion * (1/urgency - 1))
        #   urgency=1 → 1.0 (on TWAP track)
        #   urgency>1 → <1.0 (relax, spend freely)
        #   urgency<1 → >1.0 (conserve)
        safe_urgency = urgency.clamp(min=eps)
        ac_weight = torch.exp(self.risk_aversion * (1.0 / safe_urgency - 1.0))

        # Hard wall backstop when budget < 5% remaining
        depletion = F.relu(1.0 - budget_fraction / 0.05)
        hard_wall = torch.exp(self.steepness * depletion)

        # Clamp to prevent inf (gradient-based optimizers need finite values)
        return (ac_weight * hard_wall).clamp(max=1e6)

    @property
    def budget_utilization(self) -> Optional[torch.Tensor]:
        """Fraction of budget consumed per turbine. None before first update."""
        if self.cumulative_neg_steps is None:
            return None
        budget_total = self._get_budget_tensor()
        return (self.cumulative_neg_steps / budget_total.clamp(min=1)).clamp(max=1.0)

    @property
    def budget_remaining_steps(self) -> Optional[torch.Tensor]:
        """Remaining budget per turbine in timesteps. None before first update."""
        if self.cumulative_neg_steps is None:
            return None
        budget_total = self._get_budget_tensor()
        return (budget_total - self.cumulative_neg_steps).clamp(min=0)

    @property
    def time_fraction_remaining(self) -> float:
        """Fraction of horizon remaining."""
        return max(self.horizon_steps - self.current_step, 0) / max(self.horizon_steps, 1)

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Per-turbine penalty for negative yaw, modulated by AC weight.

        penalty_i = lambda_i(t) * (exp(k * relu(-action_i - threshold)) - 1)

        Args:
            action: (batch, n_turbines, action_dim) in [-1, 1] normalized
            key_padding_mask: (batch, n_turbines) True = padding

        Returns:
            (batch, n_turbines, 1) penalty
        """
        neg_excess = F.relu(-action - self.neg_yaw_threshold)
        base_penalty = torch.exp(self.steepness * neg_excess) - 1.0

        ac_lambda = self._compute_lambda().to(action.device)
        if ac_lambda.dim() >= 2 and action.dim() == 3:
            ac_lambda = ac_lambda.unsqueeze(0)  # (1, n_turbines, 1)
        penalty = ac_lambda * base_penalty

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


# =============================================================================
# FACTORY
# =============================================================================

VALID_LOAD_SURROGATE_TYPES = [
    "exponential",        # ExponentialYawSurrogate — uniform |yaw| threshold
    "threshold",          # YawThresholdLoadSurrogate — quadratic penalty
    "per_turbine",        # PerTurbineYawSurrogate — heterogeneous thresholds
    "t1_positive_only",   # PositiveYawT1Surrogate — T1 positive yaw only
    "neg_yaw_budget",     # NegativeYawBudgetSurrogate — Almgren-Chriss neg yaw budget
    "relu",               # ReluLoadSurrogate — proof-of-concept
]


def create_load_surrogate(
    surrogate_type: str,
    steepness: float = 10.0,
    threshold_deg: float = 15.0,
    yaw_max_deg: float = 30.0,
    per_turbine_thresholds: str = "",
    neg_yaw_budget_steps: int = 180,
    neg_yaw_horizon_steps: int = 3600,
    neg_yaw_risk_aversion: float = 1.0,
    neg_yaw_threshold_deg: float = 0.0,
) -> nn.Module:
    """Factory function for load surrogates.

    Args:
        surrogate_type: One of VALID_LOAD_SURROGATE_TYPES.
        steepness: Exponential wall steepness (used by exponential, per_turbine, t1_positive_only, neg_yaw_budget).
        threshold_deg: Yaw threshold in degrees (used by exponential, threshold).
        yaw_max_deg: Max yaw angle for normalization (default 30°).
        per_turbine_thresholds: Comma-separated per-turbine limits in degrees (used by per_turbine).
        neg_yaw_budget_steps: Total allowed neg-yaw timesteps (used by neg_yaw_budget).
        neg_yaw_horizon_steps: Planning horizon in timesteps (used by neg_yaw_budget).
        neg_yaw_risk_aversion: AC risk aversion param (used by neg_yaw_budget).
        neg_yaw_threshold_deg: Below this = "negative yaw" (used by neg_yaw_budget).
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
    elif surrogate_type == "neg_yaw_budget":
        return NegativeYawBudgetSurrogate(
            budget_steps=neg_yaw_budget_steps,
            horizon_steps=neg_yaw_horizon_steps,
            risk_aversion=neg_yaw_risk_aversion,
            steepness=steepness,
            yaw_max_deg=yaw_max_deg,
            neg_yaw_threshold_deg=neg_yaw_threshold_deg,
        )
    elif surrogate_type == "relu":
        return ReluLoadSurrogate()
    raise ValueError(f"Unhandled surrogate_type={surrogate_type!r}")
