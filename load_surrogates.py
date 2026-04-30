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
        schedule_type: str = "exp",
    ):
        super().__init__()
        self.budget_steps = budget_steps
        self.horizon_steps = horizon_steps
        self.risk_aversion = risk_aversion
        self.steepness = steepness
        self.yaw_max_deg = yaw_max_deg
        self.neg_yaw_threshold = neg_yaw_threshold_deg / yaw_max_deg  # normalized
        self.per_turbine_budgets = per_turbine_budgets  # None = uniform
        self.schedule_type = schedule_type  # "exp" or "inverse" (1/u)

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

        safe_urgency = urgency.clamp(min=eps)
        if self.schedule_type == "inverse":
            # Theoretically optimal under Boltzmann response: w*(u) = u^{-eta}
            # At eta=1 this is exactly 1/u. At eta=0 this is 1 (TWAP).
            ac_weight = safe_urgency.pow(-self.risk_aversion)
        else:
            # Practical approximation: exp(eta * (1/u - 1))
            # Matches 1/u to first order at u=1, better numerics near u=0
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
# DLC12 ANN-surrogate: blade-flapwise DEL (Teodor)
# =============================================================================

class FlapDELSurrogate(nn.Module):
    """Per-turbine blade-flapwise DEL via Teodor's DLC12 ANN surrogate.

    Cost c_t per turbine = DEL_flap(yaw_t, sector_flow_t) / DEL_ref.
    The surrogate output is a 10-min DEL in kNm; we treat it as a damage
    *rate* (per env step) and divide by a reference DEL so c_t is unitless
    and ~O(1) at the unconstrained operating point.

    Inputs that change per step (sector wind speed + TI + pset) are stored
    in a context buffer and refreshed by the training/eval loop via
    `update_context()`. The action (yaw) is the differentiable input used
    for energy guidance.

    Cost-only surrogate (does not depend on action_dim>1; takes the first
    action channel as yaw). For multi-DOF actions adapt accordingly.

    Usage:
        surr = FlapDELSurrogate.from_bundle(
            "checkpoints/teodor_dlc12_torch.pt",
            yaw_max_deg=30.0, del_ref=648.6)
        # each env step:
        surr.update_context(saws=arr_4, sati=arr_4, pset=arr_n)
        # then composed energy:
        e = surr.per_turbine_energy(action_norm)
    """

    def __init__(self,
                 surrogate,
                 yaw_max_deg: float = 30.0,
                 del_ref: float = 648.6,
                 output_name: str = "wrot_Bl1Rad0FlpMnt"):
        super().__init__()
        from helpers.teodor_surrogate import TeodorDLC12Surrogate
        if not isinstance(surrogate, TeodorDLC12Surrogate):
            raise TypeError("surrogate must be a TeodorDLC12Surrogate")
        if output_name not in surrogate.output_names:
            raise ValueError(
                f"output {output_name} not loaded in surrogate "
                f"(have {surrogate.output_names})")
        self.surrogate = surrogate
        self.yaw_max_deg = float(yaw_max_deg)
        self.del_ref = float(del_ref)
        self.output_name = output_name
        # Context: 9 flow/pset features per turbine. (n_turb, 9). Lazy alloc.
        self.register_buffer(
            "_ctx", torch.zeros(0, 9, dtype=torch.float32),
            persistent=False)

    @classmethod
    def from_bundle(cls,
                     bundle_path: str,
                     yaw_max_deg: float = 30.0,
                     del_ref: float = 648.6,
                     output_name: str = "wrot_Bl1Rad0FlpMnt",
                     map_location: str = "cpu"):
        from helpers.teodor_surrogate import TeodorDLC12Surrogate
        surr = TeodorDLC12Surrogate.from_bundle(
            bundle_path, outputs=[output_name],
            map_location=map_location)
        return cls(surr, yaw_max_deg=yaw_max_deg, del_ref=del_ref,
                    output_name=output_name)

    @torch.no_grad()
    def update_context(self,
                        saws: "torch.Tensor | list",
                        sati: "torch.Tensor | list",
                        pset: "torch.Tensor | list"):
        """Refresh per-turbine flow context.

        Args:
            saws: (n_turb, 4) sector wind speed [m/s], order
                  (left, right, up, down).
            sati: (n_turb, 4) sector turbulence intensity [-].
            pset: (n_turb,) power setpoint [-].
        """
        saws = torch.as_tensor(saws, dtype=torch.float32)
        sati = torch.as_tensor(sati, dtype=torch.float32)
        pset = torch.as_tensor(pset, dtype=torch.float32).reshape(-1, 1)
        if saws.shape[1] != 4 or sati.shape[1] != 4:
            raise ValueError(
                "saws/sati must be (n_turb, 4); got "
                f"{tuple(saws.shape)}, {tuple(sati.shape)}")
        ctx = torch.cat([saws, sati, pset], dim=1)  # (n_turb, 9)
        self._ctx = ctx.to(device=self.surrogate.in_mean.device)

    def _yaw_from_action(self, action: torch.Tensor) -> torch.Tensor:
        """Map normalized action [-1, 1] (..., n_turb, ad) -> yaw [deg] (..., n_turb)."""
        yaw_norm = action[..., 0]                                # take 1st channel
        return yaw_norm * self.yaw_max_deg

    def _features(self, yaw_deg: torch.Tensor) -> torch.Tensor:
        """Combine context + yaw into (..., n_turb, 10) surrogate input.

        Broadcasts context (n_turb, 9) over leading dims of yaw.
        """
        if self._ctx.numel() == 0:
            raise RuntimeError(
                "FlapDELSurrogate context not initialized. "
                "Call update_context(saws, sati, pset) once per env step.")
        n_turb_ctx = self._ctx.shape[0]
        n_turb_act = yaw_deg.shape[-1]
        if n_turb_ctx != n_turb_act:
            raise ValueError(
                f"context has {n_turb_ctx} turbines, "
                f"action has {n_turb_act}")
        # Broadcast: ctx (n_turb, 9) -> (..., n_turb, 9)
        ctx = self._ctx
        for _ in range(yaw_deg.dim() - 1):
            ctx = ctx.unsqueeze(0)
        ctx = ctx.expand(*yaw_deg.shape, 9)
        feats = torch.cat([ctx, yaw_deg.unsqueeze(-1)], dim=-1)
        return feats

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-turbine DEL-derived energy.

        Args:
            action: (batch, n_turb, action_dim) in [-1, 1].
            key_padding_mask: (batch, n_turb) True = padding.

        Returns:
            (batch, n_turb, 1) cost = DEL / del_ref (unitless, >= 0).
        """
        yaw_deg = self._yaw_from_action(action)
        feats = self._features(yaw_deg)
        # Run surrogate. Output dict with one entry.
        out = self.surrogate.predict_one(self.output_name, feats)
        cost = out / self.del_ref
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            cost = cost * mask
        return cost

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        per_turb = self.per_turbine_energy(action, key_padding_mask)
        return per_turb.sum(dim=(-2, -1), keepdim=False).unsqueeze(-1)


class FlapDELBudgetSurrogate(nn.Module):
    """Per-turbine DEL budget = total DEL incurred under no-yaw baseline.

    Cost rate c_i,t = blade-flap DEL_i(yaw, sector_flow_i, pset). The budget
    constraint per turbine is

        Σ_{t=0}^{T-1} c_i,t  ≤  B_i = Σ_{t=0}^{T-1} c^baseline_i,t

    where the baseline rollout uses zero yaw under identical inflow.
    Wake-steering moves wind onto downstream turbines and tends to raise their
    DEL above this baseline; the AC-inspired schedule penalises this when
    the cumulative cost outpaces the time-weighted-average reference.

    Combines two signals:
      1. base_penalty = DEL(candidate yaw) — diff'able in action; FlapDEL
         surrogate provides this per turbine.
      2. λ_i(t) = AC schedule from cumulative DEL vs B_i (same form as
         NegativeYawBudgetSurrogate); 1 at TWAP, large near depletion.

    Composed energy:  c̃_i = λ_i(t) · DEL_i(action) / del_ref
    so the agent prefers low-DEL actions (small yaw, off-wake direction) when
    its budget is tight, and runs unconstrained when there's headroom.

    Usage:
        surr = FlapDELBudgetSurrogate.from_bundle(
            "checkpoints/teodor_dlc12_torch.pt",
            per_turbine_budgets=[38_900.0, 25_950.0],   # kNm-step, from baseline
            horizon_steps=50, risk_aversion=1.0)
        surr.reset()
        # each env step:
        surr.update_context(saws, sati, pset)        # refreshes flow features
        a = agent.act(..., guidance_fn=surr)         # uses DEL gradient
        next_obs, ... = env.step(a)
        surr.update(saws, sati, pset, yaw_deg_taken) # accumulates realised DEL
    """

    def __init__(
        self,
        flap_del: "FlapDELSurrogate",
        per_turbine_budgets: List[float],
        horizon_steps: int,
        risk_aversion: float = 1.0,
        steepness: float = 10.0,
        del_ref: Optional[float] = None,
        schedule_type: str = "exp",
    ):
        super().__init__()
        if not isinstance(flap_del, FlapDELSurrogate):
            raise TypeError("flap_del must be a FlapDELSurrogate")
        self.flap_del = flap_del
        self.per_turbine_budgets = list(per_turbine_budgets)
        self.horizon_steps = int(horizon_steps)
        self.risk_aversion = float(risk_aversion)
        self.steepness = float(steepness)
        self.schedule_type = schedule_type
        self.del_ref = (float(del_ref) if del_ref is not None
                          else float(flap_del.del_ref))
        # State
        self.current_step: int = 0
        self.cumulative_del: Optional[torch.Tensor] = None  # (n_turb, 1)

    @classmethod
    def from_bundle(
        cls,
        bundle_path: str,
        per_turbine_budgets: List[float],
        horizon_steps: int,
        yaw_max_deg: float = 30.0,
        del_ref: float = 648.6,
        risk_aversion: float = 1.0,
        steepness: float = 10.0,
        output_name: str = "wrot_Bl1Rad0FlpMnt",
        map_location: str = "cpu",
        schedule_type: str = "exp",
    ):
        flap_del = FlapDELSurrogate.from_bundle(
            bundle_path, yaw_max_deg=yaw_max_deg, del_ref=del_ref,
            output_name=output_name, map_location=map_location)
        return cls(flap_del, per_turbine_budgets, horizon_steps,
                    risk_aversion=risk_aversion, steepness=steepness,
                    del_ref=del_ref, schedule_type=schedule_type)

    def reset(self):
        self.current_step = 0
        self.cumulative_del = None

    @torch.no_grad()
    def update_context(self, saws, sati, pset):
        """Forward to the inner FlapDEL — refreshes per-turbine flow context."""
        self.flap_del.update_context(saws, sati, pset)

    @torch.no_grad()
    def update(self, saws, sati, pset, yaw_deg):
        """Accumulate realised DEL given the action just taken.

        Args:
            saws, sati: (n_turb, 4) sector flow at this step.
            pset: (n_turb,)
            yaw_deg: (n_turb,) yaw angle taken this step.
        """
        self.update_context(saws, sati, pset)
        n_turb = self.flap_del._ctx.shape[0]
        action_dim = 1
        # Build a fake "action" with yaw already in physical degrees -> normalize.
        yaw_norm = (torch.as_tensor(yaw_deg, dtype=torch.float32)
                    / self.flap_del.yaw_max_deg)
        action = yaw_norm.reshape(1, n_turb, action_dim)
        per_turb_cost_norm = self.flap_del.per_turbine_energy(action)  # (1, n_turb, 1)
        # cost in physical kNm = norm * del_ref
        per_turb_del = per_turb_cost_norm.squeeze(0) * self.del_ref  # (n_turb, 1)
        if self.cumulative_del is None:
            self.cumulative_del = torch.zeros_like(per_turb_del)
        self.cumulative_del = self.cumulative_del + per_turb_del
        self.current_step += 1

    def _budget_tensor(self) -> torch.Tensor:
        b = torch.tensor(self.per_turbine_budgets, dtype=torch.float32)
        if self.cumulative_del is not None:
            b = b.to(self.cumulative_del.device).reshape_as(self.cumulative_del)
        return b

    def _compute_lambda(self) -> torch.Tensor:
        if self.cumulative_del is None:
            return torch.ones(1)
        eps = 1e-6
        budget_total = self._budget_tensor()
        budget_remaining = (budget_total - self.cumulative_del).clamp(min=0)
        time_remaining = max(self.horizon_steps - self.current_step, 1)

        budget_fraction = budget_remaining / budget_total.clamp(min=1.0)
        time_fraction = time_remaining / max(self.horizon_steps, 1)
        urgency = (budget_fraction / max(time_fraction, eps)).clamp(min=eps)

        if self.schedule_type == "inverse":
            ac_weight = urgency.pow(-self.risk_aversion)
        else:
            ac_weight = torch.exp(self.risk_aversion * (1.0 / urgency - 1.0))

        depletion = F.relu(1.0 - budget_fraction / 0.05)
        hard_wall = torch.exp(self.steepness * depletion)
        return (ac_weight * hard_wall).clamp(max=1e6)

    @property
    def budget_utilization(self) -> Optional[torch.Tensor]:
        if self.cumulative_del is None:
            return None
        return (self.cumulative_del / self._budget_tensor().clamp(min=1.0)
                ).clamp(max=1.0)

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """λ_i(t) · DEL_i(action) / del_ref."""
        base = self.flap_del.per_turbine_energy(action, key_padding_mask)
        lam = self._compute_lambda().to(base.device)
        if base.dim() == 3 and lam.dim() == 2:
            lam = lam.unsqueeze(0)  # (1, n_turb, 1)
        return lam * base

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
    "exponential",        # ExponentialYawSurrogate — uniform |yaw| threshold
    "threshold",          # YawThresholdLoadSurrogate — quadratic penalty
    "per_turbine",        # PerTurbineYawSurrogate — heterogeneous thresholds
    "t1_positive_only",   # PositiveYawT1Surrogate — T1 positive yaw only
    "neg_yaw_budget",     # NegativeYawBudgetSurrogate — Almgren-Chriss neg yaw budget
    "relu",               # ReluLoadSurrogate — proof-of-concept
    "flap_del",           # FlapDELSurrogate — Teodor DLC12 ANN, blade flap DEL
    "flap_del_budget",    # FlapDELBudgetSurrogate — AC-scheduled DEL budget
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
    flap_del_bundle: str = "checkpoints/teodor_dlc12_torch.pt",
    flap_del_ref: float = 648.6,
    flap_del_yaw_max_deg: float = 30.0,
    flap_del_per_turbine_budgets: str = "",      # comma list of B_i in kNm-step
    flap_del_horizon_steps: int = 200,
    flap_del_risk_aversion: float = 1.0,
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
    elif surrogate_type == "flap_del":
        return FlapDELSurrogate.from_bundle(
            flap_del_bundle,
            yaw_max_deg=flap_del_yaw_max_deg,
            del_ref=flap_del_ref,
        )
    elif surrogate_type == "flap_del_budget":
        if not flap_del_per_turbine_budgets:
            raise ValueError(
                "flap_del_budget requires --flap_del_per_turbine_budgets "
                "(comma list of B_i in kNm-step). Compute via "
                "scripts/calibrate_flap_del_budget.py.")
        budgets = [float(x) for x in flap_del_per_turbine_budgets.split(",")]
        return FlapDELBudgetSurrogate.from_bundle(
            flap_del_bundle,
            per_turbine_budgets=budgets,
            horizon_steps=flap_del_horizon_steps,
            yaw_max_deg=flap_del_yaw_max_deg,
            del_ref=flap_del_ref,
            risk_aversion=flap_del_risk_aversion,
            steepness=steepness,
        )
    raise ValueError(f"Unhandled surrogate_type={surrogate_type!r}")
