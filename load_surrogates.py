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


class DELConstraintSurrogate(nn.Module):
    """Constrain per-turbine DEL to within X% of a baseline reference.

    Uses a PyTorch port of the ANN DEL surrogate with frozen sector-averaged
    wind conditions as context. Only the yaw input carries gradients inside
    the EBT optimization loop.

    Two comparison modes:
      "per_turbine": penalty_i when DEL_agent_i / DEL_baseline_i > 1 + threshold
      "farm_max":    penalty_i when DEL_agent_i / max(DEL_baseline) > 1 + threshold

    Two penalty shapes:
      "exponential": exp(steepness * relu(ratio - (1+threshold))) - 1
      "quadratic":   steepness * relu(ratio - (1+threshold))^2

    Requires set_context() to be called each env step with the current
    sector averages and baseline DELs from LoadWrapper.
    """

    def __init__(
        self,
        torch_del_model: nn.Module,
        mode: str = "per_turbine",
        threshold_pct: float = 0.10,
        steepness: float = 10.0,
        penalty_type: str = "exponential",
        yaw_max_deg: float = 30.0,
        pset: float = 1.0,
    ):
        super().__init__()
        assert mode in ("per_turbine", "farm_max"), f"Unknown mode: {mode}"
        assert penalty_type in ("exponential", "quadratic"), f"Unknown penalty_type: {penalty_type}"

        self.torch_del_model = torch_del_model
        self.mode = mode
        self.threshold_pct = threshold_pct
        self.steepness = steepness
        self.penalty_type = penalty_type
        self.yaw_max_deg = yaw_max_deg
        self.pset = pset

        # Frozen context (set per env step, no gradient)
        self._frozen_sector_avgs: Optional[torch.Tensor] = None   # (n_turb, 8)
        self._frozen_baseline_dels: Optional[torch.Tensor] = None  # (n_turb,)

    def set_context(
        self,
        sector_avgs: torch.Tensor,
        baseline_dels: torch.Tensor,
    ) -> None:
        """Update frozen context after each env step.

        Args:
            sector_avgs: (n_turbines, 8) — [saws_L, saws_R, saws_U, saws_D,
                         sati_L, sati_R, sati_U, sati_D] in physical units.
            baseline_dels: (n_turbines,) — DEL under baseline yaws.
        """
        self._frozen_sector_avgs = sector_avgs.detach()
        self._frozen_baseline_dels = baseline_dels.detach()

    def per_turbine_energy(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Per-turbine DEL penalty energy.

        Args:
            action: (batch, n_turbines, 1) in [-1, 1] normalised action space
            key_padding_mask: (batch, n_turbines) True = padding

        Returns:
            (batch, n_turbines, 1) penalty energy per turbine
        """
        if self._frozen_sector_avgs is None or self._frozen_baseline_dels is None:
            return torch.zeros_like(action)

        batch, n_turb, _ = action.shape
        device = action.device

        # Convert normalised action to yaw degrees (only this carries gradient)
        yaw_deg = action * self.yaw_max_deg  # (batch, n_turb, 1)

        # Frozen context: broadcast to batch dimension
        sector = self._frozen_sector_avgs.to(device)  # (n_turb, 8)
        sector_batch = sector.unsqueeze(0).expand(batch, -1, -1)  # (batch, n_turb, 8)

        pset_batch = torch.full(
            (batch, n_turb, 1), self.pset, device=device, dtype=action.dtype
        )

        # Build 10-d input: [saws(4), sati(4), pset, yaw]
        model_input = torch.cat([sector_batch, pset_batch, yaw_deg], dim=-1)

        # Predict DEL — reshape to (batch*n_turb, 10) for the model
        flat_input = model_input.reshape(-1, 10)
        flat_del = self.torch_del_model(flat_input)  # (batch*n_turb, 1)
        del_agent = flat_del.reshape(batch, n_turb, 1)

        # Reference DEL
        baseline = self._frozen_baseline_dels.to(device)
        if self.mode == "per_turbine":
            del_ref = baseline.unsqueeze(0).unsqueeze(-1)  # (1, n_turb, 1)
        else:  # farm_max
            del_ref = baseline.max().view(1, 1, 1)

        # Ratio and penalty
        ratio = del_agent / del_ref.clamp(min=1e-6)
        excess = F.relu(ratio - (1.0 + self.threshold_pct))

        if self.penalty_type == "exponential":
            penalty = torch.exp(self.steepness * excess) - 1.0
        else:  # quadratic
            penalty = self.steepness * excess ** 2

        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            penalty = penalty * mask

        return penalty

    def forward(
        self,
        action: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Scalar version. Returns (batch, 1)."""
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
    "del_per_turbine",       # DELConstraintSurrogate — per-turbine DEL comparison
    "del_farm_max",          # DELConstraintSurrogate — farm-max DEL comparison
]


def create_load_surrogate(
    surrogate_type: str,
    steepness: float = 10.0,
    threshold_deg: float = 15.0,
    yaw_max_deg: float = 30.0,
    per_turbine_thresholds: str = "",
    del_model_dir: str = "",
    del_threshold_pct: float = 0.10,
    del_penalty_type: str = "exponential",
) -> nn.Module:
    """Factory function for load surrogates.

    Args:
        surrogate_type: One of VALID_LOAD_SURROGATE_TYPES.
        steepness: Exponential wall steepness or penalty scale factor.
        threshold_deg: Yaw threshold in degrees (used by exponential, threshold).
        yaw_max_deg: Max yaw angle for normalization (default 30°).
        per_turbine_thresholds: Comma-separated per-turbine limits in degrees.
        del_model_dir: Path to surrogate/ directory (for del_* types).
        del_threshold_pct: DEL increase threshold as fraction (e.g. 0.10 = 10%).
        del_penalty_type: "exponential" or "quadratic" (for del_* types).
    """
    if surrogate_type not in VALID_LOAD_SURROGATE_TYPES:
        raise ValueError(
            f"Unknown load_surrogate_type={surrogate_type!r}. "
            f"Valid: {VALID_LOAD_SURROGATE_TYPES}"
        )

    if surrogate_type.startswith("del_"):
        from pathlib import Path
        from helpers.surrogate_loads import TorchDELSurrogate
        model_dir = Path(del_model_dir)
        torch_del = TorchDELSurrogate.from_keras(
            model_dir / "models" / "ann_dlc12_out_wrot_Bl1Rad0FlpMnt_rank1.keras",
            model_dir / "scalers" / "scaler_input_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
            model_dir / "scalers" / "scaler_output_DLC12_wrot_Bl1Rad0FlpMnt.pkl",
        )
        mode = "per_turbine" if surrogate_type == "del_per_turbine" else "farm_max"
        return DELConstraintSurrogate(
            torch_del_model=torch_del,
            mode=mode,
            threshold_pct=del_threshold_pct,
            steepness=steepness,
            penalty_type=del_penalty_type,
            yaw_max_deg=yaw_max_deg,
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
