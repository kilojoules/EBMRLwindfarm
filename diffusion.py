"""
Diffusion-based EBM actor for wind farm control.

Replaces the Gaussian actor head in Transformer-SAC with a diffusion denoising
network. The transformer encoder backbone is identical — only the action head
changes from (fc_mean, fc_logstd) to an iterative denoiser.

Key capability: **post-hoc energy composition**. Train for power, then at
deployment add classifier guidance from a load surrogate to steer actions
toward safe regions, with zero retraining.

References:
    - Wang et al. 2022, "Diffusion Policies as an Expressive Policy Class
      for Offline Reinforcement Learning" (Diffusion-QL)
    - Ho et al. 2020, "Denoising Diffusion Probabilistic Models" (DDPM)
    - Song et al. 2020, "Denoising Diffusion Implicit Models" (DDIM)
    - Dhariwal & Nichol 2021, "Diffusion Models Beat GANs on Image Synthesis"
      (classifier guidance)
"""

import json
import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Callable, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import Args
from networks import (
    TransformerEncoder,
    create_positional_encoding,
    create_profile_encoding,
)


# =============================================================================
# NOISE SCHEDULES
# =============================================================================

def linear_beta_schedule(num_steps: int, beta_start: float = 0.0001, beta_end: float = 0.02) -> torch.Tensor:
    """Linear beta schedule from DDPM (Ho et al. 2020)."""
    return torch.linspace(beta_start, beta_end, num_steps)


def cosine_beta_schedule(num_steps: int, s: float = 0.008) -> torch.Tensor:
    """Cosine beta schedule from Nichol & Dhariwal 2021."""
    steps = torch.arange(num_steps + 1, dtype=torch.float64)
    f = torch.cos((steps / num_steps + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999).float()


class DDPMSchedule(nn.Module):
    """Precomputed DDPM noise schedule constants, stored as buffers."""

    def __init__(self, betas: torch.Tensor):
        super().__init__()
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.num_steps = len(betas)

        # Forward process q(x_t | x_0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))

        # Reverse process p(x_{t-1} | x_t)
        self.register_buffer("sqrt_recip_alphas", torch.sqrt(1.0 / alphas))
        self.register_buffer(
            "posterior_mean_coef1",
            betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod),
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("posterior_variance", posterior_variance)
        self.register_buffer("posterior_log_variance_clipped",
                             torch.log(posterior_variance.clamp(min=1e-20)))


# =============================================================================
# TIMESTEP EMBEDDING
# =============================================================================

def sinusoidal_timestep_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Sinusoidal positional embedding for diffusion timesteps.

    Args:
        timesteps: (batch,) integer timesteps
        dim: embedding dimension

    Returns:
        (batch, dim) embedding vectors
    """
    half_dim = dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# =============================================================================
# DENOISER MLP
# =============================================================================

class DenoisingMLP(nn.Module):
    """
    Per-turbine denoiser: predicts noise from (turbine_embedding, noisy_action, timestep_embedding).

    Shared weights across turbines to preserve permutation equivariance.
    """

    def __init__(
        self,
        embed_dim: int,
        action_dim: int,
        timestep_embed_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
    ):
        super().__init__()
        input_dim = embed_dim + action_dim + timestep_embed_dim
        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            layers.append(nn.Linear(in_d, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, action_dim))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        turbine_emb: torch.Tensor,
        noisy_action: torch.Tensor,
        timestep_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            turbine_emb: (batch, n_turbines, embed_dim)
            noisy_action: (batch, n_turbines, action_dim)
            timestep_emb: (batch, timestep_embed_dim) — broadcast to n_turbines

        Returns:
            (batch, n_turbines, action_dim) predicted noise
        """
        # Broadcast timestep embedding to all turbines
        t_emb = timestep_emb.unsqueeze(1).expand(-1, turbine_emb.shape[1], -1)
        x = torch.cat([turbine_emb, noisy_action, t_emb], dim=-1)
        return self.net(x)


# =============================================================================
# LOAD SURROGATE
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


# =============================================================================
# DIFFUSION ACTOR
# =============================================================================

class TransformerDiffusionActor(nn.Module):
    """
    Transformer-based diffusion actor for wind farm control.

    Architecture is identical to TransformerActor up to the transformer encoder.
    The Gaussian head (fc_mean, fc_logstd) is replaced with a diffusion denoiser
    that iteratively refines random noise into actions over K denoising steps.

    At inference, optional classifier guidance steers actions away from
    high-energy (unsafe) regions of a composed energy landscape.
    """

    def __init__(
        self,
        obs_dim_per_turbine: int,
        action_dim_per_turbine: int = 1,
        embed_dim: int = 128,
        pos_embed_dim: int = 32,
        num_heads: int = 4,
        num_layers: int = 2,
        mlp_ratio: float = 2.0,
        dropout: float = 0.0,
        action_scale: float = 1.0,
        action_bias: float = 0.0,
        # Positional encoding settings
        pos_encoding_type: str = "absolute_mlp",
        rel_pos_hidden_dim: int = 64,
        rel_pos_per_head: bool = True,
        pos_embedding_mode: str = "concat",
        # Profile encoding settings
        profile_encoding: Optional[str] = None,
        profile_encoder_hidden: int = 128,
        n_profile_directions: int = 360,
        profile_fusion_type: str = "add",
        profile_embed_mode: str = "add",
        # Shared profile encoders
        shared_recep_encoder: Optional[nn.Module] = None,
        shared_influence_encoder: Optional[nn.Module] = None,
        args: Optional[Args] = None,
        # Diffusion settings
        num_diffusion_steps: int = 20,
        num_inference_steps: int = 5,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        timestep_embed_dim: int = 64,
        denoiser_hidden_dim: int = 256,
        denoiser_num_layers: int = 3,
        clip_denoised: bool = True,
    ):
        super().__init__()

        self.obs_dim_per_turbine = obs_dim_per_turbine
        self.action_dim_per_turbine = action_dim_per_turbine
        self.embed_dim = embed_dim
        self.pos_encoding_type = pos_encoding_type
        self.num_diffusion_steps = num_diffusion_steps
        self.num_inference_steps = num_inference_steps
        self.clip_denoised = clip_denoised

        self.profile_encoding = profile_encoding
        self.profile_fusion_type = profile_fusion_type
        self.profile_embed_mode = profile_embed_mode

        assert profile_fusion_type in ("add", "joint")
        assert profile_embed_mode in ("add", "concat")

        # === Positional encoding (identical to TransformerActor) ===
        self.pos_encoder, self.rel_pos_bias, self.embedding_mode = \
            create_positional_encoding(
                encoding_type=pos_encoding_type,
                embed_dim=embed_dim,
                pos_embed_dim=pos_embed_dim,
                num_heads=num_heads,
                rel_pos_hidden_dim=rel_pos_hidden_dim,
                rel_pos_per_head=rel_pos_per_head,
                embedding_mode=pos_embedding_mode,
            )

        # === Profile encoding (identical to TransformerActor) ===
        if shared_recep_encoder is not None and shared_influence_encoder is not None:
            self.recep_encoder = shared_recep_encoder
            self.influence_encoder = shared_influence_encoder
        else:
            encoder_kwargs = json.loads(args.profile_encoder_kwargs) if args else {}
            self.recep_encoder, self.influence_encoder = \
                create_profile_encoding(
                    profile_type=profile_encoding,
                    embed_dim=embed_dim,
                    hidden_channels=profile_encoder_hidden,
                    **encoder_kwargs,
                )

        if profile_encoding is not None and profile_fusion_type == "joint":
            self.profile_fusion = nn.Linear(2 * embed_dim, embed_dim)

        if profile_encoding is not None and profile_embed_mode == "concat":
            self.profile_proj = nn.Linear(2 * embed_dim, embed_dim)

        # === Observation encoder (identical to TransformerActor) ===
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim_per_turbine, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # === Input projection (identical to TransformerActor) ===
        if self.embedding_mode == "concat":
            self.input_proj = nn.Linear(embed_dim + pos_embed_dim, embed_dim)
        else:
            self.input_proj = nn.Identity()

        # === Transformer encoder (identical to TransformerActor) ===
        self.transformer = TransformerEncoder(
            embed_dim, num_heads, num_layers, mlp_ratio, dropout
        )

        # === Diffusion-specific components ===
        betas = linear_beta_schedule(num_diffusion_steps, beta_start, beta_end)
        self.schedule = DDPMSchedule(betas)

        self.denoiser = DenoisingMLP(
            embed_dim=embed_dim,
            action_dim=action_dim_per_turbine,
            timestep_embed_dim=timestep_embed_dim,
            hidden_dim=denoiser_hidden_dim,
            num_layers=denoiser_num_layers,
        )
        self.timestep_embed_dim = timestep_embed_dim

        # Action scaling (identical to TransformerActor)
        self.register_buffer("action_scale", torch.tensor(action_scale, dtype=torch.float32))
        self.register_buffer("action_bias_val", torch.tensor(action_bias, dtype=torch.float32))

    # -----------------------------------------------------------------
    # Encoder (shared with critic at inference; called once per step)
    # -----------------------------------------------------------------

    def encode(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Run transformer encoder, return per-turbine embeddings.

        This is factored out so embeddings are computed once and reused
        across all K denoising steps (avoids K redundant transformer passes).

        Returns:
            h: (batch, n_turbines, embed_dim)
        """
        h = self.obs_encoder(obs)

        if self.embedding_mode == "concat" and self.pos_encoder is not None:
            pos_embed = self.pos_encoder(positions)
            h = torch.cat([h, pos_embed], dim=-1)
        elif self.embedding_mode == "add" and self.pos_encoder is not None:
            pos_embed = self.pos_encoder(positions)
            h = h + pos_embed

        h = self.input_proj(h)

        if self.recep_encoder and recep_profile is not None and influence_profile is not None:
            recep_embed = self.recep_encoder(recep_profile)
            influence_embed = self.influence_encoder(influence_profile)
            if self.profile_fusion_type == "joint":
                profile_embed = self.profile_fusion(
                    torch.cat([recep_embed, influence_embed], dim=-1)
                )
            else:
                profile_embed = recep_embed + influence_embed

            if self.profile_embed_mode == "concat":
                h = self.profile_proj(torch.cat([h, profile_embed], dim=-1))
            else:
                h = h + profile_embed

        attn_bias = None
        if self.rel_pos_bias is not None:
            attn_bias = self.rel_pos_bias(positions, key_padding_mask)

        h, _ = self.transformer(h, key_padding_mask, attn_bias, need_weights=False)
        return h

    # -----------------------------------------------------------------
    # Forward diffusion (add noise)
    # -----------------------------------------------------------------

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: add noise to clean actions.

        q(x_t | x_0) = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t]
        # Reshape for broadcasting: (batch,) -> (batch, 1, 1)
        while sqrt_alpha.dim() < x_0.dim():
            sqrt_alpha = sqrt_alpha.unsqueeze(-1)
            sqrt_one_minus_alpha = sqrt_one_minus_alpha.unsqueeze(-1)
        return sqrt_alpha * x_0 + sqrt_one_minus_alpha * noise

    # -----------------------------------------------------------------
    # Noise prediction
    # -----------------------------------------------------------------

    def predict_noise(
        self,
        turbine_emb: torch.Tensor,
        noisy_action: torch.Tensor,
        t: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise for a single denoising step."""
        t_emb = sinusoidal_timestep_embedding(t, self.timestep_embed_dim)
        eps_pred = self.denoiser(turbine_emb, noisy_action, t_emb)
        # Zero out predictions for padded turbines
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            eps_pred = eps_pred * mask
        return eps_pred

    # -----------------------------------------------------------------
    # Reverse diffusion (denoise)
    # -----------------------------------------------------------------

    def _predict_x0_from_eps(self, x_t: torch.Tensor, t: int, eps: torch.Tensor) -> torch.Tensor:
        """Predict x_0 from x_t and predicted noise (for DDIM and clipping)."""
        sqrt_alpha = self.schedule.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha = self.schedule.sqrt_one_minus_alphas_cumprod[t]
        return (x_t - sqrt_one_minus_alpha * eps) / sqrt_alpha

    def _ddpm_step(
        self, x_t: torch.Tensor, t: int, eps_pred: torch.Tensor, add_noise: bool = True,
    ) -> torch.Tensor:
        """Single DDPM reverse step: p(x_{t-1} | x_t)."""
        # Predict x_0
        x_0_pred = self._predict_x0_from_eps(x_t, t, eps_pred)
        if self.clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        # Compute posterior mean
        mean = (self.schedule.posterior_mean_coef1[t] * x_0_pred
                + self.schedule.posterior_mean_coef2[t] * x_t)

        if t == 0 or not add_noise:
            return mean

        noise = torch.randn_like(x_t)
        variance = torch.exp(0.5 * self.schedule.posterior_log_variance_clipped[t])
        return mean + variance * noise

    def _ddim_step(
        self, x_t: torch.Tensor, t: int, t_prev: int, eps_pred: torch.Tensor, eta: float = 0.0,
    ) -> torch.Tensor:
        """Single DDIM reverse step (eta=0 for deterministic)."""
        alpha_t = self.schedule.alphas_cumprod[t]
        alpha_prev = self.schedule.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Predict x_0
        x_0_pred = self._predict_x0_from_eps(x_t, t, eps_pred)
        if self.clip_denoised:
            x_0_pred = x_0_pred.clamp(-1, 1)

        # DDIM formula
        sigma = eta * torch.sqrt((1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev))
        dir_xt = torch.sqrt(1 - alpha_prev - sigma ** 2) * eps_pred
        x_prev = torch.sqrt(alpha_prev) * x_0_pred + dir_xt

        if eta > 0:
            x_prev = x_prev + sigma * torch.randn_like(x_t)

        return x_prev

    def _get_ddim_timesteps(self) -> List[int]:
        """Select evenly-spaced timesteps for DDIM acceleration."""
        step_size = self.num_diffusion_steps // self.num_inference_steps
        timesteps = list(range(0, self.num_diffusion_steps, step_size))
        return timesteps

    def denoise_chain(
        self,
        turbine_emb: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        use_ddim: bool = False,
    ) -> torch.Tensor:
        """
        Full reverse diffusion chain: x_T ~ N(0, I) -> x_0.

        For training: DDPM with full steps (differentiable).
        For inference: DDIM with fewer steps (faster).

        Returns:
            x_0: (batch, n_turbines, action_dim) clean actions in [-1, 1]
        """
        batch, n_turb, _ = turbine_emb.shape
        device = turbine_emb.device

        x_t = torch.randn(batch, n_turb, self.action_dim_per_turbine, device=device)

        # Padding mask for zeroing out padded turbines
        mask = None
        if key_padding_mask is not None:
            mask = (~key_padding_mask).unsqueeze(-1).float()
            x_t = x_t * mask

        if use_ddim:
            ddim_timesteps = self._get_ddim_timesteps()
            for i in reversed(range(len(ddim_timesteps))):
                t = ddim_timesteps[i]
                t_prev = ddim_timesteps[i - 1] if i > 0 else -1
                t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
                eps_pred = self.predict_noise(turbine_emb, x_t, t_batch, key_padding_mask)
                x_t = self._ddim_step(x_t, t, t_prev, eps_pred, eta=0.0)
                if mask is not None:
                    x_t = x_t * mask
        else:
            for t in reversed(range(self.num_diffusion_steps)):
                t_batch = torch.full((batch,), t, device=device, dtype=torch.long)
                eps_pred = self.predict_noise(turbine_emb, x_t, t_batch, key_padding_mask)
                x_t = self._ddpm_step(x_t, t, eps_pred, add_noise=(t > 0))
                if mask is not None:
                    x_t = x_t * mask

        if self.clip_denoised:
            x_t = x_t.clamp(-1, 1)

        return x_t

    def denoise_with_guidance(
        self,
        turbine_emb: torch.Tensor,
        guidance_fn: Callable,
        guidance_scale: float = 1.0,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Reverse diffusion with classifier guidance for safety composition.

        At each denoising step, the gradient of the guidance energy is
        subtracted from the action, steering sampling away from high-load regions.

        E_total(s, a) = E_power(s, a) + lambda * E_load(a)
        """
        batch, n_turb, _ = turbine_emb.shape
        device = turbine_emb.device

        x_t = torch.randn(batch, n_turb, self.action_dim_per_turbine, device=device)

        pad_mask = None
        if key_padding_mask is not None:
            pad_mask = (~key_padding_mask).unsqueeze(-1).float()
            x_t = x_t * pad_mask

        # Use DDIM for guided inference (faster)
        ddim_timesteps = self._get_ddim_timesteps()
        for i in reversed(range(len(ddim_timesteps))):
            t = ddim_timesteps[i]
            t_prev = ddim_timesteps[i - 1] if i > 0 else -1
            t_batch = torch.full((batch,), t, device=device, dtype=torch.long)

            # Compute guidance gradient (enable_grad needed since caller may use no_grad)
            with torch.enable_grad():
                x_t_grad = x_t.detach().requires_grad_(True)
                action_scaled = x_t_grad * self.action_scale + self.action_bias_val
                energy = guidance_fn(action_scaled, key_padding_mask)
                grad = torch.autograd.grad(energy.sum(), x_t_grad)[0]
            x_t = x_t.detach()

            # Normal denoising step
            eps_pred = self.predict_noise(turbine_emb, x_t, t_batch, key_padding_mask)
            x_t = self._ddim_step(x_t, t, t_prev, eps_pred, eta=0.0)

            # Apply guidance (gradient descent on load energy)
            x_t = x_t - guidance_scale * grad

            if pad_mask is not None:
                x_t = x_t * pad_mask

        if self.clip_denoised:
            x_t = x_t.clamp(-1, 1)

        return x_t

    # -----------------------------------------------------------------
    # Action interface (drop-in replacement for TransformerActor)
    # -----------------------------------------------------------------

    def get_action(
        self,
        obs: torch.Tensor,
        positions: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        deterministic: bool = False,
        recep_profile: Optional[torch.Tensor] = None,
        influence_profile: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        guidance_fn: Optional[Callable] = None,
        guidance_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Sample action via diffusion denoising.

        Drop-in replacement for TransformerActor.get_action(). Returns the same
        tuple format: (action, log_prob, mean_action, attn_weights).

        Note: log_prob is returned as zeros — diffusion has no closed-form
        log probability. This is fine because we drop the entropy term.
        """
        turbine_emb = self.encode(
            obs, positions, key_padding_mask, recep_profile, influence_profile
        )

        # Choose denoising strategy
        if guidance_fn is not None and guidance_scale > 0:
            action_raw = self.denoise_with_guidance(
                turbine_emb, guidance_fn, guidance_scale, key_padding_mask
            )
        else:
            # Use DDIM at inference (eval mode), DDPM at training
            use_ddim = not self.training
            action_raw = self.denoise_chain(turbine_emb, key_padding_mask, use_ddim=use_ddim)

        # Scale to action space: [-1, 1] -> [action_bias - action_scale, action_bias + action_scale]
        action = action_raw * self.action_scale + self.action_bias_val

        batch = obs.shape[0]
        log_prob = torch.zeros(batch, 1, device=obs.device)
        mean_action = action  # No separate "mean" for diffusion
        attn_weights: List[torch.Tensor] = []

        return action, log_prob, mean_action, attn_weights
