"""
Configuration dataclass for Transformer-SAC wind farm training.

All CLI arguments are defined here via a tyro-compatible dataclass.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class Args:
    """Command-line arguments for training."""

    # === Experiment Settings ===
    config: str = "default"  # Environment config preset
    exp_name: str = "transformer_sac_windfarm"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = True  # Enable wandb tracking
    wandb_project_name: str = "transformer_windfarm"
    wandb_entity: Optional[str] = None
    save_model: bool = True
    save_interval: int = 10000
    log_image: bool = False  # Log attention images to TensorBoard

    shuffle_turbs: bool = False  # Shuffle turbine order in obs/action
    max_episode_steps: Optional[int] = 100 # Max steps per episode (None = use env default)

    # === Receptivity Profile Settings ===
    profile_encoder_kwargs: str = "{}"  # JSON string of encoder-specific kwargs
    profile_source: str = "geometric"  # "pywake" or "geometric"
    profile_encoding_type: Optional[str] = None  # Now Optional, use None for no pos encoding
    profile_encoder_hidden: int = 128       # Hidden dim in profile encoder MLP
    rotate_profiles: bool = True            # Rotate profiles to wind-relative frame
    n_profile_directions: int = 360         # Number of directions in profile
    profile_fusion_type: str = "add"       # "add" or "joint" fusion of receptivity and influence profiles
    profile_embed_mode: str = "add"        # "add" or "concat" — how fused profile is integrated into token embedding
    share_profile_encoder: bool = False         # Whether to share weights between actor and critic for profile encoder

    # === Environment Settings ===
    turbtype: str = "DTU10MW"  # Wind turbine type
    TI_type: str = "Random"   # Turbulence intensity sampling
    dt_sim: int = 1           # Simulation timestep (seconds)
    dt_env: int = 1          # Environment timestep (seconds)
    yaw_step: float = 0.5     # Max yaw change per sim step (degrees)
    max_eps: int = 20         # Number of flow passthroughs per episode
    num_envs: int = 1         # Number of parallel environments
    wind_timeseries_csv: Optional[str] = None   # Path to wind time series CSV (overrides random wind sampling)
    wind_timeseries_random_start: bool = False  # Random start position in time series each episode

    # === Evaluation Settings ===
    eval_interval: int = 50000        # How often to evaluate (in env steps)
    eval_initial: bool = False        # Run evaluation before training starts
    num_eval_steps: int = 200         # Number of steps per evaluation episode
    viz_every_n_evals: int = 5        # Log visualization figures every N evaluations (0 = disabled)
    num_eval_episodes: int = 1        # Number of episodes per evaluation
    eval_layouts: str = ""            # Comma-separated eval layouts (empty = use training layouts)
    eval_seed: int = 42               # Seed for evaluation environments

    # === Layout Settings ===
    # Comma-separated list of layouts. Single = single-layout, Multiple = multi-layout
    layouts: str = "test_layout"  # e.g., "square_1,square_2,circular_1"

    # === Observation Settings ===
    history_length: int = 2            # Number of timesteps of history per feature
    use_wd_deviation: bool = False      # If True, convert WD to deviation from mean
    use_wind_relative_pos: bool = True  # Transform positions to wind-relative frame
    wd_scale_range: float = 90.0        # Only used if use_wd_deviation=True. Wind direction deviation range for scaling (±degrees → [-1,1])

    # === Transformer Architecture ===
    embed_dim: int = 128          # Transformer hidden dimension
    num_heads: int = 4            # Number of attention heads
    num_layers: int = 2           # Number of transformer layers
    mlp_ratio: float = 2.0        # FFN hidden dim = embed_dim * mlp_ratio
    dropout: float = 0.0          # Dropout rate (0 for RL typically)
    pos_embed_dim: int = 32       # Dimension for positional encoding


    # === Positional Encoding Settings ===
    # Options: "absolute_mlp", "relative_mlp", "relative_mlp_shared",
    #          "sinusoidal_2d",
    pos_encoding_type: Optional[str] = "relative_mlp"  # Relative pos bias in attention. Options: None, "absolute_mlp", "relative_mlp", "relative_mlp_shared", "sinusoidal_2d", etc.
    # For relative encoding: number of hidden units in the bias MLP
    rel_pos_hidden_dim: int = 64
    # For relative encoding: whether to use separate bias per head
    rel_pos_per_head: bool = True
    pos_embedding_mode: str = "concat"  # "add" or "concat" positional embedding to token (only for absolute types)

    # === Algorithm Selection ===
    algorithm: str = "sac"  # "sac" or "tqc"
    use_droq: bool = False  # Enable DroQ regularization (dropout + LayerNorm in critic MLPs)

    # === TQC Hyperparameters (only used when algorithm="tqc") ===
    tqc_n_critics: int = 5               # Number of critic networks
    tqc_n_quantiles: int = 25            # Quantiles per critic
    tqc_top_quantiles_to_drop: int = 2   # Truncation: drop top-d per-sample quantiles

    # === DroQ Hyperparameters (only used when use_droq=True) ===
    droq_dropout: float = 0.01           # Dropout rate for DroQ critic MLPs
    droq_layer_norm: bool = True         # LayerNorm in DroQ critic MLPs

    # === SAC Hyperparameters ===
    utd_ratio: float = 1.0           # Update-to-data ratio
    total_timesteps: int = 100_000
    buffer_size: int = int(1e6)
    gamma: float = 0.99           # Discount factor
    tau: float = 0.005            # Target network update rate
    batch_size: int = 256
    learning_starts: int = 5000   # Steps before training starts
    policy_lr: float = 3e-4
    q_lr: float = 3e-4
    policy_frequency: int = 2     # Policy update frequency
    target_network_frequency: int = 1
    alpha: float = 0.2            # Initial entropy coefficient
    autotune: bool = True         # Auto-tune entropy coefficient

    # === Diffusion Actor Settings ===
    actor_type: str = "gaussian"          # "gaussian", "diffusion", or "ebt"
    num_diffusion_steps: int = 20         # DDPM steps for training
    num_inference_steps: int = 10         # DDIM steps at inference
    beta_start: float = 0.0001           # Linear beta schedule start
    beta_end: float = 0.02               # Linear beta schedule end
    timestep_embed_dim: int = 64         # Sinusoidal timestep embedding dim
    denoiser_hidden_dim: int = 256       # Hidden dim of denoiser MLP
    denoiser_num_layers: int = 3         # Number of layers in denoiser MLP
    diffusion_bc_weight: float = 0.0     # Weight of diffusion BC loss (0 = pure Q-guidance)
    guidance_scale: float = 0.0          # Classifier guidance lambda at inference (0 = off)
    noise_schedule: str = "linear"        # "linear" or "cosine" beta schedule
    cosine_schedule_s: float = 0.008      # Cosine schedule offset (only used if noise_schedule="cosine")

    # === BC Weight Annealing ===
    bc_weight_start: float = 0.0          # Initial BC weight (0 = use fixed diffusion_bc_weight)
    bc_weight_end: float = 0.0            # Final BC weight after annealing
    bc_anneal_steps: int = 50000          # Steps over which to anneal BC weight
    bc_anneal_type: str = "linear"        # "linear" or "cosine" annealing curve

    # === EBT Actor Settings ===
    ebt_energy_hidden_dim: int = 256      # Hidden dim of energy MLP
    ebt_energy_num_layers: int = 3        # Number of layers in energy MLP
    ebt_opt_steps_train: int = 3          # Optimization steps during training (short for stability)
    ebt_opt_steps_eval: int = 10          # Optimization steps at inference ("think longer")
    ebt_opt_lr: float = 0.1              # Step size α for energy gradient descent
    ebt_num_candidates: int = 8           # M candidates for self-verification at inference
    ebt_langevin_noise: float = 0.01     # Noise scale for Langevin dynamics during training
    ebt_random_steps: bool = True         # Randomize optimization step count (regularization)
    ebt_random_lr: bool = True            # Randomize step size (regularization)
    ebt_energy_reg: float = 0.0           # Energy magnitude regularization weight
    load_steepness: float = 10.0          # Steepness k for exponential load surrogate wall

    # === Constraint Surrogates ===
    travel_budget_deg: float = 100.0          # Yaw travel budget per turbine (degrees over window)
    travel_budget_window: int = 100           # Rolling window size (env steps)
    travel_budget_steepness: float = 5.0      # Exponential wall steepness for travel budget
    per_turbine_thresholds: str = ""          # Comma-separated per-turbine yaw limits in degrees (for per_turbine surrogate)
    load_surrogate_type: str = "exponential"  # exponential, threshold, per_turbine, t1_positive_only, neg_yaw_budget, relu, del_per_turbine, del_farm_max
    load_del_threshold_pct: float = 0.10       # DEL increase threshold (0.10 = 10% max)
    load_del_penalty_type: str = "exponential" # DEL penalty shape: "exponential" or "quadratic"

    # === Negative Yaw Budget (Almgren-Chriss) ===
    neg_yaw_budget_hours: float = 5.0          # Negative yaw time budget per turbine (hours)
    neg_yaw_horizon_hours: float = 8760.0      # Planning horizon (hours, 8760 = 1 year)
    neg_yaw_risk_aversion: float = 1.0         # AC risk aversion (0=uniform, higher=concentrate spending)
    neg_yaw_threshold_deg: float = 0.0         # Below this = "negative yaw" (degrees)

    # === Action Regularization (for delta actions) ===
    action_reg_weight: float = 0.0        # L2 penalty on action magnitude (encourages staying put)

    # === Learning Rate Warmup ===
    lr_warmup_steps: int = 0              # Linear LR warmup steps (0 = disabled)

    # === Gradient Clipping ===
    grad_clip: bool = True
    grad_clip_max_norm: float = 1.0

    # === Fine-tuning / Resume Settings ===
    resume_checkpoint: Optional[str] = None  # Path to checkpoint .pt file for fine-tuning or resuming
    finetune_reset_actor_optimizer: int = 0     # If True, reset optimizers for fresh fine-tuning. If False, resume optimizer states too.
    finetune_reset_critic_optimizer: int = 0    # If True, reset optimizers for fresh fine-tuning. If False, resume optimizer states too.
    finetune_reset_alpha: int = 0               # If True, reset entropy coefficient. If False, keep from checkpoint.

    # === Initial Exploration Mode ===
    initial_exploration: str = "random"  # "random" = sample from action space, "policy" = use actor network (useful when resuming from checkpoint)

    # === Pretrained Encoder Loading ===
    pretrain_checkpoint: Optional[str] = None   # Path to pretrained encoder .pt from pretrain_power.py
    pretrain_freeze_steps: int = 0             # Freeze encoder for this many env steps (0 = no freeze)

    # === Action Settings ===
    action_type: str = "yaw"   # "wind" (target setpoint) or "yaw" (delta). Overridden by BC checkpoint if provided.
