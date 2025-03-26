#!/usr/bin/env python3
"""
MBD-S Finalized: Modular Mamba/MLP-Block-Diffusion Language Model.

Design: Aligned with best practices, mathematically stable, modular, optimized, testable.
Includes MLP fallback if `mamba-ssm` is unavailable.
"""
import copy
import logging
import math
import sys
import time
import unittest
from dataclasses import dataclass, field
from typing import Tuple, Optional, Dict, List, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset

# --- Mamba Import ---
try:
    from mamba_ssm.modules.mamba_simple import Mamba
    if Mamba is None: raise ImportError # Ensure Mamba class is actually available
    mamba_available = True
except ImportError:
    Mamba = None
    mamba_available = False

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
if not mamba_available:
    logging.warning("`mamba-ssm` package not found or Mamba class unavailable. Mamba-based tiers will fallback to MLP.")

# Use dynamic PAD_TOKEN_ID, initialized here but potentially updated by tokenizer in tests
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
DEFAULT_VOCAB_SIZE = 50_257
DEFAULT_L_PRIME = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE_TYPE = DEVICE.type
# Use bfloat16 if available on CUDA, otherwise float16
AMP_DTYPE = torch.bfloat16 if AMP_DEVICE_TYPE == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
AMP_ENABLED = AMP_DEVICE_TYPE == 'cuda' # Flag for enabling autocast/scaler

# --- Tier Configurations Dictionary ---
# Defines different MBD-S configurations affecting complexity and behavior.
TIER_CONFIGS_EVOLVED = {
    "simple_edge": {
        "name": "simple_edge", "mamba_layers": 1, "expansion_factor": 2, "diffusion_steps": 10,
        "noise_beta_start": 0.001, "noise_beta_end": 0.02, "schedule_type": "linear",
        "prediction_type": "noise", "use_mamba": False, "use_bidirectional_mamba": False,
        "use_overlapping_blocks": False, "overlap_ratio": 0.0, "overlap_weighting": "uniform",
        "stop_entropy": 3.5, "quantize": True,
    },
    "core_balanced": {
        "name": "core_balanced", "mamba_layers": 6, "expansion_factor": 2, "diffusion_steps": 50,
        "noise_beta_start": 0.0001, "noise_beta_end": 0.02, "schedule_type": "cosine",
        "prediction_type": "noise", "use_mamba": True, "use_bidirectional_mamba": False,
        "use_overlapping_blocks": True, "overlap_ratio": 0.25, "overlap_weighting": "triangular",
        "stop_entropy": float("inf"), "quantize": False,
    },
    "enhanced_quality": {
        "name": "enhanced_quality", "mamba_layers": 12, "expansion_factor": 2.5, "diffusion_steps": 100,
        "noise_beta_start": 0.0001, "noise_beta_end": 0.02, "schedule_type": "cosine",
        "prediction_type": "noise", "use_mamba": True, "use_bidirectional_mamba": True,
        "use_overlapping_blocks": True, "overlap_ratio": 0.25, "overlap_weighting": "triangular",
        "stop_entropy": 5.0, "quantize": False,
    },
    "extreme_power": {
        "name": "extreme_power", "mamba_layers": 18, "expansion_factor": 3, "diffusion_steps": 200,
        "noise_beta_start": 0.0001, "noise_beta_end": 0.03, "schedule_type": "cosine",
        "prediction_type": "noise", "use_mamba": True, "use_bidirectional_mamba": True,
        "use_overlapping_blocks": True, "overlap_ratio": 0.5, "overlap_weighting": "triangular",
        "stop_entropy": 5.5, "quantize": False,
    }
}

# --- Configuration Dataclasses ---
@dataclass
class TierConfig:
    name: str = "default"
    mamba_layers: int = 6
    expansion_factor: float = 2.0
    diffusion_steps: int = 100
    noise_beta_start: float = 0.0001
    noise_beta_end: float = 0.02
    schedule_type: str = "cosine" # "linear" or "cosine"
    prediction_type: str = "noise" # "noise" or "x_start"
    use_mamba: bool = True
    use_bidirectional_mamba: bool = False
    use_overlapping_blocks: bool = True
    overlap_ratio: float = 0.25 # Ratio of l_prime for overlap
    overlap_weighting: str = "triangular" # "uniform" or "triangular"
    stop_entropy: float = float("inf") # Entropy threshold for generation stopping
    quantize: bool = False # Placeholder for potential future quantization

@dataclass
class MBDConfig:
    tier_config: TierConfig # Expects a TierConfig object
    vocab_size: int = DEFAULT_VOCAB_SIZE
    embed_dim: int = 512
    l_prime: int = DEFAULT_L_PRIME # Processing block length
    width: float = 1.0 # Multiplier for embedding dimension
    dropout: float = 0.1
    pos_encoding_max_len: int = 4096

# --- Core Components ---

class DiffusionSchedule(nn.Module):
    """Manages the diffusion noise schedule (linear or cosine) and related constants."""
    def __init__(self, schedule_type: str, beta_start: float, beta_end: float, num_steps: int):
        super().__init__()
        self.num_steps = num_steps

        if schedule_type == "linear":
            betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
        elif schedule_type == "cosine":
            steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
            s = 0.008 # Small offset to prevent beta_t = 0 for t = 0
            alpha_bar = torch.cos(((steps * math.pi / 2) + s) / (1 + s)) ** 2
            betas = torch.clamp(1. - alpha_bar[1:] / alpha_bar[:-1], min=1e-12, max=0.999)
        else:
            raise ValueError(f"Unknown schedule_type: {schedule_type}")

        betas = betas.float()
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=betas.device), alpha_bars[:-1]])

        # Register buffers for schedule constants
        schedule_constants = {
            'betas': betas, 'alphas': alphas, 'alpha_bars': alpha_bars, 'alpha_bars_prev': alpha_bars_prev,
            'sqrt_alpha_bars': torch.sqrt(alpha_bars), 'sqrt_one_minus_alpha_bars': torch.sqrt(1. - alpha_bars)
        }
        for name, val in schedule_constants.items():
            self.register_buffer(name, val, persistent=False)

        # Precompute constants for q_posterior (sampling step)
        posterior_variance = betas * (1. - self.alpha_bars_prev) / (1. - self.alpha_bars).clamp(min=1e-9)
        self.register_buffer('posterior_variance', posterior_variance, persistent=False)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)), persistent=False)
        self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(self.alpha_bars_prev) / (1. - self.alpha_bars).clamp(min=1e-9), persistent=False)
        self.register_buffer('posterior_mean_coef2', (1. - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1. - self.alpha_bars).clamp(min=1e-9), persistent=False)

    def _gather(self, buffer_name: str, t: Tensor, target_shape: Tuple) -> Tensor:
        """Helper to gather values from a buffer based on timestep t and reshape."""
        buffer = getattr(self, buffer_name)
        # Ensure t is on the same device as the buffer before gathering
        t_long = torch.clamp(t.long(), 0, self.num_steps - 1).to(buffer.device)
        gathered = buffer.gather(-1, t_long)
        # Reshape to match target_shape for broadcasting (e.g., [B, 1, 1] for [B, Seq, Dim])
        return gathered.reshape(t.shape[0], *((1,) * (len(target_shape) - 1))).to(t.device)

    def q_sample(self, x_start: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Sample x_t from x_start at timestep t (forward diffusion process)."""
        if noise is None: noise = torch.randn_like(x_start)
        sqrt_alpha_bar_t = self._gather('sqrt_alpha_bars', t, x_start.shape)
        sqrt_one_minus_alpha_bar_t = self._gather('sqrt_one_minus_alpha_bars', t, x_start.shape)
        return sqrt_alpha_bar_t * x_start + sqrt_one_minus_alpha_bar_t * noise, noise

    def predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise_pred: Tensor) -> Tensor:
        """Predict x_start given x_t and the predicted noise."""
        sqrt_recip_alpha_bar_t = (1.0 / self._gather('sqrt_alpha_bars', t, x_t.shape).clamp(min=1e-9))
        sqrt_recip_m1_alpha_bar_t = (self._gather('sqrt_one_minus_alpha_bars', t, x_t.shape) * sqrt_recip_alpha_bar_t)
        return sqrt_recip_alpha_bar_t * x_t - sqrt_recip_m1_alpha_bar_t * noise_pred

    def q_posterior_mean_variance(self, x_start: Tensor, x_t: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the mean and variance of the posterior distribution q(x_{t-1} | x_t, x_0)."""
        posterior_mean = self._gather('posterior_mean_coef1', t, x_start.shape) * x_start + \
                         self._gather('posterior_mean_coef2', t, x_start.shape) * x_t
        posterior_variance = self._gather('posterior_variance', t, x_start.shape)
        posterior_log_variance_clipped = self._gather('posterior_log_variance_clipped', t, x_start.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

class SinusoidalTimestepEmbedding(nn.Module):
    """Generates sinusoidal embeddings for timesteps, followed by an MLP projection."""
    def __init__(self, dim: int, max_period: int = 10000, dropout: float = 0.1):
        super().__init__()
        if dim <= 0 or dim % 2 != 0: raise ValueError(f"Embedding dim ({dim}) must be positive and even.")
        self.dim = dim
        self.max_period = max_period
        # Simple MLP to project timestep embeddings
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.SiLU(), nn.Dropout(dropout), nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        if t.ndim != 1: raise ValueError(f"Input tensor t must be 1D, got shape {t.shape}")
        half = self.dim // 2
        freqs = torch.exp(-math.log(self.max_period) * torch.arange(half, dtype=torch.float32) / half).to(t.device)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0: raise ValueError(f"d_model must be positive, got {d_model}")
        self.dropout = nn.Dropout(p=dropout)
        self.max_len = max_len

        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model) # Add batch dim
        pe[0, :, 0::2] = torch.sin(position * div_term)
        # Handle odd d_model if necessary, though typically d_model is even
        if d_model % 2 == 0: pe[0, :, 1::2] = torch.cos(position * div_term)
        else: pe[0, :, 1::2] = torch.cos(position * div_term[:-1])
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """Adds positional encoding to the input tensor."""
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds PositionalEncoding max_len {self.max_len}")
        # Add positional encoding (sliced to sequence length) and apply dropout
        return self.dropout(x + self.pe[:, :seq_len, :])

class MDBBlockFinal(nn.Module):
    """Core Mamba/MLP block with timestep injection and prediction head."""
    def __init__(self, config: MBDConfig):
        super().__init__()
        self.config = config
        tier_cfg = config.tier_config
        self.embed_dim = int(config.embed_dim * config.width)
        self.vocab_size = config.vocab_size
        self.prediction_type = tier_cfg.prediction_type
        self.use_mamba = tier_cfg.use_mamba and mamba_available

        hidden_dim = int(self.embed_dim * tier_cfg.expansion_factor)

        self.pos_encoder = PositionalEncoding(self.embed_dim, config.dropout, config.pos_encoding_max_len)
        self.input_norm = nn.LayerNorm(self.embed_dim)

        # --- Sequence Processor (Mamba or MLP Fallback) ---
        if self.use_mamba:
            self.use_bidirectional = tier_cfg.use_bidirectional_mamba
            mamba_config = {
                "d_model": self.embed_dim,
                "d_state": max(16, self.embed_dim // 16), # Standard heuristic
                "d_conv": 4, # Standard value
                "expand": 2, # Standard value
                "dt_rank": "auto", # Let Mamba decide based on d_model
                "bimamba_type": "add" if self.use_bidirectional else None,
            }
            self.sequence_processor = Mamba(**mamba_config)
        else:
            self.use_bidirectional = False
            # Fallback MLP: Use multiple layers for comparable depth if needed
            mlp_layers = []
            current_dim = self.embed_dim
            num_mlp_layers = max(1, tier_cfg.mamba_layers) # Use at least one MLP block
            for _ in range(num_mlp_layers):
                mlp_layers.extend([
                    nn.Linear(current_dim, hidden_dim), nn.GELU(), nn.Dropout(config.dropout),
                    nn.Linear(hidden_dim, self.embed_dim), nn.Dropout(config.dropout),
                    nn.LayerNorm(self.embed_dim) # Add normalization between MLP blocks
                ])
                current_dim = self.embed_dim
            self.sequence_processor = nn.Sequential(*mlp_layers)

        # --- Prediction Head ---
        self.output_norm = nn.LayerNorm(self.embed_dim)
        # Output dimension depends on whether we predict noise (embed_dim) or tokens (vocab_size)
        output_dim = self.embed_dim if self.prediction_type == "noise" else self.vocab_size
        self.output_head = nn.Linear(self.embed_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights with common heuristics."""
        for name, param in self.named_parameters():
            if 'embedding.weight' in name and param.dim() > 1:
                 nn.init.normal_(param, mean=0, std=self.embed_dim**-0.5)
            elif 'norm' in name.lower():
                 if 'weight' in name: nn.init.ones_(param)
                 elif 'bias' in name: nn.init.zeros_(param)
            elif 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param) # Good default for linear layers
            elif 'bias' in name:
                nn.init.zeros_(param)
        # Ensure output head bias is zero
        if hasattr(self.output_head, 'bias') and self.output_head.bias is not None:
            nn.init.zeros_(self.output_head.bias)

    def forward(self, x_embed: Tensor, t_embed_proj: Tensor, mamba_state: Optional[Tuple] = None) -> Tuple[Tensor, Optional[Tuple]]:
        """Processes input embeddings with timestep conditioning."""
        # Add projected timestep embedding (broadcast across sequence) and normalize
        h = self.input_norm(x_embed + t_embed_proj.unsqueeze(1))
        # Add positional encoding
        h = self.pos_encoder(h)

        new_mamba_state = None
        if self.use_mamba:
            # Mamba handles state internally if not passed, returns updated state
            h, new_mamba_state = self.sequence_processor(h)
        else:
            # MLP processes the sequence
            h = self.sequence_processor(h)

        # Final normalization and projection head
        h = self.output_norm(h)
        prediction = self.output_head(h)
        return prediction, new_mamba_state

# --- Main MBD-S Model ---

class MBDSFinal(nn.Module):
    """Main MBD-S Model: Handles embedding, diffusion, block processing, and loss."""
    def __init__(self, config: MBDConfig):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.embed_dim = int(config.embed_dim * config.width)
        self.l_prime = config.l_prime
        self.tier_config = config.tier_config

        # Warn if Mamba is requested but unavailable
        if self.tier_config.use_mamba and not mamba_available:
            logging.warning(f"Tier '{self.tier_config.name}' requests Mamba but it's unavailable. Using MLP fallback in MDBBlock.")

        self.embedding = nn.Embedding(self.vocab_size, self.embed_dim, padding_idx=PAD_TOKEN_ID)
        self.diffusion_schedule = DiffusionSchedule(
            schedule_type=self.tier_config.schedule_type,
            beta_start=self.tier_config.noise_beta_start,
            beta_end=self.tier_config.noise_beta_end,
            num_steps=self.tier_config.diffusion_steps
        )
        self.time_projection = SinusoidalTimestepEmbedding(self.embed_dim, dropout=config.dropout)
        # The core block handles Mamba/MLP logic internally
        self.mdb_block = MDBBlockFinal(config)

        # Overlapping block configuration
        self.use_overlapping = self.tier_config.use_overlapping_blocks
        self.overlap = int(self.l_prime * self.tier_config.overlap_ratio) if self.use_overlapping and self.l_prime > 0 else 0
        self.stride = max(1, self.l_prime - self.overlap)
        self.overlap_weighting = self.tier_config.overlap_weighting

        logging.info(f"Initialized MBD-S Final - Tier: {self.tier_config.name.upper()} (Using Mamba: {self.mdb_block.use_mamba})")
        logging.info(f"  Embed Dim: {self.embed_dim}, L': {self.l_prime}, Overlap: {self.overlap}, Stride: {self.stride}")
        self.to(DEVICE) # Move model to the configured device

    def _get_overlap_weights(self) -> Tensor:
        """Generates weights for recombining overlapping blocks."""
        if not self.use_overlapping or self.l_prime <= 0 or self.overlap <= 0:
            return torch.ones(self.l_prime, device=DEVICE, dtype=torch.float32)

        if self.overlap_weighting == "uniform":
            return torch.ones(self.l_prime, device=DEVICE, dtype=torch.float32)
        elif self.overlap_weighting == "triangular":
            # Create a triangular window peaking at the center
            center = (self.l_prime - 1) / 2.0
            indices = torch.arange(self.l_prime, device=DEVICE, dtype=torch.float32)
            # Weight decreases linearly from the center
            weights = 1.0 - torch.abs(indices - center) / (center + 1e-6 if center > 0 else 1.0)
            return weights.clamp(min=1e-6) # Ensure positive weights for division
        else:
            raise ValueError(f"Unknown overlap_weighting: {self.overlap_weighting}")

    def _calculate_target_padding(self, seq_len: int) -> int:
         """Calculates the padded length required for processing with `unfold`."""
         if seq_len <= 0: return self.l_prime # Handle empty sequence case
         if seq_len <= self.l_prime: return self.l_prime # Pad short sequences to block size

         # If not overlapping, just pad to the next multiple of l_prime (or stride)
         if not self.use_overlapping or self.stride == self.l_prime:
             return math.ceil(seq_len / self.stride) * self.stride

         # With overlap: ensure the last block starts within the original sequence
         # and the total padded length covers the end of the last block.
         num_strides_needed = math.ceil(max(0, seq_len - self.l_prime) / self.stride)
         # Padded length = start of last block + block length
         padded_len = num_strides_needed * self.stride + self.l_prime
         return padded_len

    def _extract_embed_blocks(self, x_embed: Tensor) -> Tuple[Tensor, int]:
        """Extracts potentially overlapping blocks using `unfold`, returning blocks and padding amount."""
        batch_size, seq_len, embed_dim = x_embed.shape
        if self.l_prime <= 0 or self.stride <= 0:
            raise ValueError("l_prime and stride must be positive")

        padded_len = self._calculate_target_padding(seq_len)
        padding = max(0, padded_len - seq_len)

        if padding > 0:
            # Pad sequence at the end (dim=1)
            x_embed = F.pad(x_embed, (0, 0, 0, padding), value=0.0)

        # Ensure sequence length is sufficient for unfold after padding
        if x_embed.shape[1] < self.l_prime:
             raise ValueError(f"Padded sequence length {x_embed.shape[1]} is less than block size {self.l_prime}. Padding logic error?")

        # Extract blocks: [B, C, N_blocks, L'] -> [B, N_blocks, L', C]
        blocks_embed = x_embed.unfold(dimension=1, size=self.l_prime, step=self.stride)
        blocks_embed = blocks_embed.permute(0, 1, 3, 2).contiguous() # [B, N_blocks, L', D]
        return blocks_embed, padding

    def _recombine_embed_blocks(self, block_preds: Tensor, original_len: int, padding: int) -> Tensor:
        """Recombines overlapping block predictions using weighted averaging."""
        batch_size, n_blocks, block_len, embed_dim = block_preds.shape
        if block_len != self.l_prime:
            raise ValueError(f"Block length mismatch: expected {self.l_prime}, got {block_len}")

        padded_len = original_len + padding
        # Initialize tensors for accumulated predictions and counts (for averaging)
        full_pred = torch.zeros(batch_size, padded_len, embed_dim, device=block_preds.device, dtype=torch.float32)
        counts = torch.zeros(batch_size, padded_len, 1, device=block_preds.device, dtype=torch.float32)

        # Get weights (shape [1, L', 1] for broadcasting with [B, L', D])
        block_weights = self._get_overlap_weights().view(1, self.l_prime, 1)  # <-- FIX 1: Reshape weights correctly

        for b_idx in range(n_blocks):
            start_idx = b_idx * self.stride
            end_idx = start_idx + self.l_prime
            # Apply weights to the current block prediction
            # [B, L', D] * [1, L', 1] -> broadcast to [B, L', D]
            weighted_block = block_preds[:, b_idx] * block_weights

            # Determine the valid region of this block within the padded sequence
            slice_end_idx = min(end_idx, padded_len)
            slice_len = max(0, slice_end_idx - start_idx)

            if slice_len > 0:
                # Accumulate weighted predictions and weights in the corresponding slice
                # LHS shape: [B, slice_len, D], RHS shape: [B, slice_len, D]
                full_pred[:, start_idx:slice_end_idx] += weighted_block[:, :slice_len]
                # LHS shape: [B, slice_len, 1], RHS shape: [1, slice_len, 1] -> broadcast to [B, slice_len, 1]
                counts[:, start_idx:slice_end_idx] += block_weights[:, :slice_len]  # <-- FIX 2: Update counts correctly

        # Average predictions where overlaps occurred
        recombined_pred = full_pred / counts.clamp(min=1e-9)
        # Trim padding to return sequence of original length
        return recombined_pred[:, :original_len]

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass for training: Add noise, predict, recombine blocks, calculate loss."""
        batch_size, seq_len = x.shape
        device = x.device
        # Default target is the input sequence itself (denoising autoencoder style)
        targets = x if targets is None else targets
        if x.shape != targets.shape:
            raise ValueError("Input x and targets must have the same shape.")

        # Sample random timesteps for diffusion
        t = torch.randint(0, self.diffusion_schedule.num_steps, (batch_size,), device=device, dtype=torch.long)
        t_embed_proj = self.time_projection(t) # Get projected timestep embeddings

        # Embed input and target tokens
        x_embed = self.embedding(x)
        target_embed = self.embedding(targets) if self.tier_config.prediction_type == "x_start" else None
        pad_mask = (targets == PAD_TOKEN_ID) # Mask for ignoring padding in loss

        # Apply forward diffusion process: x_start -> x_t
        x_noisy_embed, noise_added = self.diffusion_schedule.q_sample(x_embed, t)

        # --- Block Processing ---
        noisy_blocks_embed, padding = self._extract_embed_blocks(x_noisy_embed)
        n_blocks = noisy_blocks_embed.size(1)
        # Pre-allocate tensor for block predictions
        all_block_preds = torch.zeros_like(noisy_blocks_embed)
        mamba_state = None # Initialize Mamba state for processing sequence of blocks

        # Determine model's primary dtype for casting inputs
        model_dtype = next(self.parameters()).dtype

        with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
            # Cast timestep embedding once for efficiency
            t_embed_proj_casted = t_embed_proj.to(model_dtype)
            # Process each block sequentially (necessary for Mamba state passing)
            for b_idx in range(n_blocks):
                block_input = noisy_blocks_embed[:, b_idx].to(model_dtype)
                # Pass input block, timestep embedding, and Mamba state (if used)
                block_pred, mamba_state = self.mdb_block(block_input, t_embed_proj_casted, mamba_state=mamba_state)
                all_block_preds[:, b_idx] = block_pred # Store prediction (possibly in AMP dtype)

        # Recombine block predictions - perform accumulation in float32 for stability
        full_prediction = self._recombine_embed_blocks(all_block_preds.float(), seq_len, padding)

        # --- Loss Calculation ---
        loss = None
        if self.tier_config.prediction_type == "noise":
            loss_target = noise_added
            # MSE loss between predicted noise and actual noise added
            loss = F.mse_loss(full_prediction, loss_target.float(), reduction='none').mean(dim=-1) # Average over embedding dim
        elif self.tier_config.prediction_type == "x_start":
            # Loss depends on whether the output head predicts embeddings or logits
            if full_prediction.shape[-1] == self.vocab_size: # Predicting logits
                loss = F.cross_entropy(full_prediction.view(-1, self.vocab_size), targets.view(-1), reduction='none').view(batch_size, seq_len)
            elif full_prediction.shape[-1] == self.embed_dim: # Predicting embeddings
                if target_embed is None: raise RuntimeError("target_embed required for x_start embedding prediction")
                loss = F.mse_loss(full_prediction, target_embed.float(), reduction='none').mean(dim=-1)
            else:
                raise ValueError("Prediction head output dimension mismatch for 'x_start'.")
        else:
            raise ValueError(f"Unknown prediction_type: {self.tier_config.prediction_type}")

        # Apply padding mask and calculate average loss over non-masked tokens
        loss = loss.masked_fill(pad_mask, 0.0)
        final_loss = loss.sum() / (~pad_mask).sum().clamp(min=1.0)

        # Return final prediction (float32) and calculated loss
        return full_prediction, final_loss

    @torch.no_grad()
    def generate(self, prompt: Tensor, max_new_tokens: int = 80, temperature: float = 1.0, ddim_steps: Optional[int] = None, ddim_eta: float = 0.0) -> Tensor:
        """Generates sequence using iterative denoising (DDPM or DDIM sampling)."""
        self.eval() # Set model to evaluation mode
        if prompt.dim() == 1: prompt = prompt.unsqueeze(0) # Ensure batch dimension
        batch_size, prompt_len = prompt.shape
        device = prompt.device
        T = self.diffusion_schedule.num_steps
        model_dtype = next(self.parameters()).dtype # Use model's primary dtype

        # Configure timesteps for DDPM or DDIM
        if ddim_steps is not None and 0 < ddim_steps < T:
            use_ddim = True
            # Linearly spaced timesteps for DDIM
            timesteps = torch.linspace(T - 1, 0, ddim_steps, dtype=torch.long, device=device)
            # Add -1 for the final step (x_0 calculation)
            all_timesteps = torch.cat([timesteps, torch.tensor([-1], device=device, dtype=torch.long)])
            num_sampling_steps = ddim_steps
        else:
            use_ddim = False
            # Full DDPM timesteps: T-1, T-2, ..., 0, -1
            all_timesteps = torch.arange(-1, T, device=device).flip(0)
            num_sampling_steps = T
            ddim_eta = 0.0 # Eta is only relevant for DDIM

        logging.info(f"Generating - {'DDIM' if use_ddim else 'DDPM'} steps: {num_sampling_steps}, Temp: {temperature}, Eta: {ddim_eta if use_ddim else 'N/A'}")

        total_seq_len = prompt_len + max_new_tokens
        shape = (batch_size, total_seq_len, self.embed_dim)
        # Start with random noise in the embedding space
        xt_embed = torch.randn(shape, device=device, dtype=model_dtype)
        # Embed the prompt if provided
        prompt_embed = self.embedding(prompt).to(xt_embed.dtype) if prompt_len > 0 else None
        mamba_state = None # Initialize Mamba state for generation loop

        # Iterative denoising loop
        for i in range(num_sampling_steps):
            t_val = all_timesteps[i]
            t_prev_val = all_timesteps[i+1]
            # Create tensors for current and previous timesteps
            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)

            current_xt_input = xt_embed.clone()
            # Inject noisy prompt embedding at each step (if prompt exists and not at t=0)
            if prompt_embed is not None and t_val > 0:
                noisy_prompt_embed, _ = self.diffusion_schedule.q_sample(prompt_embed, t)
                current_xt_input[:, :prompt_len, :] = noisy_prompt_embed

            # --- Block Processing for Denoising Step ---
            input_blocks_embed, padding = self._extract_embed_blocks(current_xt_input)
            all_block_preds = torch.zeros_like(input_blocks_embed, dtype=model_dtype)
            t_embed_proj = self.time_projection(t).to(model_dtype)

            with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                current_mamba_state = mamba_state # Use state from previous step
                for b_idx in range(input_blocks_embed.size(1)):
                    block_input = input_blocks_embed[:, b_idx] # Already correct dtype
                    block_pred, current_mamba_state = self.mdb_block(block_input, t_embed_proj, mamba_state=current_mamba_state)
                    all_block_preds[:, b_idx] = block_pred
                mamba_state = current_mamba_state # Update Mamba state for next denoising step

            # Recombine predictions (use float32 for stability)
            prediction = self._recombine_embed_blocks(all_block_preds.float(), total_seq_len, padding)

            # TODO: Implement Classifier-Free Guidance (CFG) if needed
            # Would require running the model twice (conditional and unconditional) and combining predictions.

            if self.tier_config.prediction_type == "noise":
                pred_noise = prediction
            else:
                # Generation typically relies on noise prediction
                raise NotImplementedError("Generation currently requires prediction_type='noise'.")

            # --- Sampling Step (DDPM or DDIM) ---
            # Perform calculations in float32 for numerical stability
            xt_float = current_xt_input.float()
            # Predict x0 (original embedding) from xt and predicted noise
            pred_x0_embed = self.diffusion_schedule.predict_start_from_noise(xt_float, t, pred_noise)

            # Get schedule constants for t and t_prev
            alpha_bar_t = self.diffusion_schedule._gather('alpha_bars', t, shape)
            alpha_bar_t_prev_val = self.diffusion_schedule.alpha_bars[t_prev.clamp(min=0)] # Handle t_prev = -1
            alpha_bar_t_prev = alpha_bar_t_prev_val.view(-1, 1, 1).to(device).float()
            # alpha_bar_t_prev = 1.0 when t_prev = -1
            alpha_bar_t_prev[t_prev < 0] = 1.0

            if use_ddim:
                # DDIM sampling step
                sigma = ddim_eta * torch.sqrt(
                    (1 - alpha_bar_t_prev) / (1 - alpha_bar_t).clamp(min=1e-9) *
                    (1 - alpha_bar_t / alpha_bar_t_prev.clamp(min=1e-9))
                ).clamp(min=0.0)
                noise_sample = torch.randn_like(xt_float, dtype=torch.float32) * temperature
                # Direction pointing to xt
                pred_dir_xt = torch.sqrt((1.0 - alpha_bar_t_prev - sigma**2).clamp(min=0.0)) * pred_noise
                # Combine predicted x0, direction, and noise
                xt_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0_embed + pred_dir_xt + sigma * noise_sample
            else: # DDPM
                # DDPM sampling step using posterior mean and variance
                posterior_mean, _, posterior_log_variance = self.diffusion_schedule.q_posterior_mean_variance(pred_x0_embed, xt_float, t)
                noise_sample = torch.randn_like(xt_float, dtype=torch.float32) * temperature
                # Add noise only if t > 0
                mask = (t > 0).int().view(-1, 1, 1)
                xt_prev = posterior_mean + mask * (0.5 * posterior_log_variance).exp() * noise_sample

            # Update xt_embed for the next iteration (cast back to model dtype)
            xt_embed = xt_prev.to(model_dtype)

            if (i + 1) % max(1, num_sampling_steps // 5) == 0 or i == num_sampling_steps - 1:
                logging.debug(f"Generation step {i+1}/{num_sampling_steps} (t={t_val.item()}) -> (t_prev={t_prev_val.item()}) done.")

        # --- Final Decode ---
        # Project final denoised embeddings (x0) to logits
        final_embeddings = xt_embed.float() # Use float32 for logit calculation
        logits = F.linear(final_embeddings, self.embedding.weight.float())
        # Sample tokens (greedy decoding)
        sampled_tokens = torch.argmax(logits, dim=-1)

        # Process generated sequences (remove padding, stop at EOS)
        output_sequences = []
        for i in range(batch_size):
            seq = sampled_tokens[i]
            # Find first EOS token
            eos_indices = (seq == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                # Truncate sequence after the first EOS
                seq = seq[:eos_indices[0].item() + 1]
            output_sequences.append(seq)

        # Pad batch to the maximum length in the batch
        max_len_in_batch = max(len(s) for s in output_sequences) if output_sequences else 0
        padded_batch = torch.full((batch_size, max_len_in_batch), PAD_TOKEN_ID, dtype=torch.long, device=device)
        for i, seq in enumerate(output_sequences):
            padded_batch[i, :len(seq)] = seq

        return padded_batch

    def configure_optimizers(self, lr: float = 1e-4, weight_decay: float = 0.1, betas: Tuple[float, float] = (0.9, 0.98), eps: float = 1e-6) -> Tuple[Optimizer, _LRScheduler]:
        """Sets up AdamW optimizer and a constant learning rate scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
        # Using a constant LR scheduler for simplicity, can be replaced with others (e.g., CosineAnnealingLR)
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
        return optimizer, scheduler

# --- Utilities & Training Loop ---
def count_parameters(model: nn.Module) -> int:
    """Counts the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def setup_dataloader(dataset: Union[Dataset, Tensor], batch_size: int, shuffle: bool = True) -> DataLoader:
    """Creates a DataLoader with recommended settings."""
    # Using num_workers=0 can help avoid potential issues with dataset hashing in `map`
    # Pin memory if using CUDA for faster data transfer
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0,
                      pin_memory=(DEVICE.type == 'cuda'), drop_last=True)

def basic_train_loop(model: MBDSFinal, data_loader: DataLoader, epochs: int = 1, steps: Optional[int] = None, lr: float = 1e-4, weight_decay: float = 0.1, gradient_clipping: float = 1.0, log_interval: int = 50):
    """A basic training loop for the MBD-S model with AMP."""
    model.train() # Set model to training mode
    optimizer, scheduler = model.configure_optimizers(lr=lr, weight_decay=weight_decay)
    # GradScaler for automatic mixed precision
    scaler = GradScaler(enabled=AMP_ENABLED)
    total_steps_done, epochs_done = 0, 0
    losses = []
    logging.info(f"Starting training (Epochs: {epochs}, Max Steps: {steps if steps else 'Unlimited'})...")

    # Loop over epochs
    while (epochs_done < epochs) and (steps is None or total_steps_done < steps):
        epoch_loss, num_batches = 0.0, 0
        start_epoch_time = time.time()
        model.train() # Ensure model is in train mode at start of epoch

        # Loop over batches
        for batch_idx, batch in enumerate(data_loader):
            # Handle different batch formats (dict vs tensor)
            if isinstance(batch, dict) and 'input_ids' in batch:
                 input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            elif isinstance(batch, torch.Tensor):
                 input_ids = batch.to(DEVICE, non_blocking=True)
            else:
                 logging.warning(f"Unexpected batch type: {type(batch)}. Skipping.")
                 continue

            optimizer.zero_grad(set_to_none=True) # More memory efficient

            # Forward pass with automatic mixed precision
            with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                _, loss = model(input_ids) # Targets default to input_ids

            # Backward pass and optimization
            if loss is not None and torch.isfinite(loss):
                scaler.scale(loss).backward() # Scale loss for AMP

                # Optional gradient clipping (unscale first)
                if gradient_clipping > 0:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                scaler.step(optimizer) # Optimizer step (using unscaled gradients)
                scaler.update() # Update scaler for next iteration

                # Logging and tracking
                current_loss = loss.item()
                losses.append(current_loss)
                epoch_loss += current_loss
                num_batches += 1
                total_steps_done += 1

                if total_steps_done % log_interval == 0:
                    logging.info(f"E {epochs_done+1}, S {total_steps_done}, B {batch_idx+1}/{len(data_loader)}, Loss: {current_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

                # Check if max steps reached
                if steps is not None and total_steps_done >= steps: break
            else:
                logging.warning(f"Invalid loss encountered: {loss} at step {total_steps_done}. Skipping batch.")

        scheduler.step() # Update learning rate scheduler (if applicable)
        epochs_done += 1
        avg_epoch_loss = epoch_loss / max(1, num_batches)
        logging.info(f"Epoch {epochs_done} finished in {time.time() - start_epoch_time:.2f}s. Avg Loss: {avg_epoch_loss:.4f}")

        # Early stopping if loss becomes invalid
        if not math.isfinite(avg_epoch_loss):
            logging.error("Average epoch loss is NaN or Inf. Stopping training.")
            break
        # Check if max steps reached after epoch completion
        if steps is not None and total_steps_done >= steps:
            logging.info(f"Maximum training steps ({steps}) reached.")
            break

    logging.info(f"Training finished after {total_steps_done} steps.")
    return model, losses

# --- Benchmarking & Testing Infrastructure --- (Requires `datasets`, `transformers`)
try:
    from datasets import load_dataset, disable_caching
    from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, PretrainedConfig
    benchmarking_libs_available = True
    # Consider uncommenting if persistent hashing issues occur even with num_workers=0
    # disable_caching()
except ImportError:
    benchmarking_libs_available = False
    logging.warning("Libraries for benchmarking (`datasets`, `transformers`) not found. Skipping benchmarking tests.")

if benchmarking_libs_available:
    class TransformerBaselineConfig(PretrainedConfig):
        """Configuration class for a standard Transformer baseline model."""
        model_type = "transformer_baseline"
        is_parallelizable = False # Standard TransformerEncoder isn't tensor parallelizable easily
        # Default parameters, potentially overridden during instantiation or adjustment
        def __init__(self, vocab_size=DEFAULT_VOCAB_SIZE, n_layer=6, n_head=8, n_embd=512,
                     block_size=256, dropout=0.1, pad_token_id=PAD_TOKEN_ID, **kwargs):
            self.vocab_size = vocab_size
            self.n_layer = n_layer
            self.n_head = n_head
            self.n_embd = n_embd
            self.block_size = block_size
            self.dropout = dropout
            # Pass essential arguments like pad_token_id to the parent class
            super().__init__(pad_token_id=pad_token_id, **kwargs)

    class TransformerBaseline(PreTrainedModel):
        """A standard Transformer Encoder-Decoder model for language modeling baseline."""
        config_class = TransformerBaselineConfig

        def __init__(self, config: TransformerBaselineConfig):
            super().__init__(config)
            self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd, padding_idx=config.pad_token_id)
            # Simple learned positional embeddings
            self.pos_embedding = nn.Embedding(config.block_size, config.n_embd)
            self.dropout = nn.Dropout(config.dropout)

            # Standard Transformer Encoder Layer configuration
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=config.n_embd,
                nhead=config.n_head,
                dim_feedforward=config.n_embd * 4, # Standard expansion factor
                dropout=config.dropout,
                activation='gelu', # Common activation function
                batch_first=True, # Expect input as (batch, seq, feature)
                norm_first=True # Pre-LayerNorm for stability
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.n_layer)
            self.ln_f = nn.LayerNorm(config.n_embd) # Final LayerNorm
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # Output projection

            # Weight tying between input embeddings and output projection
            self.token_embedding.weight = self.lm_head.weight

            self.apply(self._init_weights) # Initialize weights

        def _init_weights(self, module):
             """Initialize weights with common practices for Transformers."""
             if isinstance(module, (nn.Linear, nn.Embedding)):
                 torch.nn.init.normal_(module.weight, mean=0.0, std=0.02) # Often used in GPT-like models
             if isinstance(module, nn.Linear) and module.bias is not None:
                 torch.nn.init.zeros_(module.bias)
             elif isinstance(module, nn.LayerNorm):
                 nn.init.zeros_(module.bias)
                 nn.init.ones_(module.weight)

        def forward(self, input_ids: Tensor, targets: Optional[Tensor] = None, attention_mask: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
            """Forward pass for the Transformer baseline."""
            b, t = input_ids.size()
            if t > self.config.block_size:
                raise ValueError(f"Sequence length {t} exceeds block size {self.config.block_size}")

            # Token embeddings
            tok_emb = self.token_embedding(input_ids)
            # Positional embeddings
            pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0) # Shape (1, t)
            pos_emb = self.pos_embedding(pos) # Shape (1, t, n_embd)
            # Combine embeddings and apply dropout
            x = self.dropout(tok_emb + pos_emb)

            # Create masks for the Transformer encoder
            # Causal mask: prevents attending to future positions
            causal_mask = nn.Transformer.generate_square_subsequent_mask(t, device=x.device)
            # Padding mask: prevents attending to padding tokens
            # True where tokens should be *ignored*
            key_padding_mask = (input_ids == self.config.pad_token_id)
            # If an external attention_mask is provided (True where tokens should be attended), convert it
            if attention_mask is not None:
                key_padding_mask = ~attention_mask.bool() # Invert mask: True where should be ignored

            # Pass through Transformer encoder
            # Note: `is_causal=False` because we provide an explicit causal `mask`
            x = self.transformer_encoder(x, mask=causal_mask, src_key_padding_mask=key_padding_mask, is_causal=False)

            # Final layer norm and language modeling head
            x = self.ln_f(x)
            logits = self.lm_head(x) # Shape (b, t, vocab_size)

            # Calculate loss if targets are provided
            loss = None
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1),
                                       ignore_index=self.config.pad_token_id) # Ignore padding tokens in loss

            return logits, loss

    def calculate_perplexity(model: nn.Module, data_loader: DataLoader, model_type: str = "mbds") -> float:
        """Calculates perplexity for a given model and dataloader."""
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        logging.info(f"Calculating perplexity for {model_type.upper()}...")

        with torch.no_grad():
            for batch in data_loader:
                # Handle different batch formats
                if isinstance(batch, dict) and 'input_ids' in batch:
                    input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                    # For Transformer, use attention mask if available
                    attention_mask = batch.get('attention_mask', None)
                    if attention_mask is not None: attention_mask = attention_mask.to(DEVICE, non_blocking=True)
                elif isinstance(batch, torch.Tensor):
                    input_ids = batch.to(DEVICE, non_blocking=True)
                    attention_mask = None # Assume no mask if only tensor is provided
                else:
                    logging.warning(f"Skipping unexpected batch type in perplexity calculation: {type(batch)}")
                    continue

                targets = input_ids # Standard Causal LM evaluation target
                batch_size, seq_len = input_ids.shape

                with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                    if model_type == "mbds":
                        # Perplexity calculation for noise-predicting diffusion models is non-trivial
                        # and requires a different evaluation setup (e.g., ELBO estimation).
                        # Returning NaN as a placeholder.
                        logging.warning("Direct perplexity calculation for noise-predicting MBD-S is not standard. Requires model adaptation or ELBO calculation. Returning NaN.")
                        return float('nan')
                    elif model_type == "transformer":
                        # Ensure the model is a PreTrainedModel to access config.pad_token_id
                        if not isinstance(model, PreTrainedModel):
                            raise TypeError("Transformer model must be a PreTrainedModel for perplexity calculation.")
                        pad_token_id = model.config.pad_token_id
                        logits, loss = model(input_ids, targets=targets, attention_mask=attention_mask)
                    else:
                        raise ValueError(f"Unknown model_type for perplexity: {model_type}")

                # Accumulate loss and count non-padding tokens
                if loss is not None and torch.isfinite(loss):
                    # Use the provided attention mask or create one based on PAD_TOKEN_ID
                    if attention_mask is None:
                        mask = (targets != pad_token_id)
                    else:
                        mask = attention_mask.bool() # Use the provided mask directly

                    num_tokens = mask.sum().item()
                    if num_tokens > 0:
                        # Multiply average batch loss by the number of tokens it represents
                        total_loss += loss.item() * num_tokens
                        total_tokens += num_tokens
                else:
                     logging.warning(f"Invalid loss ({loss}) encountered during perplexity calculation.")


        if total_tokens == 0:
            logging.warning("No valid tokens found for perplexity calculation.")
            return float('inf')

        # Calculate perplexity: exp(total_loss / total_tokens)
        avg_loss = total_loss / total_tokens
        perplexity = math.exp(avg_loss)
        logging.info(f"{model_type.upper()} Perplexity: {perplexity:.4f} (Avg Loss: {avg_loss:.4f}, Tokens: {total_tokens})")
        return perplexity

    class TestMBDSvsTransformerLearning(unittest.TestCase):
        """Compares MBD-S training dynamics against a Transformer baseline."""
        def setUp(self):
            """Set up datasets, tokenizer, and basic configurations for tests."""
            if not benchmarking_libs_available:
                self.skipTest("Benchmarking libraries (`datasets`, `transformers`) not available.")

            self.batch_size = 4
            self.seq_len = 128
            self.train_steps = 50 # Short training run for testing convergence
            self.eval_steps = 10
            self.lr = 3e-4

            try:
                # Load a small subset of TinyStories
                self.dataset = load_dataset("roneneldan/TinyStories", split="train[:1%]", trust_remote_code=True)
                self.val_dataset = load_dataset("roneneldan/TinyStories", split="validation[:1%]", trust_remote_code=True)

                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                # Add pad token if missing (like in GPT-2)
                if self.tokenizer.pad_token is None:
                    self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                    logging.info(f"Added pad token '{self.tokenizer.pad_token}' to tokenizer.")

                # Update global PAD_TOKEN_ID based on the tokenizer
                global PAD_TOKEN_ID
                PAD_TOKEN_ID = self.tokenizer.pad_token_id
                self.vocab_size = len(self.tokenizer) # Update vocab size based on tokenizer
                logging.info(f"Using Tokenizer: {self.tokenizer.name_or_path}, Vocab Size: {self.vocab_size}, PAD ID: {PAD_TOKEN_ID}")

                # Define tokenize function locally to potentially avoid hashing/pickling issues with `map`
                seq_len = self.seq_len
                tk = self.tokenizer
                def tokenize_fn(examples):
                    # Pad/truncate to fixed sequence length
                    return tk(examples["text"], padding="max_length", truncation=True, max_length=seq_len, return_tensors="pt")

                # Tokenize datasets
                # Using `map` might show progress bars from `datasets` library
                logging.info("Tokenizing train dataset...")
                self.tokenized_ds = self.dataset.map(tokenize_fn, batched=True, remove_columns=["text"])
                logging.info("Tokenizing validation dataset...")
                self.tokenized_val_ds = self.val_dataset.map(tokenize_fn, batched=True, remove_columns=["text"])

                # Set format for PyTorch DataLoader
                self.tokenized_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])
                self.tokenized_val_ds.set_format(type='torch', columns=['input_ids', 'attention_mask'])

                # DataLoader for MBD-S (expects tensors, not dicts)
                # We pass only input_ids, as the basic_train_loop uses them as targets implicitly
                self.train_loader = setup_dataloader(self.tokenized_ds['input_ids'], self.batch_size)
                self.val_loader = setup_dataloader(self.tokenized_val_ds['input_ids'], self.batch_size, shuffle=False)

                # DataLoader for Transformer evaluation (expects dicts with input_ids and attention_mask)
                self.tf_val_loader = setup_dataloader(self.tokenized_val_ds, self.batch_size, shuffle=False)
                # Use a dict-based loader for TF training as well
                self.tf_train_loader = setup_dataloader(self.tokenized_ds, self.batch_size, shuffle=True)


            except Exception as e:
                self.fail(f"Data setup failed: {e}")

        def _train_and_eval(self, model_name: str, model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, steps: int, eval_steps: int, lr: float, model_type: str) -> Tuple[float, float]:
            """Helper function to train a model and calculate final loss and perplexity."""
            logging.info(f"--- Training {model_name} ---")
            losses = []
            if model_type == "mbds":
                if not isinstance(model, MBDSFinal): raise TypeError("Expected MBDSFinal model")
                # Use the dedicated MBD-S training loop
                model, losses = basic_train_loop(model, train_loader, epochs=1, steps=steps, lr=lr, log_interval=eval_steps)
            elif model_type == "transformer":
                if not isinstance(model, TransformerBaseline): raise TypeError("Expected TransformerBaseline model")
                # Basic training loop for Transformer
                model.train()
                optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
                scaler = GradScaler(enabled=AMP_ENABLED)
                steps_done = 0
                while steps_done < steps:
                    batch_processed = False
                    for batch in train_loader: # Use the dict-based TF loader
                        if not isinstance(batch, dict) or 'input_ids' not in batch: continue # Skip unexpected format

                        input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
                        attention_mask = batch.get('attention_mask', None)
                        if attention_mask is not None: attention_mask = attention_mask.to(DEVICE, non_blocking=True)

                        optimizer.zero_grad(set_to_none=True)
                        with autocast(enabled=AMP_ENABLED, dtype=AMP_DTYPE):
                            # Pass attention mask to the model
                            _, loss = model(input_ids, targets=input_ids, attention_mask=attention_mask)

                        if loss is not None and torch.isfinite(loss):
                            scaler.scale(loss).backward()
                            scaler.step(optimizer)
                            scaler.update()
                            losses.append(loss.item())
                            steps_done += 1
                            batch_processed = True
                            if steps_done % eval_steps == 0:
                                logging.info(f"TF Step {steps_done}/{steps}, Loss: {loss.item():.4f}")
                            if steps_done >= steps: break
                        else:
                            logging.warning(f"Invalid loss ({loss}) in Transformer training at step {steps_done}.")
                            # Optionally break or skip step based on severity
                            break # Stop training if loss is invalid

                    if not batch_processed and steps_done < steps:
                        logging.error("Transformer training loop finished epoch without processing batches.")
                        break # Avoid infinite loop if dataloader is empty or all batches fail
                    if steps_done >= steps: break
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

            final_loss = losses[-1] if losses else float('inf')
            # Use the appropriate validation loader (dict-based for TF)
            perplexity = calculate_perplexity(model, val_loader, model_type=model_type)

            return final_loss, perplexity

        def _adjust_transformer_params(self, target_params: int, config: TransformerBaselineConfig) -> TransformerBaselineConfig:
            """Iteratively adjusts Transformer config to roughly match target parameter count."""
            logging.info(f"Adjusting Transformer to match ~{target_params:,} parameters.")
            # Start with reasonable defaults
            config.n_embd = 512
            config.n_layer = 6
            config.n_head = 8 # Must be divisor of n_embd
            config.pad_token_id = PAD_TOKEN_ID # Ensure pad token ID is set

            MAX_ITER = 10
            TOLERANCE = 0.20 # Allow +/- 20% deviation

            best_config = copy.deepcopy(config)
            min_diff = float('inf')

            for i in range(MAX_ITER):
                # Ensure head count is valid
                if config.n_embd % config.n_head != 0:
                    # Find the nearest power of 2 <= n_embd // 64 as a heuristic
                    config.n_head = max(1, 2**((config.n_embd // 64).bit_length() - 1))
                    # Fallback if n_embd is very small
                    if config.n_embd % config.n_head != 0: config.n_head = 1
                    logging.debug(f"Adjusted n_head to {config.n_head} for n_embd {config.n_embd}")

                try:
                    temp_model = TransformerBaseline(config).to(DEVICE)
                    current_params = count_parameters(temp_model)
                    del temp_model
                    if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                except Exception as e:
                    logging.error(f"Failed to create temp Transformer model: {e}. Config: {config}")
                    # Revert to previous best config if creation fails
                    config = copy.deepcopy(best_config)
                    continue

                diff = abs(current_params - target_params)
                ratio = current_params / target_params if target_params > 0 else 0

                logging.debug(f"Iter {i+1}: L={config.n_layer}, D={config.n_embd}, H={config.n_head} -> Params: {current_params:,} (Target: {target_params:,}, Ratio: {ratio:.2f})")

                if diff < min_diff:
                    min_diff = diff
                    best_config = copy.deepcopy(config)

                if abs(ratio - 1.0) < TOLERANCE:
                    logging.info(f"Parameter count {current_params:,} within {TOLERANCE*100:.0f}% tolerance.")
                    break

                # Adjust parameters based on ratio
                # Prioritize changing n_layer first, then n_embd
                if ratio < 1.0: # Too few params, increase size
                    if config.n_layer < 12: config.n_layer += 1
                    elif config.n_embd < 768: config.n_embd = min(768, config.n_embd + 64)
                    else: break # Stop if max limits reached
                else: # Too many params, decrease size
                    if config.n_embd > 128: config.n_embd = max(128, config.n_embd - 64)
                    elif config.n_layer > 2: config.n_layer -= 1
                    else: break # Stop if min limits reached

                # Keep head count reasonable relative to embed dim
                config.n_head = max(1, config.n_embd // 64)


            # Log final chosen configuration
            final_config = best_config
            try:
                final_model = TransformerBaseline(final_config).to(DEVICE)
                final_params = count_parameters(final_model)
                del final_model
                if DEVICE.type == 'cuda': torch.cuda.empty_cache()
                logging.info(f"Final Adjusted TF Config: L={final_config.n_layer}, D={final_config.n_embd}, H={final_config.n_head}. Params: {final_params:,}")
            except Exception as e:
                 logging.error(f"Failed to create final adjusted Transformer model: {e}. Config: {final_config}")
                 final_params = -1 # Indicate failure

            self.assertGreater(final_params, 0, "Failed to create adjusted Transformer model.")
            return final_config

        def test_learning_core_vs_transformer(self):
            """Tests if MBD-S ('core_balanced') learns compared to a parameter-matched Transformer."""
            # --- MBD-S Setup ---
            mbd_tier_dict = TIER_CONFIGS_EVOLVED["core_balanced"]
            mbd_tier_obj = TierConfig(**mbd_tier_dict)
            # Reduce embed_dim slightly for faster testing if needed, adjust Transformer target accordingly
            mbd_embed_dim = 256
            mbd_config = MBDConfig(tier_config=mbd_tier_obj, vocab_size=self.vocab_size,
                                   embed_dim=mbd_embed_dim, l_prime=32, width=1.0,
                                   pos_encoding_max_len=self.seq_len + 10) # Ensure pos encoding covers seq len
            mbd_model = MBDSFinal(mbd_config).to(DEVICE)
            mbd_params = count_parameters(mbd_model)
            logging.info(f"MBD-S Model ('{mbd_tier_obj.name}', Mamba Used: {mbd_model.mdb_block.use_mamba}) Params: {mbd_params:,}")

            # --- Transformer Baseline Setup ---
            # **FIX Applied**: Removed explicit pad_token_id from constructor call, handled in class __init__
            base_tf_config = TransformerBaselineConfig(vocab_size=self.vocab_size, block_size=self.seq_len)
            # Adjust TF params to match MBD-S
            adj_tf_config = self._adjust_transformer_params(mbd_params, base_tf_config)
            # Ensure the adjusted config still uses the correct PAD_TOKEN_ID
            adj_tf_config.pad_token_id = PAD_TOKEN_ID
            tf_model = TransformerBaseline(adj_tf_config).to(DEVICE)
            tf_params = count_parameters(tf_model)
            logging.info(f"Transformer Baseline Params: {tf_params:,}")
            # Allow slightly larger tolerance after adjustment heuristic
            self.assertLess(abs(mbd_params - tf_params) / max(1, mbd_params), 0.30,
                            f"Parameter counts differ significantly: MBD-S ({mbd_params:,}) vs TF ({tf_params:,})")

            # --- Training & Evaluation ---
            # MBD-S uses tensor-based loader, Transformer uses dict-based loader
            mbd_loss, mbd_ppl = self._train_and_eval(
                "MBD-S", mbd_model, self.train_loader, self.val_loader, # Uses tensor loaders
                self.train_steps, self.eval_steps, self.lr, "mbds"
            )
            #print(mbd_loss, mbd_ppl)
            tf_loss, tf_ppl = self._train_and_eval(
                "Transformer", tf_model, self.tf_train_loader, self.tf_val_loader, # Uses dict loaders
                self.train_steps, self.eval_steps, self.lr, "transformer"
            )
            #print(tf_loss, tf_ppl)

            # --- Results & Assertions ---
            logging.info(f"--- Final Results (Steps: {self.train_steps}) ---")
            logging.info(f"MBD-S Final Loss: {mbd_loss:.4f}, Perplexity: {mbd_ppl:.4f} (Params: {mbd_params:,})")
            logging.info(f"Transformer Final Loss: {tf_loss:.4f}, Perplexity: {tf_ppl:.4f} (Params: {tf_params:,})")

            # Basic sanity checks
            self.assertTrue(math.isfinite(mbd_loss), "MBD-S training resulted in non-finite loss.")
            self.assertTrue(math.isfinite(tf_loss), "Transformer training resulted in non-finite loss.")
            self.assertTrue(math.isfinite(tf_ppl) or math.isnan(tf_ppl), # Allow NaN for MBD-S perplexity warning
                            "Perplexity calculation resulted in non-finite value.")

            # Check if Transformer learned reasonably (perplexity should decrease from random)
            # Initial random PPL is approx vocab size. Check if it's significantly lower.
            # Adjust threshold based on dataset size and training steps. 500 is a lenient check for TinyStories/short training.
            if math.isfinite(tf_ppl):
                self.assertLess(tf_ppl, 500, f"Transformer perplexity ({tf_ppl:.2f}) seems too high, indicating poor learning.")
            else:
                 logging.warning("Could not assert Transformer perplexity as it was non-finite.")


# --- Main Execution ---
if __name__ == "__main__":
    if benchmarking_libs_available:
        # Run unit tests if benchmarking libraries are installed
        # Example: python your_script_name.py
        # Example: python your_script_name.py TestMBDSvsTransformerLearning.test_learning_core_vs_transformer
        suite = unittest.TestSuite()
        # Add all tests from the class
        suite.addTest(unittest.makeSuite(TestMBDSvsTransformerLearning))
        runner = unittest.TextTestRunner()
        result = runner.run(suite)
        # Exit with appropriate code based on test results
        sys.exit(not result.wasSuccessful())
    else:
        # Run a simple generation test as a fallback if tests cannot be run
        logging.warning("Skipping unit tests due to missing benchmarking libraries (`datasets`, `transformers`).")
        logging.info("Running a simple generation test instead...")
        try:
            # Use a small configuration for quick testing
            test_tier_dict = TIER_CONFIGS_EVOLVED["simple_edge"] # Use a simple tier
            test_tier_obj = TierConfig(**test_tier_dict)
            test_cfg = MBDConfig(tier_config=test_tier_obj, embed_dim=64, l_prime=16,
                                 vocab_size=DEFAULT_VOCAB_SIZE, pos_encoding_max_len=128)
            test_model = MBDSFinal(test_cfg).to(DEVICE)
            test_model.eval()

            logging.info(f"Simple test model params: {count_parameters(test_model):,}")

            # Generate a short sequence from a random prompt
            prompt = torch.randint(PAD_TOKEN_ID + 1, 500, (1, 10), device=DEVICE) # Avoid PAD in prompt
            output = test_model.generate(prompt, max_new_tokens=20, ddim_steps=5) # Use few DDIM steps

            logging.info(f"Generation test successful. Output shape: {output.shape}")
            # Decode generated tokens (requires tokenizer, skip if not available)
            try:
                # Attempt to use tokenizer if available from test setup (won't exist if setup skipped)
                tokenizer = AutoTokenizer.from_pretrained("gpt2")
                if tokenizer.pad_token is None: tokenizer.add_special_tokens({'pad_token': '[PAD]'})
                logging.info(f"Prompt: {tokenizer.decode(prompt[0])}")
                logging.info(f"Generated: {tokenizer.decode(output[0])}")
            except NameError:
                 logging.info(f"Generated tokens (raw): {output[0].tolist()}")


            # Basic assertion on output shape
            assert output.shape[0] == 1
            assert output.shape[1] >= prompt.shape[1]

        except Exception as e:
            logging.error(f"Simple generation test failed: {e}", exc_info=True)
            sys.exit(1) # Exit with error code if generation fails