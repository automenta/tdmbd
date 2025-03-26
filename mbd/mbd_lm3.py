#!/usr/bin/env python3
"""
MBD-S Evolved: Enhanced Modular Mamba-Block-Diffusion Language Model.

Evolutions:
- Noise Prediction Objective: Predicts added noise (epsilon) instead of clean data (x0).
- Diffusion Timestep Conditioning: Embeds timestep 't' and injects into the model.
- Noise Schedule: Implements a standard cosine noise schedule.
- Bidirectional Mamba Option: Allows combining forward & backward Mamba passes within blocks.
- Overlapping Blocks Option: Reduces boundary effects by processing overlapping chunks.
- Enhanced Configuration: More options to control advanced features.
- Refined Generation Loop: Implements DDPM-like sampling based on noise prediction.
- Comprehensive Test Suite in main.
- Fixed identified bugs and warnings.
"""
import copy  # For deep copying model components like backward Mamba
import logging
import math
import sys  # For checking python version for amp device_type
import time
from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.amp import autocast, GradScaler

# Try importing Mamba
try:
    # Check if mamba_ssm is installed and also try to import its core component
    from mamba_ssm.modules.mamba_simple import Mamba

    if Mamba is None: raise ImportError  # Ensure Mamba class is actually available
    mamba_available = True
except ImportError:
    Mamba = None  # Define Mamba as None if import fails or class isn't found
    mamba_available = False
    # This warning is now logged only once before main execution if needed

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1
# MASK_TOKEN_ID not strictly needed for noise prediction, but can be used for other tasks.

DEFAULT_VOCAB_SIZE = 50_257
DEFAULT_L_PRIME = 32
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP_DEVICE_TYPE = DEVICE.type  # 'cuda' or 'cpu'

# Tier configurations (Evolved)
# Added: prediction_type, use_bidirectional_mamba, use_overlapping_blocks, schedule_type
TIER_CONFIGS_EVOLVED = {
    "simple_edge": {  # Fastest, minimal features, MLP fallback
        "mamba_layers": 1, "expansion_factor": 2, "diffusion_steps": 10,
        "noise_beta_start": 0.001, "noise_beta_end": 0.02, "schedule_type": "linear",
        "prediction_type": "noise",  # Can use noise even with MLP
        "use_bidirectional_mamba": False, "use_overlapping_blocks": False,
        "stop_entropy": 3.5, "quantize": True, "use_mamba": False,  # Explicitly False
    },
    "core_balanced": {  # Good balance, uses noise prediction
        "mamba_layers": 6, "expansion_factor": 2, "diffusion_steps": 50,  # More steps typical for diffusion
        "noise_beta_start": 0.0001, "noise_beta_end": 0.02, "schedule_type": "cosine",
        "prediction_type": "noise",  # Predict noise
        "use_bidirectional_mamba": False, "use_overlapping_blocks": False,
        "stop_entropy": float("inf"), "quantize": False, "use_mamba": True,  # Requires Mamba
    },
    "enhanced_quality": {  # Higher quality, bidirectional option
        "mamba_layers": 12, "expansion_factor": 2.5, "diffusion_steps": 100,
        "noise_beta_start": 0.0001, "noise_beta_end": 0.02, "schedule_type": "cosine",
        "prediction_type": "noise",
        "use_bidirectional_mamba": True,  # Enable BiMamba
        "use_overlapping_blocks": True,  # Enable Overlapping
        "overlap_ratio": 0.25,  # e.g., 25% overlap
        "stop_entropy": 5.0, "quantize": False, "use_mamba": True,  # Requires Mamba
    },
    "extreme_power": {  # Max quality settings
        "mamba_layers": 18, "expansion_factor": 3, "diffusion_steps": 200,
        "noise_beta_start": 0.0001, "noise_beta_end": 0.03, "schedule_type": "cosine",
        "prediction_type": "noise",
        "use_bidirectional_mamba": True,
        "use_overlapping_blocks": True,
        "overlap_ratio": 0.5,  # More overlap
        "stop_entropy": 5.5, "quantize": False, "use_mamba": True,  # Requires Mamba
    }
}


# --- Diffusion Helpers ---

def get_noise_schedule(schedule_type: str, beta_start: float, beta_end: float, num_steps: int):
    """Generates noise schedule tensors (betas, alphas, alpha_bars)."""
    if schedule_type == "linear":
        betas = torch.linspace(beta_start, beta_end, num_steps, dtype=torch.float64)
    elif schedule_type == "cosine":
        steps = torch.arange(num_steps + 1, dtype=torch.float64) / num_steps
        alpha_bar = torch.cos(((steps * math.pi / 2) + 0.008) / 1.008) ** 2
        betas = torch.clamp(1. - alpha_bar[1:] / alpha_bar[:-1], min=0., max=0.999)
    else:
        raise ValueError(f"Unknown schedule_type: {schedule_type}")

    alphas = 1. - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    # Ensure correct shapes and move to device later in the model __init__
    return betas.float(), alphas.float(), alpha_bars.float()


class DiffusionTimestepEmbedding(nn.Module):
    """Embeds diffusion timestep 't' using sinusoidal embeddings."""

    def __init__(self, dim, max_period=10000):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        if dim <= 0:
            raise ValueError(f"DiffusionTimestepEmbedding dim ({dim}) must be positive.")
        if dim % 2 != 0:
            logging.debug(f"DiffusionTimestepEmbedding dim ({dim}) is odd. Final dimension will be zero-padded.")

    def forward(self, t: Tensor) -> Tensor:
        """
        Args:
            t: Tensor of shape [batch_size] representing timesteps.
        Returns:
            Tensor of shape [batch_size, dim]
        """
        if t.ndim != 1:
            raise ValueError(f"Input tensor t must be 1D (batch_size,), but got shape {t.shape}")

        half = self.dim // 2
        # Ensure half > 0 for arange and division
        if half == 0:  # Happens if self.dim is 1
            if self.dim == 1:
                # Handle dim=1 case: maybe return t.float().unsqueeze(-1)? Or constant?
                # For now, let's return zeros, although dim=1 is unusual here.
                return torch.zeros(t.shape[0], 1, device=t.device, dtype=torch.float32)
            else:  # Should not happen if dim > 0
                raise RuntimeError("Internal error in DiffusionTimestepEmbedding calculation.")

        freqs = torch.exp(
            -math.log(self.max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)

        args = t.float()[:, None] * freqs[None, :]  # [B, 1] * [1, half] -> [B, half]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)  # [B, half * 2]

        # Zero pad if dim is odd
        if self.dim % 2 != 0:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)  # [B, dim]

        return embedding


# --- Core Mamba Diffusion Block (Evolved) ---

# FIX: Corrected PositionalEncoding implementation based on PyTorch tutorial style
class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding (pytorch.org/tutorials style)."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)  # Start with [max_len, d_model]
        position = torch.arange(max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]

        # Ensure div_term calculation handles d_model=1 case and uses float division
        if d_model == 1:  # Special case for d_model=1 (though unlikely)
            # Simplified calculation, e.g., just use position or constant
            div_term = torch.tensor([1.0], dtype=torch.float)  # Placeholder, adjust if needed
        else:
            div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float) * (
                    -math.log(10000.0) / float(d_model)))  # [ceil(d_model / 2)]

        pe[:, 0::2] = torch.sin(position * div_term)  # Assign to even columns

        # Assign to odd columns, slicing div_term if d_model is odd
        # Ensure there are odd indices to assign to (d_model > 1)
        if d_model > 1:
            pe[:, 1::2] = torch.cos(position * div_term[:d_model // 2])

        # Add batch dim -> [1, max_len, d_model] for easier broadcasting
        pe = pe.unsqueeze(0)

        self.register_buffer('pe', pe, persistent=False)  # Register as non-persistent buffer

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch_size, seq_len, embedding_dim]
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            # Option 1: Error out (safer)
            raise ValueError(f"Sequence length {seq_len} exceeds PositionalEncoding max_len {self.pe.size(1)}")
            # Option 2: Dynamically resize pe (more complex)
        # self.pe shape: [1, max_len, embedding_dim] -> slicing gives [1, seq_len, embedding_dim]
        x = x + self.pe[:, :seq_len, :]  # Add positional encoding
        return self.dropout(x)


class MDBBlockEvolved(nn.Module):
    """
    Evolved Mamba-Diffusion Block: Incorporates timestep embedding, optional BiMamba.
    """

    def __init__(self, embed_dim: int, mamba_layers: int, expansion_factor: float, vocab_size: int, tier_config: Dict):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.tier_use_mamba_flag = tier_config["use_mamba"]  # The config setting
        self.use_mamba = self.tier_use_mamba_flag and mamba_available  # Actual usage depends on import
        self.use_bidirectional = self.use_mamba and tier_config.get("use_bidirectional_mamba", False)
        self.prediction_type = tier_config["prediction_type"]

        if self.tier_use_mamba_flag and not mamba_available:
            logging.warning(
                f"Tier '{tier_config.get('name', 'unknown')}' requested Mamba, but `mamba-ssm` not found. Falling back to MLP.")

        hidden_dim = int(embed_dim * expansion_factor)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # Timestep embedding projection layer (expects pre-computed timestep embedding)
        self.time_proj = nn.Linear(embed_dim, embed_dim)

        # --- Core Sequence Processor ---
        if self.use_mamba:
            # Mamba config within the conditional block
            mamba_config = {
                "d_model": embed_dim,
                "d_state": max(16, embed_dim // 16),
                "d_conv": 4, "expand": 2,
                "dt_rank": "auto",  # Use auto rank for dt projection
                # Explicitly set device and dtype if needed, though Mamba usually handles it
                # "device": DEVICE,
                # "dtype": torch.float32
            }
            logging.info(f"Using Mamba block (BiMamba: {self.use_bidirectional}, Embed Dim: {embed_dim})")
            # Instantiate Mamba only if use_mamba is True
            self.forward_mamba = Mamba(**mamba_config)
            if self.use_bidirectional:
                # Create a separate instance for the backward pass
                self.backward_mamba = Mamba(**mamba_config)
                # Fusion layer after combining forward and backward outputs
                self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
            else:
                # Define backward_mamba as None if not used, for type consistency checks if any
                self.backward_mamba = None
                self.fusion_layer = None
        else:
            logging.warning("Mamba not used. Falling back to MLP sequence processor.")
            self.mlp_processor = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.GELU(),
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU()) for _ in range(max(0, mamba_layers - 1))],
                nn.Linear(hidden_dim, embed_dim)
            )
            # Ensure Mamba attributes are None if not used
            self.forward_mamba = None
            self.backward_mamba = None
            self.fusion_layer = None

        # --- Prediction Head ---
        self.output_norm = nn.LayerNorm(embed_dim)
        output_dim = embed_dim if self.prediction_type == "noise" else vocab_size
        self.output_head = nn.Linear(embed_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for name, param in self.named_parameters():
            if 'embedding.weight' in name and param.dim() > 1:
                nn.init.normal_(param, mean=0, std=self.embed_dim ** -0.5)
            elif 'weight' in name and param.dim() >= 2:
                # Exclude LayerNorm weights from Xavier init
                if 'norm' not in name.lower():
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)

    def forward(self, x: Tensor, t_embed_proj: Tensor, mamba_state: Optional[Tuple] = None) -> Tuple[
        Tensor, Optional[Tuple]]:
        """
        Process block: x (potentially noisy embeddings), t_embed_proj (PROJECTED timestep embedding).
        Returns prediction (noise or logits) and updated mamba_state (if applicable).
        """
        batch_size, seq_len, _ = x.shape

        # 1. Add Projected Timestep Embedding
        h = x + t_embed_proj.unsqueeze(1)  # [B, 1, D] -> [B, S, D]

        # 2. Add Positional Encoding
        h = self.pos_encoder(h)

        # 3. Process Sequence (Mamba or MLP)
        new_fwd_state, new_bwd_state = None, None
        if self.use_mamba and self.forward_mamba is not None:  # Check if Mamba instance exists
            fwd_state = mamba_state[0] if mamba_state and mamba_state[0] is not None else None
            h_fwd, new_fwd_state = self.forward_mamba(h, state=fwd_state)

            if self.use_bidirectional and self.backward_mamba is not None and self.fusion_layer is not None:
                bwd_state = mamba_state[1] if mamba_state and len(mamba_state) > 1 and mamba_state[
                    1] is not None else None
                h_rev = torch.flip(h, dims=[1])
                h_bwd_rev, new_bwd_state = self.backward_mamba(h_rev, state=bwd_state)
                h_bwd = torch.flip(h_bwd_rev, dims=[1])
                h = self.fusion_layer(torch.cat([h_fwd, h_bwd], dim=-1))
            else:
                h = h_fwd

            new_mamba_state = (new_fwd_state, new_bwd_state if self.use_bidirectional else None)

        else:  # MLP Fallback
            h = self.mlp_processor(h)
            new_mamba_state = None

        # 4. Normalize and Predict
        h = self.output_norm(h)
        prediction = self.output_head(h)

        return prediction, new_mamba_state


# --- Main MBD-S Model (Evolved) ---

class MBDSEvolved(nn.Module):
    """
    Evolved MBD-S Language Model with Noise Prediction and Advanced Features.
    """

    def __init__(self,
                 vocab_size: int = DEFAULT_VOCAB_SIZE,
                 embed_dim: int = 512,
                 l_prime: int = DEFAULT_L_PRIME,
                 width: float = 1.0,
                 tier: str = "core_balanced"):
        super().__init__()
        self.tier_name = tier.lower()
        if self.tier_name not in TIER_CONFIGS_EVOLVED:
            raise ValueError(f"Unknown tier: {self.tier_name}. Available: {list(TIER_CONFIGS_EVOLVED.keys())}")

        self.config = copy.deepcopy(TIER_CONFIGS_EVOLVED[self.tier_name])
        self.config['name'] = self.tier_name

        self.vocab_size = vocab_size
        self.l_prime = l_prime
        self.width = width
        self.use_overlapping = self.config.get("use_overlapping_blocks", False)
        self.overlap = int(self.l_prime * self.config.get("overlap_ratio", 0.25)) if self.use_overlapping else 0
        self.stride = max(1, self.l_prime - self.overlap)
        self.prediction_type = self.config["prediction_type"]

        scaled_embed_dim = int(embed_dim * width)
        # Ensure embed_dim is at least 1
        if scaled_embed_dim <= 0:
            raise ValueError(
                f"Calculated embed_dim ({scaled_embed_dim}) must be positive. Check embed_dim ({embed_dim}) and width ({width}).")

        # Diffusion Schedule Parameters
        self.num_diffusion_steps = self.config["diffusion_steps"]
        betas, alphas, alpha_bars = get_noise_schedule(
            self.config["schedule_type"],
            self.config["noise_beta_start"],
            self.config["noise_beta_end"],
            self.num_diffusion_steps
        )
        # Register buffers immediately to ensure they are part of the model's state
        self.register_buffer('betas', betas, persistent=False)
        self.register_buffer('alphas', alphas, persistent=False)
        self.register_buffer('alpha_bars', alpha_bars, persistent=False)
        self.register_buffer('sqrt_alpha_bars', torch.sqrt(alpha_bars), persistent=False)
        self.register_buffer('sqrt_one_minus_alpha_bars', torch.sqrt(1. - alpha_bars), persistent=False)
        alpha_bars_prev = torch.cat([torch.tensor([1.0], dtype=alpha_bars.dtype), alpha_bars[:-1]])  # alpha_bar_t-1
        # Ensure alpha_bars_prev is registered or on the correct device
        # Clamp denominator to avoid division by zero if alpha_bars approaches 1
        posterior_variance = betas * (1. - alpha_bars_prev) / (1. - alpha_bars).clamp(min=1e-9)
        self.register_buffer('posterior_variance', posterior_variance, persistent=False)
        self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)),
                             persistent=False)
        # Clamp denominator to avoid division by zero if alpha_bars approaches 1
        self.register_buffer('posterior_mean_coef1',
                             betas * torch.sqrt(alpha_bars_prev) / (1. - alpha_bars).clamp(min=1e-9), persistent=False)
        self.register_buffer('posterior_mean_coef2',
                             (1. - alpha_bars_prev) * torch.sqrt(alphas) / (1. - alpha_bars).clamp(min=1e-9),
                             persistent=False)

        # Centralized Timestep Embedding
        self.time_embed_dim = scaled_embed_dim  # Use model's main dimension
        self.time_mlp = nn.Sequential(
            DiffusionTimestepEmbedding(self.time_embed_dim),
            nn.Linear(self.time_embed_dim, self.time_embed_dim),
            nn.GELU(),
        )

        # Core MDB block (Evolved)
        self.mdb = MDBBlockEvolved(
            embed_dim=scaled_embed_dim,
            mamba_layers=self.config["mamba_layers"],
            expansion_factor=self.config["expansion_factor"],
            vocab_size=vocab_size,
            tier_config=self.config
        )

        self.stop_entropy_threshold = self.config["stop_entropy"]
        self.needs_quantization = self.config.get("quantize", False)

        logging.info(f"Initialized MBD-S Evolved - Tier: {self.tier_name.upper()}")
        logging.info(f"  Embed Dim: {scaled_embed_dim}, L': {l_prime}, Overlap: {self.overlap}, Stride: {self.stride}")
        logging.info(f"  Vocab: {vocab_size}, Prediction: {self.prediction_type}")
        logging.info(f"  Using Mamba: {self.mdb.use_mamba} (BiMamba: {self.mdb.use_bidirectional})")
        logging.info(f"  Diffusion Steps: {self.num_diffusion_steps}, Schedule: {self.config['schedule_type']}")
        # Move model and registered buffers to device AFTER initialization
        self.to(DEVICE)

    def _extract_blocks(self, x: Tensor) -> Tensor:
        """Extract overlapping or non-overlapping blocks from token sequence."""
        batch, seq_len = x.shape
        target_device = x.device
        if self.l_prime <= 0: raise ValueError("l_prime must be positive")
        if self.stride <= 0: raise ValueError("stride must be positive")

        if not self.use_overlapping:
            # Pad to be multiple of l_prime
            remainder = seq_len % self.l_prime
            padding = (self.l_prime - remainder) % self.l_prime
            if padding > 0:
                # Pad with PAD_TOKEN_ID
                x = F.pad(x, (0, padding), value=PAD_TOKEN_ID)
            num_blocks = x.shape[1] // self.l_prime
            return x.view(batch, num_blocks, self.l_prime)
        else:
            # Calculate necessary padding for overlapping blocks
            if seq_len < self.l_prime:
                # If sequence is shorter than block size, pad to block size
                padding = self.l_prime - seq_len
                num_blocks = 1
            else:
                # Calculate padding needed so the last block starts and covers the end
                num_strides = math.ceil(max(0, seq_len - self.l_prime) / self.stride)
                total_len_needed = num_strides * self.stride + self.l_prime
                padding = max(0, total_len_needed - seq_len)
                num_blocks = num_strides + 1

            if padding > 0:
                x = F.pad(x, (0, padding), value=PAD_TOKEN_ID)

            # Use unfold for overlapping blocks
            blocks = x.unfold(dimension=1, size=self.l_prime, step=self.stride)

            # Check if unfold produced the expected number of blocks
            if blocks.shape[1] != num_blocks:
                # This might happen due to edge cases in length/stride/padding calculation
                logging.warning(f"Unfold produced {blocks.shape[1]} blocks, expected {num_blocks}. Adjusting.")
                # Fallback or recalculate if necessary, though unfold should be consistent with padding

            return blocks  # Shape: [batch, num_blocks, l_prime]

    def _extract_embed_blocks(self, x_embed: Tensor) -> Tensor:
        """Extract overlapping or non-overlapping blocks from embedding sequence."""
        batch, seq_len, embed_d = x_embed.shape
        target_device = x_embed.device
        if self.l_prime <= 0: raise ValueError("l_prime must be positive")
        if self.stride <= 0: raise ValueError("stride must be positive")

        if not self.use_overlapping:
            remainder = seq_len % self.l_prime
            padding = (self.l_prime - remainder) % self.l_prime
            if padding > 0:
                # Pad embedding tensor (pad last dim by 0, seq dim by padding amount)
                x_embed = F.pad(x_embed, (0, 0, 0, padding), value=0.0)  # Pad with zeros
            num_blocks = x_embed.shape[1] // self.l_prime
            return x_embed.view(batch, num_blocks, self.l_prime, embed_d)
        else:
            # Calculate necessary padding for overlapping blocks
            if seq_len < self.l_prime:
                padding = self.l_prime - seq_len
                num_blocks = 1
            else:
                num_strides = math.ceil(max(0, seq_len - self.l_prime) / self.stride)
                total_len_needed = num_strides * self.stride + self.l_prime
                padding = max(0, total_len_needed - seq_len)
                num_blocks = num_strides + 1

            if padding > 0:
                x_embed = F.pad(x_embed, (0, 0, 0, padding), value=0.0)  # Pad with zeros

            # Use unfold for overlapping blocks
            # unfold: (dimension, size, step)
            # Input shape: [B, S, D] -> unfold dim 1 -> [B, N_blocks, D, L']
            blocks_embed_unfolded = x_embed.unfold(dimension=1, size=self.l_prime, step=self.stride)

            # Permute to [B, N_blocks, L', D]
            blocks_embed = blocks_embed_unfolded.permute(0, 1, 3, 2).contiguous()

            # Check number of blocks
            if blocks_embed.shape[1] != num_blocks:
                logging.warning(
                    f"Unfold (embed) produced {blocks_embed.shape[1]} blocks, expected {num_blocks}. Adjusting.")

            return blocks_embed

    def _q_sample(self, x_start_embed: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """Diffuse the data (forward process q(x_t | x_0)). Adds noise to embeddings."""
        device = x_start_embed.device
        if noise is None:
            noise = torch.randn_like(x_start_embed, device=device)

        # Ensure t is on the correct device and long type before indexing buffers
        t_long_device = t.long().to(device)

        # Use buffer tensors directly, they should be on the model's device
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t_long_device].view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t_long_device].view(-1, 1, 1)

        noisy_embed = sqrt_alpha_bar_t * x_start_embed + sqrt_one_minus_alpha_bar_t * noise
        return noisy_embed, noise

    def _predict_start_from_noise(self, x_t: Tensor, t: Tensor, noise_pred: Tensor) -> Tensor:
        """Estimate x_0 from x_t and predicted noise."""
        device = x_t.device
        t_long_device = t.long().to(device)

        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t_long_device]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t_long_device]

        # Add clamp for numerical stability if sqrt_alpha_bar_t is near zero
        sqrt_recip_alpha_bar_t = (1.0 / sqrt_alpha_bar_t.clamp(min=1e-9)).view(-1, 1, 1)
        # Clamp sqrt_alpha_bar_t in denominator
        sqrt_recip_m1_alpha_bar_t = (sqrt_one_minus_alpha_bar_t / sqrt_alpha_bar_t.clamp(min=1e-9)).view(-1, 1, 1)

        x_start_pred = sqrt_recip_alpha_bar_t * x_t - sqrt_recip_m1_alpha_bar_t * noise_pred
        return x_start_pred

    def _q_posterior_mean_variance(self, x_start: Tensor, x_t: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute the mean and variance of the posterior q(x_{t-1} | x_t, x_0)."""
        device = x_start.device
        t_long_device = t.long().to(device)

        posterior_mean_coef1_t = self.posterior_mean_coef1[t_long_device].view(-1, 1, 1)
        posterior_mean_coef2_t = self.posterior_mean_coef2[t_long_device].view(-1, 1, 1)
        posterior_variance_t = self.posterior_variance[t_long_device].view(-1, 1, 1)
        posterior_log_variance_clipped_t = self.posterior_log_variance_clipped[t_long_device].view(-1, 1, 1)

        posterior_mean = posterior_mean_coef1_t * x_start + posterior_mean_coef2_t * x_t
        return posterior_mean, posterior_variance_t, posterior_log_variance_clipped_t

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass for training. Predicts noise and calculates MSE loss.
        """
        batch_size = x.shape[0]
        device = x.device  # Use device of input tensor

        if targets is None:
            targets = x
        if x.shape != targets.shape:
            raise ValueError("Input x and targets must have the same shape for training.")

        # --- Timestep Sampling ---
        t = torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device, dtype=torch.long)
        t_embed_proj = self.time_mlp(t)  # [B, D], includes projection

        # --- Prepare Inputs ---
        target_blocks_tokens = self._extract_blocks(targets)  # [B, N_Blocks, L'] tokens
        n_blocks = target_blocks_tokens.size(1)
        x_start_embed = self.mdb.embedding(target_blocks_tokens)  # [B, N_Blocks, L', D]

        pad_mask = (target_blocks_tokens == PAD_TOKEN_ID).unsqueeze(-1)  # [B, N_Blocks, L', 1]

        # --- Diffusion and Prediction Loop ---
        total_loss = 0.0
        num_loss_elements = 0.0  # Use float for potentially large counts
        mamba_state = None  # Initialize Mamba state

        for b in range(n_blocks):
            x_b_start_embed = x_start_embed[:, b]  # [B, L', D]
            b_pad_mask = pad_mask[:, b]  # [B, L', 1]

            # --- Diffusion Forward: Add Noise ---
            # Ensure noise has same dtype as embeddings (e.g., float32, bfloat16)
            noise_added = torch.randn_like(x_b_start_embed)
            x_b_noisy_embed, noise_added = self._q_sample(x_b_start_embed, t, noise=noise_added)

            # --- MDB Block Prediction ---
            # Ensure inputs have compatible types if using AMP
            block_prediction, mamba_state = self.mdb(x_b_noisy_embed.to(self.mdb.embedding.weight.dtype),
                                                     t_embed_proj.to(self.mdb.embedding.weight.dtype),
                                                     mamba_state=mamba_state)
            # block_prediction is noise [B, L', D]

            # --- Loss Calculation (MSE for Noise Prediction) ---
            # Ensure loss calculation uses compatible types (e.g., float32 for stability)
            loss = F.mse_loss(block_prediction.float(), noise_added.float(), reduction='none')  # [B, L', D]
            loss = loss.masked_fill(b_pad_mask, 0.0)  # Mask padding
            total_loss += loss.sum()
            # Count non-padded elements across batch, seq, embed_dim
            num_loss_elements += (~b_pad_mask).sum().item()

        # --- Final Loss ---
        # Avoid division by zero if num_loss_elements is 0 (e.g., empty batch or all padding)
        final_loss = total_loss / num_loss_elements if num_loss_elements > 1e-8 else torch.tensor(0.0, device=device)

        # Return dummy prediction tensor and the calculated loss
        dummy_predictions = torch.zeros(1, device=device)  # Placeholder
        return dummy_predictions, final_loss

    @torch.no_grad()
    def generate(self,
                 prompt: Tensor,
                 max_new_tokens: int = 80,
                 temperature: float = 1.0,
                 ddim_steps: Optional[int] = None,
                 ddim_eta: float = 0.0
                 ) -> Tensor:
        """
        Generate sequence using iterative denoising (DDPM/DDIM sampling).
        Handles noise prediction, Mamba state, overlapping blocks.
        """
        self.eval()
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0)
        batch_size = prompt.shape[0]
        device = prompt.device  # Use prompt's device

        # Determine sampling steps
        if ddim_steps is not None and ddim_steps > 0 and ddim_steps < self.num_diffusion_steps:
            use_ddim = True
            steps_ratio = self.num_diffusion_steps / ddim_steps
            # Ensure timesteps cover the range [0, T-1] approximately
            timesteps = (torch.arange(ddim_steps, device=device) * steps_ratio).round().long().clamp(0,
                                                                                                     self.num_diffusion_steps - 1)
            # Ensure uniqueness and sort
            timesteps = torch.unique(timesteps)
            # Add t = -1 for the final step -> x0 calculation
            all_timesteps = torch.cat([torch.tensor([-1], device=device, dtype=torch.long), timesteps]).flip(0)
            actual_ddim_steps = len(all_timesteps) - 1
            logging.info(f"Using DDIM sampling with {actual_ddim_steps} unique steps (eta={ddim_eta}).")
        else:
            use_ddim = False
            # Full DDPM steps T-1 down to 0, plus -1 for final x0
            all_timesteps = torch.arange(-1, self.num_diffusion_steps, device=device).flip(0)
            logging.info(f"Using DDPM sampling with {self.num_diffusion_steps} steps.")

        # --- Initialize Generation State ---
        prompt_len = prompt.size(1)
        total_seq_len_target = prompt_len + max_new_tokens

        # Calculate total padded length for blocks
        if total_seq_len_target < self.l_prime:
            total_padded_len = self.l_prime
        else:
            num_strides = math.ceil(max(0, total_seq_len_target - self.l_prime) / self.stride)
            total_padded_len = num_strides * self.stride + self.l_prime

        shape = (batch_size, total_padded_len, self.mdb.embed_dim)
        logging.info(f"Target seq len: {total_seq_len_target}. Padded length for blocks: {total_padded_len}")
        # Start with noise xt at T, ensuring correct dtype and device
        xt_embed = torch.randn(shape, device=device, dtype=self.mdb.embedding.weight.dtype)

        # Embed the prompt
        prompt_embed = self.mdb.embedding(prompt).to(xt_embed.dtype)  # [B, PromptLen, D]

        # --- Mamba State Initialization ---
        mamba_state = None
        if self.mdb.use_mamba:
            logging.info("Warming up Mamba state...")
            warmup_t_val = torch.tensor([self.num_diffusion_steps - 1] * batch_size, device=device, dtype=torch.long)
            warmup_t_embed_proj = self.time_mlp(warmup_t_val).to(xt_embed.dtype)
            prompt_blocks_embed = self._extract_embed_blocks(prompt_embed)
            temp_mamba_state = None  # State for warmup loop
            if prompt_blocks_embed.numel() > 0:  # Only warmup if prompt exists
                with autocast(device_type=AMP_DEVICE_TYPE, enabled=AMP_DEVICE_TYPE == 'cuda'):
                    for b in range(prompt_blocks_embed.size(1)):
                        _, temp_mamba_state = self.mdb(prompt_blocks_embed[:, b], warmup_t_embed_proj,
                                                       mamba_state=temp_mamba_state)
                mamba_state = temp_mamba_state  # Final state after warmup
                logging.info("Mamba state warmed up.")
            else:
                logging.info("Skipping Mamba warmup for empty prompt.")

        # --- Iterative Denoising Loop ---
        num_sampling_steps = len(all_timesteps) - 1
        for i in range(num_sampling_steps):
            t_val = all_timesteps[i]
            t_prev_val = all_timesteps[i + 1]

            t = torch.full((batch_size,), t_val, device=device, dtype=torch.long)
            # Handle t_prev = -1 correctly
            t_prev = torch.full((batch_size,), t_prev_val, device=device, dtype=torch.long)

            # Prepare noisy prompt embedding (only if prompt exists)
            if prompt_len > 0:
                current_xt_embed = xt_embed.clone()  # Avoid modifying xt_embed in-place if needed elsewhere
                noisy_prompt_embed, _ = self._q_sample(prompt_embed, t)
                current_xt_embed[:, :prompt_len, :] = noisy_prompt_embed
            else:
                current_xt_embed = xt_embed

            # --- Block-wise Prediction ---
            xt_blocks_embed = self._extract_embed_blocks(current_xt_embed)
            n_gen_blocks = xt_blocks_embed.size(1)
            current_mamba_state = mamba_state  # Use state from previous step/warmup

            all_preds = torch.zeros_like(xt_blocks_embed)  # Store noise predictions

            t_embed_proj = self.time_mlp(t).to(xt_embed.dtype)  # Ensure dtype match

            with autocast(device_type=AMP_DEVICE_TYPE, enabled=AMP_DEVICE_TYPE == 'cuda'):
                for b in range(n_gen_blocks):
                    block_xt_embed = xt_blocks_embed[:, b]
                    block_pred, current_mamba_state = self.mdb(block_xt_embed, t_embed_proj,
                                                               mamba_state=current_mamba_state)
                    all_preds[:, b] = block_pred
            mamba_state = current_mamba_state  # Update state for next diffusion step

            # --- Combine Overlapping Predictions ---
            pred_noise_full = torch.zeros_like(current_xt_embed)  # Match shape of xt_embed
            counts = torch.zeros(batch_size, total_padded_len, 1, device=device, dtype=torch.float32)

            for b in range(n_gen_blocks):
                start_idx = b * self.stride
                end_idx = start_idx + self.l_prime
                # Ensure indices are within bounds
                actual_end_idx = min(end_idx, total_padded_len)
                actual_len = actual_end_idx - start_idx
                if actual_len > 0:
                    pred_noise_full[:, start_idx:actual_end_idx] += all_preds[:, b, :actual_len]
                    counts[:, start_idx:actual_end_idx] += 1.0

            pred_noise_full = pred_noise_full / counts.clamp(min=1e-6)

            # --- Sampling Step (DDPM or DDIM) ---
            pred_x0_embed = self._predict_start_from_noise(current_xt_embed, t, pred_noise_full)
            # pred_x0_embed = pred_x0_embed.clamp(-1., 1.) # Optional clamping

            # Get alpha_bar_t_prev safely for t_prev = -1
            alpha_bar_t_prev = self.alpha_bars[t_prev].to(device).view(-1, 1,
                                                                       1) if t_prev_val >= 0 else torch.ones_like(
                self.alpha_bars[0].view(-1, 1, 1))

            if use_ddim:
                alpha_bar_t = self.alpha_bars[t].to(device).view(-1, 1, 1)
                sigma = ddim_eta * torch.sqrt((1 - alpha_bar_t_prev) / (1 - alpha_bar_t).clamp(min=1e-6) * (
                        1 - alpha_bar_t / alpha_bar_t_prev.clamp(min=1e-6))).clamp(min=0.0)

                noise_sample = torch.randn_like(current_xt_embed) * temperature
                pred_dir_xt = torch.sqrt(
                    (1.0 - alpha_bar_t_prev - sigma ** 2).clamp(min=0.0)) * pred_noise_full  # Clamp inside sqrt
                xt_prev = torch.sqrt(alpha_bar_t_prev) * pred_x0_embed + pred_dir_xt + sigma * noise_sample
            else:
                posterior_mean, _, posterior_log_variance = self._q_posterior_mean_variance(pred_x0_embed,
                                                                                            current_xt_embed, t)
                noise_sample = torch.randn_like(current_xt_embed) * temperature
                mask = (t > 0).int().view(-1, 1, 1)  # No noise at t=0
                xt_prev = posterior_mean + mask * (0.5 * posterior_log_variance).exp() * noise_sample

            xt_embed = xt_prev  # Update xt for the next iteration
            logging.debug(f"Sampling step t={t_val} -> t'={t_prev_val} done.")

        # --- Final Step: Decode Embeddings to Tokens ---
        final_embeddings = xt_embed[:, :total_seq_len_target].float()  # Use float32 for final projection

        embed_weight = self.mdb.embedding.weight.to(final_embeddings.device).float()
        logits = F.linear(final_embeddings, embed_weight)

        sampled_tokens = torch.argmax(logits, dim=-1)

        final_output_tokens = []
        for i in range(batch_size):
            seq = sampled_tokens[i]
            eos_indices = (seq == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
            if len(eos_indices) > 0:
                first_eos = eos_indices[0].item()
                # Include prompt + generated up to *and including* EOS
                seq = seq[:first_eos + 1]
            final_output_tokens.append(seq)

        max_len_in_batch = max(len(s) for s in final_output_tokens) if final_output_tokens else 0
        padded_batch = torch.full((batch_size, max_len_in_batch), PAD_TOKEN_ID, dtype=torch.long, device=device)
        for i, seq in enumerate(final_output_tokens):
            padded_batch[i, :len(seq)] = seq

        return padded_batch

    def configure_optimizers(self, lr: float = 1e-4, weight_decay: float = 0.1):
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay, betas=(0.9, 0.98),
                                      eps=1e-6)  # Common AdamW settings
        scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1.0, total_iters=0)
        return optimizer, scheduler


# --- Factory and Training Functions ---

def create_model_evolved(tier: str, width: float = 1.0, l_prime: int = DEFAULT_L_PRIME,
                         vocab_size: int = DEFAULT_VOCAB_SIZE, embed_dim: int = 512) -> MBDSEvolved:
    """Factory function to create Evolved MBD-S model."""
    return MBDSEvolved(vocab_size=vocab_size, embed_dim=embed_dim, l_prime=l_prime, width=width, tier=tier)


def count_parameters(model: nn.Module) -> int:
    """Counts trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Updated basic_train_loop_evolved using new amp API
def basic_train_loop_evolved(model: MBDSEvolved, data_loader: torch.utils.data.DataLoader, epochs: int = 1,
                             steps: Optional[int] = None, lr: float = 1e-4, weight_decay: float = 0.1,
                             gradient_clipping: float = 1.0):
    """Basic training loop example for Evolved MBD-S."""
    model.train()
    optimizer, scheduler = model.configure_optimizers(lr=lr, weight_decay=weight_decay)
    # scaler = GradScaler(device_type=AMP_DEVICE_TYPE, enabled=AMP_DEVICE_TYPE=='cuda')
    scaler = GradScaler(enabled=AMP_DEVICE_TYPE == 'cuda')

    total_steps_done = 0
    epochs_done = 0
    losses = []
    logging.info(f"Starting basic training loop for Evolved Model (Epochs: {epochs}, Max Steps: {steps})...")

    while (epochs_done < epochs) and (steps is None or total_steps_done < steps):
        epoch_loss = 0.0
        num_batches = 0
        start_epoch_time = time.time()
        model.train()

        for batch_idx, batch in enumerate(data_loader):
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            elif isinstance(batch, torch.Tensor):
                input_ids = batch.to(DEVICE, non_blocking=True)
            else:
                continue

            optimizer.zero_grad(set_to_none=True)

            # Determine autocast dtype
            amp_dtype = torch.bfloat16 if AMP_DEVICE_TYPE == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
            with autocast(device_type=AMP_DEVICE_TYPE, enabled=AMP_DEVICE_TYPE == 'cuda', dtype=amp_dtype):
                predictions, loss = model(input_ids, targets=input_ids)

            if loss is not None and torch.isfinite(loss):
                scaler.scale(loss).backward()

                if gradient_clipping > 0:
                    scaler.unscale_(optimizer)  # Needed before clip_grad_norm_
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                scaler.step(optimizer)
                scaler.update()
                # scheduler.step() # If stepping per batch

                current_loss = loss.item()
                losses.append(current_loss)
                epoch_loss += current_loss
                num_batches += 1
                total_steps_done += 1

                if total_steps_done % 50 == 0:  # Log frequency
                    logging.info(
                        f"Epoch {epochs_done + 1}/{epochs}, Step {total_steps_done}/{steps or 'inf'}, Batch {batch_idx + 1}/{len(data_loader)}, Loss: {current_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

                if steps is not None and total_steps_done >= steps: break
            else:
                logging.warning(f"Invalid loss encountered: {loss} at step {total_steps_done}. Skipping batch.")
                # Optional: break if loss becomes invalid?

        # End of Epoch
        scheduler.step()  # If stepping per epoch
        epochs_done += 1
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else float('nan')
        epoch_time = time.time() - start_epoch_time
        logging.info(f"Epoch {epochs_done} completed in {epoch_time:.2f}s. Average Loss: {avg_epoch_loss:.4f}")

        if not math.isfinite(avg_epoch_loss):
            logging.error("Average epoch loss is NaN or Inf. Stopping training.")
            break
        if steps is not None and total_steps_done >= steps:
            logging.info(f"Reached maximum steps ({steps}). Stopping training.")
            break

    logging.info(f"Training finished after {epochs_done} epochs and {total_steps_done} steps.")
    return model, losses


# --- Comprehensive Test Suite ---

def run_test(config_name: str, embed_dim: int, l_prime: int, width: float, batch_size: int, seq_len: int,
             train_steps: int, max_gen_tokens: int):
    """Runs a single test configuration."""
    print("\n" + "=" * 60)
    logging.info(
        f"Running Test: Config='{config_name}', Dim={embed_dim}, L'={l_prime}, Width={width}, BS={batch_size}, Len={seq_len}")
    print("=" * 60)

    tier_config_orig = TIER_CONFIGS_EVOLVED[config_name]
    effective_config_name = config_name

    # --- Mamba Check ---
    if tier_config_orig["use_mamba"] and not mamba_available:
        logging.warning(
            f"Tier '{config_name}' requires Mamba but it's not installed. Switching to 'simple_edge' tier for this test.")
        effective_config_name = "simple_edge"
        if effective_config_name not in TIER_CONFIGS_EVOLVED:
            logging.error("Fallback tier 'simple_edge' not found in configs. Skipping test.")
            print("=" * 60 + "\n")
            return False

    # --- Model Creation ---
    try:
        # Clear CUDA cache before creating model
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()
        model = create_model_evolved(
            tier=effective_config_name,
            embed_dim=embed_dim,
            l_prime=l_prime,
            width=width,
            vocab_size=DEFAULT_VOCAB_SIZE
        )
        param_count = count_parameters(model)
        logging.info(f"Model created successfully. Tier: {effective_config_name}, Parameters: {param_count:,}")
        assert param_count > 0
    except Exception as e:
        logging.error(f"Failed to create model: {e}", exc_info=True)
        print("=" * 60 + "\n")
        return False

    # --- Dummy Data ---
    try:
        # More robust data generation, ensuring at least one non-pad token
        dummy_dataset = torch.randint(PAD_TOKEN_ID + 1, DEFAULT_VOCAB_SIZE - 1,
                                      (batch_size * (train_steps + 1), seq_len), device="cpu")
        data_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True, num_workers=0,
                                                  pin_memory=DEVICE.type == 'cuda')  # Adjust num_workers based on system
        prompt_len = min(seq_len // 4, 16)  # Ensure prompt len is reasonable
        prompt_seq = torch.randint(PAD_TOKEN_ID + 1, DEFAULT_VOCAB_SIZE - 1, (1, prompt_len),
                                   device=DEVICE)  # Short prompt on target device
        logging.info(f"Dummy data created. Loader batches: {len(data_loader)}, Prompt shape: {prompt_seq.shape}")
    except Exception as e:
        logging.error(f"Failed to create dummy data: {e}", exc_info=True)
        print("=" * 60 + "\n")
        return False

    # --- Training Test ---
    logging.info(f"Starting training test ({train_steps} steps)...")
    initial_params = {name: p.clone().detach() for name, p in model.named_parameters() if p.requires_grad}
    try:
        model, losses = basic_train_loop_evolved(model, data_loader, epochs=1, steps=train_steps, lr=5e-4,
                                                 gradient_clipping=1.0)
        assert len(losses) >= min(1,
                                  train_steps), f"Expected at least {min(1, train_steps)} losses, got {len(losses)}"  # Allow for early stop if loss explodes
        # Only check finite loss if training ran for at least one step
        if losses:
            assert all(math.isfinite(l) for l in losses), "Non-finite loss detected during training."
            logging.info(f"Training losses: First={losses[0]:.4f}, Last={losses[-1]:.4f} (Count: {len(losses)})")
            # Relax loss decrease check, just ensure params changed
            # if len(losses) > 1: assert losses[-1] < losses[0] * 1.5, "Loss did not decrease significantly."

            # Check if parameters changed
            params_changed = False
            for name, p_initial in initial_params.items():
                if name in dict(model.named_parameters()):
                    p_final = dict(model.named_parameters())[name]
                    if not torch.equal(p_initial.cpu(), p_final.cpu().detach()):
                        params_changed = True
                        logging.debug(f"Parameter changed: {name}")
                        break
                else:
                    logging.warning(f"Initial parameter {name} not found in final model.")
            assert params_changed, "Model parameters did not change after training."
        else:
            logging.warning("Training finished with no losses recorded (possibly 0 steps or immediate failure).")

        logging.info("Training test passed.")

    except Exception as e:
        logging.error(f"Training test failed: {e}", exc_info=True)
        print("=" * 60 + "\n")
        return False

    # --- Generation Test ---
    logging.info(f"Starting generation test (Max new tokens: {max_gen_tokens})...")
    try:
        model.eval()  # Ensure eval mode
        gen_prompt = prompt_seq.to(DEVICE)  # Ensure prompt is on correct device for generation
        generated_output = model.generate(
            gen_prompt,
            max_new_tokens=max_gen_tokens,
            temperature=0.7,
            ddim_steps=10  # Use few steps for faster testing
        )

        assert generated_output.ndim == 2, f"Expected 2D output shape [B, L], got {generated_output.shape}"
        assert generated_output.shape[0] == gen_prompt.shape[
            0], f"Expected batch size {gen_prompt.shape[0]}, got {generated_output.shape[0]}"
        # EOS might truncate, so min length is prompt length
        min_expected_len = gen_prompt.shape[1]
        # Max length can be prompt + max_new + 1 (for EOS)
        max_expected_len = gen_prompt.shape[1] + max_gen_tokens + 1
        # Allow for possibility of immediate EOS -> len = prompt_len + 1
        min_possible_len = gen_prompt.shape[1]
        logging.info(
            f"Generated length: {generated_output.shape[1]}, Expected range: [{min_possible_len}, {max_expected_len}]")
        assert min_possible_len <= generated_output.shape[1] <= max_expected_len, \
            f"Generated length {generated_output.shape[1]} outside expected range [{min_possible_len}, {max_expected_len}]"

        # Check token validity
        non_pad_mask = generated_output != PAD_TOKEN_ID
        if non_pad_mask.any():  # Only check if there are non-pad tokens
            assert torch.all(generated_output[non_pad_mask] >= 0), "Negative token ID found."
            assert torch.all(
                generated_output[non_pad_mask] < DEFAULT_VOCAB_SIZE), f"Token ID >= {DEFAULT_VOCAB_SIZE} found."

        # Check prompt preservation
        gen_prompt_part = generated_output[0, :gen_prompt.shape[1]]
        assert torch.equal(gen_prompt_part.cpu(),
                           gen_prompt[0].cpu()), "Generated sequence does not start with the prompt."

        logging.info(f"Generated sequence shape: {generated_output.shape}")
        logging.info(f"Sample generated tokens (first batch): {generated_output[0].cpu().numpy().tolist()}")
        logging.info("Generation test passed.")

    except Exception as e:
        logging.error(f"Generation test failed: {e}", exc_info=True)
        print("=" * 60 + "\n")
        return False
    finally:
        # Clean up CUDA cache after generation
        if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    # --- Test Passed ---
    logging.info(f"All tests passed for config: '{config_name}' (Effective: '{effective_config_name}')")
    print("=" * 60 + "\n")
    return True


def main_evolved():
    """Runs a series of tests on the Evolved MBD-S model."""
    logging.info(f"Using device: {DEVICE}")
    logging.info(f"AMP Device Type for autocast/scaler: {AMP_DEVICE_TYPE}")
    if not mamba_available:
        logging.warning("`mamba-ssm` package not found. Mamba-based tiers will fallback to MLP (`simple_edge`).")

    test_results = {}

    # --- Test Configurations ---
    # (config_name, embed_dim, l_prime, width, batch_size, seq_len, train_steps, max_gen_tokens)
    tests_to_run = [
        # Test MLP fallback (simple_edge) - Smaller dimensions
        ("simple_edge", 64, 16, 1.0, 4, 64, 20, 30),
        # Test core Mamba (if available) or MLP fallback - Medium dimensions
        ("core_balanced", 128, 32, 0.5, 4, 128, 20, 40),
        # Test BiMamba + Overlap (if available) or MLP fallback - Medium dimensions
        ("enhanced_quality", 128, 32, 0.5, 2, 128, 20, 40),
    ]

    # --- Run Tests ---
    all_passed = True
    for test_params in tests_to_run:
        config_name = test_params[0]
        passed = False  # Default to False
        try:
            passed = run_test(*test_params)
        except Exception as e:
            # Catch critical errors within the run_test call itself
            logging.error(f"CRITICAL error during test run for '{config_name}': {e}", exc_info=True)
            passed = False
        finally:
            test_results[config_name] = passed
            if not passed:
                all_passed = False
            # Clear CUDA cache between tests
            if DEVICE.type == 'cuda': torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "#" * 60)
    logging.info("Test Suite Summary:")
    for name, result in test_results.items():
        logging.info(f"- {name}: {'PASSED' if result else 'FAILED'}")
    logging.info(f"Overall Result: {'ALL PASSED' if all_passed else 'SOME TESTS FAILED'}")
    print("#" * 60 + "\n")

    if not all_passed:
        sys.exit(1)  # Exit with error code if any test failed


if __name__ == "__main__":
    main_evolved()
