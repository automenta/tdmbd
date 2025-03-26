#!/usr/bin/env python3
"""SSMDiffusionLM: State-Space Model Diffusion Language Model."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch.cuda.amp import autocast
from torch import Tensor
import math
import time
import numpy as np
from tqdm.notebook import tqdm # Use standard tqdm if not in notebook

# Constants
VOCAB_SIZE = 50_000  # Default vocabulary size
DEFAULT_L_PRIME = 16  # Default block size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Add [MASK] token? Diffusion works on embeddings, maybe not needed like in BERT
# Add special tokens like [PAD], [BOS], [EOS]
PAD_TOKEN_ID = 0
BOS_TOKEN_ID = 1
EOS_TOKEN_ID = 2
# Adjust vocab size if needed: VOCAB_SIZE = 50_003

# --- Diffusion Schedule ---
def linear_beta_schedule(timesteps: int, beta_start: float = 1e-4, beta_end: float = 0.02) -> Tensor:
    return torch.linspace(beta_start, beta_end, timesteps, device=DEVICE)

def get_diffusion_params(betas: Tensor) -> dict:
    """Precompute diffusion parameters."""
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

    return {
        "betas": betas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas": sqrt_recip_alphas,
        "posterior_variance": posterior_variance,
    }

def extract(a: Tensor, t: Tensor, x_shape: Tuple[int, ...]) -> Tensor:
    """Extract values from a batch of diffusion parameters based on timesteps t."""
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu()) # Get params corresponding to timesteps
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# --- Positional and Time Embeddings ---
class SinusoidalPositionalEmbedding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # [1, max_len, dim]
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        # x shape: [batch, seq_len, dim]
        return self.pe[:, :x.size(1), :]

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        half_dim = dim // 2
        # Original Transformer PE constants: log(10000) / (half_dim - 1)
        # DDPM uses: -log(10000) / half_dim
        self.inv_freq = 1.0 / (10000 ** (torch.arange(0, half_dim, dtype=torch.float32) / half_dim))
        self.register_buffer('inv_freq_buf', self.inv_freq) # Use buffer

    def forward(self, t: Tensor) -> Tensor:
        # t shape: [batch]
        freqs = torch.outer(t.float().to(self.inv_freq_buf.device), self.inv_freq_buf) # [batch, half_dim]
        embeddings = torch.cat([freqs.sin(), freqs.cos()], dim=-1) # [batch, dim]
        # Ensure correct dimensionality if dim is odd
        if self.dim % 2 == 1:
             embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings

# --- Scratch Mamba-like SSM Block ---
class SimpleMambaBlock(nn.Module):
    """
    Simplified Mamba block implementation focusing on the selective scan (S6).
    Uses sequential scan (efficient for inference, slower for training).
    Does NOT implement the full hardware-aware parallel scan optimizations.
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)

        # Linear projections for x, z
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=True,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
        )
        # Activation
        self.activation = nn.SiLU()

        # Projections for SSM parameters (dt, B, C)
        # dt_rank = ceil(d_model / 16) - let's keep it simple for now
        self.dt_rank = math.ceil(self.d_model / 16)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + d_state * 2, bias=False)

        # SSM State parameters (A, D)
        # A islog-parameterized for stability
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)))
        # D is a learned skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, dtype=torch.float32))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _ssm(self, x: Tensor, dt: Tensor, A: Tensor, B: Tensor, C: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Run the selective SSM scan sequentially.
        Args:
            x: Input sequence, shape (batch, seq_len, d_inner)
            dt: Time step delta, shape (batch, seq_len, d_inner)
            A: State transition matrix (diagonal), shape (d_inner, d_state)
            B: Input matrix, shape (batch, seq_len, d_state)
            C: Output matrix, shape (batch, seq_len, d_state)
        Returns:
            y: Output sequence, shape (batch, seq_len, d_inner)
            last_state: Final hidden state, shape (batch, d_inner, d_state)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[-1]

        # Discretize A and B
        # dt is broadcasted across d_state
        delta_A = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0)) # (batch, seq_len, d_inner, d_state)
        delta_B = dt.unsqueeze(-1) * B.unsqueeze(2) # (batch, seq_len, d_inner, d_state)

        h = torch.zeros(batch, d_inner, d_state, device=x.device) # Initial state
        ys = []

        # Sequential scan
        for i in range(seq_len):
            # Update state: h_t = dA * h_{t-1} + dB * x_t
            h = delta_A[:, i] * h + delta_B[:, i] * x[:, i].unsqueeze(-1) # Broadcast x across d_state
            # Calculate output: y_t = C_t * h_t
            y_i = (C[:, i].unsqueeze(1) @ h.unsqueeze(-1)).squeeze(-1).squeeze(-1) # (batch, d_inner) - einsum might be cleaner
            ys.append(y_i)

        y = torch.stack(ys, dim=1) # (batch, seq_len, d_inner)
        return y, h # Return final state

    def forward(self, x: Tensor, h_prev: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x: Input tensor shape (batch, seq_len, d_model)
            h_prev: Previous hidden state from last block, shape (batch, d_inner, d_state)
        Returns:
            output: Output tensor shape (batch, seq_len, d_model)
            h_last: Last hidden state shape (batch, d_inner, d_state)
        """
        batch, seq_len, d_model = x.shape
        assert d_model == self.d_model

        # Input projection and split into x, z
        xz = self.in_proj(x)
        x_in, z = xz.chunk(2, dim=-1) # Both (batch, seq_len, d_inner)

        # 1D convolution: needs (batch, channels, length)
        x_conv = x_in.transpose(1, 2) # (batch, d_inner, seq_len)
        x_conv = self.conv1d(x_conv)[:, :, :seq_len] # Apply conv, truncate padding
        x_conv = x_conv.transpose(1, 2) # (batch, seq_len, d_inner)

        # Apply activation
        x_activated = self.activation(x_conv)

        # Project for SSM parameters
        x_ssm_params = self.x_proj(x_activated) # (batch, seq_len, dt_rank + 2 * d_state)
        dt, B, C = torch.split(
            x_ssm_params,
            [self.dt_rank, self.d_state, self.d_state],
            dim=-1
        ) # dt:(b,l,rank), B:(b,l,state), C:(b,l,state)

        # Parameterize dt: softplus ensures positivity
        dt = F.softplus(dt)

        # Get discretized A (using self.A_log)
        A = -torch.exp(self.A_log.float()) # (d_inner, d_state)

        # Run SSM scan
        if h_prev is not None:
            raise NotImplementedError("Passing initial state `h_prev` to sequential scan needs careful handling. Skipping for now.")
            # If implementing: requires modifying the _ssm loop or initialization.

        # Note: The original Mamba paper uses a more complex parameterization for dt involving dt_rank
        # Simplifying here by directly using dt after softplus. This might affect performance.
        # Also, B and C are sequence-varying, as in Mamba.

        # Run sequential SSM scan
        y, h_last = self._ssm(x_activated, dt, A, B, C)

        # Add skip connection D * x_in (where x_in was before conv/activation)
        y = y + x_in * self.D

        # Apply gating z
        output = y * self.activation(z) # Element-wise multiply

        # Output projection
        output = self.out_proj(output)

        return output, h_last

# --- Main SSM Diffusion Block ---
class SSMDiffusionBlock(nn.Module):
    """Core block combining SSM, time embedding, and denoising."""
    def __init__(self, embed_dim: int, ssm_state_dim: int, ssm_conv_dim: int, ssm_expand: int, time_embed_dim: int):
        super().__init__()
        self.ssm = SimpleMambaBlock(
            d_model=embed_dim,
            d_state=ssm_state_dim,
            d_conv=ssm_conv_dim,
            expand=ssm_expand
        )
        self.time_mlp = nn.Sequential(
            nn.Linear(time_embed_dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Denoising network predicts x0_embed directly
        # Input: SSM output + Time embedding
        self.denoise_mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_t_embed: Tensor, time_emb: Tensor, h_prev: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Args:
            x_t_embed: Noisy embedding block (batch, l_prime, embed_dim)
            time_emb: Diffusion time embedding (batch, time_embed_dim)
            h_prev: Previous SSM state (batch, d_inner, ssm_state_dim)

        Returns:
            pred_x0_embed: Predicted clean embedding block (batch, l_prime, embed_dim)
            h_last: Last SSM state (batch, d_inner, ssm_state_dim)
        """
        # Process sequence with SSM
        ssm_out, h_last = self.ssm(self.norm(x_t_embed), h_prev) # Apply norm before SSM

        # Incorporate time embedding
        time_cond = self.time_mlp(time_emb).unsqueeze(1) # (batch, 1, embed_dim)
        combined = ssm_out + time_cond # Add time conditioning

        # Predict clean embeddings
        pred_x0_embed = self.denoise_mlp(combined)

        return pred_x0_embed, h_last

# --- SSMDiffusionLM Model ---
class SSMDiffusionLM(nn.Module):
    """Block-based Language Model using SSM and Diffusion."""

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 256,
                 l_prime: int = DEFAULT_L_PRIME,
                 ssm_state_dim: int = 16,
                 ssm_conv_dim: int = 4,
                 ssm_expand: int = 2,
                 diffusion_timesteps: int = 100, # Number of diffusion steps T
                 beta_schedule: str = 'linear',
                 ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.l_prime = l_prime
        self.diffusion_timesteps = diffusion_timesteps

        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        self.pos_embedding = SinusoidalPositionalEmbedding(embed_dim, max_len=l_prime * 2) # PE within blocks
        self.time_embedding = TimeEmbedding(embed_dim) # Time embedding dimension matches model dim

        self.ssm_diffusion_block = SSMDiffusionBlock(
            embed_dim=embed_dim,
            ssm_state_dim=ssm_state_dim,
            ssm_conv_dim=ssm_conv_dim,
            ssm_expand=ssm_expand,
            time_embed_dim=embed_dim # Use model dim for time embedding MLP input
        )

        # Output projection (often tied to token_embedding.weight)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        # Weight tying
        self.output_projection.weight = self.token_embedding.weight

        # --- Diffusion Parameters ---
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(diffusion_timesteps)
        else:
            raise NotImplementedError(f"Beta schedule {beta_schedule} not implemented")

        diff_params = get_diffusion_params(betas)
        for name, param in diff_params.items():
            self.register_buffer(name, param)

        self.to(DEVICE)

    def _split_blocks(self, x: Tensor) -> List[Tensor]:
        """Split sequence into blocks of size l_prime."""
        batch, seq_len = x.shape
        # Pad sequence to be divisible by l_prime
        pad_len = (self.l_prime - (seq_len % self.l_prime)) % self.l_prime
        # Pad with PAD_TOKEN_ID
        x_padded = F.pad(x, (0, pad_len), value=PAD_TOKEN_ID)
        return x_padded.view(batch, -1, self.l_prime)

    def _unsplit_blocks(self, blocks: Tensor, original_len: int) -> Tensor:
        """Combine blocks back into a sequence, removing padding."""
        batch, n_blocks, l_prime = blocks.shape
        flat = blocks.view(batch, -1)
        return flat[:, :original_len]

    def q_sample(self, x_start_embed: Tensor, t: Tensor, noise: Optional[Tensor] = None) -> Tensor:
        """Forward diffusion process: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_start_embed)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start_embed.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start_embed.shape)

        return sqrt_alphas_cumprod_t * x_start_embed + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """Calculate diffusion loss (predicting x0)."""
        batch, seq_len = x_start.shape
        x_blocks = self._split_blocks(x_start) # (batch, n_blocks, l_prime)
        batch, n_blocks, l_prime = x_blocks.shape

        # Embed blocks
        x0_embed_blocks = self.token_embedding(x_blocks) # (batch, n_blocks, l_prime, embed_dim)

        # Sample noise and create noisy blocks
        noise = torch.randn_like(x0_embed_blocks)
        # Sample single t per sequence, apply to all blocks for simplicity
        # Alternative: Sample t per block
        t_batch = torch.randint(0, self.diffusion_timesteps, (batch,), device=DEVICE).long()

        # Need to reshape t_batch to match noise/x0 shape for q_sample
        t_expanded = t_batch.view(batch, 1, 1, 1).expand(batch, n_blocks, l_prime, 1)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t_batch, (batch, 1, 1, self.embed_dim))
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t_batch, (batch, 1, 1, self.embed_dim))
        xt_embed_blocks = sqrt_alphas_cumprod_t * x0_embed_blocks + sqrt_one_minus_alphas_cumprod_t * noise

        # Get time embeddings
        time_emb = self.time_embedding(t_batch) # (batch, time_embed_dim)

        # Process blocks sequentially, propagating SSM state
        h_prev = None
        pred_x0_embed_list = []
        total_loss = 0.0
        ssm_states = [] # Keep track of states if needed later

        for b in range(n_blocks):
            xt_b_embed = xt_embed_blocks[:, b] # (batch, l_prime, embed_dim)
            x0_b_embed = x0_embed_blocks[:, b] # Target

            # Add positional encoding within the block
            pos_emb = self.pos_embedding(xt_b_embed)
            xt_b_embed_pos = xt_b_embed + pos_emb

            # Predict x0 for this block
            pred_x0_b_embed, h_last = self.ssm_diffusion_block(xt_b_embed_pos, time_emb, h_prev)
            pred_x0_embed_list.append(pred_x0_b_embed)

            # Calculate loss for this block (MSE on embeddings)
            loss_b = F.mse_loss(pred_x0_b_embed, x0_b_embed)
            total_loss += loss_b

            # Update state for next block - detach to prevent gradients flowing across long sequences?
            # Mamba typically doesn't detach state during training. Let's keep it attached.
            h_prev = h_last
            ssm_states.append(h_last)

        avg_loss = total_loss / n_blocks
        all_pred_x0_embed = torch.stack(pred_x0_embed_list, dim=1) # (batch, n_blocks, l_prime, embed_dim)

        return avg_loss, all_pred_x0_embed

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Forward pass for training."""
        # For diffusion, we need targets (clean data) and predicted clean data
        # Targets are the input 'x' itself.
        batch_size = x.shape[0]
        # Sample timesteps t
        t = torch.randint(0, self.diffusion_timesteps, (batch_size,), device=x.device).long()
        # Calculate loss
        loss, _ = self.p_losses(x, t)
        # During training, we don't typically return logits, just the loss
        return None, loss # Return None for logits as loss is primary output

    # --- Sampling / Generation ---

    @torch.no_grad()
    def p_sample(self, xt_embed_block: Tensor, t: Tensor, t_index: int, h_prev: Optional[Tensor]) -> Tuple[Tensor, Tensor, Tensor]:
        """Single step of denoising: p(x_{t-1} | x_t)."""
        batch_size = xt_embed_block.shape[0]
        time_emb = self.time_embedding(t) # Get time embedding for current t

        # Add positional encoding
        pos_emb = self.pos_embedding(xt_embed_block)
        xt_embed_block_pos = xt_embed_block + pos_emb

        # Predict x0 using the network
        pred_x0_embed, h_last = self.ssm_diffusion_block(xt_embed_block_pos, time_emb, h_prev)

        # Use DDPM sampling formula to get x_{t-1}
        betas_t = extract(self.betas, t, xt_embed_block.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, xt_embed_block.shape)
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, xt_embed_block.shape)
        posterior_variance_t = extract(self.posterior_variance, t, xt_embed_block.shape)

        # Equation 11 in DDPM paper (simplified):
        # mean = sqrt_recip_alphas_t * (xt_embed_block - betas_t * pred_noise / sqrt_one_minus_alphas_cumprod_t)
        # where pred_noise = (xt_embed_block - sqrt_alphas_cumprod_t * pred_x0) / sqrt_one_minus_alphas_cumprod_t

        # Alternative: Use predicted x0 (more stable for L_simple objective)
        # Appendix B in DDPM paper or Ho et al. 2020 Eq. 15 rearranged for x0 prediction
        alpha_t = 1.0 - betas_t
        alphas_cumprod_t = extract(self.alphas_cumprod, t, xt_embed_block.shape)
        alphas_cumprod_prev_t = extract(F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0), t, xt_embed_block.shape)

        pred_x0_coeff = torch.sqrt(alphas_cumprod_prev_t) * betas_t / (1.0 - alphas_cumprod_t)
        xt_coeff = torch.sqrt(alpha_t) * (1.0 - alphas_cumprod_prev_t) / (1.0 - alphas_cumprod_t)

        mean = pred_x0_coeff * pred_x0_embed + xt_coeff * xt_embed_block

        if t_index == 0:
            return mean, pred_x0_embed, h_last # No noise added at the last step
        else:
            noise = torch.randn_like(xt_embed_block)
            # Use posterior_variance or beta_tilde (variance derived from beta)
            variance = torch.sqrt(posterior_variance_t) * noise # Use clipped variance for stability
            return mean + variance, pred_x0_embed, h_last

    @torch.no_grad()
    def generate(self, prompt: Tensor, max_blocks: int = 10, temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
        """Generate sequence block-by-block using DDPM sampling."""
        self.eval()
        batch_size = prompt.shape[0]
        original_len = prompt.shape[1]

        # 1. Process prompt to get initial SSM state
        prompt_blocks = self._split_blocks(prompt) # (batch, n_prompt_blocks, l_prime)
        prompt_embed = self.token_embedding(prompt_blocks)
        h_current = None
        for b in range(prompt_blocks.shape[1]):
            block_embed = prompt_embed[:, b]
            pos_emb = self.pos_embedding(block_embed)
            # Pass through SSM part only to update state
            # We need a dummy time embedding; t=0 might be suitable
            time_emb_dummy = self.time_embedding(torch.zeros(batch_size, device=DEVICE).long())
            # Use the main block, but ignore the prediction, just get the state
            _, h_current = self.ssm_diffusion_block(block_embed + pos_emb, time_emb_dummy, h_current)

        generated_tok_ids = [prompt_blocks[:, i] for i in range(prompt_blocks.shape[1])]

        # 2. Generate new blocks iteratively
        for _ in range(max_blocks):
            # Start with noise for the new block
            xt_embed = torch.randn(batch_size, self.l_prime, self.embed_dim, device=DEVICE)

            # Denoising loop from T down to 0
            for i in tqdm(reversed(range(0, self.diffusion_timesteps)), desc="Sampling Block", total=self.diffusion_timesteps, leave=False):
                t = torch.full((batch_size,), i, device=DEVICE, dtype=torch.long)
                xt_embed, pred_x0_embed, h_last_step = self.p_sample(xt_embed, t, i, h_current)
                # Note: h_current should ideally be constant during the diffusion sampling of *one* block
                # h_last_step is the state *after* processing xt_embed at step t, maybe not needed here.

            # At the end (t=0), xt_embed is ~ x0_embed
            final_pred_x0_embed = xt_embed # Or use pred_x0_embed from last step? Usually xt_embed@t=0

            # Project to logits
            logits = self.output_projection(final_pred_x0_embed) # (batch, l_prime, vocab_size)

            # Apply sampling strategy (temperature, top-k)
            logits = logits / temperature
            if top_k is not None:
                v, _ = torch.topk(logits, top_k, dim=-1)
                logits[logits < v[..., -1:]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            sampled_ids = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(batch_size, self.l_prime) # (batch, l_prime)

            # Check for EOS token - maybe stop generation for specific sequences in batch?
            # This basic implementation generates full blocks.

            generated_tok_ids.append(sampled_ids)

            # 3. Update SSM state using the *sampled* block for the next generation step
            sampled_embed = self.token_embedding(sampled_ids)
            pos_emb = self.pos_embedding(sampled_embed)
            time_emb_dummy = self.time_embedding(torch.zeros(batch_size, device=DEVICE).long())
            _, h_current = self.ssm_diffusion_block(sampled_embed + pos_emb, time_emb_dummy, h_current)

        # Concatenate all blocks and remove padding/prompt overlap
        all_blocks_tensor = torch.stack(generated_tok_ids, dim=1) # (batch, n_total_blocks, l_prime)
        # Calculate expected final length
        final_len = original_len + max_blocks * self.l_prime
        # Unsplit/flatten and trim
        output_sequence = self._unsplit_blocks(all_blocks_tensor, final_len)

        return output_sequence


# === Analysis and Comparison Setup ===

# --- Baseline Model: Simple Transformer Decoder ---
class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_heads, n_layers, dim_feedforward, max_seq_len=512):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.token_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        self.pos_embedding = SinusoidalPositionalEmbedding(embed_dim, max_len=max_seq_len)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation='relu',
            batch_first=True, # Crucial!
            norm_first=True # Pre-LN is often more stable
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.output_projection = nn.Linear(embed_dim, vocab_size)
        self.output_projection.weight = self.token_embedding.weight # Weight tying
        self.max_seq_len = max_seq_len
        self.to(DEVICE)

    def _generate_square_subsequent_mask(self, sz: int) -> Tensor:
        """Generates an upper-triangular matrix of -inf, with zeros on diag."""
        return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1).to(DEVICE)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        # x shape: [batch, seq_len]
        seq_len = x.size(1)
        if seq_len > self.max_seq_len:
             # Simple truncation, could use sliding window etc. for longer seqs
            x = x[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        causal_mask = self._generate_square_subsequent_mask(seq_len)
        padding_mask = (x == PAD_TOKEN_ID) # [batch, seq_len], True where padded

        embed = self.token_embedding(x) * math.sqrt(self.embed_dim) # Scaling
        pos_embed = self.pos_embedding(embed)
        x_in = embed + pos_emb

        transformer_output = self.transformer_encoder(
            x_in,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        )
        logits = self.output_projection(transformer_output) # [batch, seq_len, vocab_size]

        loss = None
        if targets is not None:
            # Shift logits and targets for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_targets = targets[..., 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_targets.view(-1), ignore_index=PAD_TOKEN_ID)

        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: Tensor, max_new_tokens: int = 50, temperature: float = 1.0, top_k: Optional[int] = None) -> Tensor:
        self.eval()
        current_tokens = prompt.clone() # [batch, seq_len]

        for _ in range(max_new_tokens):
            input_ids = current_tokens
            if input_ids.size(1) > self.max_seq_len:
                input_ids = input_ids[:, -self.max_seq_len:] # Keep only last part

            logits, _ = self.forward(input_ids) # Get logits for the whole sequence
            next_token_logits = logits[:, -1, :] # Logits for the very next token [batch, vocab_size]

            # Apply sampling
            next_token_logits = next_token_logits / temperature
            if top_k is not None:
                v, _ = torch.topk(next_token_logits, top_k, dim=-1)
                next_token_logits[next_token_logits < v[..., -1:]] = -float('Inf')

            probs = F.softmax(next_token_logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1) # [batch, 1]

            current_tokens = torch.cat([current_tokens, next_token_id], dim=1)

            # Simple stopping condition (optional)
            if (next_token_id == EOS_TOKEN_ID).all():
                 break

        return current_tokens

# --- Evaluation Functions ---

def calculate_perplexity(model: nn.Module, data_loader: torch.utils.data.DataLoader, model_type: str) -> float:
    """Calculate perplexity on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Calculating Perplexity"):
            # Assuming batch is a tensor of shape [batch_size, seq_len]
            input_ids = batch.to(DEVICE)
            targets = input_ids # Use input as target for LM

            if model_type == 'ssm_diffusion':
                 # Need to run the forward diffusion process and get loss
                 t = torch.randint(0, model.diffusion_timesteps, (input_ids.shape[0],), device=DEVICE).long()
                 loss, _ = model.p_losses(input_ids, t)
                 # MSE loss isn't directly perplexity. We need cross-entropy.
                 # For PPL, maybe run generation for 1 step (t=0 prediction) and calc CE?
                 # Or add a separate CE head trained alongside? This complicates things.
                 # Let's just report the MSE loss for diffusion for now, PPL is tricky.
                 # OR: approximate PPL using the final projection after sampling t=0
                 xt_embed = model.token_embedding(input_ids)
                 logits_approx = model.output_projection(xt_embed) # Very rough approx
                 targets_shifted = targets[..., 1:].contiguous()
                 logits_shifted = logits_approx[..., :-1, :].contiguous()
                 loss = F.cross_entropy(logits_shifted.view(-1, model.vocab_size), targets_shifted.view(-1), ignore_index=PAD_TOKEN_ID)

            elif model_type == 'transformer':
                _, loss = model(input_ids, targets) # Model calculates CE loss internally

            else: # MBD-S style model (if we adapt it)
                # Assuming MBD-S returns CE loss
                _, loss = model(input_ids, targets)

            # Calculate number of non-padding tokens in the target part
            mask = (targets[:, 1:] != PAD_TOKEN_ID).view(-1)
            num_tokens = mask.sum().item()

            # Accumulate loss weighted by number of tokens
            total_loss += loss.item() * num_tokens # Use item() to free graph
            total_tokens += num_tokens

    if total_tokens == 0: return float('inf')
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def measure_speed(model: nn.Module, input_data: Tensor, generation_func, gen_args: dict, num_runs: int = 10) -> Tuple[float, float]:
    """Measure training step time and generation time."""
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) # Dummy optimizer

    # Measure training step time
    start_times = []
    end_times = []
    if isinstance(model, SSMDiffusionLM): # Diffusion training step
        t = torch.randint(0, model.diffusion_timesteps, (input_data.shape[0],), device=DEVICE).long()
        for _ in range(num_runs + 1): # +1 for warmup
            start = time.time()
            optimizer.zero_grad()
            loss, _ = model.p_losses(input_data, t)
            loss.backward()
            optimizer.step()
            end = time.time()
            if _ > 0: # Skip warmup
              start_times.append(start)
              end_times.append(end)
    else: # Standard LM training step
       targets = input_data
       for _ in range(num_runs + 1): # +1 for warmup
            start = time.time()
            optimizer.zero_grad()
            _, loss = model(input_data, targets)
            loss.backward()
            optimizer.step()
            end = time.time()
            if _ > 0: # Skip warmup
               start_times.append(start)
               end_times.append(end)

    train_time_avg = np.mean(np.array(end_times) - np.array(start_times))

    # Measure generation time
    model.eval()
    gen_start_times = []
    gen_end_times = []
    with torch.no_grad():
        for _ in range(num_runs + 1): # +1 for warmup
            start = time.time()
            _ = generation_func(input_data, **gen_args)
            end = time.time()
            if _ > 0: # Skip warmup
                gen_start_times.append(start)
                gen_end_times.append(end)

    gen_time_avg = np.mean(np.array(gen_end_times) - np.array(gen_start_times))

    return train_time_avg, gen_time_avg


# --- Main Analysis Function ---
def run_analysis():
    print(f"Using device: {DEVICE}")

    # --- 1. Setup: Data, Models ---
    SEQ_LEN = 64
    L_PRIME = 16 # Block size for SSMDiffusionLM
    BATCH_SIZE = 16 # Keep small for demo
    VOCAB_SIZE = 1000 # Small vocab for faster testing
    EMBED_DIM = 128 # Small embed dim
    N_HEADS = 4
    N_LAYERS = 4
    DIM_FF = EMBED_DIM * 4
    SSM_STATE = 16
    SSM_CONV = 4
    SSM_EXPAND = 2
    DIFFUSION_T = 50 # Fewer steps for faster testing

    # Dummy Data (replace with real data loader, e.g., TinyShakespeare)
    dummy_data = torch.randint(PAD_TOKEN_ID + 3, VOCAB_SIZE, (BATCH_SIZE * 10, SEQ_LEN), device=DEVICE) # 10 batches
    dummy_loader = torch.utils.data.DataLoader(dummy_data, batch_size=BATCH_SIZE)
    prompt = dummy_data[:BATCH_SIZE // 2, :SEQ_LEN // 4].clone() # Use half batch for generation prompt

    print("\n--- Model Initialization ---")
    # SSM Diffusion Model
    ssm_model = SSMDiffusionLM(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, l_prime=L_PRIME,
        ssm_state_dim=SSM_STATE, ssm_conv_dim=SSM_CONV, ssm_expand=SSM_EXPAND,
        diffusion_timesteps=DIFFUSION_T
    ).to(DEVICE)
    print(f"SSMDiffusionLM parameters: {sum(p.numel() for p in ssm_model.parameters() if p.requires_grad):,}")

    # Transformer Baseline
    transformer_model = SimpleTransformerLM(
        vocab_size=VOCAB_SIZE, embed_dim=EMBED_DIM, n_heads=N_HEADS, n_layers=N_LAYERS,
        dim_feedforward=DIM_FF, max_seq_len=SEQ_LEN * 2 # Allow generation beyond initial length
    ).to(DEVICE)
    print(f"SimpleTransformerLM parameters: {sum(p.numel() for p in transformer_model.parameters() if p.requires_grad):,}")

    # Optional: MBD-S Baseline (using the provided code, adapted slightly)
    # Need to ensure vocab size matches, create_model exists, etc.
    try:
        # Assuming the provided MBD-S code is in a file named mbd_s_original.py
        # from mbd_s_original import create_model as create_mbds_model
        # mbds_model = create_mbds_model(tier='core', width=0.5, l_prime=L_PRIME, vocab_size=VOCAB_SIZE).to(DEVICE) # Adjust width for param count
        # print(f"MBD-S (Core) parameters: {sum(p.numel() for p in mbds_model.parameters() if p.requires_grad):,}")
        mbds_model = None # Placeholder if not running MBD-S
    except ImportError:
        print("MBD-S model code not found, skipping comparison.")
        mbds_model = None

    models = {
        "SSM_Diffusion": ssm_model,
        "Transformer": transformer_model,
        # "MBD_S_Core": mbds_model # Add if available
    }

    # --- 2. Training (Short Demo) ---
    print("\n--- Training Loop (Demo) ---")
    EPOCHS = 2 # Very few epochs for demonstration
    LR = 3e-4
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=LR) for name, model in models.items() if model is not None}

    for epoch in range(EPOCHS):
        for name, model in models.items():
            if model is None: continue
            model.train()
            epoch_loss = 0
            batch_count = 0
            for batch_data in tqdm(dummy_loader, desc=f"Epoch {epoch+1}/{EPOCHS} - {name}", leave=False):
                input_ids = batch_data.to(DEVICE)
                optimizers[name].zero_grad()

                if name == "SSM_Diffusion":
                    t = torch.randint(0, model.diffusion_timesteps, (input_ids.shape[0],), device=DEVICE).long()
                    loss, _ = model.p_losses(input_ids, t)
                else: # Transformer or MBD-S
                    targets = input_ids
                    _, loss = model(input_ids, targets)

                loss.backward()
                optimizers[name].step()
                epoch_loss += loss.item()
                batch_count += 1
            avg_epoch_loss = epoch_loss / batch_count
            print(f"Epoch {epoch+1} - {name} Avg Loss: {avg_epoch_loss:.4f}")

    # --- 3. Evaluation: Perplexity ---
    print("\n--- Perplexity Evaluation ---")
    ppl_results = {}
    for name, model in models.items():
        if model is None: continue
        if name == "SSM_Diffusion":
             # PPL calculation for diffusion is complex/approximate. Report MSE loss instead?
             print(f"SSM_Diffusion: PPL calculation is approximate/complex. Reporting final training loss as proxy.")
             ppl_results[name] = avg_epoch_loss # Using last training loss as a rough proxy
        else:
             ppl = calculate_perplexity(model, dummy_loader, 'transformer' if name=='Transformer' else 'mbd_s')
             ppl_results[name] = ppl
             print(f"{name} Perplexity: {ppl:.2f}")

    # --- 4. Evaluation: Speed ---
    print("\n--- Speed Evaluation ---")
    speed_results = {}
    gen_args_ssm = {"max_blocks": 2, "temperature": 0.7}
    gen_args_transformer = {"max_new_tokens": L_PRIME * 2, "temperature": 0.7} # Generate similar length
    # gen_args_mbds = {"max_blocks": 2, "temperature": 0.7}

    test_batch = next(iter(dummy_loader)).to(DEVICE)

    for name, model in models.items():
        if model is None: continue
        gen_func = model.generate
        gen_args = {}
        if name == "SSM_Diffusion": gen_args = gen_args_ssm
        elif name == "Transformer": gen_args = gen_args_transformer
        # elif name == "MBD_S_Core": gen_args = gen_args_mbds

        train_speed, gen_speed = measure_speed(model, test_batch, gen_func, gen_args)
        speed_results[name] = {"train_step_ms": train_speed * 1000, "generation_s": gen_speed}
        print(f"{name} - Avg Train Step Time: {train_speed*1000:.2f} ms")
        print(f"{name} - Avg Generation Time (for ~{L_PRIME*2} tokens): {gen_speed:.4f} s")

    # --- 5. Evaluation: Generation Quality (Qualitative) ---
    print("\n--- Generation Examples ---")
    gen_args_ssm["max_blocks"] = 4 # Generate longer sequence
    gen_args_transformer["max_new_tokens"] = L_PRIME * 4

    for name, model in models.items():
         if model is None: continue
         model.eval()
         with torch.no_grad():
            gen_func = model.generate
            gen_args = {}
            if name == "SSM_Diffusion": gen_args = gen_args_ssm
            elif name == "Transformer": gen_args = gen_args_transformer
            # elif name == "MBD_S_Core": gen_args = gen_args_mbds

            print(f"\n{name} Generation:")
            print(f"Prompt (IDs): {prompt[0].tolist()}")
            generated_ids = gen_func(prompt[:1], **gen_args) # Generate for first prompt in batch
            print(f"Generated (IDs): {generated_ids[0].tolist()}")
            # Add token decoding here if you have a tokenizer
            # print(f"Generated (Text): {tokenizer.decode(generated_ids[0].tolist())}")

    # --- 6. Analysis Summary ---
    print("\n--- Analysis Summary ---")
    print("Perplexity (Lower is Better):")
    for name, ppl in ppl_results.items():
        note = "(Approx. Loss)" if name == "SSM_Diffusion" else ""
        print(f"  {name}: {ppl:.2f} {note}")

    print("\nSpeed (Lower is Better):")
    for name, speed in speed_results.items():
        print(f"  {name}: Train Step={speed['train_step_ms']:.2f} ms, Generation={speed['generation_s']:.4f} s")

    print("\nQualitative Generation:")
    print("  Review the generated token ID sequences above.")

    print("\n--- Further Tuning & Amplification ---")
    print("Based on these results, consider:")
    print("1. Scaling: Increase embed_dim, ssm_state_dim, layers, heads.")
    print("2. Diffusion Steps: Adjust `diffusion_timesteps` (more steps might improve quality but slow sampling).")
    print("3. SSM Implementation: A parallel scan implementation would drastically speed up training.")
    print("4. Block Size (`l_prime`): Larger blocks capture more local context but increase computation per block.")
    print("5. Training Data/Time: Train on a larger, more realistic dataset for longer.")
    print("6. Hyperparameters: Tune learning rate, beta schedule, temperature, top-k.")
    print("7. Context Propagation: Investigate detaching SSM state `h` between blocks during training if memory becomes an issue.")


if __name__ == "__main__":
    run_analysis()