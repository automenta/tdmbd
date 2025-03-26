#!/usr/bin/env python3
"""
HydraScale V2.1: Improved Analysis and Tuning Guidance for a
Language Model using Selective Scan (SSM) and Discrete Diffusion.

This version focuses on:
1. Enhancing the analysis section with detailed tuning guidance.
2. Providing clearer comparative metrics against a Transformer baseline.
3. Refining code for clarity, compactness, and best practices.
4. Maintaining the core SSM and Diffusion mechanisms.

**Key Limitation**: The `forward` pass of `SelectiveScan` uses a Python loop
for the scan operation. This is **computationally inefficient for training**
compared to parallel scan implementations (e.g., in CUDA). However, the
`step` method uses the efficient recurrent formulation for generation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.cuda.amp import autocast, GradScaler
from torch import Tensor
import math
import time
import numpy as np
from tqdm import tqdm  # Optional: for progress bars

# --- Constants ---
VOCAB_SIZE = 50_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_L_PRIME = 0  # Currently unused, full sequence processing

# --- Custom SSM Implementation (Inspired by Mamba/S6) ---
class SelectiveScan(nn.Module):
    """
    Custom implementation of the Selective Scan mechanism (S6).

    Uses a sequential scan in `forward` (SLOW for training) and an efficient
    recurrent calculation in `step` (FAST for inference).

    Args:
        embed_dim (int): Input/output dimension (D).
        state_dim (int): State dimension (N). Typically small (e.g., 16, 32).
        d_conv (int): Local convolution width (e.g., 3, 4).
        dt_rank (int or str): Rank for delta_t projection ('auto' or int).
                         'auto' defaults to ceil(embed_dim / 16).
        bias (bool): Whether to include bias in linear layers.
    """

    def __init__(self, embed_dim: int, state_dim: int = 16, d_conv: int = 4, dt_rank: str = 'auto', bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.d_conv = d_conv
        self.dt_rank = math.ceil(embed_dim / 16) if dt_rank == 'auto' else dt_rank
        self.bias = bias

        # Input-dependent projections (Linear layers)
        self.in_proj = nn.Linear(embed_dim, 2 * embed_dim + self.dt_rank + 2 * self.state_dim, bias=bias)

        # Local convolution (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim, bias=bias,
            kernel_size=d_conv, groups=embed_dim, padding=d_conv - 1,
        )

        # Delta_t projection (learnable)
        self.dt_proj = nn.Linear(self.dt_rank, embed_dim, bias=True)
        # Initialize dt_proj bias near zero for stability following Mamba init
        nn.init.constant_(self.dt_proj.bias, 0.0) # Small positive init might also work

        # State matrix A (learnable log-diagonal)
        # Initialized following Mamba: Range 1..N for diversity
        self.A_log = nn.Parameter(
             torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).unsqueeze(0).repeat(embed_dim, 1) # [D, N]
        )
        # Ensure A is negative (stability)
        self.A = nn.Parameter(-torch.exp(self.A_log.float()), requires_grad=False) # [D, N] Cached negative A

        # Residual D parameter (learnable)
        self.D = nn.Parameter(torch.ones(embed_dim))  # [D]

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _compute_A_tilde(self, dt: Tensor) -> Tensor:
        """ Discretize continuous A using Zero-Order Hold (ZOH). dt: [B, L, D] """
        # A is pre-computed and stored as negative exponentiated log
        # A_tilde = exp(dt * A)
        A_tilde = torch.exp(dt.unsqueeze(-1) * self.A.unsqueeze(0).unsqueeze(1)) # [B, L, D, N]
        return A_tilde

    def _compute_B_tilde(self, dt: Tensor, A_tilde: Tensor) -> Tensor:
        """ Discretize continuous B using ZOH variant. dt: [B, L, D], A_tilde: [B, L, D, N] """
        # B_tilde = (exp(dt * A) - 1) / A * B = (A_tilde - 1) / A * B
        # We compute the (A_tilde - 1) / A part here. B is input-dependent.
        # Handle A=0 case numerically (shouldn't happen with exp init)
        A_unsqueeze = self.A.unsqueeze(0).unsqueeze(1) # [1, 1, D, N]
        B_tilde_base = (A_tilde - 1) / (A_unsqueeze + 1e-10) # Avoid division by zero [B, L, D, N]
        # Mamba paper suggests B_tilde ≈ dt * B for small dt. Here we use the exact formula base.
        # The input-dependent B_proj will be multiplied later.
        return B_tilde_base

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using **sequential scan**. SLOW for training.
        x: [batch, seq_len, embed_dim] (B, L, D)
        """
        batch, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        # --- Input Projections ---
        # Project x to get inputs for convolution (x_conv_in), gating (z), and SSM params (dt, B, C)
        projected = self.in_proj(x) # [B, L, 2D + dt_rank + 2N]
        x_conv_in, z, dt_unproj, B_proj, C_proj = torch.split(
            projected,
            [self.embed_dim, self.embed_dim, self.dt_rank, self.state_dim, self.state_dim],
            dim=-1
        )
        # dt_unproj: [B, L, dt_rank], B_proj: [B, L, N], C_proj: [B, L, N], z: [B, L, D]

        # --- Convolution Path ---
        # Apply causal convolution
        x_conv = self.conv1d(x_conv_in.transpose(1, 2)) # [B, D, L + pad]
        x_conv = x_conv[:, :, :seq_len] # Remove padding: [B, D, L]
        # Apply activation (SiLU/Swish)
        u = F.silu(x_conv.transpose(1, 2)) # [B, L, D] - Input 'u' to the scan

        # --- SSM Path ---
        # Compute delta_t (input-dependent timestep)
        dt = F.softplus(self.dt_proj(dt_unproj)) # [B, L, D] - Ensure positivity

        # Discretize A and B based on dt
        A_tilde = self._compute_A_tilde(dt)      # [B, L, D, N] - State transition matrix
        B_tilde_base = self._compute_B_tilde(dt, A_tilde) # [B, L, D, N] - Base for input matrix

        # Combine B_tilde_base with input-dependent B_proj
        # B_tilde = B_tilde_base * B_proj (needs careful broadcast/einsum)
        # Input term in scan: B_tilde * u = (B_tilde_base * B_proj) * u
        # We will compute this inside the loop for clarity.

        # *** Sequential Scan Placeholder ***
        # SLOW: Iterates through sequence length. Replace with parallel scan for training.
        h = torch.zeros(batch, self.embed_dim, self.state_dim, device=x.device, dtype=A_tilde.dtype) # [B, D, N] State
        ys = []
        for t in range(seq_len):
            A_t = A_tilde[:, t, :, :]      # [B, D, N]
            B_base_t = B_tilde_base[:, t, :, :]  # [B, D, N]
            B_proj_t = B_proj[:, t, :]      # [B, N]
            C_proj_t = C_proj[:, t, :]      # [B, N]
            u_t = u[:, t, :]              # [B, D]

            # Calculate effective B_tilde for this step
            # B_tilde_t = B_base_t * B_proj_t.unsqueeze(1) # [B, D, N] * [B, 1, N] -> [B, D, N]
            B_tilde_t = torch.einsum('bdn,bn->bdn', B_base_t, B_proj_t) # Cleaner

            # State Update: h_t = A_t * h_{t-1} + B_tilde_t * u_t
            # Input term: einsum('bdn,bd->bdn', B_tilde_t, u_t)
            input_term = torch.einsum('bdn,bd->bdn', B_tilde_t, u_t) # [B, D, N]
            h = A_t * h + input_term  # Update state [B, D, N]

            # Output Calculation: y_t = C_proj_t * h_t (sum over state dim N)
            y_t = torch.einsum('bn,bdn->bd', C_proj_t, h) # [B, D]
            ys.append(y_t)

        y = torch.stack(ys, dim=1) # [B, L, D] - Combined output sequence
        # *** End Sequential Scan Placeholder ***

        # Add residual D term (skip connection based on 'u')
        y = y + u * self.D.unsqueeze(0).unsqueeze(1) # [B, L, D]

        # Apply gating (similar to GLU)
        y = y * F.silu(z) # [B, L, D]

        # Final output projection
        y = self.out_proj(y) # [B, L, D]
        return y

    def step(self, x_step: Tensor, h_prev: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Recurrent step function for efficient inference/generation.
        Processes one input step `x_step` and updates the hidden state `h_prev`
        and convolution state `conv_state`.

        Args:
            x_step: [batch, embed_dim] (B, D) - Current input element
            h_prev: [batch, embed_dim, state_dim] (B, D, N) - Previous SSM state
            conv_state: [batch, embed_dim, d_conv - 1] - Previous conv inputs

        Returns:
            y_step: [batch, embed_dim] (B, D) - Output for current step
            h: [batch, embed_dim, state_dim] (B, D, N) - New SSM state
            new_conv_state: [batch, embed_dim, d_conv - 1] - Updated conv state
        """
        batch, embed_dim = x_step.shape

        # --- Input Projections for this step ---
        projected = self.in_proj(x_step) # [B, 2D + dt_rank + 2N]
        x_conv_in, z, dt_unproj, B_proj, C_proj = torch.split(
            projected,
            [self.embed_dim, self.embed_dim, self.dt_rank, self.state_dim, self.state_dim],
            dim=-1
        ) # Shapes: x_conv_in[B,D], z[B,D], dt_unproj[B,dt_rank], B_proj[B,N], C_proj[B,N]

        # --- Handle Convolution State ---
        conv_input = torch.cat([conv_state, x_conv_in.unsqueeze(2)], dim=2) # [B, D, d_conv]
        new_conv_state = conv_input[:, :, 1:] # Update state for next step [B, D, k-1]

        # Apply convolution for this step
        conv_out = F.conv1d(
            conv_input,
            self.conv1d.weight, # [D, 1, k]
            self.conv1d.bias,   # [D]
            padding=0,
            groups=self.embed_dim
        ).squeeze(-1) # Output: [B, D]

        # Activation 'u' from convolution output
        u = F.silu(conv_out) # [B, D]

        # --- SSM Step Logic ---
        # Compute delta_t for this step
        dt = F.softplus(self.dt_proj(dt_unproj)) # [B, D]

        # Discretize A, B for this step (simplified notation)
        A_tilde_step = torch.exp(dt.unsqueeze(-1) * self.A.unsqueeze(0)) # [B, D, N]
        # B_tilde_base calculation
        A_unsqueeze = self.A.unsqueeze(0) # [1, D, N]
        B_tilde_base_step = (A_tilde_step - 1) / (A_unsqueeze + 1e-10) # [B, D, N]

        # Combine B_tilde_base with input-dependent B_proj
        # B_tilde_step = B_tilde_base_step * B_proj.unsqueeze(1)
        B_tilde_step = torch.einsum('bdn,bn->bdn', B_tilde_base_step, B_proj) # [B, D, N]

        # State Update: h_t = A_t * h_{t-1} + B_t * u_t
        # Input term einsum: einsum('bdn,bd->bdn', B_tilde_step, u)
        input_term = torch.einsum('bdn,bd->bdn', B_tilde_step, u) # [B, D, N]
        h = A_tilde_step * h_prev + input_term # [B, D, N] - New state

        # Output Calculation: y_t = C_proj_t * h_t + D * u_t
        # Output einsum: einsum('bn, bdn -> bd', C_proj, h)
        y = torch.einsum('bn,bdn->bd', C_proj, h) # [B, D]
        y = y + u * self.D.unsqueeze(0) # Add residual D term [B, D]

        # Apply gating
        y = y * F.silu(z) # [B, D]

        # Final output projection
        y_step = self.out_proj(y) # [B, D]

        return y_step, h, new_conv_state

# --- HydraScale Components ---

class HydraBlock(nn.Module):
    """ Combines Selective Scan with LayerNorm and MLP in a Pre-Norm residual block. """
    def __init__(self, embed_dim: int, mlp_mult: int = 4, **ssm_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ssm = SelectiveScan(embed_dim, **ssm_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = embed_dim * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(), # Or nn.SiLU()
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm architecture: Residual -> Norm -> Module -> Output
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x_step: Tensor, ssm_state: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Step function for recurrent inference, managing states for one block. """
        # Input: x_step [B, D], ssm_state [B, D, N], conv_state [B, D, k-1]
        # Output: y_step [B, D], new_ssm_state [B, D, N], new_conv_state [B, D, k-1]

        # SSM part (Pre-Norm)
        residual = x_step
        x_norm1 = self.norm1(x_step)
        ssm_out, ssm_state_new, conv_state_new = self.ssm.step(x_norm1, ssm_state, conv_state)
        x = residual + ssm_out  # Apply residual after SSM

        # MLP part (Pre-Norm)
        residual = x
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        y_step = residual + mlp_out # Apply residual after MLP

        return y_step, ssm_state_new, conv_state_new

def sinusoidal_embedding(timesteps: Tensor, embedding_dim: int):
    """ Sinusoidal time embedding, adapted from Attention is All You Need. """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    # Ensure timesteps are float for multiplication
    emb = timesteps.float().unsqueeze(1) * emb.unsqueeze(0) # [B, half_dim]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # [B, embedding_dim or embedding_dim-1]
    if embedding_dim % 2 == 1:  # zero pad if dim is odd
        emb = F.pad(emb, (0, 1))
    return emb # [B, embedding_dim]

class HydraScaleLM(nn.Module):
    """
    HydraScale Language Model using Selective Scan blocks and Discrete Diffusion.

    Predicts the original sequence `x_0` given a noisy sequence `x_t` and timestep `t`.
    Uses efficient recurrent steps (`step` method) for generation.

    Args:
        vocab_size (int): Size of the vocabulary.
        embed_dim (int): Dimension of token embeddings and hidden states (D).
        depth (int): Number of HydraBlocks (layers).
        ssm_state_dim (int): State dimension (N) for the SSM core.
        mlp_mult (int): Multiplier for the MLP hidden dimension within blocks.
        num_diffusion_timesteps (int): Number of diffusion steps (T).
        noise_schedule (str): Type of noise schedule ('cosine' or 'linear').
        ssm_d_conv (int): Convolution kernel size in SSM blocks.
        ssm_dt_rank (int or str): Rank for dt projection in SSM blocks.
        l_prime (int): Unused parameter (legacy).
    """
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 512,
                 depth: int = 6,
                 ssm_state_dim: int = 16,
                 mlp_mult: int = 4,
                 num_diffusion_timesteps: int = 100,
                 noise_schedule: str = 'cosine',
                 ssm_d_conv: int = 4,
                 ssm_dt_rank: str = 'auto',
                 l_prime: int = DEFAULT_L_PRIME): # l_prime unused
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_timesteps = num_diffusion_timesteps
        self.mask_token_id = vocab_size  # Use vocab_size index as the [MASK] id
        self.ssm_d_conv = ssm_d_conv  # Needed for state initialization

        # Token embedding (includes [MASK] token)
        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim)

        # Time embedding MLP projection
        self.time_embedding_dim = embed_dim # Use same dim for simplicity
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim),
        )

        # HydraBlocks (SSM core layers)
        ssm_kwargs = {'d_conv': ssm_d_conv, 'dt_rank': ssm_dt_rank, 'state_dim': ssm_state_dim}
        self.layers = nn.ModuleList([
            HydraBlock(embed_dim, mlp_mult=mlp_mult, **ssm_kwargs)
            for _ in range(depth)
        ])

        # Output layers
        self.norm_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size) # Predict original token logits

        # --- Diffusion Schedule Setup ---
        if noise_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_diffusion_timesteps)
        elif noise_schedule == 'linear':
            betas = self._linear_beta_schedule(num_diffusion_timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        # Precompute sqrt terms often used in sampling/analysis (optional here)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        self.to(DEVICE)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """ Cosine schedule (improved stability). """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        """ Linear schedule. """
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def _get_mask_prob_from_time(self, t: Tensor) -> Tensor:
        """ Determine mask probability based on timestep t using noise level. """
        # Probability related to sqrt(1 - alpha_bar_t), the noise level
        # Using sqrt_one_minus_alphas_cumprod for masking probability
        mask_prob = self.sqrt_one_minus_alphas_cumprod[t] # Use precomputed [B]
        return mask_prob.to(DEVICE) # Ensure device match

    def _mask_tokens(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ Apply masking based on timestep t. x_0: [B, L], t: [B] """
        batch_size, seq_len = x_0.shape
        # Get mask probability per sequence in batch, expand to seq_len
        mask_prob = self._get_mask_prob_from_time(t).view(batch_size, 1).expand(batch_size, seq_len) # [B, L]

        # Generate random noise and create mask based on probability
        rand_noise = torch.rand_like(x_0, dtype=torch.float32)
        mask = rand_noise < mask_prob # Boolean mask [B, L]

        # Replace tokens with [MASK] ID where mask is True
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t, mask # Return noisy sequence and the mask itself

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass for training/evaluation. Predicts logits for the original token x_0.
        Uses the **SLOW** sequential scan implementation within `SelectiveScan`.

        Args:
            x: [batch, seq_len] - Noisy input sequence (x_t).
            t: [batch] - Timestep for each sequence in the batch.

        Returns:
            logits: [batch, seq_len, vocab_size] - Predicted logits for x_0.
        """
        batch_size, seq_len = x.shape

        # 1. Embed tokens
        token_emb = self.token_embedding(x)  # [B, L, D]

        # 2. Embed timesteps
        time_emb = sinusoidal_embedding(t, self.time_embedding_dim)  # [B, D_time]
        time_emb = self.time_mlp(time_emb)  # Project time embedding [B, D]
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]

        # 3. Combine embeddings (simple addition)
        h = token_emb + time_emb  # [B, L, D]

        # 4. Apply HydraBlocks (uses SLOW sequential scan in training)
        # Layer outputs are implicitly passed to the next layer
        for layer in self.layers:
            h = layer(h)

        # 5. Final normalization and LM head
        h = self.norm_out(h)
        logits = self.lm_head(h)  # [B, L, VocabSize]

        return logits

    def compute_loss(self, x_0: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Computes the training loss for a batch of original sequences x_0.
        1. Samples timestep t.
        2. Creates noisy input x_t by masking x_0.
        3. Predicts logits for x_0 using the model (calls forward).
        4. Computes CrossEntropy loss between predicted logits and original x_0.
        """
        batch_size = x_0.shape[0]
        # 1. Sample timesteps t ~ Uniform(0, T-1)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=DEVICE).long()

        # 2. Create noisy input x_t by masking x_0 based on t
        x_t, mask = self._mask_tokens(x_0, t)

        # 3. Predict logits for x_0 using the model
        # Requires autocast context if using mixed precision
        pred_logits = self.forward(x_t, t)  # [B, L, V]

        # 4. Compute loss: CrossEntropy between pred_logits and x_0 (target)
        # Loss is calculated over all tokens (masked and unmasked predicted)
        loss = F.cross_entropy(pred_logits.view(-1, self.vocab_size), x_0.view(-1))

        # Optional: Compute loss only on masked tokens (might stabilize training)
        # loss = F.cross_entropy(pred_logits.view(-1, self.vocab_size), x_0.view(-1), reduction='none')
        # loss = (loss * mask.view(-1)).sum() / (mask.sum() + 1e-8)

        return loss, pred_logits

    @torch.no_grad()
    def generate(self,
                 prompt: Tensor,
                 num_tokens_to_generate: int,
                 num_sampling_steps: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None) -> Tensor:
        """
        Generate sequence using iterative denoising via **efficient recurrent steps**.

        Args:
            prompt (Tensor): Input prompt sequence [batch, prompt_len].
            num_tokens_to_generate (int): Number of tokens to generate after prompt.
            num_sampling_steps (Optional[int]): Number of diffusion steps for sampling.
                                                Defaults to `self.num_timesteps`. Fewer steps (like DDIM)
                                                can speed up generation but may affect quality.
            temperature (float): Sampling temperature. 0 means argmax.
            top_k (Optional[int]): If set, limits sampling to top-k logits.

        Returns:
            Tensor: Generated sequence [batch, prompt_len + num_tokens_to_generate].
        """
        self.eval()
        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + num_tokens_to_generate
        sampling_steps = num_sampling_steps if num_sampling_steps is not None else self.num_timesteps

        # Initialize sequence with prompt and [MASK] tokens for the rest
        x_gen = torch.full((batch_size, total_len), self.mask_token_id, dtype=torch.long, device=DEVICE)
        x_gen[:, :prompt_len] = prompt

        # --- Initialize Recurrent States (SSM hidden states and Conv states) ---
        ssm_states = []  # List of states per layer [B, D, N]
        conv_states = [] # List of conv states per layer [B, D, k-1]
        for layer in self.layers:
            ssm_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.state_dim, device=DEVICE, dtype=torch.float32)
            # Conv state requires correct shape based on d_conv
            conv_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.d_conv - 1, device=DEVICE, dtype=torch.float32)
            ssm_states.append(ssm_state)
            conv_states.append(conv_state)

        # --- Process Prompt to Warm Up States (Efficiently using step) ---
        if prompt_len > 0:
            # Use a fixed time (e.g., t=0) or average time embedding? Let's use t=0 for simplicity.
            t_prompt = torch.zeros(batch_size, device=DEVICE, dtype=torch.long)
            time_emb_prompt = sinusoidal_embedding(t_prompt, self.time_embedding_dim)
            time_emb_prompt = self.time_mlp(time_emb_prompt) # [B, D]

            # Iterate through prompt tokens using the step function
            for t_idx in range(prompt_len):
                x_step = self.token_embedding(prompt[:, t_idx]) # [B, D]
                h_step = x_step + time_emb_prompt # Add time embedding [B, D]

                # Pass through layers, updating states in-place in the lists
                with autocast(enabled=DEVICE.type == 'cuda'): # Use AMP if available
                    for i, layer in enumerate(self.layers):
                        h_step, ssm_states[i], conv_states[i] = layer.step(h_step, ssm_states[i], conv_states[i])
                # Final h_step after processing prompt is not stored, only states matter

        # --- Iterative Denoising Generation Loop ---
        # Define the sequence of timesteps for sampling (e.g., T-1 down to 0)
        time_seq = torch.linspace(self.num_timesteps - 1, 0, sampling_steps, device=DEVICE).long()

        # tqdm progress bar for generation steps
        for i in tqdm(range(sampling_steps), desc="Generating", disable=batch_size > 1 or sampling_steps < 5):
            t_current = time_seq[i].expand(batch_size) # Current timestep [B]

            # Embed the *current* state of the generated sequence `x_gen`
            token_emb = self.token_embedding(x_gen)  # [B, L_total, D]

            # Get time embedding for the current diffusion step
            time_emb = sinusoidal_embedding(t_current, self.time_embedding_dim)
            time_emb = self.time_mlp(time_emb) # [B, D]

            # --- Run Recurrent Steps over the sequence using `step` ---
            # Use copies of states to avoid modifying the prompt-warmed states permanently
            ssm_states_step = [s.clone() for s in ssm_states]
            conv_states_step = [c.clone() for c in conv_states]

            # Store hidden states after each step to apply final norm/head
            hidden_states_gen = [] # Stores final layer output h_step for each position

            # Process the entire sequence step-by-step efficiently
            for t_idx in range(total_len):
                x_step = token_emb[:, t_idx, :] # [B, D]
                h_step = x_step + time_emb # Add time embedding [B, D]

                # Pass through layers, updating step-specific states
                with autocast(enabled=DEVICE.type == 'cuda'):
                    for layer_idx, layer in enumerate(self.layers):
                        h_step, ssm_states_step[layer_idx], conv_states_step[layer_idx] = \
                            layer.step(h_step, ssm_states_step[layer_idx], conv_states_step[layer_idx])

                # Store the output hidden state for this position
                hidden_states_gen.append(h_step)

            # Combine hidden states for the sequence
            h_final_sequence = torch.stack(hidden_states_gen, dim=1) # [B, L_total, D]

            # Apply final normalization and LM head to get logits for x_0 prediction
            h_final_sequence = self.norm_out(h_final_sequence)
            logits = self.lm_head(h_final_sequence) # [B, L_total, V]

            # --- Sampling Strategy ---
            # We only need to sample for the positions that were originally masked
            logits_to_sample = logits[:, prompt_len:, :] # [B, num_gen, V]

            if temperature == 0:  # Argmax sampling (deterministic)
                sampled_ids = torch.argmax(logits_to_sample, dim=-1)  # [B, num_gen]
            else:
                # Apply temperature scaling
                logits_to_sample = logits_to_sample / temperature

                # Apply top-k filtering
                if top_k is not None and top_k > 0:
                    v = logits_to_sample.size(-1)
                    k = min(top_k, v)
                    # Efficient top-k: compute top-k values, set others to -inf
                    indices_to_remove = logits_to_sample < torch.topk(logits_to_sample, k, dim=-1)[0][..., -1, None]
                    logits_to_sample = logits_to_sample.masked_fill(indices_to_remove, -float('Inf'))

                # Sample token ids from the filtered logits distribution
                probs = F.softmax(logits_to_sample.view(-1, self.vocab_size), dim=-1) # Flatten for multinomial [B * num_gen, V]
                sampled_ids = torch.multinomial(probs, num_samples=1).view(batch_size, -1)  # [B, num_gen]

            # Update the generated part of the sequence for the *next* diffusion step
            # This sampled sequence becomes the input for the next denoising step
            # Only update if not the last step (last step gives the final result)
            if i < sampling_steps - 1:
                x_gen[:, prompt_len:] = sampled_ids
            else:
                # Final step -> this is the definitive result
                x_gen[:, prompt_len:] = sampled_ids

        return x_gen


# --- Evaluation and Analysis Functions ---

def count_parameters(model: nn.Module) -> int:
    """ Counts the number of trainable parameters in a model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate_perplexity(model: nn.Module, data_loader: List[Tensor], num_batches: Optional[int] = None,
                        model_type="hydra"):
    """
    Calculate perplexity on a dataset using the model's loss mechanism.

    For HydraScale (Diffusion): Perplexity is calculated based on the model's ability
    to predict the original sequence `x_0` from noisy inputs `x_t` averaged over
    random timesteps `t`. This reflects the training objective.

    For Transformer (Causal LM): Perplexity is the standard next-token prediction perplexity.

    WARNING: For HydraScale, this uses `compute_loss`, which calls the **SLOW**
             `forward` pass with sequential scan. Evaluation can be time-consuming.
    """
    model.eval()
    total_neg_log_likelihood = 0.0
    total_tokens = 0
    actual_batches = 0

    print(f"Starting Perplexity evaluation for {model_type.upper()}...")
    if model_type == "hydra":
        print("NOTE: HydraScale perplexity uses the diffusion loss objective and SLOW forward pass.")

    # Determine number of batches to process
    num_total_batches = len(data_loader)
    batches_to_process = num_batches if num_batches is not None else num_total_batches
    batches_to_process = min(batches_to_process, num_total_batches)

    if batches_to_process == 0:
        return float('inf'), 0.0 # Avoid division by zero if no data

    with tqdm(data_loader[:batches_to_process], desc=f"Perplexity Eval ({model_type})") as pbar:
        for batch in pbar:
            x0 = batch.to(DEVICE)
            if x0.nelement() == 0: continue

            batch_size, seq_len = x0.shape
            num_tokens_in_batch = x0.numel()

            with autocast(enabled=DEVICE.type == 'cuda'):
                if model_type == "hydra":
                    # Hydra's loss is CE over all tokens, predicting x0 from xt
                    loss, _ = model.compute_loss(x0)
                elif model_type == "transformer":
                    # Transformer's loss is CE for next token prediction
                    loss, _ = model.compute_loss(x0) # Input x0 -> predicts x0[1:] from x0[:-1]
                else:
                    raise ValueError(f"Unknown model type for perplexity: {model_type}")

            # Accumulate total negative log likelihood (NLL)
            # loss is usually mean NLL per token. Total NLL = loss * num_tokens
            total_neg_log_likelihood += loss.item() * num_tokens_in_batch
            total_tokens += num_tokens_in_batch
            actual_batches += 1

            pbar.set_postfix({"Avg Batch Loss": f"{loss.item():.4f}"})

    if total_tokens == 0: return float('inf'), total_neg_log_likelihood
    avg_nll = total_neg_log_likelihood / total_tokens
    perplexity = math.exp(avg_nll)
    return perplexity, avg_nll

@torch.no_grad()
def measure_generation_speed(model: nn.Module, prompt_len=16, gen_len=64, batch_size=4, num_repeats=5,
                             model_type="hydra", **gen_kwargs):
    """
    Measure generation throughput (tokens/sec) and latency (ms/token).
    Uses the model's `generate` method.
    """
    model.eval()
    # Generate a consistent random prompt for fair comparison
    torch.manual_seed(123) # Ensure same prompt for both models
    prompt = torch.randint(0, VOCAB_SIZE - 1, (batch_size, prompt_len), device=DEVICE)
    print(f"\n--- Measuring {model_type.upper()} Generation Speed ---")
    print(f"Config: Prompt Len={prompt_len}, Gen Len={gen_len}, Batch Size={batch_size}, Repeats={num_repeats}")
    print(f"Device: {DEVICE}")

    # Warmup run (important for GPU timing)
    print("Warmup...")
    _ = model.generate(prompt, num_tokens_to_generate=gen_len, **gen_kwargs)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    print("Warmup complete.")

    # Timed runs
    start_time = time.perf_counter()
    generated_outputs = []
    for i in range(num_repeats):
        gen_output = model.generate(prompt, num_tokens_to_generate=gen_len, **gen_kwargs)
        generated_outputs.append(gen_output) # Store if needed later
        if DEVICE.type == 'cuda': torch.cuda.synchronize() # Ensure GPU work is done
        print(f"  Repeat {i+1}/{num_repeats} completed.", end='\r')
    end_time = time.perf_counter()
    print("\nTiming complete.")

    total_time = end_time - start_time
    avg_time_per_run = total_time / num_repeats
    total_tokens_generated_per_run = batch_size * gen_len
    # Total tokens generated across all runs / total time
    tokens_per_second = (total_tokens_generated_per_run * num_repeats) / total_time
    # Average time for one run / tokens generated in that run * 1000 for ms
    latency_ms_per_token = (avg_time_per_run / total_tokens_generated_per_run) * 1000 if total_tokens_generated_per_run > 0 else float('inf')
    # Latency per sequence generated
    latency_ms_per_sequence = (avg_time_per_run / batch_size) * 1000 if batch_size > 0 else float('inf')


    print(f"\nResults ({model_type.upper()}):")
    print(f"  Avg time/run: {avg_time_per_run:.4f} s")
    print(f"  Throughput (Tokens/sec): {tokens_per_second:.2f}")
    print(f"  Latency (ms/token generated): {latency_ms_per_token:.2f} ms")
    print(f"  Latency (ms/sequence generated): {latency_ms_per_sequence:.2f} ms")

    # Show a snippet of the first generated output (token IDs)
    print(f"\nExample Generated IDs (first sequence, last few tokens):")
    print(f"  ...{generated_outputs[0][0, -min(gen_len, 10):].tolist()}")

    return avg_time_per_run, tokens_per_second, latency_ms_per_token


# --- Comparison Baseline: Standard Transformer LM ---
class SimpleTransformerLM(nn.Module):
    """ Standard Causal Transformer LM for baseline comparison. """
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, dim_feedforward, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        # Embedding layer
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Simple fixed sinusoidal positional encoding added to embeddings
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        # Standard Transformer Encoder stack (using NormFirst for stability)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout,
                                                   activation='gelu', batch_first=True, norm_first=True)
        encoder_norm = nn.LayerNorm(embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=encoder_norm)
        # Output head
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.to(DEVICE)

        self._init_weights()

    def _init_weights(self):
        # Standard initialization practices
        nn.init.normal_(self.pos_encoder, std=0.02)
        self.token_embedding.weight.data.uniform_(-0.02, 0.02)
        self.lm_head.bias.data.zero_()
        self.lm_head.weight.data.uniform_(-0.02, 0.02)

    def forward(self, src: Tensor) -> Tensor:
        """ Forward pass with causal masking. """
        batch_size, seq_len = src.shape
        # Limit sequence length if needed (can impact performance if truncated often)
        if seq_len > self.max_seq_len:
            src = src[:, -self.max_seq_len:] # Take the last part of the sequence
            seq_len = self.max_seq_len

        # Embedding and positional encoding
        src_emb = self.token_embedding(src) * math.sqrt(self.embed_dim) # Scale embedding
        src_emb = src_emb + self.pos_encoder[:, :seq_len, :] # Add positional encoding

        # Create causal mask (prevents attending to future tokens)
        # mask shape [L, L] or [N*num_heads, L, L] depending on implementation
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=DEVICE)

        # Pass through transformer encoder
        output = self.transformer_encoder(src_emb, mask=mask, is_causal=False) # `is_causal=False` as mask is provided

        # Project to vocabulary size
        logits = self.lm_head(output)
        return logits

    def compute_loss(self, x_0: Tensor) -> Tuple[Tensor, Tensor]:
        """ Standard Causal LM loss: Predict next token. """
        # Input: x_0 sequence [B, L]
        # Target: x_0 sequence shifted left [B, L]
        inp = x_0[:, :-1] # Input tokens [B, L-1]
        tgt = x_0[:, 1:]  # Target tokens (next token prediction) [B, L-1]

        # Get logits for input sequence
        logits = self.forward(inp) # Output logits [B, L-1, V]

        # Compute cross-entropy loss comparing logits with target tokens
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        return loss, logits

    @torch.no_grad()
    def generate(self, prompt: Tensor, num_tokens_to_generate: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> Tensor:
        """ Autoregressive generation for Transformer. """
        self.eval()
        generated = prompt # Start with the prompt [B, L_prompt]

        for _ in range(num_tokens_to_generate):
            # Get current context (limit to max_seq_len for efficiency and pos encoding)
            context = generated[:, -self.max_seq_len:]

            # Get logits for the next token prediction
            logits = self.forward(context)[:, -1, :] # Only need the last token's logits [B, V]

            # Apply sampling strategy (argmax, temperature, top-k)
            if temperature == 0:  # Argmax (deterministic)
                next_token = torch.argmax(logits, dim=-1, keepdim=True) # [B, 1]
            else:
                if temperature > 0: # Avoid division by zero if temp=0 used for argmax logic
                    logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v = logits.size(-1)
                    k = min(top_k, v)
                    indices_to_remove = logits < torch.topk(logits, k, dim=-1)[0][..., -1, None]
                    logits.masked_fill_(indices_to_remove, -float('Inf'))

                probs = F.softmax(logits, dim=-1) # [B, V]
                next_token = torch.multinomial(probs, num_samples=1) # [B, 1]

            # Append the sampled token to the generated sequence
            generated = torch.cat((generated, next_token), dim=1) # [B, L_prompt + current_gen_len]

        return generated


# --- Main Execution & Analysis ---
def main():
    print(f"--- HydraScale V2.1 Demo ---")
    print(f"Using device: {DEVICE}")
    print("\n" + "="*60)
    print("IMPORTANT NOTE:")
    print("HydraScale's `forward` pass uses a **sequential scan** (Python loop).")
    print("This makes **training significantly slower** than optimized parallel scans.")
    print("However, `generate` uses the efficient **recurrent step** calculation.")
    print("="*60 + "\n")

    # --- Configuration ---
    # Model Size (Small for Demo)
    HYDRA_EMBED_DIM = 256
    HYDRA_DEPTH = 4
    HYDRA_SSM_STATE = 16
    HYDRA_SSM_DCONV = 3
    HYDRA_DT_RANK = 'auto' # math.ceil(HYDRA_EMBED_DIM / 16)
    HYDRA_MLP_MULT = 4
    HYDRA_DIFFUSION_STEPS = 50

    TF_EMBED_DIM = 256
    TF_NHEAD = 4
    TF_FFN_DIM = TF_EMBED_DIM * 2 # Smaller FFN for param matching
    TF_LAYERS = 3
    TF_MAX_LEN = 512 # Max sequence length for Transformer

    # Data/Training Config (Small for Demo)
    SEQ_LEN = 128
    BATCH_SIZE = 8  # Reduced further for potential memory constraints
    NUM_BATCHES_DATA = 50 # Number of batches for dummy data
    NUM_TRAIN_STEPS = 20  # Slightly increased training steps for demo
    NUM_EVAL_BATCHES = 15 # Batches for evaluation

    # Generation Config
    GEN_PROMPT_LEN = 16
    GEN_LEN = 64
    GEN_BATCH_SIZE = 4
    GEN_REPEATS = 5
    GEN_SAMPLING_STEPS = 25 # Fewer steps for HydraScale generation (faster)

    # --- Generate Dummy Data ---
    print("Generating dummy data...")
    # Ensure reproducibility for data generation
    torch.manual_seed(42)
    dummy_data = [torch.randint(0, VOCAB_SIZE - 1, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
                  for _ in range(NUM_BATCHES_DATA)]
    split_idx = NUM_BATCHES_DATA // 2
    train_loader = dummy_data[:split_idx]
    eval_loader = dummy_data[split_idx:]
    print(f"Dummy data created: {len(train_loader)} train batches, {len(eval_loader)} eval batches.")

    # --- Model Initialization ---
    print("\nInitializing models...")
    # HydraScale Model
    hydra_model = HydraScaleLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=HYDRA_EMBED_DIM,
        depth=HYDRA_DEPTH,
        ssm_state_dim=HYDRA_SSM_STATE,
        ssm_d_conv=HYDRA_SSM_DCONV,
        ssm_dt_rank=HYDRA_DT_RANK,
        mlp_mult=HYDRA_MLP_MULT,
        num_diffusion_timesteps=HYDRA_DIFFUSION_STEPS
    ).to(DEVICE)

    # Transformer Baseline Model
    transformer_model = SimpleTransformerLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=TF_EMBED_DIM,
        nhead=TF_NHEAD,
        num_layers=TF_LAYERS,
        dim_feedforward=TF_FFN_DIM,
        max_seq_len=TF_MAX_LEN
    ).to(DEVICE)

    # --- Parameter Count Comparison ---
    hydra_params = count_parameters(hydra_model)
    tf_params = count_parameters(transformer_model)
    print(f"\n--- Model Parameters ---")
    print(f"HydraScale Params : {hydra_params / 1e6:.2f} M")
    print(f"Transformer Params: {tf_params / 1e6:.2f} M")
    print(f"Parameter ratio (Hydra/TF): {hydra_params / tf_params:.2f}")


    # --- Training Demo (Few Steps) ---
    print(f"\n--- Training Demo ({NUM_TRAIN_STEPS} Steps) ---")
    # Use AdamW optimizer
    hydra_optimizer = torch.optim.AdamW(hydra_model.parameters(), lr=5e-4, weight_decay=0.01)
    tf_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=5e-4, weight_decay=0.01)
    # Use GradScaler for mixed precision (speeds up training, reduces memory)
    scaler = GradScaler(enabled=DEVICE.type == 'cuda')

    hydra_model.train()
    transformer_model.train()
    train_start_time = time.perf_counter()

    for i in range(NUM_TRAIN_STEPS):
        # Cycle through limited dummy data
        batch_idx = i % len(train_loader)
        batch = train_loader[batch_idx]

        # Hydra Training Step
        hydra_optimizer.zero_grad()
        with autocast(enabled=DEVICE.type == 'cuda'):
            loss_hydra, _ = hydra_model.compute_loss(batch)
        scaler.scale(loss_hydra).backward()
        scaler.step(hydra_optimizer)
        scaler.update() # Update scaler for next iteration

        # Transformer Training Step
        tf_optimizer.zero_grad()
        with autocast(enabled=DEVICE.type == 'cuda'):
            loss_tf, _ = transformer_model.compute_loss(batch)
        scaler.scale(loss_tf).backward()
        scaler.step(tf_optimizer)
        scaler.update()

        if (i + 1) % 5 == 0 or i == NUM_TRAIN_STEPS - 1:
            print(f"Step {i + 1:>{len(str(NUM_TRAIN_STEPS))}}/{NUM_TRAIN_STEPS}: Hydra Loss={loss_hydra.item():.4f}, TF Loss={loss_tf.item():.4f}")

    train_end_time = time.perf_counter()
    print(f"Training demo finished in {train_end_time - train_start_time:.2f} seconds.")
    print(f"WARNING: HydraScale training time is impacted by the slow sequential scan.")


    # --- Evaluation: Perplexity ---
    print("\n--- Evaluation: Perplexity ---")
    # Calculate Perplexity for HydraScale
    # This is slow due to sequential scan in forward pass used by compute_loss
    perplexity_hydra, nll_hydra = evaluate_perplexity(
        hydra_model, eval_loader, NUM_EVAL_BATCHES, model_type="hydra"
    )
    print(f"HydraScale Final Perplexity : {perplexity_hydra:.2f} (Avg NLL: {nll_hydra:.4f})")
    print(f"(Note: Based on {NUM_TRAIN_STEPS} training steps & {NUM_EVAL_BATCHES} eval batches)")

    # Calculate Perplexity for Transformer
    perplexity_tf, nll_tf = evaluate_perplexity(
        transformer_model, eval_loader, NUM_EVAL_BATCHES, model_type="transformer"
    )
    print(f"Transformer Final Perplexity: {perplexity_tf:.2f} (Avg NLL: {nll_tf:.4f})")
    print(f"(Note: Based on {NUM_TRAIN_STEPS} training steps & {NUM_EVAL_BATCHES} eval batches)")


    # --- Evaluation: Generation Speed ---
    print("\n--- Evaluation: Generation Speed ---")
    # Measure HydraScale Generation Speed (should be efficient)
    hydra_gen_time, hydra_tokens_sec, hydra_latency_token = measure_generation_speed(
        hydra_model,
        prompt_len=GEN_PROMPT_LEN,
        gen_len=GEN_LEN,
        batch_size=GEN_BATCH_SIZE,
        num_repeats=GEN_REPEATS,
        model_type="hydra",
        # Pass generation specific args for Hydra
        num_sampling_steps=GEN_SAMPLING_STEPS,
        temperature=0.8, top_k=50
    )

    # Measure Transformer Generation Speed
    tf_gen_time, tf_tokens_sec, tf_latency_token = measure_generation_speed(
        transformer_model,
        prompt_len=GEN_PROMPT_LEN,
        gen_len=GEN_LEN,
        batch_size=GEN_BATCH_SIZE,
        num_repeats=GEN_REPEATS,
        model_type="transformer",
        # Pass generation specific args for Transformer
        temperature=0.8, top_k=50
    )

    # --- Generation Speed Summary ---
    print("\n--- Generation Speed Summary ---")
    print(f"Metric                    | HydraScale        | Transformer       | Ratio (Hydra/TF)")
    print(f"--------------------------|-------------------|-------------------|-----------------")
    print(f"Avg Time/Run (s)          | {hydra_gen_time:<17.4f} | {tf_gen_time:<17.4f} | {hydra_gen_time / tf_gen_time:.2f}")
    print(f"Throughput (Tokens/sec)   | {hydra_tokens_sec:<17.2f} | {tf_tokens_sec:<17.2f} | {hydra_tokens_sec / tf_tokens_sec:.2f}")
    print(f"Latency (ms/Token)        | {hydra_latency_token:<17.2f} | {tf_latency_token:<17.2f} | {hydra_latency_token / tf_latency_token:.2f}")
    print(f"\nNote: Hydra generation uses {GEN_SAMPLING_STEPS} sampling steps.")
    print(f"      Transformer uses standard autoregressive generation.")
    print(f"      Lower Latency and Higher Throughput are better.")

    # --- Qualitative Generation Example ---
    print("\n--- Qualitative Generation Example ---")
    # Use a fixed, simple prompt for comparison
    # Need a simple tokenizer or map integers to chars for readability
    # Placeholder: Use integers, assume 101=BOS, 102=EOS, common words
    # Example prompt: "The quick brown fox" (ids are arbitrary placeholders)
    torch.manual_seed(777) # Seed for generation comparison consistency
    prompt_ids = [101, 1996, 4248, 2829, 4419, 102] # Placeholder IDs
    prompt_example = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)
    print(f"Prompt Token IDs: {prompt_example.tolist()}")
    # Placeholder for decoding - replace with actual tokenizer if available
    print(f"Prompt Text (Placeholder): <Start> The quick brown fox <End>")

    print("\nGenerating with HydraScale...")
    generated_hydra = hydra_model.generate(
        prompt_example,
        num_tokens_to_generate=32,
        num_sampling_steps=GEN_SAMPLING_STEPS, # Use specified sampling steps
        temperature=0.8,
        top_k=50
    )
    print(f"HydraScale Output IDs: {generated_hydra.tolist()}")
    print(f"  Output Text (Placeholder): <See Token IDs above>") # Add decoding here

    print("\nGenerating with Transformer...")
    generated_tf = transformer_model.generate(
        prompt_example,
        num_tokens_to_generate=32,
        temperature=0.8,
        top_k=50
    )
    print(f"Transformer Output IDs: {generated_tf.tolist()}")
    print(f"  Output Text (Placeholder): <See Token IDs above>") # Add decoding here

    print("\n--- Analysis and Tuning Guide ---")
    print("""
    Model Power & Tuning:
    1. SSM Core (`SelectiveScan`):
       - `state_dim` (N): Controls the size of the hidden state. Larger N allows capturing
         more complex long-range dependencies but increases computation/memory. Typical
         values: 16, 32, 64. Tune this carefully.
       - `d_conv` (Convolution Width): Controls the local receptive field before the scan.
         Larger values capture more local context. Typical: 3, 4, 5.
       - `dt_rank`: Rank for projecting the time-step `dt`. 'auto' is a good starting point.
         Lower rank might save params but could limit expressiveness.
       - Initialization: `A_log` and `dt_proj` initialization can affect stability. The
         current setup follows Mamba recommendations.

    2. Diffusion Mechanism:
       - `num_diffusion_timesteps` (T): Number of steps in the noise schedule during training.
         More steps can lead to better modeling but require more computation.
       - `noise_schedule`: 'cosine' is often preferred over 'linear' for stability.
       - Masking Probability: Derived from `sqrt(1 - alpha_bar)`. Controls noise level.
       - Generation Sampling Steps (`num_sampling_steps`): Can be fewer than training steps (T)
         to speed up generation (like DDIM). E.g., T=100, sampling_steps=25. This is a key
         trade-off between speed and quality.

    3. Overall Architecture:
       - `embed_dim` (D): Main dimension. Larger D increases model capacity significantly.
       - `depth`: Number of layers. Deeper models can learn more complex functions.
       - `mlp_mult`: Expansion factor in MLP blocks. Affects parameter count and capacity.

    Performance Considerations:
    - Training Speed: Limited by the **sequential scan** in `forward`. Requires custom
      parallel implementation (CUDA) for competitive training speed.
    - Inference Speed (Generation): Uses the **efficient recurrent `step`** method.
      Should be significantly faster per step than Transformer's O(N^2) attention,
      especially for long sequences. HydraScale's diffusion requires multiple sampling steps,
      while Transformer generates token-by-token. The overall generation speed depends
      on `num_sampling_steps` (Hydra) vs `num_tokens_to_generate` (Transformer) and the
      cost per step/token.
    - Perplexity: HydraScale's perplexity metric reflects the diffusion training objective,
      not standard causal LM perplexity. Direct comparison needs care.

    Tuning Strategy:
    - Start with a small model configuration (like the demo).
    - Tune SSM parameters (`state_dim`, `d_conv`) first.
    - Adjust diffusion steps (`num_diffusion_timesteps`, `num_sampling_steps`) for
      desired quality/speed trade-off.
    - Scale `embed_dim` and `depth` for capacity.
    - Monitor training loss and generation quality.
    """)

    print("--- Demo Complete ---")

if __name__ == "__main__":
    main()