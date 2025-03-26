#!/usr/bin/env python3
"""
HydraScale V2: A Novel Language Model inspired by MBD-S,
using a custom Selective Scan (SSM) core and discrete diffusion.
Fixed evaluation function naming for compatibility. Enhanced clarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List
from torch.cuda.amp import autocast
from torch import Tensor
import math
import time
import numpy as np
from tqdm import tqdm  # Optional: for progress bars

# --- Constants ---
VOCAB_SIZE = 50_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_L_PRIME = 0  # Default: Process full sequence (0 means no blocking)


# --- Custom SSM Implementation (Inspired by Mamba) ---
# WARNING: The forward pass uses a sequential scan placeholder for demonstration.
# A parallel scan implementation (like in original Mamba) is crucial for training speed.
class SelectiveScan(nn.Module):
    """
    Custom implementation of the Selective Scan mechanism (S6).
    Uses sequential scan in forward (SLOW for training) and recurrent in step (fast for inference).
    Args:
        embed_dim (int): Input/output dimension (D).
        state_dim (int): State dimension (N).
        d_conv (int): Local convolution width.
        dt_rank (int): Rank for delta_t projection ('auto' or int).
        bias (bool): Whether to include bias in linear layers.
    """

    def __init__(self, embed_dim: int, state_dim: int = 16, d_conv: int = 4, dt_rank: str = 'auto', bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.d_conv = d_conv
        self.dt_rank = math.ceil(embed_dim / 16) if dt_rank == 'auto' else dt_rank
        self.bias = bias

        # Input-dependent projections
        # Proj for main pathway x -> x'
        self.in_proj_x = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Proj for gating z
        self.in_proj_z = nn.Linear(embed_dim, embed_dim, bias=bias)
        # Proj for dt, B, C parameters
        self.in_proj_params = nn.Linear(embed_dim, self.dt_rank + 2 * self.state_dim, bias=bias)

        # Local convolution (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim, bias=bias,
            kernel_size=d_conv, groups=embed_dim, padding=d_conv - 1,
        )

        # Delta_t projection
        self.dt_proj = nn.Linear(self.dt_rank, embed_dim, bias=True)

        # State matrix A (learnable log-diagonal) and residual D
        self.A_log = nn.Parameter(
            torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).unsqueeze(0).repeat(embed_dim, 1))  # [D, N]
        self.D = nn.Parameter(torch.ones(embed_dim))  # [D] initialized to 1

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _compute_A_tilde(self, dt: Tensor) -> Tensor:
        """ Discretize continuous A using Zero-Order Hold (ZOH). dt: [B, L, D] """
        A = -torch.exp(self.A_log.float())  # [D, N]
        A_tilde = torch.exp(A.unsqueeze(0).unsqueeze(1) * dt.unsqueeze(-1))  # [B, L, D, N]
        return A_tilde

    def _compute_B_tilde(self, dt: Tensor, A_tilde: Tensor) -> Tensor:
        """ Discretize continuous B using ZOH variant. dt: [B, L, D], A_tilde: [B, L, D, N] """
        A = -torch.exp(self.A_log.float())  # [D, N]
        # Approximated ZOH for B: (e^(dt*A) - 1) / A ~= dt when dt is small
        # B_tilde = dt.unsqueeze(-1) * B_proj.unsqueeze(2) ? Needs B proj first.
        # Let's use the (A_tilde - 1) / A formula, scaled by dt later if needed or implicitly handled
        # Handle A=0 case numerically
        A_unsqueeze = A.unsqueeze(0).unsqueeze(1)  # [1, 1, D, N]
        B_tilde = (A_tilde - 1) / (A_unsqueeze + 1e-8)  # [B, L, D, N]
        # Mamba paper implies B_tilde = (e^{dt A} - 1)/A * B. Our B comes from input proj.
        # Let's scale the input term later: dt * B_proj * u
        return B_tilde  # [B, L, D, N]

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using **sequential scan**. SLOW for training.
        x: [batch, seq_len, embed_dim] (B, L, D)
        """
        batch, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim

        # Input projections
        x_proj = self.in_proj_x(x)  # [B, L, D]
        z = self.in_proj_z(x)  # [B, L, D] (for gating)

        params = self.in_proj_params(x)  # [B, L, dt_rank + 2N]
        dt_unproj, B_proj, C_proj = torch.split(params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        # dt_unproj: [B, L, dt_rank], B_proj: [B, L, N], C_proj: [B, L, N]

        dt = F.softplus(self.dt_proj(dt_unproj))  # [B, L, D] - Projected and constrained delta_t

        # Convolution and SiLU activation
        x_conv = self.conv1d(x_proj.transpose(1, 2))  # [B, D, L + pad]
        x_conv = x_conv[:, :, :seq_len]  # Remove padding: [B, D, L]
        u = F.silu(x_conv.transpose(1, 2))  # [B, L, D] - Input to the scan 'u'

        # Discretize A, B
        A_tilde = self._compute_A_tilde(dt)  # [B, L, D, N]
        B_tilde = self._compute_B_tilde(dt, A_tilde)  # [B, L, D, N]

        # *** Sequential Scan Placeholder ***
        h = torch.zeros(batch, embed_dim, self.state_dim, device=x.device, dtype=A_tilde.dtype)  # [B, D, N] State
        ys = []
        for t in range(seq_len):
            A_t = A_tilde[:, t, :, :]  # [B, D, N]
            B_t = B_tilde[:, t, :, :]  # [B, D, N] (Discretized B base)
            B_proj_t = B_proj[:, t, :]  # [B, N] (Input-dependent B modulator)
            C_proj_t = C_proj[:, t, :]  # [B, N] (Input-dependent C modulator)
            u_t = u[:, t, :]  # [B, D] (Input to scan for this step)

            # State Update: h_t = A_t * h_{t-1} + (B_t * B_proj_t) * u_t
            # The input term needs careful einsum: (B_tilde * B_proj) influences state based on u_t
            # Input term: einsum('bdn,bn,bd->bdn', B_t, B_proj_t, u_t)
            input_term = torch.einsum('bdn,bn,bd->bdn', B_t, B_proj_t, u_t)  # [B, D, N]
            h = A_t * h + input_term  # Update state [B, D, N]

            # Output Calculation: y_t = C_proj_t * h_t
            # Output term einsum: einsum('bn, bdn -> bd', C_proj_t, h)
            y_t = torch.einsum('bn,bdn->bd', C_proj_t, h)  # [B, D]
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # [B, L, D] - Combined output sequence
        # *** End Sequential Scan Placeholder ***

        # Add residual D term (skip connection based on 'u')
        y = y + u * self.D.unsqueeze(0).unsqueeze(1)  # [B, L, D]

        # Apply gating
        y = y * F.silu(z)  # [B, L, D]

        # Final output projection
        y = self.out_proj(y)  # [B, L, D]
        return y

    def step(self, x_step: Tensor, h_prev: Tensor, conv_state: Optional[Tensor] = None) -> Tuple[
        Tensor, Tensor, Tensor]:
        """
        Recurrent step function for efficient inference/generation.
        x_step: [batch, embed_dim] (B, D) - Current input element
        h_prev: [batch, embed_dim, state_dim] (B, D, N) - Previous state
        conv_state: [batch, embed_dim, d_conv - 1] - Previous conv inputs
        Returns:
            y_step: [batch, embed_dim] (B, D) - Output for current step
            h: [batch, embed_dim, state_dim] (B, D, N) - New state
            new_conv_state: [batch, embed_dim, d_conv - 1] - Updated conv state
        """
        batch, embed_dim = x_step.shape

        # --- Handle Convolution State ---
        if conv_state is None:
            conv_state = torch.zeros(batch, embed_dim, self.d_conv - 1, device=x_step.device, dtype=x_step.dtype)

        conv_input = torch.cat([conv_state, x_step.unsqueeze(2)], dim=2)  # [B, D, d_conv]
        new_conv_state = conv_input[:, :, 1:]  # Update state for next step

        # Apply convolution
        # Need conv1d weights: [D, 1, k]
        # Need conv1d bias: [D]
        conv_weights = self.conv1d.weight.squeeze(1)  # [D, k]
        # Einsum: conv_out[b, d] = sum_k ( conv_input[b, d, k] * conv_weights[d, k] )
        # Reverse weights for convolution: torch.sum(conv_input * conv_weights.flip(dims=[-1]), dim=-1)
        conv_out = torch.einsum('bdk,dk->bd', conv_input, conv_weights)  # Direct correlation, not conv. Need flip?
        # Conv1d applies correlation. Let's use functional conv1d for correctness.
        # Input needs shape [B, D, k] for conv1d
        conv_out_func = F.conv1d(
            conv_input,
            self.conv1d.weight,  # [D_out, D_in/groups, k] -> [D, 1, k]
            self.conv1d.bias,  # [D_out] -> [D]
            padding=0,
            groups=self.embed_dim
        ).squeeze(-1)  # Output: [B, D]

        # --- SSM Step Logic ---
        # Input projections
        # x_proj is needed for conv, but u comes from conv_out. Use x_step for params.
        z = self.in_proj_z(x_step)  # [B, D] (for gating)

        params = self.in_proj_params(x_step)  # [B, dt_rank + 2N]
        dt_unproj, B_proj, C_proj = torch.split(params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        # dt_unproj: [B, dt_rank], B_proj: [B, N], C_proj: [B, N]

        dt = F.softplus(self.dt_proj(dt_unproj))  # [B, D] - Projected delta_t for this step

        # Activation 'u' from convolution output
        u = F.silu(conv_out_func)  # [B, D]

        # Discretize A, B for this step
        A = -torch.exp(self.A_log.float())  # [D, N]
        A_tilde_step = torch.exp(A.unsqueeze(0) * dt.unsqueeze(-1))  # [B, D, N]
        # B_tilde = (A_tilde - 1) / A (handled via input term scaling)
        # B_tilde base calculation (optional, can combine in input term)
        A_unsqueeze = A.unsqueeze(0)  # [1, D, N]
        B_tilde_step = (A_tilde_step - 1) / (A_unsqueeze + 1e-8)  # [B, D, N]

        # State Update: h_t = A_t * h_{t-1} + (B_t * B_proj_t) * u_t
        # Input term einsum: einsum('bdn,bn,bd->bdn', B_tilde_step, B_proj, u)
        input_term = torch.einsum('bdn,bn,bd->bdn', B_tilde_step, B_proj, u)  # [B, D, N]
        h = A_tilde_step * h_prev + input_term  # [B, D, N] - New state

        # Output Calculation: y_t = C_proj_t * h_t + D * u_t
        # Output einsum: einsum('bn, bdn -> bd', C_proj, h)
        y = torch.einsum('bn,bdn->bd', C_proj, h)  # [B, D]
        y = y + u * self.D.unsqueeze(0)  # Add residual D term [B, D]

        # Apply gating
        y = y * F.silu(z)  # [B, D]

        # Final output projection
        y_step = self.out_proj(y)  # [B, D]

        return y_step, h, new_conv_state


# --- HydraScale Components ---

class HydraBlock(nn.Module):
    """ Combines Selective Scan with LayerNorm and MLP. """

    def __init__(self, embed_dim: int, mlp_mult: int = 4, **ssm_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ssm = SelectiveScan(embed_dim, **ssm_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = embed_dim * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),  # Or SiLU
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        # Pre-norm architecture
        x = x + self.ssm(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def step(self, x_step: Tensor, ssm_state: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Step function for recurrent inference, managing states. """
        # Input: x_step [B, D], ssm_state [B, D, N], conv_state [B, D, k-1]
        # Output: y_step [B, D], new_ssm_state [B, D, N], new_conv_state [B, D, k-1]

        # SSM part
        residual = x_step
        x_norm1 = self.norm1(x_step)
        ssm_out, ssm_state_new, conv_state_new = self.ssm.step(x_norm1, ssm_state, conv_state)
        x = residual + ssm_out  # Apply residual after SSM

        # MLP part (operates only on current step's features)
        residual = x
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        y_step = residual + mlp_out  # Apply residual after MLP

        return y_step, ssm_state_new, conv_state_new


def sinusoidal_embedding(timesteps: Tensor, embedding_dim: int):
    """ Sinusoidal time embedding. """
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, device=timesteps.device, dtype=torch.float32) * -emb)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad if dim is odd
        emb = F.pad(emb, (0, 1))
    return emb


class HydraScaleLM(nn.Module):
    """
    HydraScale Language Model using Selective Scan and Discrete Diffusion.
    Args are same as previous version.
    """

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 512,
                 depth: int = 6,
                 ssm_state_dim: int = 16,
                 mlp_mult: int = 4,
                 l_prime: int = DEFAULT_L_PRIME,  # Currently unused, processes full sequence
                 num_diffusion_timesteps: int = 100,
                 noise_schedule: str = 'cosine',
                 ssm_d_conv: int = 4,
                 ssm_dt_rank: str = 'auto'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.l_prime = l_prime
        self.num_timesteps = num_diffusion_timesteps
        self.mask_token_id = vocab_size  # Use vocab_size as the [MASK] id
        self.ssm_d_conv = ssm_d_conv  # Needed for state initialization

        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim)  # +1 for [MASK] token

        # Time embedding
        self.time_embedding_dim = embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim),
        )

        # HydraBlocks (SSM core)
        ssm_kwargs = {'d_conv': ssm_d_conv, 'dt_rank': ssm_dt_rank, 'state_dim': ssm_state_dim}
        self.layers = nn.ModuleList([
            HydraBlock(embed_dim, mlp_mult=mlp_mult, **ssm_kwargs)
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # --- Diffusion Schedule ---
        if noise_schedule == 'cosine':
            betas = self._cosine_beta_schedule(num_diffusion_timesteps)
        elif noise_schedule == 'linear':
            betas = self._linear_beta_schedule(num_diffusion_timesteps)
        else:
            raise ValueError(f"Unknown noise schedule: {noise_schedule}")

        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        # Precompute sqrt(alphas_cumprod) and sqrt(1-alphas_cumprod) if needed for sampling later

        self.to(DEVICE)

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """ Cosine schedule as proposed in https://arxiv.org/abs/2102.09672 """
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def _linear_beta_schedule(self, timesteps, beta_start=0.0001, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)

    def _get_mask_prob_from_time(self, t: Tensor) -> Tensor:
        """ Determine mask probability based on timestep t. """
        # Option 1: Linear
        # mask_prob = (t.float() + 1) / self.num_timesteps

        # Option 2: Based on noise level (1 - sqrt(alpha_bar))
        alpha_bar_t = self.alphas_cumprod[t]  # [B]
        mask_prob = 1.0 - torch.sqrt(alpha_bar_t)

        return mask_prob.to(DEVICE)  # Ensure device match

    def _mask_tokens(self, x_0: Tensor, t: Tensor) -> Tensor:
        """ Apply masking based on timestep t. x_0: [B, L], t: [B] """
        batch_size, seq_len = x_0.shape
        mask_prob = self._get_mask_prob_from_time(t).view(batch_size, 1).expand(batch_size, seq_len)  # [B, L]
        rand_noise = torch.rand_like(x_0, dtype=torch.float32)
        mask = rand_noise < mask_prob
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t, mask

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass for training/evaluation. Predicts logits for the original token x_0.
        x: [batch, seq_len] - Noisy input (x_t)
        t: [batch] - Timestep for each sequence in the batch
        """
        batch_size, seq_len = x.shape

        # 1. Embed tokens
        token_emb = self.token_embedding(x)  # [B, L, D]

        # 2. Embed timesteps
        time_emb = sinusoidal_embedding(t, self.time_embedding_dim)  # [B, D_time]
        time_emb = self.time_mlp(time_emb)  # [B, D_time] -> [B, D]
        time_emb = time_emb.unsqueeze(1).expand(-1, seq_len, -1)  # [B, L, D]

        # 3. Combine embeddings
        h = token_emb + time_emb  # [B, L, D]

        # 4. Apply HydraBlocks (Uses SLOW sequential scan in training)
        for layer in self.layers:
            h = layer(h)

        # 5. Final normalization and LM head
        h = self.norm_out(h)
        logits = self.lm_head(h)  # [B, L, VocabSize]

        return logits

    def compute_loss(self, x_0: Tensor) -> Tuple[Tensor, Tensor]:
        """ Training step: Sample t, create x_t, predict x_0, compute loss. """
        batch_size = x_0.shape[0]
        # 1. Sample timesteps t ~ Uniform(0, T-1)
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=DEVICE).long()
        # 2. Create noisy input x_t by masking x_0 based on t
        x_t, mask = self._mask_tokens(x_0, t)
        # 3. Predict logits for x_0 using the model
        pred_logits = self.forward(x_t, t)  # [B, L, V]
        # 4. Compute loss: CrossEntropy between pred_logits and x_0 (target)
        loss = F.cross_entropy(pred_logits.view(-1, self.vocab_size), x_0.view(-1))
        return loss, pred_logits

    @torch.no_grad()
    def generate(self,
                 prompt: Tensor,
                 num_tokens_to_generate: int,
                 num_sampling_steps: Optional[int] = None,
                 temperature: float = 0.7,
                 top_k: Optional[int] = 40) -> Tensor:
        """
        Generate sequence using iterative denoising via efficient recurrent steps.
        """
        self.eval()
        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + num_tokens_to_generate
        sampling_steps = num_sampling_steps if num_sampling_steps is not None else self.num_timesteps

        # Initialize sequence with prompt and [MASK] tokens
        x_gen = torch.full((batch_size, total_len), self.mask_token_id, dtype=torch.long, device=DEVICE)
        x_gen[:, :prompt_len] = prompt

        # --- Initialize Recurrent States ---
        ssm_states = []  # List of states per layer
        conv_states = []  # List of conv states per layer
        for layer in self.layers:
            ssm_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.state_dim, device=DEVICE, dtype=torch.float32)
            conv_state = torch.zeros(batch_size, self.embed_dim, self.ssm_d_conv - 1, device=DEVICE,
                                     dtype=torch.float32)
            ssm_states.append(ssm_state)
            conv_states.append(conv_state)

        # --- Process Prompt to Warm Up States ---
        if prompt_len > 0:
            prompt_emb = self.token_embedding(prompt)  # [B, L_p, D]
            # No time embedding needed for prompt processing? Or use t=0? Let's use t=0.
            t_prompt = torch.zeros(batch_size, device=DEVICE, dtype=torch.long)
            time_emb_prompt = sinusoidal_embedding(t_prompt, self.time_embedding_dim)
            time_emb_prompt = self.time_mlp(time_emb_prompt).unsqueeze(1)  # [B, 1, D]

            h = prompt_emb  # [B, L_p, D] - Start with token embeddings
            for t_idx in range(prompt_len):
                h_step = h[:, t_idx, :] + time_emb_prompt.squeeze(1)  # Add time emb to current token [B, D]
                with autocast(enabled=DEVICE.type == 'cuda'):
                    for i, layer in enumerate(self.layers):
                        h_step, ssm_states[i], conv_states[i] = layer.step(h_step, ssm_states[i], conv_states[i])
                # Output h_step is not directly used here, only state update matters

        # --- Iterative Denoising Generation ---
        time_seq = torch.linspace(self.num_timesteps - 1, 0, sampling_steps, device=DEVICE).long()
        indices_to_generate = list(range(prompt_len, total_len))

        for i in tqdm(range(sampling_steps), desc="Generating", disable=batch_size > 1):
            t_current = time_seq[i].expand(batch_size)  # Current timestep [B]

            # Embed the *current* state of the sequence (including previous predictions)
            token_emb = self.token_embedding(x_gen)  # [B, L, D]

            # Get time embedding for the current step
            time_emb = sinusoidal_embedding(t_current, self.time_embedding_dim)
            time_emb = self.time_mlp(time_emb).unsqueeze(1)  # [B, 1, D]

            # --- Run Recurrent Steps over generated part ---
            # We need logits only for the generated part, using updated states
            # Re-initialize states? No, continue from prompt states.
            ssm_states_step = [s.clone() for s in ssm_states]  # Copy states for this diffusion step
            conv_states_step = [c.clone() for c in conv_states]

            logits_gen = []  # Store logits for generated positions
            h_step = None  # Needs initialization if prompt_len == 0
            if prompt_len > 0:
                # Get the last hidden state after processing the prompt
                # Re-run the last step of prompt processing to get h_step?
                # Simplified: assume the state captures history, start from first generated token.
                # Let's recalculate the last h_step from prompt if needed.
                last_prompt_token_emb = token_emb[:, prompt_len - 1, :]
                h_step = last_prompt_token_emb + time_emb.squeeze(1)  # Approx input to first gen step
                with autocast(enabled=DEVICE.type == 'cuda'):
                    for layer_idx, layer in enumerate(self.layers):
                        # Use the states *after* prompt processing
                        h_step, _, _ = layer.step(h_step, ssm_states[layer_idx], conv_states[layer_idx])
            else:
                # Start from zero input? Or a start token embedding?
                # Assume generation starts from scratch, need initial h_step
                # Use time embedding as initial input?
                h_step = time_emb.squeeze(1)  # [B, D]

            for t_idx in indices_to_generate:
                current_token_emb = token_emb[:, t_idx, :]  # [B, D]
                h_step = current_token_emb + time_emb.squeeze(1)  # Add time emb [B, D]

                with autocast(enabled=DEVICE.type == 'cuda'):
                    for layer_idx, layer in enumerate(self.layers):
                        h_step, ssm_states_step[layer_idx], conv_states_step[layer_idx] = \
                            layer.step(h_step, ssm_states_step[layer_idx], conv_states_step[layer_idx])

                # Final normalization and LM head for this step
                h_final = self.norm_out(h_step)
                logits_step = self.lm_head(h_final)  # [B, V]
                logits_gen.append(logits_step)

            # Combine logits for the generated part
            logits_to_sample = torch.stack(logits_gen, dim=1)  # [B, num_gen, V]

            # --- Sampling Strategy ---
            if temperature == 0:  # Argmax sampling
                sampled_ids = torch.argmax(logits_to_sample, dim=-1)  # [B, num_gen]
            else:
                # Apply temperature
                logits_to_sample = logits_to_sample / temperature
                # Apply top-k
                if top_k is not None and top_k > 0:
                    v = logits_to_sample.size(-1)
                    k = min(top_k, v)
                    logits_flat = logits_to_sample.view(-1, v)  # [B * num_gen, V]
                    indices_to_remove = logits_flat < torch.topk(logits_flat, k, dim=-1)[0][..., -1, None]
                    logits_flat[indices_to_remove] = -float('Inf')
                    # Reshape not needed if using view(-1, v)

                # Sample token ids
                probs = F.softmax(logits_flat, dim=-1)  # [B * num_gen, V]
                sampled_ids = torch.multinomial(probs, num_samples=1).view(batch_size, -1)  # [B, num_gen]

            # Update the generated sequence part for the *next* diffusion step
            # Only update if not the last step (which gives the final result)
            if i < sampling_steps - 1:
                x_gen[:, prompt_len:] = sampled_ids
            else:
                # Final step -> this is the result
                x_gen[:, prompt_len:] = sampled_ids

        return x_gen


# --- Evaluation and Analysis Functions ---

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def evaluate_perplexity(model: nn.Module, data_loader: List[Tensor], num_batches: Optional[int] = None,
                        model_type="hydra"):
    """ Calculate perplexity on a dataset using the model's loss mechanism. """
    model.eval()
    total_loss = 0
    total_tokens = 0
    actual_batches = 0
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader, desc="Perplexity Eval")):
            if num_batches is not None and i >= num_batches:
                break

            x0 = batch.to(DEVICE)
            if x0.nelement() == 0: continue

            with autocast(enabled=DEVICE.type == 'cuda'):
                if model_type == "hydra":
                    loss, _ = model.compute_loss(x0)  # Uses internal sampling of t
                elif model_type == "transformer":
                    loss, _ = model.compute_loss(x0)  # Standard causal LM loss
                else:
                    raise ValueError("Unknown model type")

            # Loss is usually mean over batch*seq_len tokens
            # For perplexity need sum of NLL / total tokens
            # If loss is mean NLL, total NLL = loss * num_tokens_in_batch
            num_tokens_in_batch = x0.numel()
            total_loss += loss.item() * num_tokens_in_batch
            total_tokens += num_tokens_in_batch
            actual_batches += 1

    if total_tokens == 0: return float('inf')
    avg_neg_log_likelihood = total_loss / total_tokens
    perplexity = math.exp(avg_neg_log_likelihood)
    return perplexity


def measure_generation_speed(model: nn.Module, prompt_len=16, gen_len=64, batch_size=4, num_repeats=1,
                             model_type="hydra"):
    """ Measure generation throughput/latency. """
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE - 1, (batch_size, prompt_len), device=DEVICE)
    print('prompt', prompt)

    # Warmup
    print('warming up')
    _ = model.generate(prompt, num_tokens_to_generate=gen_len)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()

    print('warmed up...ready')
    start_time = time.time()
    for _ in range(num_repeats):
        _ = model.generate(prompt, num_tokens_to_generate=gen_len)
        #TODO _ = tokenizer.decode(_)
        print('  generated', _)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
    end_time = time.time()

    total_time = end_time - start_time
    avg_time_per_batch = total_time / num_repeats
    total_tokens_generated_run = batch_size * gen_len
    tokens_per_second = (total_tokens_generated_run * num_repeats) / total_time
    latency_ms_per_token = (avg_time_per_batch / total_tokens_generated_run) * 1000 if total_tokens_generated_run > 0 else float('inf')

    print(f"\n--- {model_type.upper()} Generation Speed ---")
    print(f"Prompt={prompt_len}, Gen={gen_len}, Batch={batch_size}, Repeats={num_repeats}")
    print(f"Avg time/batch: {avg_time_per_batch:.4f} s")
    print(f"Tokens/second: {tokens_per_second:.2f}")
    print(f"Latency/token: {latency_ms_per_token:.2f} ms")
    return avg_time_per_batch, tokens_per_second


# --- Comparison Baseline: Transformer ---
class SimpleTransformerLM(nn.Module):
    # Same as previous implementation
    def __init__(self, vocab_size, embed_dim, nhead, num_layers, dim_feedforward, max_seq_len=512, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, nhead, dim_feedforward, dropout, batch_first=True,
                                                   norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers, norm=nn.LayerNorm(embed_dim))
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.to(DEVICE)

    def forward(self, src: Tensor) -> Tensor:
        seq_len = src.shape[1]
        if seq_len > self.max_seq_len:
            src = src[:, -self.max_seq_len:]  # Truncate input
            seq_len = self.max_seq_len

        src_emb = self.token_embedding(src) * math.sqrt(self.embed_dim)
        src_emb = src_emb + self.pos_encoder[:, :seq_len, :]  # Add positional encoding

        # Causal mask
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=DEVICE)

        output = self.transformer_encoder(src_emb, mask=mask)
        logits = self.lm_head(output)
        return logits

    def compute_loss(self, x_0: Tensor) -> Tuple[Tensor, Tensor]:
        """Standard Causal LM loss."""
        inp = x_0[:, :-1]
        tgt = x_0[:, 1:]
        logits = self.forward(inp)  # Logits for predicting next token
        # Align logits and targets
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        return loss, logits

    @torch.no_grad()
    def generate(self, prompt: Tensor, num_tokens_to_generate: int, temperature: float = 0.7,
                 top_k: Optional[int] = 40) -> Tensor:
        self.eval()
        generated = prompt
        for _ in range(num_tokens_to_generate):
            context = generated
            # Limit context to max_seq_len for efficiency and pos encoding
            if context.shape[1] > self.max_seq_len:
                context = context[:, -self.max_seq_len:]

            logits = self.forward(context)[:, -1, :]  # Get logits for the very last token prediction

            if temperature == 0:  # Argmax
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                if temperature > 0:
                    logits = logits / temperature
                if top_k is not None and top_k > 0:
                    v = logits.size(-1)
                    k = min(top_k, v)
                    indices_to_remove = logits < torch.topk(logits, k, dim=-1)[0][..., -1, None]
                    logits[indices_to_remove] = -float('Inf')

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            generated = torch.cat((generated, next_token), dim=1)
        return generated


# --- Main Execution & Testing ---
def main():
    print(f"Using device: {DEVICE}")
    print("WARNING: HydraScale uses a **sequential** SSM scan for its forward pass.")
    print("         Training will be very slow. Generation uses efficient recurrent steps.")

    # --- Configuration ---
    SEQ_LEN = 128
    BATCH_SIZE = 16  # Reduce if OOM
    NUM_BATCHES_DATA = 50  # Number of batches for dummy data
    NUM_TRAIN_STEPS = 10  # Reduced training steps for demo
    NUM_EVAL_BATCHES = 10  # Batches for evaluation

    # Generate Dummy Data (List of tensors to simulate a loader)
    print("Generating dummy data...")
    dummy_data = [torch.randint(0, VOCAB_SIZE - 1, (BATCH_SIZE, SEQ_LEN), device=DEVICE)
                  for _ in range(NUM_BATCHES_DATA)]
    train_loader = dummy_data[:NUM_BATCHES_DATA // 2]
    eval_loader = dummy_data[NUM_BATCHES_DATA // 2:]

    # --- Model Definitions ---
    print("\nInitializing models...")
    # HydraScale (Small)
    hydra_model = HydraScaleLM(
        vocab_size=VOCAB_SIZE,
        embed_dim=256, depth=4,
        num_diffusion_timesteps=50, ssm_d_conv=3
    ).to(DEVICE)

    # Transformer Baseline (Comparable Params)
    tf_embed = 256;
    tf_nhead = 4;
    tf_ffn = tf_embed * 2;
    tf_layers = 3;
    tf_max_len = 512
    transformer_model = SimpleTransformerLM(VOCAB_SIZE, tf_embed, tf_nhead, tf_layers, tf_ffn,
                                            max_seq_len=tf_max_len).to(DEVICE)

    print(f"HydraScale Params: {count_parameters(hydra_model) / 1e6:.2f} M")
    print(f"Transformer Params: {count_parameters(transformer_model) / 1e6:.2f} M")

    # --- Training (Few Steps Demo) ---
    print(f"\n--- Training Demo ({NUM_TRAIN_STEPS} Steps) ---")
    hydra_optimizer = torch.optim.AdamW(hydra_model.parameters(), lr=1e-4)
    tf_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=1e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=DEVICE.type == 'cuda')  # For mixed precision

    hydra_model.train()
    transformer_model.train()
    for i in range(NUM_TRAIN_STEPS):
        batch = train_loader[i % len(train_loader)]  # Cycle through train data

        # Hydra Training Step
        hydra_optimizer.zero_grad()
        with autocast(enabled=DEVICE.type == 'cuda'):
            loss_hydra, _ = hydra_model.compute_loss(batch)
        # loss_hydra.backward() # Replace with scaler
        scaler.scale(loss_hydra).backward()
        # hydra_optimizer.step() # Replace with scaler
        scaler.step(hydra_optimizer)
        scaler.update()  # Needed for next iteration

        # Transformer Training Step
        tf_optimizer.zero_grad()
        with autocast(enabled=DEVICE.type == 'cuda'):
            loss_tf, _ = transformer_model.compute_loss(batch)
        # loss_tf.backward()
        scaler.scale(loss_tf).backward()
        # tf_optimizer.step()
        scaler.step(tf_optimizer)
        scaler.update()

        if (i + 1) % 5 == 0 or i == NUM_TRAIN_STEPS - 1:
            print(f"Step {i + 1}: Hydra Loss={loss_hydra.item():.4f}, TF Loss={loss_tf.item():.4f}")

    # --- Evaluation ---
    print("\n--- Evaluation ---")
    # Perplexity
    print("Calculating Perplexity (HydraScale)...")
    # NOTE: This uses compute_loss which calls the SLOW forward pass multiple times.
    perplexity_hydra = evaluate_perplexity(hydra_model, eval_loader, NUM_EVAL_BATCHES, model_type="hydra")
    print(f"HydraScale Perplexity: {perplexity_hydra:.2f} (Note: Based on limited training & slow evaluation)")

    print("\nCalculating Perplexity (Transformer)...")
    perplexity_tf = evaluate_perplexity(transformer_model, eval_loader, NUM_EVAL_BATCHES, model_type="transformer")
    print(f"Transformer Perplexity: {perplexity_tf:.2f} (Note: Based on limited training)")

    # Generation Speed
    print("\nMeasure Generation Speed (Hydra)...")
    measure_generation_speed(hydra_model, prompt_len=16, gen_len=64, batch_size=4, model_type="hydra")
    print("\nMeasure Generation Speed (Transformer)...")
    measure_generation_speed(transformer_model, prompt_len=16, gen_len=64, batch_size=4, model_type="transformer")

    # --- Qualitative Generation Example ---
    print("\n--- Generation Example ---")
    prompt_example = torch.tensor([[101, 7592, 1010, 2026, 3899, 2003, 1012, 0]], dtype=torch.long,
                                  device=DEVICE)  # Example prompt
    print(f"Prompt: {prompt_example.tolist()}")

    print("\nGenerating with HydraScale...")
    generated_hydra = hydra_model.generate(prompt_example, num_tokens_to_generate=32, temperature=0.8, top_k=50)
    print(f"HydraScale Output: {generated_hydra.tolist()}")

    print("\nGenerating with Transformer...")
    generated_tf = transformer_model.generate(prompt_example, num_tokens_to_generate=32, temperature=0.8, top_k=50)
    print(f"Transformer Output: {generated_tf.tolist()}")


if __name__ == "__main__":
    main()