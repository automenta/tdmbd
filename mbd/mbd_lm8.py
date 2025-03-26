#!/usr/bin/env python3
"""
HydraScale V3.1: Language Model with Self-Contained Selective Scan Core
and Discrete Diffusion. Fixes Time MLP initialization. Enhances output visibility.
Replaces external mamba_ssm dependency with an internal implementation.
Features efficient recurrent inference but uses a sequential scan for training
(training performance bottleneck). Includes enhanced analysis and comparison.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
from torch.cuda.amp import autocast, GradScaler
from torch import Tensor
import math
import time
import numpy as np
from tqdm import tqdm
import warnings

# --- Constants ---
VOCAB_SIZE = 50_000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_L_PRIME = 0  # Default: Process full sequence (0 means no blocking)

# --- Self-Contained Selective Scan (S6) Implementation ---
class SelectiveScan(nn.Module):
    """
    Self-contained implementation of the Selective Scan mechanism (S6),
    inspired by Mamba, without external mamba_ssm dependency.

    Uses a sequential scan in the `forward` pass (SLOW FOR TRAINING) and an
    efficient recurrent calculation in the `step` function (FAST FOR INFERENCE).

    Args:
        embed_dim (int): Input/output dimension (D).
        state_dim (int): State dimension (N). Typically small (e.g., 16).
        d_conv (int): Local convolution width.
        dt_rank (int | str): Rank for delta_t projection. 'auto' defaults to ceil(embed_dim / 16).
        bias (bool): Whether to include bias in linear layers.
    """
    def __init__(self, embed_dim: int, state_dim: int = 16, d_conv: int = 4, dt_rank: str | int = 'auto', bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.d_conv = d_conv
        self.dt_rank = math.ceil(embed_dim / 16) if dt_rank == 'auto' else dt_rank
        self.bias = bias

        # Input-dependent projections: x -> (x', z, dt, B, C)
        self.in_proj_x = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_z = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_params = nn.Linear(embed_dim, self.dt_rank + 2 * self.state_dim, bias=False) # Usually no bias for params

        # Local convolution (depthwise)
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim, bias=bias,
            kernel_size=d_conv, groups=embed_dim,
            padding=d_conv - 1,
        )

        # Delta_t projection (from dt_rank to embed_dim)
        self.dt_proj = nn.Linear(self.dt_rank, embed_dim, bias=True) # Bias is important for dt

        # State matrix A (learnable log-diagonal) and residual D (learnable)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).unsqueeze(0).repeat(embed_dim, 1)) # Shape: [D, N]
        self.D = nn.Parameter(torch.ones(embed_dim)) # Shape: [D]

        # Output projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # Initialize dt_proj bias
        with torch.no_grad():
            dt_init_std = self.dt_rank**-0.5
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            inv_softplus_target = math.log(math.expm1(0.01)) # Target dt ~ 0.01
            self.dt_proj.bias.data.fill_(inv_softplus_target)


    def _compute_A_tilde(self, dt: Tensor, A_log: Tensor) -> Tensor:
        """ Discretize continuous A using Zero-Order Hold (ZOH). dt: [B, L, D] """
        A = -torch.exp(A_log.float()) # Shape: [D, N]
        A_tilde = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(1)) # Shape: [B, L, D, N]
        return A_tilde

    def _compute_B_tilde(self, dt: Tensor, A_log: Tensor, A_tilde: Optional[Tensor] = None) -> Tensor:
        """ Discretize continuous B using ZOH variant. B is treated as identity here. dt: [B, L, D] """
        A = -torch.exp(A_log.float()) # Shape: [D, N]
        if A_tilde is None:
             A_tilde = self._compute_A_tilde(dt, A_log) # Shape: [B, L, D, N]

        A_unsqueezed = A.unsqueeze(0).unsqueeze(1) # Shape: [1, 1, D, N]
        is_zero_A = torch.abs(A_unsqueezed) < 1e-8
        B_tilde = torch.where(
            is_zero_A,
            dt.unsqueeze(-1), # Broadcast dt to [B, L, D, N]
            (A_tilde - 1) / (A_unsqueezed + 1e-10) # Add epsilon for stability
        ) # Shape: [B, L, D, N]
        return B_tilde

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass using **sequential scan**. This is SLOW for training.
        Args: x (Tensor): Input tensor [B, L, D]
        Returns: Tensor: Output tensor [B, L, D]
        """
        batch, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
             raise ValueError(f"Input embed_dim ({embed_dim}) doesn't match model embed_dim ({self.embed_dim})")

        # --- Input Projections ---
        x_res = self.in_proj_x(x) # [B, L, D]
        z = self.in_proj_z(x)     # [B, L, D]
        params = self.in_proj_params(x) # [B, L, dt_rank + 2*N]
        dt_unproj, B_proj, C_proj = torch.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_unproj)) # [B, L, D]

        # --- Convolution ---
        x_conv = self.conv1d(x_res.transpose(1, 2)) # [B, D, L + (k-1)]
        x_conv = x_conv[:, :, :seq_len] # [B, D, L]
        u = F.silu(x_conv.transpose(1, 2)) # [B, L, D]

        # --- Prepare SSM Parameters ---
        A_tilde = self._compute_A_tilde(dt, self.A_log) # [B, L, D, N]
        B_tilde = self._compute_B_tilde(dt, self.A_log, A_tilde) # [B, L, D, N]

        # --- Sequential Scan ---
        h = torch.zeros(batch, self.embed_dim, self.state_dim, device=x.device, dtype=A_tilde.dtype) # [B, D, N]
        ys = []
        for t in range(seq_len):
            A_t = A_tilde[:, t, :, :] # [B, D, N]
            B_t = B_tilde[:, t, :, :] # [B, D, N]
            B_proj_t = B_proj[:, t, :] # [B, N]
            C_proj_t = C_proj[:, t, :] # [B, N]
            u_t = u[:, t, :]           # [B, D]

            input_term = torch.einsum('bdn, bn, bd -> bdn', B_t, B_proj_t, u_t)
            h = A_t * h + input_term # Update state [B, D, N]

            y_t = torch.einsum('bn, bdn -> bd', C_proj_t, h)
            ys.append(y_t)

        y = torch.stack(ys, dim=1) # [B, L, D]

        # --- Residual and Gating ---
        y = y + u * self.D.unsqueeze(0).unsqueeze(1) # [B, L, D]
        y = y * F.silu(z) # [B, L, D]

        # --- Output Projection ---
        y_out = self.out_proj(y) # [B, L, D]
        return y_out

    def step(self, x_step: Tensor, h_prev: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Recurrent step function for efficient inference/generation (O(1) per step).
        Args:
            x_step (Tensor): Input [B, D]
            h_prev (Tensor): Previous state [B, D, N]
            conv_state (Tensor): Conv state [B, D, K-1]
        Returns: Tuple[Tensor, Tensor, Tensor]: y_step [B, D], h [B, D, N], new_conv_state [B, D, K-1]
        """
        batch, embed_dim = x_step.shape

        # --- Convolution Step ---
        conv_input = torch.cat([conv_state, x_step.unsqueeze(2)], dim=2) # [B, D, K]
        new_conv_state = conv_input[:, :, 1:] # [B, D, K-1]
        conv_out = F.conv1d(
            conv_input, weight=self.conv1d.weight, bias=self.conv1d.bias,
            groups=self.embed_dim, padding=0
        ).squeeze(-1) # [B, D]
        u_step = F.silu(conv_out) # [B, D]

        # --- Input Projections for SSM Params ---
        z_step = self.in_proj_z(x_step)     # [B, D]
        params = self.in_proj_params(x_step) # [B, dt_rank + 2*N]
        dt_unproj, B_proj, C_proj = torch.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        dt_step = F.softplus(self.dt_proj(dt_unproj)) # [B, D]

        # --- Discretize A and B for this step ---
        dt_for_compute = dt_step.unsqueeze(1) # [B, 1, D]
        A_tilde_step = self._compute_A_tilde(dt_for_compute, self.A_log).squeeze(1) # [B, D, N]
        B_tilde_step = self._compute_B_tilde(dt_for_compute, self.A_log, A_tilde_step.unsqueeze(1)).squeeze(1) # [B, D, N]

        # --- Recurrent State Update ---
        input_term = torch.einsum('bdn, bn, bd -> bdn', B_tilde_step, B_proj, u_step)
        h = A_tilde_step * h_prev + input_term # New state [B, D, N]

        # --- Output Calculation ---
        y = torch.einsum('bn, bdn -> bd', C_proj, h)
        y = y + u_step * self.D.unsqueeze(0) # [B, D]

        # --- Gating and Output Projection ---
        y = y * F.silu(z_step) # [B, D]
        y_step = self.out_proj(y) # [B, D]

        return y_step, h, new_conv_state


# --- HydraScale Components ---

class HydraBlock(nn.Module):
    """ Combines Selective Scan with LayerNorm and MLP (standard block structure). """
    def __init__(self, embed_dim: int, mlp_mult: int = 4, ssm_kwargs: Dict[str, Any] = {}):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ssm = SelectiveScan(embed_dim, **ssm_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = embed_dim * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, embed_dim)
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        x = self.ssm(x)
        x = residual + x
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        return x

    def step(self, x_step: Tensor, ssm_state: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Step function for recurrent inference, managing states. """
        residual = x_step
        x_norm1 = self.norm1(x_step)
        ssm_out, ssm_state_new, conv_state_new = self.ssm.step(x_norm1, ssm_state, conv_state)
        x = residual + ssm_out
        residual = x
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        y_step = residual + mlp_out
        return y_step, ssm_state_new, conv_state_new

def sinusoidal_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    """ Sinusoidal time embedding. Computes static embeddings. """
    if timesteps.ndim > 1:
        timesteps = timesteps.squeeze(-1) # Ensure timesteps are 1D: [B]
    if not hasattr(sinusoidal_embedding, 'pe') or sinusoidal_embedding.pe.size(1) != embedding_dim:
        # Cache the positional encoding matrix if dim changes or not cached
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        # Ensure device is consistent during calculation if possible, or use timesteps.device
        _device = timesteps.device
        emb = torch.exp(torch.arange(half_dim, device=_device, dtype=torch.float32) * -emb)
        sinusoidal_embedding.pe = emb.unsqueeze(0) # Shape [1, half_dim]
        print(f"  (Re)Calculating sinusoidal embedding matrix for dim={embedding_dim} on {timesteps.device}")


    # Use cached embedding matrix 'pe'
    emb = timesteps.float().unsqueeze(1) * sinusoidal_embedding.pe.to(timesteps.device) # [B, 1] * [1, H/2] -> [B, H/2]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # [B, H]
    if embedding_dim % 2 == 1: # zero pad if dim is odd
        emb = F.pad(emb, (0, 1))
    return emb


class HydraScaleLM(nn.Module):
    """
    HydraScale Language Model using the self-contained Selective Scan
    and Discrete Diffusion. Corrected Time MLP initialization.

    Args are same as V3.
    """
    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 512,
                 depth: int = 6,
                 mlp_mult: int = 4,
                 num_diffusion_timesteps: int = 100,
                 noise_schedule: str = 'cosine',
                 ssm_state_dim: int = 16,
                 ssm_d_conv: int = 4,
                 ssm_dt_rank: str | int = 'auto'):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_timesteps = num_diffusion_timesteps
        self.mask_token_id = vocab_size
        self.ssm_d_conv = ssm_d_conv

        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim) # +1 for [MASK] token

        # Time embedding MLP - Takes sinusoidal embedding as input
        self.time_embedding_dim = embed_dim
        # **FIX:** Removed sinusoidal_embedding function from Sequential
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4), # Input dim matches sinusoidal output
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim),
        )

        # HydraBlocks (SSM core)
        ssm_kwargs = {'state_dim': ssm_state_dim, 'd_conv': ssm_d_conv, 'dt_rank': ssm_dt_rank}
        self.layers = nn.ModuleList([
            HydraBlock(embed_dim, mlp_mult=mlp_mult, ssm_kwargs=ssm_kwargs)
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size)

        # --- Diffusion Schedule ---
        betas = self._calculate_betas(num_diffusion_timesteps, noise_schedule)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        self.to(DEVICE)
        print(f"HydraScaleLM initialized on {DEVICE}.")
        warnings.warn(
            "HydraScaleLM training uses a sequential scan in its forward pass, "
            "which will be significantly slower than optimized parallel scan implementations (like CUDA Mamba). "
            "Generation/inference uses an efficient recurrent step."
        )

    def _calculate_betas(self, timesteps, schedule='cosine', s=0.008, beta_start=0.0001, beta_end=0.02):
        # Same as before
        if schedule == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999).float()
        elif schedule == 'linear':
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}")

    def _extract(self, a: Tensor, t: Tensor, x_shape: Tuple[int, ...]) -> Tensor:
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    def _get_mask_prob_from_time(self, t: Tensor) -> Tensor:
        alpha_bar_t = self.alphas_cumprod.gather(-1, t)
        mask_prob = 1.0 - torch.sqrt(alpha_bar_t)
        return mask_prob

    def _mask_tokens(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len = x_0.shape
        mask_prob = self._get_mask_prob_from_time(t)
        mask_prob_expanded = mask_prob.view(batch_size, 1).expand(batch_size, seq_len)
        rand_noise = torch.rand_like(x_0, dtype=torch.float32)
        mask = rand_noise < mask_prob_expanded
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t, mask

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """
        Forward pass for training/evaluation (SLOW sequential scan).
        Args: x [B, L], t [B]
        Returns: logits [B, L, V]
        """
        batch_size, seq_len = x.shape

        # 1. Embed tokens
        token_emb = self.token_embedding(x) # [B, L, D]

        # 2. Embed timesteps and add
        # **FIX:** Explicitly call sinusoidal_embedding *before* time_mlp
        time_emb_sin = sinusoidal_embedding(t, self.time_embedding_dim) # [B, D_time]
        time_emb = self.time_mlp(time_emb_sin) # [B, D]
        h = token_emb + time_emb.unsqueeze(1) # Add time embedding [B, L, D]

        # 3. Apply HydraBlocks (Core SSM/MLP layers - uses slow forward)
        for layer in self.layers:
            h = layer(h)

        # 4. Final normalization and LM head
        h = self.norm_out(h)
        logits = self.lm_head(h) # [B, L, V]
        return logits

    def compute_loss(self, x_0: Tensor) -> Tuple[Tensor, float]:
        """ Computes the diffusion training loss and accuracy on masked tokens. """
        batch_size, seq_len = x_0.shape
        if x_0.nelement() == 0:
             return torch.tensor(0.0, device=DEVICE, requires_grad=True), 0.0

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=DEVICE).long()
        x_t, mask = self._mask_tokens(x_0, t)
        pred_logits = self.forward(x_t, t) # [B, L, V] - SLOW STEP

        masked_logits = pred_logits[mask]
        masked_targets = x_0[mask]

        if masked_targets.numel() == 0:
            loss = torch.tensor(0.0, device=DEVICE, requires_grad=True)
            accuracy = 1.0
        else:
            loss = F.cross_entropy(masked_logits, masked_targets)
            with torch.no_grad():
                correct_preds = (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
                accuracy = correct_preds / masked_targets.numel()

        return loss, accuracy

    @torch.no_grad()
    def _predict_x0_from_logits(self, x_t: Tensor, t: Tensor, logits: Tensor,
                                sampling_mode: str = 'argmax', temperature: float = 1.0, top_k: Optional[int] = None
                               ) -> Tensor:
        """ Samples or takes argmax from logits to predict x_0, only updating masks. """
        batch_size, seq_len, vocab_size = logits.shape

        if sampling_mode == 'argmax':
            pred_ids = torch.argmax(logits, dim=-1) # [B, L]
        elif sampling_mode in ['multinomial', 'topk']:
            logits_flat = logits.view(-1, vocab_size) # [B*L, V]
            if temperature > 0 and temperature != 1.0:
                logits_flat = logits_flat / temperature
            if top_k is not None and top_k > 0:
                k = min(top_k, vocab_size)
                kth_vals, _ = torch.topk(logits_flat, k, dim=-1)
                kth_vals_min = kth_vals[..., -1, None]
                indices_to_remove = logits_flat < kth_vals_min
                logits_flat.masked_fill_(indices_to_remove, -float('Inf'))

            probs = F.softmax(logits_flat, dim=-1) # [B*L, V]
            # Handle potential NaN probs if all logits become -inf (e.g., from extreme top-k)
            probs = torch.nan_to_num(probs, nan=0.0)
            # Ensure there's some probability mass, otherwise multinomial fails
            # If a row sums to 0, sample uniformly? Or assign to a default token?
            # For simplicity, let's re-normalize rows that sum to 0 to uniform.
            row_sums = probs.sum(dim=-1, keepdim=True)
            zero_rows = (row_sums <= 1e-9)
            uniform_probs = torch.full_like(probs, 1.0/vocab_size)
            probs = torch.where(zero_rows, uniform_probs, probs)
            # Renormalize just in case: probs /= probs.sum(dim=-1, keepdim=True) -> Already handled by softmax

            pred_ids_flat = torch.multinomial(probs, num_samples=1).squeeze(-1) # [B*L]
            pred_ids = pred_ids_flat.view(batch_size, seq_len) # [B, L]
        else:
             raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        mask = (x_t == self.mask_token_id)
        return torch.where(mask, pred_ids, x_t)

    @torch.no_grad()
    def generate(self,
                 prompt: Tensor,
                 num_tokens_to_generate: int,
                 num_sampling_steps: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 sampling_mode: str = 'topk'
                 ) -> Tensor:
        """
        Generate sequence using iterative denoising via the EFFICIENT recurrent `step` function.
        Args are same as V3.
        Returns: Generated sequence [B, L_prompt + num_tokens_to_generate]
        """
        self.eval()
        batch_size, prompt_len = prompt.shape
        total_len = prompt_len + num_tokens_to_generate
        sampling_steps = num_sampling_steps if num_sampling_steps is not None else self.num_timesteps
        if sampling_steps > self.num_timesteps:
            warnings.warn(f"num_sampling_steps ({sampling_steps}) > model.num_timesteps ({self.num_timesteps}). Clamping.")
            sampling_steps = self.num_timesteps

        x_gen = torch.full((batch_size, total_len), self.mask_token_id, dtype=torch.long, device=DEVICE)
        x_gen[:, :prompt_len] = prompt.to(DEVICE)

        layer_states: List[Tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            ssm_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.state_dim, device=DEVICE, dtype=torch.float32)
            conv_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.d_conv - 1, device=DEVICE, dtype=torch.float32)
            layer_states.append((ssm_state, conv_state))

        time_indices = torch.linspace(self.num_timesteps - 1, 0, sampling_steps, device=DEVICE).long()

        for i in tqdm(range(sampling_steps), desc="Generating", disable=batch_size > 1, leave=False):
            t_current_val = time_indices[i]
            t_current = t_current_val.expand(batch_size)

            current_layer_states = [(s[0].clone(), s[1].clone()) for s in layer_states]
            token_emb = self.token_embedding(x_gen) # [B, L_total, D]

            # **FIX:** Explicitly call sinusoidal_embedding *before* time_mlp
            time_emb_sin = sinusoidal_embedding(t_current, self.time_embedding_dim) # [B, D_time]
            time_emb = self.time_mlp(time_emb_sin) # [B, D]
            # time_emb_expanded = time_emb.unsqueeze(1) # [B, 1, D] - Not needed for step input

            all_logits_step = []
            with autocast(enabled=(DEVICE.type == 'cuda')):
                for token_idx in range(total_len):
                    # Input for this token step: token embedding + time embedding
                    x_step = token_emb[:, token_idx, :] + time_emb # [B, D] + [B, D] = [B, D]

                    for layer_idx, layer in enumerate(self.layers):
                        ssm_state, conv_state = current_layer_states[layer_idx]
                        x_step, ssm_state_new, conv_state_new = layer.step(x_step, ssm_state, conv_state)
                        current_layer_states[layer_idx] = (ssm_state_new, conv_state_new)

                    h_final = self.norm_out(x_step)
                    logits_token = self.lm_head(h_final)
                    all_logits_step.append(logits_token)

            logits = torch.stack(all_logits_step, dim=1) # [B, L_total, V]

            # Use argmax for the first few steps for stability? Or based on 't'?
            current_sampling_mode = sampling_mode
            # Optional heuristic: Use argmax when noise level is high (early steps)
            # if t_current_val > self.num_timesteps // 2 :
            #    current_sampling_mode = 'argmax'

            predicted_x0 = self._predict_x0_from_logits(
                x_gen, t_current, logits,
                sampling_mode=current_sampling_mode,
                temperature=temperature, top_k=top_k
            )
            x_gen = predicted_x0

        return x_gen


# --- Evaluation and Analysis Functions ---
# (count_parameters, evaluate_perplexity, measure_generation_speed are unchanged)
@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    """ Counts the number of trainable parameters in a model. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

@torch.no_grad()
def evaluate_perplexity(model: nn.Module, data_loader: List[Tensor], num_batches: Optional[int] = None,
                        model_type="hydra") -> float:
    """ Calculate perplexity on a dataset. """
    model.eval()
    total_neg_log_likelihood = 0.0
    total_tokens = 0
    actual_batches = 0
    if not data_loader: return float('inf')
    max_batches = len(data_loader) if num_batches is None else min(num_batches, len(data_loader))
    desc = f"Perplexity Eval ({model_type})"

    for i, batch_cpu in enumerate(tqdm(data_loader[:max_batches], desc=desc, leave=False)):
        if batch_cpu is None or batch_cpu.nelement() == 0: continue
        x0 = batch_cpu.to(DEVICE) # Move batch to device
        batch_size, seq_len = x0.shape
        num_tokens_in_batch = x0.numel()
        if num_tokens_in_batch == 0: continue

        try:
            with autocast(enabled=(DEVICE.type == 'cuda')):
                if model_type == "hydra":
                    loss, acc = model.compute_loss(x0)
                    # Use loss on masked tokens as average NLL estimate per token
                    if not torch.isnan(loss):
                        total_neg_log_likelihood += loss.item() * num_tokens_in_batch # Scale by total tokens for approx total NLL
                        total_tokens += num_tokens_in_batch
                elif model_type == "transformer":
                    if seq_len < 2: continue # Skip batches too short for causal LM loss
                    loss, _ = model.compute_loss(x0) # Standard Causal LM loss
                    num_predicted_tokens = batch_size * (seq_len - 1)
                    if not torch.isnan(loss) and num_predicted_tokens > 0:
                       total_neg_log_likelihood += loss.item() * num_predicted_tokens
                       total_tokens += num_predicted_tokens
                else:
                    raise ValueError(f"Unknown model type for perplexity: {model_type}")
            actual_batches += 1
        except Exception as e:
            print(f"Error during perplexity evaluation (batch {i}, type {model_type}): {e}")
            continue # Skip batch on error

    if total_tokens == 0: return float('inf')
    avg_neg_log_likelihood = total_neg_log_likelihood / total_tokens
    avg_neg_log_likelihood = min(avg_neg_log_likelihood, 700) # Clamp for exp
    try:
        perplexity = math.exp(avg_neg_log_likelihood)
    except OverflowError:
        perplexity = float('inf')
    return perplexity


@torch.no_grad()
def measure_generation_speed(model: nn.Module, prompt_len: int = 16, gen_len: int = 64,
                             batch_size: int = 4, num_repeats: int = 10, model_type: str = "hydra",
                             gen_kwargs: Dict[str, Any] = {} ) -> Dict[str, float]:
    """ Measure generation throughput and latency. """
    model.eval()
    prompt = torch.randint(0, VOCAB_SIZE - 1, (batch_size, prompt_len), device=DEVICE)
    total_gen_len = prompt_len + gen_len

    print(f"  Warmup ({model_type})...", end="")
    for _ in range(max(1, num_repeats // 5)):
         _ = model.generate(prompt, num_tokens_to_generate=gen_len, **gen_kwargs)
    if DEVICE.type == 'cuda': torch.cuda.synchronize()
    print(" Done.")

    start_time = time.perf_counter()
    for i in range(num_repeats):
        _ = model.generate(prompt, num_tokens_to_generate=gen_len, **gen_kwargs)
        if DEVICE.type == 'cuda': torch.cuda.synchronize()
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time_per_run = total_time / num_repeats
    total_new_tokens_generated_run = batch_size * gen_len
    total_new_tokens_all_runs = total_new_tokens_generated_run * num_repeats

    tokens_per_second = total_new_tokens_all_runs / total_time if total_time > 0 else float('inf')
    latency_ms_per_token = (avg_time_per_run / total_new_tokens_generated_run) * 1000 if total_new_tokens_generated_run > 0 else float('inf')

    results = {
        "avg_time_per_run_s": avg_time_per_run,
        "tokens_per_second": tokens_per_second,
        "latency_ms_per_token": latency_ms_per_token,
    }
    print(f"  {model_type.upper()} Speed Results:")
    print(f"    Prompt={prompt_len}, Gen={gen_len}, Batch={batch_size}, Repeats={num_repeats}")
    print(f"    Avg time/run: {results['avg_time_per_run_s']:.4f} s")
    print(f"    New tokens/sec: {results['tokens_per_second']:.2f}")
    print(f"    Latency/new token: {results['latency_ms_per_token']:.2f} ms")
    return results


# --- Comparison Baseline: Transformer (Simplified) ---
# (SimpleTransformerLM class definition is unchanged)
class SimpleTransformerLM(nn.Module):
    """ Standard Decoder-only Transformer for baseline comparison. """
    def __init__(self, vocab_size: int, embed_dim: int, nhead: int, num_layers: int,
                 dim_feedforward: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True
        )
        encoder_norm = nn.LayerNorm(embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        self.to(DEVICE)
        print(f"SimpleTransformerLM initialized on {DEVICE}.")


    def _generate_causal_mask(self, sz: int, device: torch.device) -> Tensor:
        return torch.triu(torch.full((sz, sz), -float('inf'), device=device), diagonal=1)

    def forward(self, src: Tensor) -> Tensor:
        batch_size, seq_len = src.shape
        if seq_len > self.max_seq_len:
            src = src[:, -self.max_seq_len:]
            seq_len = self.max_seq_len
            # warnings.warn(f"Input sequence truncated to max_seq_len ({self.max_seq_len})") # Can be noisy

        src_emb = self.token_embedding(src) * math.sqrt(self.embed_dim)
        pos_emb = self.pos_encoder[:, :seq_len, :]
        # Ensure pos_emb device matches src_emb device if they can differ
        src_combined = src_emb + pos_emb.to(src_emb.device)
        src_combined = self.dropout(src_combined)

        causal_mask = self._generate_causal_mask(seq_len, device=src.device)

        output = self.transformer_encoder(src_combined, mask=causal_mask, is_causal=False)
        logits = self.lm_head(output)
        return logits

    def compute_loss(self, x_0: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        if x_0.shape[1] < 2:
            return torch.tensor(0.0, device=DEVICE, requires_grad=True), None
        inp = x_0[:, :-1]
        tgt = x_0[:, 1:]
        logits = self.forward(inp)
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), tgt.reshape(-1))
        return loss, logits

    @torch.no_grad()
    def generate(self, prompt: Tensor, num_tokens_to_generate: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> Tensor:
        self.eval()
        generated_ids = prompt.to(DEVICE)
        batch_size = prompt.shape[0]

        for _ in range(num_tokens_to_generate):
            context = generated_ids
            if context.shape[1] > self.max_seq_len:
                context = context[:, -self.max_seq_len:]

            with autocast(enabled=(DEVICE.type == 'cuda')):
                logits = self.forward(context)
                next_token_logits = logits[:, -1, :]

            if temperature == 0:
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else:
                if temperature != 1.0:
                    next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v = next_token_logits.size(-1)
                    k = min(top_k, v)
                    kth_vals, _ = torch.topk(next_token_logits, k, dim=-1)
                    kth_vals_min = kth_vals[:, -1, None]
                    indices_to_remove = next_token_logits < kth_vals_min
                    next_token_logits.masked_fill_(indices_to_remove, -float('Inf'))

                probs = F.softmax(next_token_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0) # Handle potential NaNs
                # If all probs are 0, sample uniformly?
                zero_probs = (probs.sum(dim=-1, keepdim=True) < 1e-9)
                uniform_dist = torch.full_like(probs, 1.0 / probs.shape[-1])
                probs = torch.where(zero_probs, uniform_dist, probs)

                next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        return generated_ids


# --- Main Execution & Analysis ---
def main():
    print(f"Executing on device: {DEVICE}")
    print("-" * 60)

    # --- Configuration ---
    # Data params
    SEQ_LEN = 128
    BATCH_SIZE = 8
    NUM_BATCHES_DATA = 50
    # Model params (Small for demo)
    EMBED_DIM = 256
    DEPTH = 4
    MLP_MULT = 4
    SSM_STATE = 16
    SSM_DCONV = 4
    SSM_DT_RANK = 'auto' # ~ 16
    # Transformer params
    TF_NHEAD = 4
    TF_FFN = EMBED_DIM * MLP_MULT
    TF_LAYERS = DEPTH
    TF_MAX_LEN = 512
    # Training/Eval params
    NUM_TRAIN_STEPS = 25
    NUM_EVAL_BATCHES = 10
    LR = 3e-4
    # Generation params
    GEN_PROMPT_LEN = 16
    GEN_NEW_TOKENS = 48
    GEN_BATCH_SIZE = 4
    GEN_REPEATS = 10

    print("Generating dummy data...")
    # Keep data on CPU, move to GPU per batch
    dummy_data = [torch.randint(0, VOCAB_SIZE - 1, (BATCH_SIZE, SEQ_LEN), device='cpu') for _ in range(NUM_BATCHES_DATA)]
    train_loader = dummy_data[:NUM_BATCHES_DATA // 2]
    eval_loader = dummy_data[NUM_BATCHES_DATA // 2:]
    print(f"Dummy data: {len(train_loader)} train batches, {len(eval_loader)} eval batches.")
    print("-" * 60)

    print("Initializing models...")
    hydra_model_config = {
        "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM, "depth": DEPTH,
        "mlp_mult": MLP_MULT, "num_diffusion_timesteps": 50, "noise_schedule": "cosine",
        "ssm_state_dim": SSM_STATE, "ssm_d_conv": SSM_DCONV, "ssm_dt_rank": SSM_DT_RANK
    }
    hydra_model = HydraScaleLM(**hydra_model_config).to(DEVICE)

    tf_model_config = {
        "vocab_size": VOCAB_SIZE, "embed_dim": EMBED_DIM, "nhead": TF_NHEAD,
        "num_layers": TF_LAYERS, "dim_feedforward": TF_FFN, "max_seq_len": TF_MAX_LEN
    }
    transformer_model = SimpleTransformerLM(**tf_model_config).to(DEVICE)

    print(f"HydraScale Params: {count_parameters(hydra_model) / 1e6:.2f} M")
    print(f"Transformer Params: {count_parameters(transformer_model) / 1e6:.2f} M")
    print("-" * 60)

    print(f"--- Training Demo ({NUM_TRAIN_STEPS} Steps) ---")
    hydra_optimizer = torch.optim.AdamW(hydra_model.parameters(), lr=LR)
    tf_optimizer = torch.optim.AdamW(transformer_model.parameters(), lr=LR)
    scaler = GradScaler(enabled=(DEVICE.type == 'cuda'))
    hydra_losses, hydra_accs, tf_losses = [], [], []

    for i in tqdm(range(NUM_TRAIN_STEPS), desc="Training Steps"):
        batch_cpu = train_loader[i % len(train_loader)]
        batch = batch_cpu.to(DEVICE) # Move data to GPU for this batch

        # Hydra Training Step
        hydra_model.train()
        hydra_optimizer.zero_grad()
        with autocast(enabled=(DEVICE.type == 'cuda')):
            loss_hydra, acc_hydra = hydra_model.compute_loss(batch)
        if not torch.isnan(loss_hydra) and not torch.isinf(loss_hydra):
            scaler.scale(loss_hydra).backward()
            scaler.step(hydra_optimizer)
            scaler.update()
            hydra_losses.append(loss_hydra.item())
            hydra_accs.append(acc_hydra)
        else:
            print(f"Warning: Skipping Hydra update at step {i+1} due to NaN/Inf loss.")
            hydra_optimizer.zero_grad() # Ensure grads are cleared if step is skipped


        # Transformer Training Step
        transformer_model.train()
        tf_optimizer.zero_grad()
        with autocast(enabled=(DEVICE.type == 'cuda')):
            loss_tf, _ = transformer_model.compute_loss(batch)
        if not torch.isnan(loss_tf) and not torch.isinf(loss_tf):
            scaler.scale(loss_tf).backward()
            scaler.step(tf_optimizer)
            scaler.update()
            tf_losses.append(loss_tf.item())
        else:
             print(f"Warning: Skipping Transformer update at step {i+1} due to NaN/Inf loss.")
             tf_optimizer.zero_grad()


        if (i + 1) % 5 == 0 or i == NUM_TRAIN_STEPS - 1:
             hydra_loss_avg = np.mean(hydra_losses[-5:]) if hydra_losses else float('nan')
             hydra_acc_avg = np.mean(hydra_accs[-5:]) if hydra_accs else float('nan')
             tf_loss_avg = np.mean(tf_losses[-5:]) if tf_losses else float('nan')
             tqdm.write(f"  Step {i + 1:>{len(str(NUM_TRAIN_STEPS))}}: Hydra Loss={hydra_loss_avg:.4f} Acc={hydra_acc_avg*100:.2f}%, TF Loss={tf_loss_avg:.4f}")

    print("Training Demo Complete.")
    print("-" * 60)

    print("--- Evaluation ---")
    print("Calculating Perplexity (HydraScale - Approx)...")
    perplexity_hydra = evaluate_perplexity(hydra_model, eval_loader, NUM_EVAL_BATCHES, model_type="hydra")
    print(f"  HydraScale Approx Perplexity: {perplexity_hydra:.2f}")

    print("\nCalculating Perplexity (Transformer)...")
    perplexity_tf = evaluate_perplexity(transformer_model, eval_loader, NUM_EVAL_BATCHES, model_type="transformer")
    print(f"  Transformer Perplexity: {perplexity_tf:.2f}")
    print("-" * 60)

    print("--- Generation Speed Analysis ---")
    print("Measure Generation Speed (Hydra - using efficient recurrent step)...")
    hydra_gen_kwargs = {"num_sampling_steps": hydra_model.num_timesteps, "sampling_mode": "topk", "top_k": 50, "temperature": 0.8}
    speed_results_hydra = measure_generation_speed(hydra_model,
                                                   prompt_len=GEN_PROMPT_LEN, gen_len=GEN_NEW_TOKENS,
                                                   batch_size=GEN_BATCH_SIZE, num_repeats=GEN_REPEATS,
                                                   model_type="hydra", gen_kwargs=hydra_gen_kwargs)

    print("\nMeasure Generation Speed (Transformer - token by token)...")
    tf_gen_kwargs = {"top_k": 50, "temperature": 0.8}
    speed_results_tf = measure_generation_speed(transformer_model,
                                                prompt_len=GEN_PROMPT_LEN, gen_len=GEN_NEW_TOKENS,
                                                batch_size=GEN_BATCH_SIZE, num_repeats=GEN_REPEATS,
                                                model_type="transformer", gen_kwargs=tf_gen_kwargs)
    print("-" * 60)

    print("--- Generation Example ---")
    # Example prompt tokens (e.g., representing "The quick brown fox")
    prompt_example_ids = [101, 1996, 4248, 2829, 4419, 2183, 102] # Using common BPE IDs for demo
    prompt_example = torch.tensor([prompt_example_ids] * GEN_BATCH_SIZE, dtype=torch.long, device='cpu') # Use same batch size as speed test? No, use 1 for clarity.
    prompt_example = torch.tensor([prompt_example_ids], dtype=torch.long).to(DEVICE) # B=1

    print(f"Prompt Tokens (IDs): {prompt_example.tolist()}")
    # Note: To see actual text, a tokenizer corresponding to VOCAB_SIZE would be needed.

    print("\nGenerating with HydraScale...")
    generated_hydra = hydra_model.generate(prompt_example, num_tokens_to_generate=GEN_NEW_TOKENS, **hydra_gen_kwargs)
    # Print only the first example if batch size > 1 was used in generate call
    print(f"  HydraScale Output Tokens (IDs): {generated_hydra[0].tolist()}") # Print first item in batch

    print("\nGenerating with Transformer...")
    generated_tf = transformer_model.generate(prompt_example, num_tokens_to_generate=GEN_NEW_TOKENS, **tf_gen_kwargs)
    print(f"  Transformer Output Tokens (IDs): {generated_tf[0].tolist()}") # Print first item in batch
    print("-" * 60)

    print("Analysis Complete.")


if __name__ == "__main__":
    main()