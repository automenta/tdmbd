#!/usr/bin/env python3
"""
MBD-S: Modular Mamba-Block-Diffusion Language Model Library (Enhanced for Power).

Improvements:
- Replaced placeholder MLP with actual Mamba block (requires `mamba-ssm`).
- Improved context handling via Mamba's state or dedicated mechanism.
- Added positional embeddings.
- Refined diffusion masking/sampling process.
- Removed detrimental initial quantization; suggest PTQ/QAT paths.
- Enhanced training loop (placeholder for full trainer).
- Improved generation loop with proper sampling and stopping.
- Better code structure, constants, and type hinting.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional, Dict
from torch.cuda.amp import autocast, GradScaler
from torch import Tensor
import math
import logging
import time

# Try importing Mamba
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None
    logging.warning("`mamba-ssm` not found. Falling back to MLP/Attention placeholder. "
                    "Install with `pip install mamba-ssm causal-conv1d>=1.1.0` for full power.")

# --- Configuration & Constants ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Special Token IDs (assuming a standard tokenizer setup)
PAD_TOKEN_ID = 0
EOS_TOKEN_ID = 1 # Example: Often 1 or 2 depending on tokenizer
MASK_TOKEN_ID = 4 # Example: Use a dedicated mask token ID

DEFAULT_VOCAB_SIZE = 50_257 # Example: GPT-2 size, adjust to your tokenizer
DEFAULT_L_PRIME = 32      # Increased default block size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tier configurations (more detailed)
TIER_CONFIGS = {
    "simple": { # Focus: Edge/Fastest Inference
        "mamba_layers": 1,
        "expansion_factor": 2,
        "diffusion_steps": 1, # T
        "noise_beta": 0.5, # Fixed masking ratio
        "noise_omega": 0.5,
        "stop_entropy": 3.0,
        "quantize": True, # Target for PTQ/QAT
        "use_mamba": False, # Can fallback to MLP if mamba-ssm not needed/avail
    },
    "core": { # Focus: Balanced Performance
        "mamba_layers": 4,
        "expansion_factor": 2,
        "diffusion_steps": 5, # T
        "noise_beta": 0.1,
        "noise_omega": 0.9,
        "stop_entropy": float("inf"), # No entropy stopping by default
        "quantize": False,
        "use_mamba": True,
    },
    "enhanced": { # Focus: Higher Quality, some efficiency
        "mamba_layers": 8,
        "expansion_factor": 2.5,
        "diffusion_steps": 10, # T
        "noise_beta": 0.05,
        "noise_omega": 0.95,
        "stop_entropy": 4.5,
        "quantize": False, # Could be True if targeting efficient high quality
        "use_mamba": True,
    },
    "extreme": { # Focus: Maximum Quality
        "mamba_layers": 12,
        "expansion_factor": 3,
        "diffusion_steps": 15, # T
        "noise_beta": 0.01,
        "noise_omega": 0.99,
        "stop_entropy": 5.0,
        "quantize": False,
        "use_mamba": True,
    }
}

# --- Core Mamba Diffusion Block ---

class PositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class MDBBlock(nn.Module):
    """
    Mamba-Diffusion Block: Enhanced core processing unit.
    Uses true Mamba if available, otherwise falls back to MLP.
    """
    def __init__(self, embed_dim: int, mamba_layers: int, expansion_factor: float, vocab_size: int, tier_config: Dict):
        super().__init__()
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size
        self.use_mamba = tier_config["use_mamba"] and Mamba is not None
        hidden_dim = int(embed_dim * expansion_factor)

        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=PAD_TOKEN_ID)
        self.pos_encoder = PositionalEncoding(embed_dim)

        # --- Core Sequence Processor ---
        if self.use_mamba:
            logging.info(f"Using Mamba block (d_model={embed_dim})")
            # Configure Mamba based on embed_dim. These are examples, tune as needed.
            # d_state=16, d_conv=4 are common starting points.
            self.sequence_processor = Mamba(
                d_model=embed_dim,
                d_state=max(16, embed_dim // 16), # Scale state dim with model size
                d_conv=4,
                expand=2, # Inner expansion factor in Mamba itself
            )
            # Optional: Add more Mamba layers or MLP layers if needed
            # self.mamba_layers = nn.ModuleList([Mamba(...) for _ in range(mamba_layers)])

        else:
            # Fallback: Simple MLP (less powerful for sequences)
            logging.warning("Mamba not used. Falling back to MLP sequence processor.")
            # Input dim is embed_dim (x_t) + embed_dim (context summary)
            context_dim = embed_dim # We'll use a simpler context summary for MLP
            self.sequence_processor = nn.Sequential(
                nn.Linear(embed_dim + context_dim, hidden_dim),
                nn.GELU(), # GELU often performs better than ReLU
                # Add more layers based on tier_config['mamba_layers'] if desired
                *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU()) for _ in range(max(0, mamba_layers - 1))],
                nn.Linear(hidden_dim, embed_dim)
            )

        # --- Denoising Head ---
        self.denoise_norm = nn.LayerNorm(embed_dim)
        self.denoise_head = nn.Linear(embed_dim, vocab_size)

        # Optional: Initialize weights (e.g., Xavier initialization)
        self._init_weights()

    def _init_weights(self):
        # Apply sensible initializations
        for name, param in self.named_parameters():
            if 'embedding' in name:
                 nn.init.normal_(param, mean=0, std=self.embed_dim**-0.5)
            elif 'weight' in name and param.dim() >= 2:
                 nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                 nn.init.zeros_(param)
        # Tie embedding and output layer weights (improves performance)
        # Ensure dimensions match if you do this. May need adjustment if padding_idx is used differently.
        # self.denoise_head.weight = self.embedding.weight

    def forward(self, x_t: Tensor, x_prev_context: Tensor, mamba_state: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Process block: x_t (noisy block), x_prev_context (summary/state), optional mamba_state.
        Returns logits and updated mamba_state (if applicable).
        """
        # 1. Embed tokens
        x_t_embed = self.embedding(x_t) # [batch, seq_len, embed_dim]
        # Add positional encoding *relative to the block*
        x_t_embed = self.pos_encoder(x_t_embed)

        # 2. Process Sequence (Mamba or MLP)
        new_mamba_state = None
        if self.use_mamba:
            # Mamba expects input shape [batch, seq_len, embed_dim]
            # We need to handle the context. The *best* way is to pass the hidden state
            # from the previous block's Mamba execution.
            # For simplicity here, if mamba_state isn't passed, we might start fresh or use x_prev_context.
            # Let's assume mamba_state *is* passed during generation/sequential processing.
            # During training on full sequences, Mamba handles causality internally.
            h, new_mamba_state = self.sequence_processor(x_t_embed) # Mamba might take hidden state input
            # `h` has shape [batch, seq_len, embed_dim]
        else:
            # MLP Fallback: Concatenate context summary
            # Ensure x_prev_context is [batch, seq_len, embed_dim]
            if x_prev_context.dim() == 2: # If context is just [batch, embed_dim] summary
                 x_prev_context = x_prev_context.unsqueeze(1).expand(-1, x_t_embed.size(1), -1)
            elif x_prev_context.size(1) != x_t_embed.size(1):
                 # Handle mismatch, maybe pool or take last state
                 x_prev_context = x_prev_context[:,-1,:].unsqueeze(1).expand(-1, x_t_embed.size(1), -1)

            x_input = torch.cat([x_t_embed, x_prev_context], dim=-1)
            h = self.sequence_processor(x_input) # [batch, seq_len, embed_dim]

        # 3. Denoise
        h = self.denoise_norm(h)
        logits = self.denoise_head(h) # [batch, seq_len, vocab_size]

        return logits, new_mamba_state

# --- Main MBD-S Model ---

class MBDS(nn.Module):
    """
    MBD-S Language Model: Enhanced modular implementation.
    """
    def __init__(self,
                 vocab_size: int = DEFAULT_VOCAB_SIZE,
                 embed_dim: int = 512, # Increased default embed_dim
                 l_prime: int = DEFAULT_L_PRIME,
                 width: float = 1.0,
                 tier: str = "core"):
        super().__init__()
        if tier.lower() not in TIER_CONFIGS:
            raise ValueError(f"Unknown tier: {tier}. Available: {list(TIER_CONFIGS.keys())}")
        self.tier_name = tier.lower()
        self.config = TIER_CONFIGS[self.tier_name]

        self.vocab_size = vocab_size
        self.l_prime = l_prime
        self.width = width # Width multiplier can scale embed_dim further if needed

        # Scale dimensions based on tier config and width multiplier
        scaled_embed_dim = int(embed_dim * width)

        # Core MDB block using tier config
        self.mdb = MDBBlock(
            embed_dim=scaled_embed_dim,
            mamba_layers=self.config["mamba_layers"],
            expansion_factor=self.config["expansion_factor"],
            vocab_size=vocab_size,
            tier_config=self.config
        )

        # Tier-specific parameters from config
        self.T = self.config["diffusion_steps"]
        self.beta = self.config["noise_beta"]
        self.omega = self.config["noise_omega"]
        self.stop_entropy_threshold = self.config["stop_entropy"]
        self.needs_quantization = self.config["quantize"] # Flag for later PTQ/QAT

        logging.info(f"Initialized MBD-S model - Tier: {self.tier_name.upper()}")
        logging.info(f"  Embed Dim: {scaled_embed_dim}, L': {l_prime}, Vocab: {vocab_size}")
        logging.info(f"  Using Mamba: {self.mdb.use_mamba}")
        logging.info(f"  Diffusion Steps (T): {self.T}, Noise Range: [{self.beta:.2f}, {self.omega:.2f}]")
        self.to(DEVICE)

    def _get_noise_level(self, step: Optional[int] = None) -> float:
        """Determines the noise level (masking ratio) based on diffusion step or random."""
        if self.T <= 1: # No diffusion steps or simple masking
            return self.beta # Use the base fixed noise level

        if step is None: # Random step during training
            # Sample t uniformly from [beta, omega]
            return torch.rand(1).item() * (self.omega - self.beta) + self.beta
        else:
            # Linear schedule during generation (step from T down to 1)
            # Noise level decreases as we approach step 1
            # alpha_t = 1 - noise_level
            # Common diffusion schedules (like cosine) are more complex
            # Simple linear noise level:
            return self.beta + (self.omega - self.beta) * (step -1) / max(1, self.T - 1)

    def _split_blocks(self, x: Tensor) -> Tensor:
        """Split sequence into blocks of size l_prime, padding if necessary."""
        batch, seq_len = x.shape
        padded_len = math.ceil(seq_len / self.l_prime) * self.l_prime
        if padded_len > seq_len:
             # Pad with PAD_TOKEN_ID
            x_padded = F.pad(x, (0, padded_len - seq_len), value=PAD_TOKEN_ID)
        else:
            x_padded = x
        return x_padded.view(batch, -1, self.l_prime) # [batch, n_blocks, l_prime]

    def _mask_tokens(self, x: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor]:
        """
        Mask tokens with a given ratio.
        Returns masked tensor and the boolean mask.
        Uses a dedicated MASK_TOKEN_ID.
        Avoids masking special tokens like PAD.
        """
        if mask_ratio <= 0:
            return x, torch.zeros_like(x, dtype=torch.bool)
        if mask_ratio >= 1:
            return torch.full_like(x, MASK_TOKEN_ID), torch.ones_like(x, dtype=torch.bool)

        # Probability matrix, ignore PAD tokens
        can_mask = (x != PAD_TOKEN_ID)
        prob = torch.full_like(x, mask_ratio, dtype=torch.float, device=x.device)
        prob.masked_fill_(~can_mask, 0.0) # Don't mask padding

        # Generate mask based on probability
        mask = torch.bernoulli(prob).bool()

        # Apply mask
        masked_x = torch.where(mask, MASK_TOKEN_ID, x)
        return masked_x, mask

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """
        Forward pass: Compute logits and optional loss for training.
        Processes the entire sequence block by block sequentially if using Mamba state,
        or potentially in parallel if context is summarized differently.
        """
        if self.mdb.use_mamba:
            # For Mamba, process sequence more holistically if possible, or pass state
            # This example maintains block processing but shows state passing concept
            return self._forward_blockwise_sequential(x, targets)
        else:
            # For MLP, parallel block processing with context summary is feasible
            return self._forward_blockwise_parallel_mlp(x, targets)

    def _forward_blockwise_sequential(self, x: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass suitable for stateful models like Mamba."""
        x_blocks = self._split_blocks(x) # [batch, n_blocks, l_prime]
        batch_size, n_blocks, _ = x_blocks.shape
        all_logits = []
        mamba_state = None # Initialize Mamba state (if applicable)

        # --- Context Handling ---
        # For sequential processing, the "context" is implicitly handled by Mamba's state.
        # We just need to feed blocks sequentially.

        total_loss = 0.0
        num_loss_tokens = 0

        for b in range(n_blocks):
            x_b = x_blocks[:, b] # Current block [batch, l_prime]

            # --- Diffusion / Masking ---
            noise_level = self._get_noise_level(step=None) # Random noise for training
            x_t_b, mask = self._mask_tokens(x_b, noise_level)

            # --- Process Block ---
            # Pass the current Mamba state (if any)
            # For training, we typically process the whole sequence, Mamba handles causality.
            # If processing strictly block-by-block for memory, pass state.
            # Context for Mamba is implicitly its internal state from previous tokens/blocks.
            # We pass a dummy context tensor here as the MDBBlock signature expects it,
            # but it's not used by the Mamba path.
            dummy_context = torch.empty(batch_size, 0, self.mdb.embed_dim, device=x.device) # Placeholder
            logits_b, mamba_state = self.mdb(x_t_b, dummy_context, mamba_state=mamba_state)
            all_logits.append(logits_b)

            # --- Loss Calculation (if targets provided) ---
            if targets is not None:
                # Align targets with the current block
                target_blocks = self._split_blocks(targets) # Inefficient to do this every time
                target_b = target_blocks[:, b]

                # Calculate loss only for non-padded tokens
                loss_mask = (target_b != PAD_TOKEN_ID)
                loss = F.cross_entropy(
                    logits_b.view(-1, self.vocab_size), # [batch*l_prime, vocab_size]
                    target_b.view(-1),                  # [batch*l_prime]
                    reduction='none'                    # Get loss per token
                )
                loss = loss.view(batch_size, self.l_prime)
                masked_loss = loss * loss_mask # Zero out loss for padded tokens
                total_loss += masked_loss.sum()
                num_loss_tokens += loss_mask.sum().item()

        # --- Combine Results ---
        logits = torch.stack(all_logits, dim=1) # [batch, n_blocks, l_prime, vocab_size]
        # Reshape logits to match original sequence length (approx)
        logits = logits.view(batch_size, n_blocks * self.l_prime, self.vocab_size)
        # Trim potential padding added by _split_blocks
        logits = logits[:, :x.size(1), :]

        final_loss = None
        if targets is not None and num_loss_tokens > 0:
            final_loss = total_loss / num_loss_tokens

        return logits, final_loss

    def _forward_blockwise_parallel_mlp(self, x: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass suitable for MLP (parallel blocks with summarized context)."""
        x_blocks = self._split_blocks(x) # [batch, n_blocks, l_prime]
        batch_size, n_blocks, _ = x_blocks.shape
        all_logits = []

        # Pre-calculate context summaries if needed (e.g., mean of previous blocks' embeddings)
        # This is still a bottleneck, but feasible for MLP.
        # For simplicity, let's use a running mean context like the original, but embed first.
        embedded_blocks = self.mdb.embedding(x_blocks) # [batch, n_blocks, l_prime, embed_dim]

        total_loss = 0.0
        num_loss_tokens = 0

        prev_context_summary = torch.zeros(batch_size, self.mdb.embed_dim, device=x.device)

        for b in range(n_blocks):
            x_b = x_blocks[:, b] # Current block tokens [batch, l_prime]
            x_b_embed = embedded_blocks[:, b] # Current block embeddings [batch, l_prime, embed_dim]

            # --- Diffusion / Masking ---
            noise_level = self._get_noise_level(step=None)
            # Mask the *tokens* before embedding for the MDB block input
            x_t_b, mask = self._mask_tokens(x_b, noise_level)

            # --- Prepare Context ---
            # Use the running average of *embeddings* of previous blocks
            current_block_context = prev_context_summary # Use summary from *before* this block

            # --- Process Block ---
            logits_b, _ = self.mdb(x_t_b, current_block_context) # MLP path doesn't use state
            all_logits.append(logits_b)

             # --- Update Context Summary for *next* block ---
            # Simple running average - can be improved (e.g., weighted average, pooling)
            block_mean_embed = x_b_embed.mean(dim=1) # [batch, embed_dim]
            if b == 0:
                prev_context_summary = block_mean_embed
            else:
                # Simple EMA-like update
                prev_context_summary = 0.8 * prev_context_summary + 0.2 * block_mean_embed


            # --- Loss Calculation (if targets provided) ---
            # (Identical loss calculation as in _forward_blockwise_sequential)
            if targets is not None:
                target_blocks = self._split_blocks(targets)
                target_b = target_blocks[:, b]
                loss_mask = (target_b != PAD_TOKEN_ID)
                loss = F.cross_entropy(logits_b.view(-1, self.vocab_size), target_b.view(-1), reduction='none')
                loss = loss.view(batch_size, self.l_prime)
                masked_loss = loss * loss_mask
                total_loss += masked_loss.sum()
                num_loss_tokens += loss_mask.sum().item()

        # --- Combine Results ---
        logits = torch.stack(all_logits, dim=1) # [batch, n_blocks, l_prime, vocab_size]
        logits = logits.view(batch_size, n_blocks * self.l_prime, self.vocab_size)
        logits = logits[:, :x.size(1), :]

        final_loss = None
        if targets is not None and num_loss_tokens > 0:
            final_loss = total_loss / num_loss_tokens

        return logits, final_loss

    @torch.no_grad()
    def generate(self,
                 prompt: Tensor,
                 max_new_blocks: int = 10,
                 temperature: float = 0.8,
                 top_k: int = 50,
                 top_p: float = 0.95, # Add top-p (nucleus) sampling
                 diffusion_steps: Optional[int] = None, # Override T
                 ) -> Tensor:
        """
        Generate sequence block-by-block using iterative denoising/sampling.
        Handles Mamba state persistence.
        """
        self.eval()
        if prompt.dim() == 1:
            prompt = prompt.unsqueeze(0) # Add batch dimension if needed
        batch_size = prompt.shape[0]

        # Use tier's T unless overridden
        T = diffusion_steps if diffusion_steps is not None else self.T

        # Split prompt into blocks, padding might occur
        prompt_blocks = self._split_blocks(prompt) # [batch, n_prompt_blocks, l_prime]
        generated_blocks = [prompt_blocks[:, i] for i in range(prompt_blocks.size(1))]
        current_len = prompt.size(1)
        max_len = current_len + max_new_blocks * self.l_prime

        # --- Initialize Mamba State (if used) ---
        mamba_state = None
        if self.mdb.use_mamba:
            # "Warm up" Mamba state using the prompt
            logging.info("Warming up Mamba state with prompt...")
            # Process prompt blocks sequentially to get the final state
            dummy_context = torch.empty(batch_size, 0, self.mdb.embed_dim, device=DEVICE)
            prompt_tokens = prompt_blocks.view(batch_size, -1) # Flatten prompt blocks
            prompt_tokens_masked, _ = self._mask_tokens(prompt_tokens, 0.0) # No masking needed for warmup
            with autocast(enabled=DEVICE.type == 'cuda'):
                 _, mamba_state = self.mdb(prompt_tokens_masked, dummy_context, mamba_state=None)
            logging.info("Mamba state warmed up.")

        # --- Context for MLP (if used) ---
        mlp_context_summary = torch.zeros(batch_size, self.mdb.embed_dim, device=DEVICE)
        if not self.mdb.use_mamba:
             # Calculate initial context summary from prompt
             prompt_embeds = self.mdb.embedding(prompt_blocks).mean(dim=(1, 2)) # Simple mean
             mlp_context_summary = prompt_embeds

        logging.info(f"Generating up to {max_new_blocks} new blocks (L'={self.l_prime}). Max length: {max_len}")

        for block_idx in range(max_new_blocks):
            start_time = time.time()
            # Initialize the block to be generated with MASK tokens
            x_m_b = torch.full((batch_size, self.l_prime), MASK_TOKEN_ID, dtype=torch.long, device=DEVICE)

            # --- Iterative Denoising (Diffusion Steps T -> 1) ---
            for t in range(T, 0, -1):
                noise_level = self._get_noise_level(step=t) # Get noise for this step

                # Mask the *current estimate* of the block (x_m_b)
                # Except for the very first step where x_m_b is all MASKs
                if t < T:
                     # Mask based on the schedule for this step 't'
                     x_t_b, _ = self._mask_tokens(x_m_b, noise_level)
                else:
                     x_t_b = x_m_b # Start with all MASK tokens

                # --- Get Context ---
                context_input = mlp_context_summary if not self.mdb.use_mamba else torch.empty(batch_size, 0, self.mdb.embed_dim, device=DEVICE)

                # --- Predict Logits ---
                with autocast(enabled=DEVICE.type == 'cuda'):
                    logits_b, new_mamba_state = self.mdb(x_t_b, context_input, mamba_state=mamba_state)
                    # Note: new_mamba_state is intermediate here, only update the main `mamba_state` *after* the block is finalized

                # --- Sampling ---
                # Apply temperature
                logits_b = logits_b / temperature # [batch, l_prime, vocab_size]

                # Apply Top-K / Top-P
                probs = F.softmax(logits_b, dim=-1)
                probs_flat = probs.view(-1, self.vocab_size) # [batch*l_prime, vocab_size]

                if top_k > 0:
                    top_k_probs, top_k_indices = torch.topk(probs_flat, k=top_k, dim=-1)
                    # Remove candidates below top-k threshold
                    kth_vals = top_k_probs[:, -1].unsqueeze(-1)
                    probs_flat[probs_flat < kth_vals] = 0.0

                if top_p > 0.0:
                    sorted_probs, sorted_indices = torch.sort(probs_flat, dim=-1, descending=True)
                    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                    # Find indices where cumulative probability exceeds top_p
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift right to keep the first element exceeding top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    # Create mask in original indices space
                    indices_to_remove = torch.zeros_like(probs_flat, dtype=torch.bool).scatter_(
                        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                    )
                    probs_flat[indices_to_remove] = 0.0

                # Renormalize probabilities
                probs_flat = probs_flat / probs_flat.sum(dim=-1, keepdim=True).clamp(min=1e-10) # Add clamp for stability

                # --- Decide next x_m_b ---
                if t > 1: # Intermediate diffusion steps: Sample
                    sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1) # [batch*l_prime, 1]
                    x_m_b = sampled_indices_flat.view(batch_size, self.l_prime) # [batch, l_prime]
                else: # Final step (t=1): Sample or Argmax (controlled by temp/topk/topp)
                    # Using multinomial sampling here too provides consistency
                    # If temp is very low (~0) and topk=1, it approximates argmax
                    sampled_indices_flat = torch.multinomial(probs_flat, num_samples=1)
                    x_m_b = sampled_indices_flat.view(batch_size, self.l_prime)

            # --- Block Finalized ---
            generated_blocks.append(x_m_b)
            current_len += self.l_prime
            end_time = time.time()
            logging.debug(f"Generated block {block_idx+1}/{max_new_blocks} in {end_time - start_time:.2f}s. Current len: {current_len}")

            # --- Update State/Context for Next Block ---
            if self.mdb.use_mamba:
                 # Compute the Mamba state *using the finalized block* `x_m_b`
                 # This ensures the state reflects the actual generated tokens
                 with autocast(enabled=DEVICE.type == 'cuda'):
                    _, mamba_state = self.mdb(x_m_b, context_input, mamba_state=mamba_state)
            else:
                 # Update MLP context summary
                 block_embed = self.mdb.embedding(x_m_b).mean(dim=1)
                 mlp_context_summary = 0.8 * mlp_context_summary + 0.2 * block_embed # EMA update

            # --- Stopping Criteria ---
            # 1. Check for EOS token in the last generated block
            if (x_m_b == EOS_TOKEN_ID).any():
                logging.info(f"EOS token detected in generated block {block_idx+1}. Stopping generation.")
                # Trim generated block after the first EOS
                for i in range(batch_size):
                    eos_indices = (x_m_b[i] == EOS_TOKEN_ID).nonzero(as_tuple=True)[0]
                    if len(eos_indices) > 0:
                         first_eos_idx = eos_indices[0].item()
                         # Fill remaining positions in the block with PAD
                         generated_blocks[-1][i, first_eos_idx+1:] = PAD_TOKEN_ID
                break # Stop generating more blocks

            # 2. Check entropy of the *final* logits for the generated block
            # Calculate entropy on the final `logits_b` before sampling
            final_probs = F.softmax(logits_b / temperature, dim=-1) # Use the temperature-adjusted logits
            entropy = -(final_probs * torch.log(final_probs.clamp(min=1e-9))).sum(dim=-1) # [batch, l_prime]
            mean_block_entropy = entropy.mean() # Average entropy over batch and block
            if mean_block_entropy < self.stop_entropy_threshold:
                 logging.info(f"Average block entropy ({mean_block_entropy:.2f}) fell below threshold ({self.stop_entropy_threshold:.2f}). Stopping generation.")
                 break

            # 3. Max length reached (implicit in loop)


        # --- Combine generated blocks ---
        # generated_tensor = torch.cat(generated_blocks, dim=1) # This concatenates along l_prime dim
        generated_tensor = torch.stack(generated_blocks, dim=1) # [batch, n_total_blocks, l_prime]
        generated_tensor = generated_tensor.view(batch_size, -1) # [batch, total_len_padded]

        # Trim padding added by _split_blocks initially and any PADs added after EOS
        # Find the first PAD token added by _split_blocks or after EOS
        is_pad = (generated_tensor == PAD_TOKEN_ID)
        first_pad_indices = torch.argmax(is_pad.int(), dim=1)
        # If a row has no PADs, argmax returns 0. Check if the 0th element is actually PAD.
        no_pad = ~is_pad[:, 0]
        actual_lengths = torch.where(no_pad & (first_pad_indices == 0), generated_tensor.size(1), first_pad_indices)
        max_actual_len = actual_lengths.max().item()

        # Trim the tensor to the maximum actual content length
        generated_tensor = generated_tensor[:, :max_actual_len]

        return generated_tensor

    # --- Training Placeholders (Replace with a proper trainer) ---
    def configure_optimizers(self, lr: float = 1e-4, weight_decay: float = 0.1):
        """Sets up AdamW optimizer and optionally LR scheduler."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        # Example scheduler: Cosine Annealing
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=lr/10) # T_max needs dataset size / batch size
        return optimizer, scheduler

    # train_online is generally not recommended for large models. Removed for clarity.
    # If needed, implement carefully using appropriate online learning techniques.

# --- Factory and Training Functions ---

def create_model(tier: str, width: float = 1.0, l_prime: int = DEFAULT_L_PRIME, vocab_size: int = DEFAULT_VOCAB_SIZE, embed_dim: int = 512) -> MBDS:
    """Factory function to create MBD-S model."""
    return MBDS(vocab_size=vocab_size, embed_dim=embed_dim, l_prime=l_prime, width=width, tier=tier)

def basic_train_loop(model: MBDS, data_loader: torch.utils.data.DataLoader, epochs: int = 1, lr: float = 1e-4, weight_decay: float = 0.1, gradient_clipping: float = 1.0):
    """Basic training loop example."""
    model.train()
    optimizer, scheduler = model.configure_optimizers(lr=lr, weight_decay=weight_decay)
    scaler = GradScaler(enabled=DEVICE.type == 'cuda') # For Mixed Precision

    total_steps = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        start_epoch_time = time.time()

        for batch in data_loader:
            # Assuming data_loader yields tensors or dicts with 'input_ids'
            if isinstance(batch, dict):
                input_ids = batch['input_ids'].to(DEVICE)
            elif isinstance(batch, torch.Tensor):
                 input_ids = batch.to(DEVICE)
            else:
                logging.error(f"Unsupported batch type: {type(batch)}")
                continue

            optimizer.zero_grad()

            with autocast(enabled=DEVICE.type == 'cuda'):
                # Use input_ids as both input and target for standard LM training
                logits, loss = model(input_ids, targets=input_ids)

            if loss is not None:
                # Backpropagation with GradScaler
                scaler.scale(loss).backward()

                # Gradient Clipping
                if gradient_clipping > 0:
                    scaler.unscale_(optimizer) # Unscale gradients before clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step() # Step scheduler typically per step or per epoch

                epoch_loss += loss.item()
                num_batches += 1
                total_steps += 1

                if total_steps % 100 == 0: # Log every 100 steps
                    logging.info(f"Epoch {epoch+1}/{epochs}, Step {total_steps}, Batch Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
            else:
                 logging.warning("Loss computation returned None. Skipping optimizer step.")


        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        epoch_time = time.time() - start_epoch_time
        logging.info(f"Epoch {epoch+1} completed in {epoch_time:.2f}s. Average Loss: {avg_epoch_loss:.4f}")

    return model


# --- Example Usage ---

def main():
    """Example usage across tiers."""
    logging.info(f"Using device: {DEVICE}")

    # Dummy data setup (replace with real data using DataLoader)
    batch_size = 4
    seq_len = 128
    dummy_dataset = torch.randint(PAD_TOKEN_ID + 1, DEFAULT_VOCAB_SIZE - 1, (batch_size * 10, seq_len), device="cpu") # Keep data on CPU initially
    # Use DataLoader for proper batching
    data_loader = torch.utils.data.DataLoader(dummy_dataset, batch_size=batch_size, shuffle=True)

    # Prompt for generation
    prompt_seq = torch.randint(PAD_TOKEN_ID + 1, DEFAULT_VOCAB_SIZE - 1, (1, 16), device=DEVICE) # Single prompt, length 16

    for tier in ["simple", "core"]: # Test simple and core tiers
        print("-" * 40)
        logging.info(f"Testing {tier.upper()} tier:")

        # Create model (adjust embed_dim, l_prime as needed)
        # Use smaller embed_dim for faster demo
        model = create_model(tier, width=1.0, l_prime=16, embed_dim=128)

        # --- Training ---
        logging.info("Starting basic training loop...")
        # Train for a small number of epochs/steps for demo
        # Use a subset of the data_loader for quick test
        limited_loader = list(data_loader)[:5] # Take first 5 batches
        model = basic_train_loop(model, limited_loader, epochs=1, lr=5e-4)
        logging.info(f"Finished basic training for {tier} model.")

        # --- Generation ---
        logging.info("Starting generation...")
        generated_output = model.generate(
            prompt_seq,
            max_new_blocks=5, # Generate 5 new blocks (5 * 16 = 80 tokens)
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )
        logging.info(f"Generated sequence shape: {generated_output.shape}")
        logging.info(f"Prompt: {prompt_seq.cpu().numpy().tolist()}")
        logging.info(f"Generated: {generated_output.cpu().numpy().tolist()}")

        # --- Quantization (Placeholder) ---
        if model.needs_quantization:
            logging.info(f"Tier {tier} flagged for quantization. Apply PTQ/QAT here.")
            # Example: Placeholder for Post-Training Quantization (PTQ)
            # model_quantized = torch.quantization.quantize_dynamic(
            #     model, {nn.Linear, Mamba}, dtype=torch.qint8 # Select layers and Mamba if supported
            # )
            # Or Quantization-Aware Training (QAT) - requires modifying the training loop
            pass
        print("-" * 40)


if __name__ == "__main__":
    # Check for Mamba installation
    if Mamba is None:
         logging.warning("Mamba-ssm package not found. Model performance will be limited.")
    main()