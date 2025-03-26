#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✨ EDGE REVOLUTION: HydraScale v3.4 vs Transformer ✨

Objective: Achieve Maximum Awesomeness. Leverage Text Diffusion (HydraScale)
           with State-Space Models (SSMs) to crush Transformers,
           parameter-for-parameter, bringing AI power back to the Edge.

Methodology:
- HydraScale Model: Uses Selective Scan (S6) SSM core.
    - Training: Discrete Diffusion (Mask-Predict Objective). The model learns
                to predict original tokens given a partially masked sequence
                and a noise level (timestep).
    - Inference: Efficient recurrent generation via the `step` method.
- Transformer Baseline: Standard Causal Language Model for comparison.
- Dataset: WikiText-2 (tokenized with GPT2 tokenizer).
- Optimization: Multi-objective Optuna (NSGA-II) balancing Accuracy vs. Parameters.
- Monitoring: Live Rich dashboard for real-time trial tracking & text samples.
- Enhancements: Mixed precision, gradient clipping, dynamic vocab size,
              robust error handling, improved efficiency, and ✨ style ✨.
"""

# --- Core Imports ---
import gc
import math
import time
import warnings
import traceback # For detailed error logging
from threading import Lock
from typing import Optional, Tuple, List, Dict, Any, Union

# --- ML/Data Handling Imports ---
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.cuda.amp import GradScaler, autocast # Updated AMP API

# --- Ecosystem Imports ---
import optuna
from optuna.visualization import plot_pareto_front, plot_param_importances

# --- UI/Utilities Imports ---
from rich.console import Console
from rich.live import Live
from rich.table import Table
# Use notebook tqdm for better integration if in notebook, otherwise standard tqdm
try:
    # Check if running in an IPython environment (like Jupyter)
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        from tqdm.notebook import tqdm
        print("Using tqdm.notebook for progress bars.")
    else:
        from tqdm import tqdm
        print("Using standard tqdm for progress bars.")
except NameError:
    from tqdm import tqdm # Fallback for standard Python environment
    print("Using standard tqdm for progress bars.")
except ImportError:
    print("WARNING: tqdm not found. Progress bars will be basic.")
    # Dummy tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        print("Info: tqdm not installed, progress bar disabled.")
        return iterable

# --- Dependency Check & Installation Hint ---
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("\n" + "="*60)
    print("ERROR: Missing essential libraries!")
    print("Please install them using pip:")
    print("  pip install datasets transformers torch optuna rich tqdm plotly kaleido")
    print("="*60 + "\n")
    exit(1)

# --- Constants & Configuration ---
# Basic Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42 # For reproducibility in sampling/initialization (where applicable)

# Dataset & Tokenizer
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1" # Standard benchmark
TOKENIZER_NAME = "gpt2"             # Common & effective tokenizer

# Model/Training Hyperparameters (Tuneable Ranges defined in Optuna Objective)
SEQ_LEN = 16                        # Sequence length during training/eval
BATCH_SIZE = 64                     # Adjust based on GPU VRAM
GRAD_CLIP_NORM = 1.0                # Max norm for gradient clipping (prevents explosions)

# Optuna Study Configuration
OPTUNA_N_TRIALS = 16                # Increase trials for better search space coverage
NUM_DATA_LOAD_BATCHES = 600         # Batches to preload (reduce disk I/O during trials)
NUM_TRAIN_STEPS_PER_TRIAL = 100    # Steps per trial (balance speed & convergence signal)
NUM_EVAL_BATCHES_PER_TRIAL = 30     # Batches for validation check during training

# Text Generation Configuration
GENERATION_MAX_NEW = 50             # Max new tokens for sample generation
GENERATION_PROMPT = "Edge computing will revolutionize AI by" # Focused prompt

# --- Seed for Reproducibility (where possible) ---
# Note: Full reproducibility on GPU is hard due to CUDA non-determinism
# torch.manual_seed(SEED)
# np.random.seed(SEED)
# if torch.cuda.is_available():
#     torch.cuda.manual_seed_all(SEED)
#     # These can impact performance, use if strict reproducibility is critical
#     # torch.backends.cudnn.deterministic = True
#     # torch.backends.cudnn.benchmark = False

# --- Initialize Tokenizer ---
print(f"\n🚀 Loading tokenizer: {TOKENIZER_NAME}...")
try:
    # trust_remote_code=True might be needed for newer/custom tokenizers
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME, trust_remote_code=True)
except Exception as e:
    print(f"❌ Error loading tokenizer {TOKENIZER_NAME}: {e}")
    print("Check tokenizer name and internet connection.")
    exit(1)

# Handle Padding Token (Crucial for batching)
if tokenizer.pad_token_id is None:
    if tokenizer.eos_token_id is not None:
        print("⚠️ Tokenizer lacks pad_token. Using eos_token as pad_token.")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    else:
        # Add a new pad token if neither exists (rare case)
        print("⚠️ Tokenizer lacks both pad_token and eos_token. Adding a new '[PAD]' token.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        # Note: Model embedding layers will need resizing if vocab size changes here.
        # This is handled dynamically in model initializations now.

VOCAB_SIZE = len(tokenizer) # Get vocab size *after* potential token additions
print(f"✅ Tokenizer loaded. Vocab size: {VOCAB_SIZE}, Pad token ID: {tokenizer.pad_token_id}")

# --- Data Handling ---
# Cache loaded data in memory to speed up trials (assumes dataset fits reasonably)
_data_cache: Dict[str, List[Tensor]] = {}

def prepare_data(dataset_name: str, dataset_config: str, tokenizer: AutoTokenizer, seq_len: int, num_batches: int, batch_size: int, split="train", force_reload=False) -> List[Tensor]:
    """Loads, tokenizes, chunks, and prepares data batches. Caches results."""
    cache_key = f"{dataset_name}-{dataset_config}-{split}-{seq_len}-{num_batches}-{batch_size}"
    if not force_reload and cache_key in _data_cache:
        print(f"   ↳ Using cached data for '{split}' split.")
        return _data_cache[cache_key]

    print(f"   ↳ Loading dataset {dataset_name} ({dataset_config}) - split: {split}...")
    try:
        # trust_remote_code=True is often necessary nowadays
        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True)
    except Exception as e:
        print(f"   ⚠️ Error loading dataset with config '{dataset_config}': {e}")
        print("      Attempting to load without config...")
        try:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True)
        except Exception as e2:
             print(f"   ❌ FATAL: Failed to load dataset '{dataset_name}'. Error: {e2}")
             return [] # Return empty list on failure

    print(f"   ↳ Tokenizing text for '{split}' split (using multiple processes)...")
    def tokenize_function(examples):
        texts = [text if isinstance(text, str) else "" for text in examples.get("text", [])]
        # Tokenize without adding special tokens or truncating here
        return tokenizer(texts, add_special_tokens=False, truncation=False)

    try:
        # Adjust num_proc based on available CPU cores for faster mapping
        num_proc = min(max(1, torch.multiprocessing.cpu_count() // 2), 8)
    except NotImplementedError:
        num_proc = 4 # Sensible default

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=dataset.column_names, # Keep only token IDs
        desc=f"Tokenizing '{split}'"
    )

    print(f"   ↳ Concatenating and chunking into sequences of length {seq_len}...")
    # --- CORRECTED group_texts function ---
    def group_texts(examples):
        # Concatenate all texts within the batch 'examples'
        # Assumes keys like 'input_ids', 'attention_mask' are present
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}

        # Calculate total length based on input_ids
        total_length = len(concatenated_examples.get('input_ids', []))
        if total_length == 0:
            # If input_ids is empty, return empty lists for all keys
            return {k: [] for k in examples.keys()}

        # Drop the small remainder to ensure sequences are full length
        total_length = (total_length // seq_len) * seq_len

        # Chunk *all* columns based on the calculated total_length
        result = {}
        for k, v in concatenated_examples.items():
            if len(v) > 0: # Process only non-empty columns
                 # Ensure we only take up to the calculated total_length before chunking
                 result[k] = [v[i : i + seq_len] for i in range(0, total_length, seq_len)]
            else:
                 result[k] = [] # Keep empty columns empty

        # Optional: Add labels for standard LM loss (though not used by Hydra)
        # if 'input_ids' in result:
        #     result["labels"] = result["input_ids"].copy()

        return result
    # --- End of CORRECTED group_texts function ---

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Chunking '{split}'"
        # No remove_columns here - let the function return the processed columns
    )


    print(f"   ↳ Preparing {num_batches} batches of size {batch_size} for '{split}'...")
    data_batches: List[Tensor] = []
    total_sequences = len(lm_datasets)
    num_sequences_to_process = min(num_batches * batch_size, total_sequences)

    # Use tqdm for clear progress on batch creation
    indices = range(0, num_sequences_to_process, batch_size)
    for i in tqdm(indices, desc=f"Creating '{split}' batches", leave=False, unit="batch"):
        end_idx = min(i + batch_size, total_sequences)
        batch_ids = lm_datasets[i:end_idx]['input_ids']

        if not batch_ids: continue # Skip empty slices

        current_batch_size = len(batch_ids)
        # Pad the last batch if it's smaller than the target batch_size
        if current_batch_size < batch_size:
            padding_needed = batch_size - current_batch_size
            # Ensure pad tensor dimensions match seq_len
            pad_tensor = torch.full((padding_needed, seq_len), tokenizer.pad_token_id, dtype=torch.long)
            batch_tensor = torch.tensor(batch_ids, dtype=torch.long)
            batch_tensor = torch.cat([batch_tensor, pad_tensor], dim=0)
        else:
            batch_tensor = torch.tensor(batch_ids, dtype=torch.long)

        # Final sanity check for sequence length
        if batch_tensor.shape[1] == seq_len:
            # Store batches on CPU initially to conserve GPU memory
            data_batches.append(batch_tensor.cpu())
        else:
            # This should ideally not happen with the chunking logic
            print(f"   ⚠️ Warning: Skipping batch with incorrect sequence length {batch_tensor.shape[1]} (expected {seq_len}). Check data processing pipeline.")

        if len(data_batches) >= num_batches:
            break # Stop once requested number of batches is reached

    if not data_batches:
         print(f"   ⚠️ WARNING: No data batches were created for '{split}' split. Check dataset content and parameters (seq_len, batch_size).")
    else:
         print(f"   ✅ Loaded {len(data_batches)} batches ({len(data_batches) * batch_size} sequences target) for '{split}' split.")

    _data_cache[cache_key] = data_batches
    return data_batches

# --- Selective Scan (S6 / Mamba Block) Implementation ---
# Core SSM logic inspired by Mamba (Gu & Dao, 2023)
# Provides efficient recurrence for inference and allows parallel training scans (though implemented sequentially here for clarity)
class SelectiveScan(nn.Module):
    """ Simplified Selective Scan (S6) Block inspired by Mamba. """
    def __init__(self, embed_dim: int, state_dim: int = 16, d_conv: int = 4, dt_rank: Union[str, int] = 'auto', bias=False):
        super().__init__()
        self.embed_dim = embed_dim  # D (Input/Output Dimension)
        self.state_dim = state_dim  # N (SSM State Dimension)
        self.d_conv = d_conv        # Kernel size for causal convolution
        # Rank for dt projection; 'auto' scales dt_rank with embed_dim (common heuristic)
        self.dt_rank = math.ceil(embed_dim / 16) if dt_rank == 'auto' else dt_rank
        self.bias = bias

        # Input projections: x->x', x->z, x->(Δt, B, C)
        # Different linear layers for projecting input 'x'
        self.in_proj_x = nn.Linear(embed_dim, embed_dim, bias=bias) # Residual path projection
        self.in_proj_z = nn.Linear(embed_dim, embed_dim, bias=bias) # Gating projection
        # Projects input 'x' to parameters for the SSM dynamics
        self.in_proj_params = nn.Linear(embed_dim, self.dt_rank + 2 * self.state_dim, bias=False)

        # Causal Convolution (Depthwise): Captures local context before SSM scan
        self.conv1d = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim, bias=bias,
            kernel_size=d_conv, groups=embed_dim, # Depthwise conv
            padding=d_conv - 1 # Causal padding: ensures output at time t depends only on inputs up to t
        )

        # Projection for Δt (time step delta): Maps from dt_rank to embed_dim
        self.dt_proj = nn.Linear(self.dt_rank, embed_dim, bias=True)

        # SSM State Matrix A (log-parameterized, diagonal): Models state decay/evolution
        # Shape: [D, N] - Independent dynamics per dimension D over state N
        # Initialized to represent decaying states (negative real part)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).unsqueeze(0).repeat(embed_dim, 1))

        # Output Skip Connection Matrix D (learnable per dimension)
        self.D = nn.Parameter(torch.ones(embed_dim)) # Shape: [D]

        # Final Output Projection
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        # --- Custom Initializations ---
        # Critical for stable training, especially for dt_proj
        with torch.no_grad():
            # Initialize dt_proj weights for small initial dt values
            dt_init_std = self.dt_rank**-0.5 * 0.1 # Smaller init variance
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            # Initialize bias for softplus to output small values (e.g., ~0.001 to 0.01)
            # Target value for softplus(bias) = target_dt -> bias = log(expm1(target_dt))
            target_dt = 0.001
            inv_softplus_target = math.log(math.expm1(target_dt))
            self.dt_proj.bias.data.fill_(inv_softplus_target)
            # Initialize A_log towards negative values (stability) - already done by log(1...N)

    def _compute_A_tilde(self, dt: Tensor, A_log: Tensor) -> Tensor:
        """ Computes discretized A: A_tilde = exp(Δt * A). Uses Bilinear approx? No, direct exp. """
        # A is diagonal, A = -exp(A_log) -> Ensures stability (negative real part)
        # A shape: [D, N]
        # dt shape: [B, L, D]
        A = -torch.exp(A_log.float()) # Ensure A is negative
        # Reshape for broadcasting: dt [B, L, D, 1], A [1, 1, D, N]
        # Result A_tilde: [B, L, D, N]
        A_tilde = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(1))
        return A_tilde

    def _compute_B_tilde(self, dt: Tensor, A_log: Tensor, A_tilde: Optional[Tensor] = None) -> Tensor:
        """ Computes discretized B: B_tilde = (A_tilde - 1) / A. (Approximation for B=1) """
        A = -torch.exp(A_log.float()) # Shape [D, N]
        if A_tilde is None:
            A_tilde = self._compute_A_tilde(dt, A_log) # Shape [B, L, D, N]

        # Reshape A for broadcasting: [1, 1, D, N]
        A_unsqueezed = A.unsqueeze(0).unsqueeze(1)
        # Handle A near zero using first-order Taylor expansion: (e^(dt*A) - 1)/A ≈ dt
        is_zero_A = torch.abs(A_unsqueezed) < 1e-8 # Threshold for near-zero
        # Reshape dt for broadcasting: [B, L, D, 1]
        # Formula: (A_tilde - 1) / A
        B_tilde = torch.where(
            is_zero_A,
            dt.unsqueeze(-1), # Approximation B_tilde ≈ dt when A is near zero
            (A_tilde - 1) / (A_unsqueezed + 1e-10) # Add epsilon for numerical stability
        ) # Result B_tilde: [B, L, D, N]
        return B_tilde

    def forward(self, x: Tensor) -> Tensor:
        """ Training Forward Pass (Sequential Scan).
            NOTE: This is a reference implementation. For production speed,
                  a parallel scan algorithm (like in the official Mamba repo) is needed.
        """
        batch, seq_len, embed_dim = x.shape
        assert embed_dim == self.embed_dim, f"Input dim {embed_dim} != Model dim {self.embed_dim}"

        # 1. Input Projections & Residual Path
        x_proj = self.in_proj_x(x) # Residual connection starts here
        z = self.in_proj_z(x)     # Gating mechanism input

        # Project input 'x' to get parameters (dt_rank, B_proj, C_proj)
        params = self.in_proj_params(x) # Shape: [B, L, dt_rank + 2*N]
        # Split the projected parameters
        dt_unproj, B_proj, C_proj = torch.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        ) # Shapes: [B, L, dt_r], [B, L, N], [B, L, N]

        # 2. Compute Δt (learnable time step delta) via projection and softplus
        # Softplus ensures dt > 0
        dt = F.softplus(self.dt_proj(dt_unproj)) # Shape: [B, L, D]

        # 3. Apply 1D Causal Convolution + SiLU Activation
        # Input x shape: [B, L, D] -> Transpose for Conv1d: [B, D, L]
        x_conv = self.conv1d(x.transpose(1, 2)) # Output shape includes padding: [B, D, L + K - 1]
        # Remove padding to keep sequence length L: Output shape: [B, D, L]
        x_conv = x_conv[:, :, :seq_len]
        # Apply SiLU activation (Swish) and transpose back: [B, D, L] -> [B, L, D]
        u = F.silu(x_conv.transpose(1, 2))

        # 4. Compute Discretized SSM Parameters (A_tilde, B_tilde) based on Δt
        A_tilde = self._compute_A_tilde(dt, self.A_log) # Shape: [B, L, D, N]
        B_tilde = self._compute_B_tilde(dt, self.A_log, A_tilde) # Shape: [B, L, D, N]

        # 5. Perform Sequential Scan (State Recurrence)
        # This loop simulates the RNN behavior: h_t = A_tilde_t * h_{t-1} + (B_tilde_t * B_proj_t) * u_t
        #                                     y_t = (C_proj_t * h_t)
        # Initialize hidden state h: Shape [B, D, N]
        h = torch.zeros(batch, self.embed_dim, self.state_dim, device=x.device, dtype=A_tilde.dtype)
        ys: List[Tensor] = [] # Store per-step outputs

        # Iterate through sequence length
        for t in range(seq_len):
            # Get parameters for the current timestep t
            A_t = A_tilde[:, t, :, :]     # [B, D, N]
            B_t = B_tilde[:, t, :, :]     # [B, D, N]
            B_proj_t = B_proj[:, t, :]    # [B, N] (Projected B component, shared across D)
            C_proj_t = C_proj[:, t, :]    # [B, N] (Projected C component, shared across D)
            u_t = u[:, t, :]              # [B, D] (Input after convolution/activation)

            # Calculate the input term for the state update
            # Einsum computes: (B_t * B_proj_t) * u_t element-wise per batch, dim, state
            # B_t: [B, D, N], B_proj_t: [B, N], u_t: [B, D] -> input_term: [B, D, N]
            input_term = torch.einsum('bdn, bn, bd -> bdn', B_t, B_proj_t, u_t)

            # Update the hidden state h: h_t = A_t * h_{t-1} + input_term
            h = A_t * h + input_term # Shape: [B, D, N]

            # Compute the output y for this step using the updated state h
            # Einsum computes: Sum_n (C_proj_t_n * h_bdn) for each batch and dimension D
            # C_proj_t: [B, N], h: [B, D, N] -> y_t: [B, D]
            y_t = torch.einsum('bn, bdn -> bd', C_proj_t, h) # Shape: [B, D]
            ys.append(y_t)

        # Stack the outputs along the sequence length dimension
        y = torch.stack(ys, dim=1) # Shape: [B, L, D]

        # 6. Add Skip Connection (u * D)
        # D is learnable, shape [D]. Unsqueeze for broadcasting: [1, 1, D]
        y = y + u * self.D.unsqueeze(0).unsqueeze(1)

        # 7. Apply Gating Mechanism (y * SiLU(z))
        y = y * F.silu(z) # Element-wise multiplication; Shape: [B, L, D]

        # 8. Final Output Projection
        y_out = self.out_proj(y) # Shape: [B, L, D]
        return y_out

    def step(self, x_step: Tensor, h_prev: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Inference Step (Recurrent) - Optimized for single-step generation. """
        # x_step: Input for the current timestep [B, D]
        # h_prev: Previous SSM hidden state [B, D, N]
        # conv_state: Previous convolution state [B, D, K-1] (where K = kernel_size)

        batch, embed_dim = x_step.shape

        # 1. Convolution Step (Update convolution state and get current output)
        # Append current input x_step to the end of the convolution state buffer
        # conv_state shape: [B, D, K-1], x_step shape: [B, D] -> unsqueeze to [B, D, 1]
        conv_input = torch.cat([conv_state, x_step.unsqueeze(2)], dim=2) # Shape [B, D, K]
        # Compute convolution output using the updated buffer
        conv_out = F.conv1d(
            conv_input, weight=self.conv1d.weight, bias=self.conv1d.bias,
            groups=self.embed_dim, # Depthwise conv
            padding=0 # No padding needed as input is already shaped correctly
        ).squeeze(-1) # Output shape [B, D, 1] -> [B, D]
        # Update convolution state for the *next* step (slide window)
        new_conv_state = conv_input[:, :, 1:] # Shape [B, D, K-1]
        # Apply SiLU activation to convolution output -> u_step
        u_step = F.silu(conv_out) # Shape [B, D]

        # 2. Input Projections for Current Step
        z_step = self.in_proj_z(x_step) # Gating input [B, D]
        params = self.in_proj_params(x_step) # Projected params [B, dt_rank + 2*N]
        dt_unproj, B_proj, C_proj = torch.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        ) # Shapes: [B, dt_r], [B, N], [B, N]

        # 3. Compute Δt for Current Step
        dt_step = F.softplus(self.dt_proj(dt_unproj)) # Shape [B, D]

        # 4. Compute Discretized A_tilde, B_tilde for Current Step
        # Need to add a dummy sequence length dimension (L=1) for compatibility
        dt_for_compute = dt_step.unsqueeze(1) # Shape [B, 1, D]
        # A_tilde_step: [B, 1, D, N] -> squeeze to [B, D, N]
        A_tilde_step = self._compute_A_tilde(dt_for_compute, self.A_log).squeeze(1)
        # B_tilde_step: [B, 1, D, N] -> squeeze to [B, D, N]
        B_tilde_step = self._compute_B_tilde(dt_for_compute, self.A_log, A_tilde_step.unsqueeze(1)).squeeze(1)

        # 5. Update SSM State (Recurrence: h_t = A_tilde_t * h_{t-1} + B_tilde_t * B_proj_t * u_t)
        # Shapes: B_tilde [B,D,N], B_proj [B,N], u_step [B,D] -> input_term [B,D,N]
        input_term = torch.einsum('bdn, bn, bd -> bdn', B_tilde_step, B_proj, u_step)
        # Update state using previous state h_prev
        h = A_tilde_step * h_prev + input_term # Shape [B, D, N]

        # 6. Compute Output for Current Step (y_t = C_proj_t * h_t)
        # Shapes: C_proj [B, N], h [B, D, N] -> y [B, D]
        y = torch.einsum('bn, bdn -> bd', C_proj, h) # Shape [B, D]

        # 7. Add Skip Connection (D term)
        # Shapes: y [B, D], u_step [B, D], D [D] -> unsqueezed to [1, D]
        y = y + u_step * self.D.unsqueeze(0) # Shape [B, D]

        # 8. Apply Gating Mechanism
        y = y * F.silu(z_step) # Shape [B, D]

        # 9. Final Output Projection
        y_step = self.out_proj(y) # Shape [B, D]

        # Return current output, updated SSM state, and updated conv state
        return y_step, h, new_conv_state

# --- HydraScale Components ---
class HydraBlock(nn.Module):
    """ A single block of the HydraScale model, combining Selective Scan (S6)
        with a standard MLP layer, using Pre-Layer Normalization.
    """
    def __init__(self, embed_dim: int, mlp_mult: int = 4, ssm_kwargs: Dict[str, Any] = {}):
        super().__init__()
        # Pre-Normalization variant: Norm -> Module -> Residual Add
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ssm = SelectiveScan(embed_dim, **ssm_kwargs)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = embed_dim * mlp_mult
        # Standard MLP block with GELU activation
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, embed_dim)
            # Consider adding Dropout here if needed: nn.Dropout(dropout_rate)
        )

    def forward(self, x: Tensor) -> Tensor:
        """ Forward pass through the HydraBlock. """
        # SSM Path (Pre-LN structure)
        residual = x
        x_norm = self.norm1(x)
        ssm_out = self.ssm(x_norm)
        x = residual + ssm_out # First residual connection

        # MLP Path (Pre-LN structure)
        residual = x
        x_norm = self.norm2(x)
        mlp_out = self.mlp(x_norm)
        x = residual + mlp_out # Second residual connection
        return x

    def step(self, x_step: Tensor, ssm_state: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """ Inference step for the block, using the S6 `step` method. """
        # SSM Path (Pre-LN structure for single step)
        residual = x_step
        x_norm1 = self.norm1(x_step)
        ssm_out, ssm_state_new, conv_state_new = self.ssm.step(x_norm1, ssm_state, conv_state)
        x = residual + ssm_out # Add residual

        # MLP Path (Pre-LN structure for single step)
        residual = x
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        y_step = residual + mlp_out # Add residual

        return y_step, ssm_state_new, conv_state_new

# Sinusoidal Time Embedding Helper (with caching)
_sinusoidal_embedding_cache: Dict[Tuple[int, torch.device], Tensor] = {}

def sinusoidal_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    """ Creates sinusoidal time embeddings for diffusion models. Caches the frequency matrix. """
    # Input timesteps shape: [B] or [B, 1]
    if timesteps.ndim > 1: timesteps = timesteps.squeeze(-1) # Ensure shape [B]
    device = timesteps.device
    key = (embedding_dim, device) # Cache key based on dim and device

    # Retrieve or compute the frequency embedding matrix
    if key not in _sinusoidal_embedding_cache:
        # print(f"   Computing sinusoidal embedding matrix cache (dim={embedding_dim}, device={device})...")
        half_dim = embedding_dim // 2
        # Standard frequency calculation: 1 / (10000^(2i / D))
        # exponents = torch.arange(half_dim, device=device, dtype=torch.float32) * -(math.log(10000.0) / (half_dim - 1))
        # Use slightly modified frequency range based on common practice
        log_timescale_increment = math.log(10000.0) / half_dim
        inv_timescales = torch.exp(-log_timescale_increment * torch.arange(half_dim, device=device, dtype=torch.float32))

        _sinusoidal_embedding_cache[key] = inv_timescales.unsqueeze(0) # Shape [1, half_dim]

    # Get cached frequencies matrix
    cached_freqs = _sinusoidal_embedding_cache[key] # Shape [1, H/2]
    # Calculate arguments for sin/cos: timesteps * frequencies
    # timesteps: [B] -> [B, 1]
    args = timesteps.float().unsqueeze(1) * cached_freqs # Shape [B, H/2]
    # Compute embeddings: concatenate sin(args) and cos(args)
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # Shape [B, H]
    # Handle odd embedding dimensions by padding the last dimension with zero
    if embedding_dim % 2 == 1:
        embedding = F.pad(embedding, (0, 1)) # Pad last dimension (right side)
    return embedding


# --- HydraScale Language Model Definition ---
class HydraScaleLM(nn.Module):
    """
    HydraScale Language Model: Text Diffusion using Selective Scan blocks.

    Architecture:
    1. Token Embedding (+ Time Embedding)
    2. Sequence of HydraBlocks (SSM + MLP with Pre-LN)
    3. Final Layer Normalization
    4. LM Head (Projection to Vocabulary)

    Training: Mask-Predict Diffusion Objective
    Inference: Recurrent Step-by-Step Generation
    """
    def __init__(self,
                 vocab_size: int,              # Original vocabulary size from tokenizer
                 embed_dim: int = 512,         # Model's main embedding dimension (D)
                 depth: int = 6,               # Number of HydraBlocks (Layers)
                 mlp_mult: int = 4,            # Multiplier for MLP hidden dimension
                 num_diffusion_timesteps: int = 100, # Number of noise levels (T)
                 noise_schedule: str = 'cosine',   # Noise schedule type ('cosine' or 'linear')
                 ssm_state_dim: int = 16,      # SSM state dimension (N)
                 ssm_d_conv: int = 4,          # SSM convolution kernel size (K)
                 ssm_dt_rank: Union[str, int] = 'auto'): # SSM dt projection rank
        super().__init__()
        # Store config
        self.effective_vocab_size = vocab_size # Size of the original vocabulary
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_timesteps = num_diffusion_timesteps
        # Define a unique ID for the [MASK] token (outside the regular vocab)
        self.mask_token_id = self.effective_vocab_size
        self.ssm_d_conv = ssm_d_conv # Needed for state initialization during generation

        # --- Model Layers ---
        # Embedding layer: Includes original vocab + 1 for the [MASK] token
        self.token_embedding = nn.Embedding(self.effective_vocab_size + 1, embed_dim)

        # Time Embedding MLP: Processes sinusoidal time embeddings
        self.time_embedding_dim = embed_dim # Often set equal to main embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4),
            nn.SiLU(), # Commonly used activation for time embeddings
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim),
        )

        # Stack of HydraBlocks (SSM + MLP)
        ssm_kwargs = {'state_dim': ssm_state_dim, 'd_conv': ssm_d_conv, 'dt_rank': ssm_dt_rank}
        self.layers = nn.ModuleList([
            HydraBlock(embed_dim, mlp_mult=mlp_mult, ssm_kwargs=ssm_kwargs)
            for _ in range(depth)
        ])

        # Final normalization and output projection
        self.norm_out = nn.LayerNorm(embed_dim)
        # LM head projects back to the *original* vocabulary size (predicts original tokens)
        self.lm_head = nn.Linear(embed_dim, self.effective_vocab_size)

        # --- Diffusion Schedule Parameters ---
        # Register diffusion parameters as buffers (not trained, but part of state)
        betas = self._calculate_betas(num_diffusion_timesteps, noise_schedule)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        # Precompute square roots for efficiency in loss/sampling calculations
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        # --- Initialization & Info ---
        # Apply custom weight initialization if desired (e.g., Xavier/Kaiming)
        # self.apply(self._init_weights)
        print(f"✨ HydraScaleLM Initialized ✨ (D={embed_dim}, Depth={depth}, SSM_N={ssm_state_dim}, ConvK={ssm_d_conv}, T={num_diffusion_timesteps}). Vocab: {self.effective_vocab_size}.")
        # warnings.warn( # Inform user about sequential training scan
        #     "Note: HydraScaleLM training uses a sequential scan (for clarity), which is slower than optimized parallel scans. Inference uses the efficient recurrent `step` method.",
        #     UserWarning
        # )

    # Optional: Custom weight initialization
    # def _init_weights(self, module):
    #     if isinstance(module, nn.Linear):
    #         nn.init.xavier_uniform_(module.weight)
    #         if module.bias is not None:
    #             nn.init.zeros_(module.bias)
    #     elif isinstance(module, nn.Embedding):
    #         nn.init.normal_(module.weight, mean=0.0, std=0.02)
    #     elif isinstance(module, nn.LayerNorm):
    #         nn.init.ones_(module.weight)
    #         nn.init.zeros_(module.bias)

    def _calculate_betas(self, timesteps: int, schedule: str ='cosine', s: float = 0.008, beta_start: float = 0.0001, beta_end: float = 0.02) -> Tensor:
        """ Calculates beta values for the noise schedule (variance schedule). """
        if schedule == 'cosine':
            # Cosine schedule (from Nichol & Dhariwal, 2021) - often preferred
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
            # Calculate alphas_cumprod based on cosine function
            alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0] # Normalize to start at 1
            # Derive betas from alphas_cumprod: beta_t = 1 - alpha_t / alpha_{t-1}
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            # Clip betas to prevent numerical issues (e.g., beta near 0 or 1)
            return torch.clip(betas, min=1e-6, max=0.999).float() # Ensure float32
        elif schedule == 'linear':
            # Linear schedule (simpler, but sometimes less effective)
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        else:
            raise ValueError(f"Unknown noise schedule: {schedule}. Choose 'cosine' or 'linear'.")

    def _get_mask_prob_from_time(self, t: Tensor) -> Tensor:
        """ Heuristic to determine masking probability based on timestep t.
            Higher t (more noise) should correspond to higher mask probability.
            Uses sqrt(alpha_cumprod_t) relation.
        """
        # Gather alpha_cumprod values for the sampled timesteps 't'
        # Ensure 't' is clamped within valid range [0, T-1]
        t_clamped = torch.clamp(t, 0, self.num_timesteps - 1)
        alpha_bar_t = self.alphas_cumprod.gather(dim=0, index=t_clamped) # Shape [B]
        # Mask probability increases as alpha_bar decreases (more noise = lower alpha_bar)
        # Example heuristic: mask_prob = 1.0 - sqrt(alpha_bar_t)
        mask_prob = 1.0 - torch.sqrt(alpha_bar_t)
        # Ensure probability is within [0, 1], clamp potentially needed if schedule is weird
        # return torch.clamp(mask_prob, 0.0, 1.0)
        return mask_prob # Shape [B]

    def _mask_tokens(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        """ Creates the masked input x_t by replacing some tokens in x_0 with [MASK]."""
        batch_size, seq_len = x_0.shape
        # Get masking probability for each sequence in the batch based on its timestep t
        mask_prob = self._get_mask_prob_from_time(t) # Shape [B]
        # Expand probability to match input shape [B, L] for element-wise comparison
        mask_prob_expanded = mask_prob.view(batch_size, 1).expand(-1, seq_len)

        # Generate random noise in [0, 1) for masking decision
        rand_noise = torch.rand_like(x_0, dtype=torch.float32)
        # Determine mask locations: where noise < probability AND token is not padding
        is_padding = (x_0 == tokenizer.pad_token_id)
        # Apply mask only to non-padding tokens based on probability
        mask = (rand_noise < mask_prob_expanded) & (~is_padding)

        # Replace masked locations in x_0 with the mask_token_id to create x_t
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t, mask # Return masked input and the boolean mask

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        """ Denoising Forward Pass: Predicts logits for the original tokens x_0
            given the masked/noisy input x_t and the timestep t.
        """
        batch_size, seq_len = x_t.shape
        # Sanity check input token IDs
        if x_t.max() >= self.token_embedding.num_embeddings:
             max_id = x_t.max().item()
             emb_size = self.token_embedding.num_embeddings
             raise ValueError(
                 f"Input token ID {max_id} is out of bounds for embedding layer size {emb_size}. "
                 f"Vocab size: {self.effective_vocab_size}, Mask ID: {self.mask_token_id}."
                 " Check input data and mask token ID assignment."
             )

        # 1. Embed tokens (x_t contains original and [MASK] tokens)
        token_emb = self.token_embedding(x_t) # Shape: [B, L, D]

        # 2. Embed timesteps 't'
        time_emb_sin = sinusoidal_embedding(t, self.time_embedding_dim) # Shape: [B, D_time]
        time_emb = self.time_mlp(time_emb_sin) # Shape: [B, D]

        # 3. Combine token and time embeddings
        # Add time embedding to token embeddings (broadcasts along sequence length L)
        h = token_emb + time_emb.unsqueeze(1) # Shape: [B, L, D]

        # 4. Pass through the stack of HydraBlocks
        for layer in self.layers:
            h = layer(h) # Shape remains [B, L, D]

        # 5. Final Layer Normalization
        h = self.norm_out(h) # Shape: [B, L, D]

        # 6. Project to Vocabulary Logits (Predicting the *original* tokens)
        logits = self.lm_head(h) # Shape: [B, L, V_orig] (predicts original vocab)
        return logits

    def compute_loss(self, x_0: Tensor, return_acc: bool = True) -> Union[Tensor, Tuple[Tensor, float]]:
        """ Computes the Mask-Predict Diffusion training loss. """
        batch_size, seq_len = x_0.shape
        # Handle empty batch case
        if batch_size == 0 or x_0.nelement() == 0:
            loss = torch.tensor(0.0, device=x_0.device, requires_grad=True)
            # Ensure loss is connected to graph for backprop if needed (multiply by 0 * param)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            return (loss, 1.0) if return_acc else loss # Accuracy is trivially 1 if no elements

        # 1. Sample random timesteps t for each sequence in the batch
        # Ensure timesteps are within the valid range [0, T-1]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device).long()

        # 2. Create masked input x_t and the corresponding boolean mask
        x_t, mask = self._mask_tokens(x_0, t) # `mask` is True where x_0 was replaced by [MASK]

        # 3. Predict logits for the original tokens using the model
        pred_logits = self.forward(x_t, t) # Shape: [B, L, V_orig]

        # 4. Calculate CrossEntropy loss ONLY on the masked positions
        # Select the logits and target tokens corresponding to the masked positions
        masked_logits = pred_logits[mask] # Shape: [NumMaskedTokens, V_orig]
        masked_targets = x_0[mask]       # Shape: [NumMaskedTokens]

        if masked_targets.numel() == 0:
            # If no tokens were masked (e.g., t=0 or batch was all padding)
            # Loss is zero, but ensure it has a gradient node
            loss = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            accuracy = 1.0 # Accuracy is 100% if no targets to predict
        else:
            # Compute CrossEntropy loss between predicted logits and original tokens at masked positions
            loss = F.cross_entropy(masked_logits, masked_targets)
            # 5. Calculate accuracy on masked tokens (optional)
            accuracy = 0.0
            if return_acc:
                with torch.no_grad():
                    correct_preds = (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
                    accuracy = correct_preds / masked_targets.numel()

        # Handle potential NaN/Inf loss during training (can indicate instability)
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn("🤯 NaN or Inf loss detected in HydraScale compute_loss. Replacing with zero loss.", RuntimeWarning)
            loss = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            accuracy = 0.0 # Penalize accuracy significantly if loss is invalid

        return (loss, accuracy) if return_acc else loss

    @torch.no_grad()
    def _predict_x0_from_logits(self, x_t: Tensor, logits: Tensor,
                                sampling_mode: str = 'argmax', temperature: float = 1.0, top_k: Optional[int] = None
                               ) -> Tensor:
        """ Predicts the original tokens x_0 given the model's output logits.
            Applies sampling strategy (argmax, top-k) only to masked positions.
        """
        batch_size, seq_len, vocab_size = logits.shape
        assert vocab_size == self.effective_vocab_size, "Logits should have original vocab size"

        # Apply sampling strategy (argmax, top-k, etc.)
        if sampling_mode == 'argmax':
            pred_ids = torch.argmax(logits, dim=-1) # Greedy: take most likely token [B, L]
        elif sampling_mode in ['multinomial', 'topk']:
            # Prepare logits for sampling (flatten, apply temp, top-k)
            logits_flat = logits.view(-1, vocab_size) # Shape [B*L, V]

            # Apply temperature scaling (if temp > 0 and != 1)
            if temperature > 0 and temperature != 1.0:
                logits_flat = logits_flat / temperature

            # Apply Top-K filtering (if specified)
            if top_k is not None and top_k > 0 and sampling_mode == 'topk':
                k = min(top_k, vocab_size) # Ensure k is valid
                # Find the k-th largest logit value for each position
                kth_vals, _ = torch.topk(logits_flat, k, dim=-1)
                kth_vals_min = kth_vals[..., -1, None] # Shape [B*L, 1]
                # Mask out logits smaller than the k-th largest by setting to -infinity
                logits_flat.masked_fill_(logits_flat < kth_vals_min, -float('Inf'))

            # Convert logits to probabilities via Softmax
            probs = F.softmax(logits_flat, dim=-1)

            # Handle potential numerical issues (NaNs, zero rows after filtering)
            probs = torch.nan_to_num(probs, nan=0.0) # Replace NaNs with 0
            row_sums = probs.sum(dim=-1, keepdim=True)
            # Check for rows where all probabilities are zero or near-zero
            zero_rows_mask = (row_sums <= 1e-9)
            # If a row sums to zero, replace with uniform distribution to allow sampling
            uniform_probs = torch.full_like(probs, 1.0 / vocab_size)
            probs = torch.where(zero_rows_mask, uniform_probs, probs)
            # Re-normalize probabilities to ensure they sum to 1 after potential corrections
            probs /= probs.sum(dim=-1, keepdim=True)

            # Sample token IDs from the probability distribution
            pred_ids_flat = torch.multinomial(probs, num_samples=1).squeeze(-1) # Shape [B*L]
            pred_ids = pred_ids_flat.view(batch_size, seq_len) # Reshape to [B, L]
        else:
            raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        # --- Crucial Step: Only update the masked positions ---
        # Identify which tokens in the input `x_t` were masks
        mask = (x_t == self.mask_token_id)
        # Combine the original unmasked tokens from `x_t` with the predicted tokens `pred_ids` at masked positions
        predicted_x0 = torch.where(mask, pred_ids, x_t)

        return predicted_x0

    @torch.no_grad()
    def generate(self,
                 prompt_ids: Tensor, # Input prompt token IDs, Shape: [B, L_prompt]
                 num_tokens_to_generate: int,
                 num_sampling_steps: Optional[int] = None, # Diffusion steps for generation (<= T)
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 sampling_mode: str = 'topk' # 'argmax', 'multinomial', 'topk'
                 ) -> Tensor:
        """ Generates text using iterative refinement based on the diffusion process. """
        self.eval() # Set model to evaluation mode
        batch_size, prompt_len = prompt_ids.shape
        total_len = prompt_len + num_tokens_to_generate

        # Determine the number of diffusion steps for generation
        sampling_steps = num_sampling_steps if num_sampling_steps is not None else self.num_timesteps
        sampling_steps = min(sampling_steps, self.num_timesteps) # Cannot exceed trained T
        if sampling_steps <= 0: sampling_steps = 1 # Need at least one step

        # Initialize the sequence: Starts with the prompt followed by [MASK] tokens
        x_gen = torch.full((batch_size, total_len), self.mask_token_id,
                           dtype=torch.long, device=prompt_ids.device)
        x_gen[:, :prompt_len] = prompt_ids # Place prompt at the beginning

        # Initialize recurrent states (SSM state 'h' and Conv state) for each HydraBlock layer
        layer_states: List[Tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            # SSM hidden state h: [B, D, N]
            ssm_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.state_dim,
                                    device=prompt_ids.device, dtype=torch.float32)
            # Convolution state buffer: [B, D, K-1]
            conv_state = torch.zeros(batch_size, self.embed_dim, self.ssm_d_conv - 1,
                                     device=prompt_ids.device, dtype=torch.float32)
            layer_states.append((ssm_state, conv_state))

        # Define the diffusion timesteps to iterate through (from T-1 down to 0)
        # Linspace ensures inclusion of 0 if sampling_steps > 1
        time_indices = torch.linspace(self.num_timesteps - 1, 0, sampling_steps,
                                      device=prompt_ids.device).long()

        # --- Iterative Refinement Loop (Reverse Diffusion) ---
        # Optional tqdm progress bar for generation steps
        gen_iterator = time_indices
        # gen_iterator = tqdm(time_indices, desc="Hydra Gen Steps", leave=False, total=sampling_steps, unit="step")

        for t_val in gen_iterator:
            t_current = t_val.expand(batch_size) # Shape [B]

            # --- Predict x0 using the efficient recurrent `step` method ---
            # We must process the *entire* current sequence `x_gen` token-by-token
            # to update the recurrent states (SSM and Conv) correctly.

            # Clone states for this diffusion step to avoid interference if needed
            # (step method should ideally return new states without modifying input)
            current_layer_states = [(s[0].clone(), s[1].clone()) for s in layer_states]

            # Get token embeddings for the current state of x_gen
            token_emb = self.token_embedding(x_gen) # Shape [B, L_total, D]
            # Get time embedding for the current timestep t
            time_emb_sin = sinusoidal_embedding(t_current, self.time_embedding_dim) # [B, D_time]
            time_emb = self.time_mlp(time_emb_sin) # [B, D]

            all_logits_step: List[Tensor] = [] # Store logits for each position in the sequence

            # Iterate through each token position in the sequence (recurrently)
            for token_idx in range(total_len):
                # Input to the first layer for this token: token emb + time emb
                x_step = token_emb[:, token_idx, :] + time_emb # Shape [B, D]

                # Pass through each HydraBlock layer using its recurrent `step` method
                for layer_idx, layer in enumerate(self.layers):
                    ssm_state, conv_state = current_layer_states[layer_idx]
                    # Apply layer step -> get output for this pos, and updated states
                    x_step, ssm_state_new, conv_state_new = layer.step(x_step, ssm_state, conv_state)
                    # Update the states for the *next token position* within this diffusion step
                    current_layer_states[layer_idx] = (ssm_state_new, conv_state_new)

                # Final normalization and LM head projection for this token position
                h_final = self.norm_out(x_step) # Shape [B, D]
                logits_token = self.lm_head(h_final) # Shape [B, V_orig]
                all_logits_step.append(logits_token)

            # Combine logits from all token positions for this diffusion step
            logits = torch.stack(all_logits_step, dim=1) # Shape [B, L_total, V_orig]

            # --- Predict the 'clean' sequence x0 based on the calculated logits ---
            predicted_x0 = self._predict_x0_from_logits(
                x_gen, # Current state x_t (contains prompt, generated, and masks)
                logits, # Model's prediction of original token logits
                sampling_mode=sampling_mode,
                temperature=temperature,
                top_k=top_k
            )

            # --- Update x_gen: Only fill in the [MASK] tokens with predictions ---
            # This is the core refinement step: keep prompt/already generated tokens,
            # replace masks based on the prediction for this diffusion level.
            mask_for_update = (x_gen == self.mask_token_id)
            x_gen = torch.where(mask_for_update, predicted_x0, x_gen)

            # Optional Advanced Sampling (DDIM / Reparameterization):
            # Instead of just predicting x0, one could use DDIM-like steps
            # involving predicted x0, x_t, and t to sample x_{t-1}.
            # This is more complex for discrete token space. Sticking to the
            # predict-x0-and-fill-masks approach for simplicity and robustness.

        # Final check: Ensure the original prompt remains unmodified
        x_gen[:, :prompt_len] = prompt_ids
        self.train() # Return model to training mode after generation
        return x_gen # Return the fully generated sequence


# --- Comparison Baseline: Standard Transformer Language Model ---
class SimpleTransformerLM(nn.Module):
    """ A standard Decoder-only Transformer for Causal Language Modeling baseline. """
    def __init__(self, vocab_size: int, embed_dim: int, nhead: int, num_layers: int,
                 dim_feedforward: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.effective_vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len # Max context window size

        # Token Embedding Layer
        self.token_embedding = nn.Embedding(self.effective_vocab_size, embed_dim)
        # Positional Encoding (Learned Absolute)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.normal_(self.pos_encoder, std=0.02) # Standard initialization

        self.dropout = nn.Dropout(dropout)

        # Transformer Encoder Layer Configuration (using Pre-LN / NormFirst)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=F.gelu, # Common activation choice
            batch_first=True,  # Expect input shape [Batch, Sequence, Dim]
            norm_first=True    # Apply LayerNorm before attention/FFN (often trains better)
        )
        # Final LayerNorm after all encoder layers
        encoder_norm = nn.LayerNorm(embed_dim)
        # Stack of Transformer Encoder Layers
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=encoder_norm
        )
        # Output Projection Layer (LM Head)
        self.lm_head = nn.Linear(embed_dim, self.effective_vocab_size)

        # Optional: Weight Tying (tie input embedding and output projection weights)
        # Often improves performance and saves parameters.
        # self.token_embedding.weight = self.lm_head.weight

        print(f"🔩 SimpleTransformerLM Initialized (D={embed_dim}, Layers={num_layers}, Head={nhead}, FFN={dim_feedforward}, MaxLen={max_seq_len}). Vocab: {self.effective_vocab_size}.")

    def _generate_causal_mask(self, sz: int, device: torch.device) -> Tensor:
        """Generates an upper-triangular mask for causal self-attention."""
        # Mask shape [sz, sz]. Value `float('-inf')` indicates masking.
        mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device), diagonal=1)
        return mask

    def forward(self, src: Tensor) -> Tensor:
        """ Standard Transformer forward pass for causal language modeling. """
        batch_size, seq_len = src.shape
        # Truncate input sequence if it exceeds the model's max sequence length
        if seq_len > self.max_seq_len:
            src = src[:, -self.max_seq_len:] # Take only the last `max_seq_len` tokens
            seq_len = self.max_seq_len

        # Sanity check input token IDs
        if src.max() >= self.token_embedding.num_embeddings:
             max_id = src.max().item()
             emb_size = self.token_embedding.num_embeddings
             raise ValueError(f"Input token ID {max_id} >= embedding size {emb_size}.")

        # 1. Embed tokens and add positional encodings
        # Scale token embeddings (common practice)
        src_emb = self.token_embedding(src) * math.sqrt(self.embed_dim)
        # Add positional encodings (up to current sequence length)
        pos_emb = self.pos_encoder[:, :seq_len, :]
        src_combined = src_emb + pos_emb.to(src_emb.device) # Ensure pos_emb is on correct device
        src_combined = self.dropout(src_combined) # Apply dropout

        # 2. Prepare masks for the Transformer Encoder
        # Causal mask: Prevents attention to future tokens (ensures autoregressive behavior)
        causal_mask = self._generate_causal_mask(seq_len, device=src.device) # Shape [L, L]

        # Padding mask: Prevents attention to padding tokens
        # `src_key_padding_mask` expects shape [B, L], with `True` indicating padding positions.
        padding_mask = (src == tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None # Shape [B, L]

        # 3. Pass through the Transformer Encoder stack
        # `mask` is the causal mask, `src_key_padding_mask` handles padding.
        output = self.transformer_encoder(
            src_combined,
            mask=causal_mask,
            src_key_padding_mask=padding_mask
        ) # Output shape: [B, L, D]

        # 4. Project to vocabulary logits using the LM head
        logits = self.lm_head(output) # Shape: [B, L, V]
        return logits

    def compute_loss(self, x_0: Tensor, return_acc: bool = True) -> Union[Tensor, Tuple[Tensor, float]]:
        """ Computes standard Causal Language Modeling loss (predict next token). """
        batch_size, seq_len = x_0.shape
        # Need at least 2 tokens to predict the next token
        if seq_len < 2:
            loss = torch.tensor(0.0, device=x_0.device, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            return (loss, 1.0) if return_acc else loss

        # Prepare inputs and targets for next-token prediction
        # Input: Sequence from start up to the second-to-last token
        inp = x_0[:, :-1].contiguous() # Shape [B, L-1]
        # Target: Sequence from the second token to the end
        tgt = x_0[:, 1:].contiguous()  # Shape [B, L-1]

        # Get logits by passing the input sequence through the model
        logits = self.forward(inp) # Shape [B, L-1, V]

        # Calculate CrossEntropy loss
        # Reshape logits to [B*(L-1), V] and targets to [B*(L-1)]
        # `ignore_index` prevents loss calculation for padding tokens in the target sequence
        loss = F.cross_entropy(
            logits.view(-1, self.effective_vocab_size),
            tgt.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        # Calculate accuracy (optional)
        accuracy = 0.0
        if return_acc:
            with torch.no_grad():
                # Consider only non-padding target tokens for accuracy
                valid_targets_mask = (tgt.view(-1) != tokenizer.pad_token_id)
                num_valid_targets = valid_targets_mask.sum().item()

                if num_valid_targets > 0:
                    # Get predictions (most likely token ID)
                    pred_ids = logits.view(-1, self.effective_vocab_size).argmax(dim=-1)
                    # Select predictions and targets where the target is not padding
                    valid_preds = pred_ids[valid_targets_mask]
                    valid_targets = tgt.view(-1)[valid_targets_mask]
                    # Calculate number of correct predictions
                    correct_preds = (valid_preds == valid_targets).sum().item()
                    accuracy = correct_preds / num_valid_targets
                else:
                    # If all targets are padding, accuracy is trivially 100% (or undefined, use 1.0)
                    accuracy = 1.0

        # Handle potential NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn("🤯 NaN or Inf loss detected in Transformer compute_loss. Replacing with zero loss.", RuntimeWarning)
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            accuracy = 0.0

        return (loss, accuracy) if return_acc else loss

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, num_tokens_to_generate: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> Tensor:
        """ Generates text autoregressively, token by token. """
        self.eval() # Set model to evaluation mode
        generated_ids = prompt_ids.to(DEVICE) # Start with the prompt, move to device
        batch_size = prompt_ids.shape[0]

        # Autoregressive generation loop
        gen_iterator = range(num_tokens_to_generate)
        # gen_iterator = tqdm(range(num_tokens_to_generate), desc="Transformer Gen Steps", leave=False, total=num_tokens_to_generate, unit="token")

        for _ in gen_iterator:
            # Current context is all previously generated tokens
            context = generated_ids
            # Truncate context if it exceeds max_seq_len (sliding window)
            if context.shape[1] > self.max_seq_len:
                context = context[:, -self.max_seq_len:] # Keep only the most recent tokens

            # Use mixed precision for generation if on CUDA
            with autocast(enabled=(DEVICE.type == 'cuda')):
                # Get logits for the current context
                logits = self.forward(context) # Shape [B, L_context, V]
                # We only need the logits for the very last token prediction
                next_token_logits = logits[:, -1, :] # Shape [B, V]

            # --- Sampling Logic ---
            # Apply temperature scaling (if temp > 0 and != 1)
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply Top-K filtering (if specified)
            if top_k is not None and top_k > 0:
                v = next_token_logits.size(-1) # Vocab size
                k = min(top_k, v)
                # Find the k-th largest logit values
                kth_vals, _ = torch.topk(next_token_logits, k, dim=-1) # Shape [B, k]
                kth_vals_min = kth_vals[:, -1, None] # Shape [B, 1] (value of the k-th token)
                # Mask logits below the k-th value by setting to -infinity
                indices_to_remove = next_token_logits < kth_vals_min
                next_token_logits.masked_fill_(indices_to_remove, -float('Inf'))

            # Convert filtered logits to probabilities via Softmax
            probs = F.softmax(next_token_logits, dim=-1) # Shape [B, V]

            # Sample the next token ID
            # Use argmax if temperature is zero or negative (greedy decoding)
            if temperature <= 0:
                next_token_id = torch.argmax(probs, dim=-1, keepdim=True) # Shape [B, 1]
            else:
                # Handle potential numerical issues (NaNs, zero rows after filtering)
                probs = torch.nan_to_num(probs, nan=0.0)
                row_sums = probs.sum(dim=-1, keepdim=True)
                zero_rows_mask = (row_sums <= 1e-9)
                uniform_probs = torch.full_like(probs, 1.0 / probs.shape[-1])
                probs = torch.where(zero_rows_mask, uniform_probs, probs)
                probs /= probs.sum(dim=-1, keepdim=True) # Re-normalize
                # Sample from the multinomial distribution
                next_token_id = torch.multinomial(probs, num_samples=1) # Shape [B, 1]

            # Append the predicted token ID to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1) # Shape [B, L_current + 1]

        self.train() # Return model to training mode
        return generated_ids


# --- Utility Functions ---
@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    """ Counts the total number of trainable parameters in a PyTorch module. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_sample_text(model: nn.Module, tokenizer: AutoTokenizer, prompt_text: str, device: torch.device, model_type: str, max_new: int = 48, gen_kwargs: Optional[Dict] = None) -> str:
    """Generates a text sample from a given model and decodes it."""
    if gen_kwargs is None: gen_kwargs = {}
    model.eval() # Ensure model is in evaluation mode

    try:
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        if prompt_ids.shape[1] == 0:
             print("⚠️ Warning: Encoded prompt is empty. Using a default start token if possible.")
             # Attempt to use BOS token if available, otherwise handle error
             bos_token_id = getattr(tokenizer, 'bos_token_id', None)
             if bos_token_id is not None:
                  prompt_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
             else:
                  return "[Error: Empty prompt and no BOS token]"


        # Define default generation parameters (can be overridden)
        default_kwargs = {"temperature": 0.8, "top_k": 50}
        current_gen_kwargs = default_kwargs.copy()

        if model_type == "hydra":
            # Hydra specific defaults/overrides
            hydra_defaults = {"num_sampling_steps": min(getattr(model, 'num_timesteps', 100)//2, 50), "sampling_mode": "topk"} # Faster sampling
            current_gen_kwargs.update(hydra_defaults)
            current_gen_kwargs.update(gen_kwargs) # User kwargs override all
            gen_func = model.generate
        elif model_type == "transformer":
            current_gen_kwargs.update(gen_kwargs) # User kwargs override defaults
            gen_func = model.generate
        else:
            return f"[Error: Invalid model_type '{model_type}']"

        # Generate using mixed precision on CUDA for speed
        with autocast(enabled=(device.type == 'cuda')):
            output_ids = gen_func(prompt_ids, num_tokens_to_generate=max_new, **current_gen_kwargs)

        # Decode only the newly generated part (excluding the prompt)
        generated_part_ids = output_ids[0, prompt_ids.shape[1]:]
        # skip_special_tokens=True removes padding, EOS, etc.
        generated_text = tokenizer.decode(generated_part_ids, skip_special_tokens=True)

        # Combine prompt and generated text, clean up whitespace
        full_text = (prompt_text + generated_text).replace("\n", " ").strip()
        return full_text

    except Exception as e:
        print(f"\n❌ Error during text generation for {model_type}:")
        traceback.print_exc() # Print detailed traceback for debugging
        return f"[{model_type.capitalize()} Gen Error: {type(e).__name__}]"
    finally:
        model.train() # Ensure model is returned to training mode


# --- Training & Evaluation Helpers for Optuna ---

def train_eval_trial(
    trial: optuna.Trial, # Pass trial object (needed for pruning check, even if report isn't used)
    model: nn.Module,
    train_loader: List[Tensor],
    eval_loader: List[Tensor], # Used only for pruning check now
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    num_train_steps: int,
    num_eval_batches: int, # Number of batches for validation check
    device: torch.device,
    model_type: str
) -> Tuple[float, float]:
    """
    Trains the model for `num_train_steps`, performs periodic evaluation checks
    (primarily for Optuna's pruning mechanism if supported), and returns the
    final average training loss and accuracy.
    """
    model.train()
    total_loss = total_accuracy = 0.0
    steps_done = 0

    data_iter = iter(train_loader) # Use iterator for efficient batch loading

    train_iterator = tqdm(range(num_train_steps), desc=f"Trial {trial.number} Train {model_type[:5].upper()}", leave=False, unit="step")

    # --- Training Loop ---
    for step in train_iterator:
        try:
            batch_cpu = next(data_iter)
            # Move batch to GPU asynchronously if possible
            batch = batch_cpu.to(device, non_blocking=True)
        except StopIteration:
            # Reset iterator if dataset runs out before steps are completed
            # print(f"\n   🔄 Resetting training data iterator at step {step}...")
            data_iter = iter(train_loader)
            try:
                batch_cpu = next(data_iter)
                batch = batch_cpu.to(device, non_blocking=True)
            except StopIteration:
                print("\n   ⚠️ WARNING: Training data loader exhausted prematurely. Stopping trial.")
                break # Not enough data

        if batch.nelement() == 0: continue # Skip potentially empty batches

        # --- Forward & Backward Pass ---
        optimizer.zero_grad(set_to_none=True) # More memory efficient than setting to zero

        try:
            # Use mixed precision (autocast) for forward pass on CUDA
            with autocast(enabled=(device.type == 'cuda')):
                loss, accuracy = model.compute_loss(batch, return_acc=True)

            # Check for invalid loss values (NaN or Inf)
            if not torch.isfinite(loss):
                print(f"\n   ⚠️ Warning: Invalid loss ({loss.item()}) detected at step {step}. Skipping update.")
                # Skip backward/step, do not accumulate metrics for this step
                # Consider pruning if this happens frequently
                # if step > 50: # Avoid pruning on initial instability
                #      raise optuna.TrialPruned(f"Invalid loss at step {step}")
                continue

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # Gradient Clipping (applied *before* optimizer step, *after* unscaling)
            scaler.unscale_(optimizer) # Unscale gradients inplace
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            # Optimizer Step (updates model parameters)
            scaler.step(optimizer)

            # Update GradScaler's scale factor for next iteration
            scaler.update()

            # --- Accumulate Metrics & Update Progress Bar ---
            current_loss = loss.item()
            total_loss += current_loss
            total_accuracy += accuracy
            steps_done += 1

            # Update tqdm postfix less frequently for performance
            if step % 50 == 0 or step == num_train_steps - 1:
                 train_iterator.set_postfix({
                     "Loss": f"{current_loss:.3f}",
                     "Acc": f"{accuracy:.3f}",
                     "Scale": f"{scaler.get_scale():.1f}" # Show grad scaler factor
                 })

        # --- Error Handling (OOM) ---
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n   ❌ CUDA OOM encountered during training step {step} for Trial {trial.number}.")
                model_details = f"Model: {model_type}, EmbedDim: {getattr(model, 'embed_dim', 'N/A')}, Depth: {getattr(model, 'depth', 'N/A')}"
                print(f"      {model_details}")
                # Clean up memory aggressively
                del batch, loss
                optimizer.zero_grad(set_to_none=True)
                gc.collect()
                torch.cuda.empty_cache()
                # Prune the trial due to OOM
                raise optuna.TrialPruned(f"OOM at step {step}") from e
            else:
                # Re-raise other runtime errors
                print(f"\n   ❌ Runtime Error during training step {step} for Trial {trial.number}:")
                traceback.print_exc()
                raise e

        # --- Intermediate Evaluation & Pruning (Adapted for Multi-Objective) ---
        # `trial.report` is NOT supported for multi-objective.
        # We rely on the pruner (e.g., MedianPruner) comparing *completed* trial values.
        # However, we still need `trial.should_prune()` check. Optuna's pruners
        # often work by looking at the history of completed trials even without `report`.
        eval_interval = max(150, num_train_steps // 5) # Check pruning ~5 times or every 150 steps
        if (step + 1) % eval_interval == 0:
             # Check if Optuna's pruner suggests pruning based on completed trials so far
             if trial.should_prune():
                  # Clean up before pruning
                  del batch
                  if 'loss' in locals(): del loss
                  gc.collect()
                  torch.cuda.empty_cache()
                  raise optuna.TrialPruned(f"Pruned at step {step+1} by pruner decision.")

             # Optional: Run a quick validation check here *without* reporting to trial.report
             # This can give insights but won't directly influence the pruner in this setup.
             # ... (validation loop similar to previous version, but don't call trial.report) ...

    # --- End of Training Loop ---
    # Calculate final average training metrics
    avg_loss = total_loss / steps_done if steps_done > 0 else 0.0
    avg_accuracy = total_accuracy / steps_done if steps_done > 0 else 0.0

    # Ensure metrics are standard Python floats
    return float(avg_loss), float(avg_accuracy)


# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, train_data: List[Tensor], eval_data: List[Tensor], fixed_seq_len: int, device: torch.device) -> Tuple[float, float]:
    """
    Optuna objective function for optimizing HydraScale vs Transformer.
    Optimizes for:
        1. Accuracy (Maximize) - Primary performance metric (using training avg acc).
        2. Parameter Count (Minimize) - Efficiency metric.
    Returns: Tuple[float, float] -> (accuracy, parameter_count)
    """
    print(f"\n--- 🚀 Starting Trial {trial.number} ---")
    # Initialize variables for robust cleanup in `finally` block
    model = None
    optimizer = None
    scaler = None
    param_count = float('inf') # Default to worst case for minimization objective
    final_accuracy = 0.0       # Default to worst case for maximization objective

    try:
        # --- Suggest Model Type ---
        model_type = trial.suggest_categorical("model_type", ["hydra", "transformer"])
        print(f"   🔬 Model Type: {model_type.upper()}")

        # --- Suggest Common Hyperparameters ---
        lr = trial.suggest_float("lr", 1e-5, 8e-4, log=True) # Learning rate range
        # Embed dim: Use powers of 2 or steps? Steps allow finer control. Ensure divisibility by common head counts.
        # Keep embed_dim options reasonable for edge devices / moderate GPUs
        embed_dim_options = [d for d in range(128, 768 + 1, 64) if d % 8 == 0] # Divisible by 8 for heads
        embed_dim = trial.suggest_categorical("embed_dim", embed_dim_options)
        depth = trial.suggest_int("depth", 2, 8) # Number of layers

        print(f"   ⚙️ Common Params: LR={lr:.2e}, EmbedDim={embed_dim}, Depth={depth}")

        model_config = {"vocab_size": VOCAB_SIZE} # Start building config dict

        # --- Model Specific Hyperparameters & Instantiation ---
        if model_type == "hydra":
            mlp_mult = trial.suggest_categorical("mlp_mult", [2, 3, 4]) # MLP expansion factor
            ssm_state_dim = trial.suggest_categorical("ssm_state_dim", [8, 16, 24, 32]) # SSM state size N
            ssm_d_conv = trial.suggest_categorical("ssm_d_conv", [3, 4, 5]) # Conv kernel size K
            num_diffusion_timesteps = trial.suggest_categorical("num_diffusion_timesteps", [50, 100, 150]) # Diffusion T

            print(f"   ⚙️ Hydra Params: MLP_Mult={mlp_mult}, SSM_N={ssm_state_dim}, ConvK={ssm_d_conv}, T={num_diffusion_timesteps}")

            model_config.update({
                "embed_dim": embed_dim,
                "depth": depth,
                "mlp_mult": mlp_mult,
                "num_diffusion_timesteps": num_diffusion_timesteps,
                "ssm_state_dim": ssm_state_dim,
                "ssm_d_conv": ssm_d_conv,
                "ssm_dt_rank": 'auto' # Keep dt_rank 'auto' for simplicity
            })
            # Instantiate HydraScale model
            model = HydraScaleLM(**model_config).to(device)

        elif model_type == "transformer":
            # Suggest nhead from a FIXED list first (Fixes the dynamic categorical error)
            fixed_nhead_options = [h for h in [2, 4, 8, 12, 16] if embed_dim >= h] # Must be <= embed_dim
            if not fixed_nhead_options: # Should not happen if embed_dim >= 128
                 raise ValueError(f"No valid nhead options for embed_dim {embed_dim}")
            nhead = trial.suggest_categorical("nhead", fixed_nhead_options)

            # NOW, check for compatibility and prune if embed_dim is not divisible by nhead
            if embed_dim % nhead != 0:
                raise optuna.TrialPruned(
                    f"Incompatible combo: embed_dim={embed_dim} not divisible by nhead={nhead}. Pruning."
                )

            # Feedforward dimension multiplier
            ffn_mult = trial.suggest_categorical("ffn_mult", [2, 3, 4, 6])
            dim_feedforward = embed_dim * ffn_mult
            # Dropout rate
            dropout = trial.suggest_float("dropout", 0.05, 0.25)
            # Max sequence length for positional embeddings
            max_seq_len = fixed_seq_len * 4 # Allow longer context than training length

            print(f"   ⚙️ Transformer Params: Heads={nhead}, FFN_Mult={ffn_mult} ({dim_feedforward}), Dropout={dropout:.3f}")

            model_config.update({
                 "embed_dim": embed_dim, "nhead": nhead, "num_layers": depth,
                 "dim_feedforward": dim_feedforward, "max_seq_len": max_seq_len,
                 "dropout": dropout
            })
            # Instantiate Transformer model
            model = SimpleTransformerLM(**model_config).to(device)

        # --- Calculate Parameter Count (Objective 2) ---
        param_count = count_parameters(model) # Get raw parameter count
        param_count_M = param_count / 1e6
        trial.set_user_attr("param_count_M", param_count_M) # Store Millions for display
        trial.set_user_attr("config", model_config) # Store full config for review
        print(f"   ✅ Model Instantiated ({model.__class__.__name__}). Parameters: {param_count_M:.3f} M ({param_count:,})")

        # --- Training & Evaluation ---
        # AdamW optimizer with weight decay
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # GradScaler for mixed precision - Use updated API
        scaler = GradScaler(enabled=(device.type == 'cuda'))

        print(f"   ⏳ Starting training ({NUM_TRAIN_STEPS_PER_TRIAL} steps)...")
        start_time = time.time()
        # Run the training loop
        avg_loss, avg_accuracy = train_eval_trial(
            trial, # Pass trial for pruning checks
            model, train_data, eval_data, optimizer, scaler,
            num_train_steps=NUM_TRAIN_STEPS_PER_TRIAL,
            num_eval_batches=NUM_EVAL_BATCHES_PER_TRIAL,
            device=device, model_type=model_type
        )
        end_time = time.time()
        print(f"   ⏱️ Training finished in {end_time - start_time:.2f} seconds.")
        print(f"   📊 Final Avg Train Loss: {avg_loss:.4f}, Final Avg Train Accuracy: {avg_accuracy:.4f}")

        # --- Final Accuracy (Objective 1) ---
        final_accuracy = avg_accuracy # Use average training accuracy as the metric
        trial.set_user_attr("final_accuracy", final_accuracy) # Store for display/analysis
        trial.set_user_attr("final_loss", avg_loss)

        # --- Generate Text Sample (Qualitative Check) ---
        generated_text_sample = "[Generation Skipped (Low Acc)]"
        # Generate sample only if training showed some learning (accuracy > threshold)
        # Avoid generating if accuracy is NaN or very low.
        if final_accuracy > 0.02 and not np.isnan(final_accuracy):
            print("   📝 Generating text sample...")
            # Use consistent sampling parameters for comparison across trials
            gen_kwargs = {"temperature": 0.8, "top_k": 50}
            generated_text_sample = generate_sample_text(
                model, tokenizer, GENERATION_PROMPT, device, model_type,
                max_new=GENERATION_MAX_NEW, gen_kwargs=gen_kwargs
            )
            # Clean and truncate sample for dashboard display
            generated_text_sample = generated_text_sample.replace("\n", " ").strip()
            if len(generated_text_sample) > 150:
                generated_text_sample = generated_text_sample[:147] + "..."
            print(f"      Sample: {generated_text_sample}")
        else:
             print(f"   📝 Skipping text generation due to low/NaN accuracy ({final_accuracy:.4f}).")

        trial.set_user_attr("generated_sample", generated_text_sample)

        # --- Return Objectives: (Accuracy, Parameter Count) ---
        # Ensure accuracy is a valid float for Optuna
        if np.isnan(final_accuracy):
            print("   ⚠️ Warning: Final accuracy is NaN. Reporting 0.0 to Optuna.")
            final_accuracy = 0.0

        print(f"--- ✅ Trial {trial.number} Complete: Acc={final_accuracy:.4f}, Params={param_count_M:.3f}M ---")
        # Return the raw parameter count for the objective (minimization)
        return float(final_accuracy), float(param_count)

    # --- Error Handling & Pruning ---
    except optuna.TrialPruned as e:
        # Handle pruning exceptions gracefully
        print(f"--- ✂️ Trial {trial.number} Pruned: {e} ---")
        # Let Optuna handle the pruning logic, just re-raise
        raise e
    except Exception as e:
        # Catch any other unexpected errors during the trial
        print(f"\n!!!!!!!! 💥 TRIAL {trial.number} FAILED 💥 !!!!!!!!")
        traceback.print_exc() # Print detailed traceback
        print(f"--- ❌ Trial {trial.number} Failed. Reporting worst objective values. ---")
        # Return worst possible values for Optuna objectives
        final_accuracy = 0.0
        param_count = float('inf')
        # Ensure the tuple return matches the expected types
        return float(final_accuracy), float(param_count)

    # --- Resource Cleanup (Ensures GPU memory is freed) ---
    finally:
        print(f"   🧹 Cleaning up resources for Trial {trial.number}...")
        del model, optimizer, scaler # Explicitly delete large objects
        gc.collect()              # Trigger Python garbage collection
        if device.type == 'cuda':
            torch.cuda.empty_cache()  # Clear PyTorch's CUDA cache
        print(f"   -> Cleanup complete for Trial {trial.number}.")


# --- Live Dashboard Callback & Setup ---
dashboard_data: Dict[int, Dict[str, Any]] = {} # Shared dict for dashboard state
dashboard_lock = Lock() # Mutex for thread-safe access to dashboard_data
# console = Console(width=200) # Rich console instance
console = Console() # Use automatic width detection

def make_dashboard_table() -> Table:
    """Creates the Rich Table object for the live dashboard display."""
    table = Table(title=f"🤖 Optuna Live: HydraScale vs Transformer ({DATASET_NAME}@{DEVICE}) 🤖",
                  caption="Goal: Top-Left Corner (High Accuracy, Low Params)",
                  expand=True) # Allow table to expand to console width
    table.add_column("Trial", justify="right", style="cyan", no_wrap=True, width=5)
    table.add_column("Model", style="magenta", width=11) # Fixed width?
    table.add_column("State", style="yellow", width=10)
    table.add_column("Acc (%)", justify="right", style="green", width=8)
    table.add_column("Params (M)", justify="right", style="blue", width=10)
    table.add_column("LR", justify="right", style="dim cyan", width=10)
    table.add_column("Config", style="dim", width=30, overflow="fold") # Key config parts
    table.add_column("Sample Output / Status", style="white", min_width=40, overflow="fold") # Flexible width
    return table

class DashboardCallback:
    """ Optuna callback to update shared data used by the Rich live display. """
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        with dashboard_lock: # Ensure thread-safe update
            accuracy, params_raw = (None, None)
            state = trial.state.name # Get trial state (RUNNING, COMPLETE, FAIL, PRUNED)

            # Extract objective values if the trial is complete
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
                # trial.values should be [accuracy, param_count]
                accuracy = trial.values[0] * 100 if trial.values[0] is not None else None
                params_raw = trial.values[1] if trial.values[1] is not None else None
            # Handle failed/pruned trials - use user attrs or defaults
            elif trial.state in [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED]:
                # Use stored final accuracy if available, else 0
                accuracy = trial.user_attrs.get("final_accuracy", 0.0) * 100
                # Use stored param count (in M) if available, convert back to raw, else Inf
                params_m_stored = trial.user_attrs.get("param_count_M")
                params_raw = params_m_stored * 1e6 if params_m_stored is not None else float('inf')
                # If values exist (e.g., failed after first report), prefer them
                if trial.values and trial.values[1] is not None:
                    params_raw = trial.values[1]

            # Convert raw param count to Millions for display
            params_m = params_raw / 1e6 if params_raw is not None and params_raw != float('inf') else None

            # --- Get Key Hyperparameters for Display ---
            model_type = trial.params.get("model_type", "N/A")
            lr = trial.params.get("lr", float('nan'))
            embed_dim = trial.params.get("embed_dim", -1)
            depth = trial.params.get("depth", -1)

            # Build concise config string
            config_str = f"D={embed_dim}, L={depth}"
            if model_type == "hydra":
                ssm_n = trial.params.get("ssm_state_dim", "?")
                t_steps = trial.params.get("num_diffusion_timesteps", "?")
                convk = trial.params.get("ssm_d_conv", "?")
                config_str += f", N={ssm_n}, K={convk}, T={t_steps}"
            elif model_type == "transformer":
                nhead = trial.params.get("nhead", "?")
                ffn_m = trial.params.get("ffn_mult", "?")
                config_str += f", H={nhead}, Fx{ffn_m}"

            # --- Get Generated Sample / Status ---
            sample_or_status = trial.user_attrs.get("generated_sample", "")
            # Add status messages based on trial state
            if state == "RUNNING" and not sample_or_status:
                 sample_or_status = "[Training...]"
            elif state == "FAIL":
                 sample_or_status = f"[Failed: {trial.user_attrs.get('fail_reason', 'Unknown')}]"
            elif state == "PRUNED":
                 sample_or_status = "[Pruned ✂️]"
            elif state == "COMPLETE" and not sample_or_status:
                 sample_or_status = "[No Sample / Low Acc]"

            # --- Store data for the dashboard ---
            # Use trial number as the dictionary key
            dashboard_data[trial.number] = {
                "model": model_type.upper(),
                "state": state,
                "accuracy": accuracy,
                "params": params_m, # Store in Millions
                "lr": lr,
                "config": config_str,
                "sample": sample_or_status,
            }

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" ✨ Initializing Edge Revolution Optimization ✨ ".center(80, "="))
    print("=" * 80)
    # Print key configuration details
    print(f"▶ Device: {DEVICE.type.upper()}")
    if DEVICE.type == 'cuda': print(f"  ▶ CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"▶ Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    print(f"▶ Tokenizer: {TOKENIZER_NAME} (Vocab Size: {VOCAB_SIZE})")
    print(f"▶ Training Params: SeqLen={SEQ_LEN}, BatchSize={BATCH_SIZE}, GradClip={GRAD_CLIP_NORM}")
    print(f"▶ Optuna: Trials={OPTUNA_N_TRIALS}, Steps/Trial={NUM_TRAIN_STEPS_PER_TRIAL}")
    print("-" * 80)

    # --- Load Data (Once before study starts) ---
    print("⏳ Loading and preparing data...")
    start_data_time = time.time()
    # Load training data
    train_data = prepare_data(DATASET_NAME, DATASET_CONFIG, tokenizer, SEQ_LEN, NUM_DATA_LOAD_BATCHES, BATCH_SIZE, split="train")
    # Load validation data (used for pruning checks if enabled, and final eval potentially)
    eval_data = prepare_data(DATASET_NAME, DATASET_CONFIG, tokenizer, SEQ_LEN, NUM_DATA_LOAD_BATCHES // 2, BATCH_SIZE, split="validation")
    end_data_time = time.time()

    # Exit if data loading failed
    if not train_data or not eval_data:
        print("\n❌ FATAL: Failed to load sufficient data. Check dataset paths, names, and disk space. Exiting.")
        exit(1)
    print(f"✅ Data loading complete in {end_data_time - start_data_time:.2f} seconds.")
    print("-" * 80)

    # --- Setup Optuna Study ---
    print("🛠️ Setting up Optuna study...")
    # Use NSGA-II sampler for effective multi-objective optimization
    sampler = optuna.samplers.NSGAIISampler(
        # population_size=30, # Adjust population size based on computational budget
        seed=SEED # Seed the sampler for reproducibility
    )
    # Use TPESampler as an alternative (might require different pruning strategy)
    # sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=SEED, n_startup_trials=10)

    # Use a pruner - MedianPruner is simple and often effective. It typically prunes based
    # on the *first* objective (Accuracy) by comparing intermediate results (if reported via trial.report)
    # or final results of completed trials. Since trial.report is disabled for multi-obj,
    # it will primarily work based on completed trials.
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5,     # Allow first few trials to complete regardless
        n_warmup_steps=NUM_TRAIN_STEPS_PER_TRIAL // 3, # Don't prune early in training
        interval_steps=150       # Check for pruning periodically (corresponds to eval_interval in train_eval_trial)
    )

    # Create the Optuna study object
    try:
        study = optuna.create_study(
            directions=["maximize", "minimize"], # [Objective 1: Accuracy, Objective 2: Params]
            study_name="Edge_Revolution_v3_4",
            sampler=sampler,
            pruner=pruner,
            # Optional: Use persistent storage (SQLite DB) to resume study
            # storage="sqlite:///edge_revolution_optuna.db",
            # load_if_exists=True # Set to True to resume from DB
        )
    except Exception as e:
        print(f"❌ Error creating/loading Optuna study: {e}")
        print("   If using storage, check DB file path and permissions.")
        exit(1)

    # Instantiate the dashboard callback
    callback = DashboardCallback()

    # --- Run Optimization with Live Dashboard ---
    print(f"🏁 Starting Optuna optimization ({OPTUNA_N_TRIALS} trials)...")
    # Use Rich Live context manager to manage the dashboard display
    live_table = make_dashboard_table()
    try:
        with Live(live_table, console=console, refresh_per_second=1.5, vertical_overflow="visible") as live:
            # Define the objective function lambda to pass static arguments
            objective_func = lambda trial: objective(trial, train_data, eval_data, SEQ_LEN, DEVICE)

            # === Start Optuna Optimization Loop ===
            study.optimize(
                objective_func,
                n_trials=OPTUNA_N_TRIALS,
                callbacks=[callback],      # Register dashboard callback
                gc_after_trial=True,      # Recommended for GPU memory management
                show_progress_bar=False   # Disable Optuna's default tqdm bar
            )
            # === Optimization Finished ===

            # Keep the final dashboard state visible for a moment
            # Update the table one last time with final states
            final_table = make_dashboard_table() # Recreate table
            with dashboard_lock: # Access shared data safely
                 # Sort trials by number for consistent display order
                 sorted_trial_nums = sorted(dashboard_data.keys())
                 for trial_num in sorted_trial_nums:
                      data = dashboard_data[trial_num]
                      # Format data for display
                      state_color = "green" if data['state'] == "COMPLETE" else ("red" if data['state'] in ["FAIL", "PRUNED"] else "yellow")
                      acc_str = f"{data['accuracy']:.2f}" if data['accuracy'] is not None else "N/A"
                      params_str = f"{data['params']:.3f}" if data['params'] is not None else "N/A"
                      lr_str = f"{data['lr']:.1e}" if data.get('lr') and not np.isnan(data['lr']) else "N/A"
                      # Add row to the final table
                      final_table.add_row(
                          str(trial_num), data['model'], f"[{state_color}]{data['state']}[/]",
                          acc_str, params_str, lr_str, data.get('config', 'N/A'), data.get('sample', '')
                      )
            live.update(final_table) # Display the final complete table
            time.sleep(5) # Pause for 5 seconds

    except KeyboardInterrupt:
         console.print("\n\n[bold yellow]🛑 Optimization interrupted by user (Ctrl+C).[/bold yellow]")
    except Exception as e:
         console.print("\n\n[bold red]💥 An unexpected error occurred during optimization:[/bold red]")
         traceback.print_exc()
    finally:
         # Ensure the Live display is properly stopped if needed
         # (usually handled by `with Live`, but good practice)
         console.print("\n✨ Live display stopped.")


    # --- Final Results Analysis ---
    print("\n" + "=" * 80)
    print("📊 Optimization Analysis 📊".center(80, "="))
    print(f"Total number of trials in study: {len(study.trials)}")

    # Filter for successfully completed trials
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if not complete_trials:
        print("\n[bold red]No trials completed successfully. Cannot show Pareto front.[/bold red]")
    else:
        print(f"\n✅ Number of successfully completed trials: {len(complete_trials)}")
        print("\n" + "-" * 30 + " Pareto Front Trials " + "-" * 30)
        print("🌟 Trials representing the best Accuracy vs. Parameter trade-offs found:")

        # Get the Pareto optimal trials from the study
        pareto_trials = sorted(study.best_trials, key=lambda t: t.values[1]) # Sort by Params (Objective 2)

        for trial in pareto_trials:
            # Extract values safely
            acc = trial.values[0] * 100 if trial.values and trial.values[0] is not None else float('nan')
            params_raw = trial.values[1] if trial.values and trial.values[1] is not None else float('inf')
            params_m = params_raw / 1e6 if params_raw != float('inf') else float('inf')
            model_type = trial.params.get("model_type", "N/A").upper()
            lr = trial.params.get("lr", float('nan'))
            config_str = dashboard_data.get(trial.number, {}).get('config', 'N/A') # Get config str from dashboard data

            console.print(f"\n[bold cyan]  Trial {trial.number}[/bold cyan] ([magenta]{model_type}[/magenta])")
            console.print(f"    [green]Accuracy = {acc:.3f}%[/green], [blue]Params = {params_m:.3f} M[/blue] ({int(params_raw):,})")
            console.print(f"    [dim]Config: LR={lr:.2e}, {config_str}[/dim]")
            # console.print(f"    Full Params: {trial.params}") # Uncomment for all hyperparameters
            console.print(f"    [white]Sample: {trial.user_attrs.get('generated_sample', '[N/A]')}[/white]")

    print("\n" + "-" * 80)

    # --- Visualizations (Optional, requires Plotly) ---
    try:
        if optuna.visualization.is_available() and complete_trials:
            console.print("\n📈 Generating visualization plots (requires plotly)...")

            # Define target formatting for plots
            target_names = ["Accuracy (%)", "Parameters (M)"]
            targets_format = lambda t: (t.values[0] * 100, t.values[1] / 1e6) # Acc in %, Params in M

            # Plot 1: Pareto Front
            fig1 = optuna.visualization.plot_pareto_front(
                study,
                targets=targets_format,
                target_names=target_names,
                # Color points by model type for easy comparison
                #color_axis={"name": "Model Type",
                #            "values": [t.params.get("model_type", "N/A").upper() for t in complete_trials],
                #            # Define consistent colors (e.g., Blue for Hydra, Orange for Transformer)
                #            "colorscale": [['HYDRA', '#1f77b4'], ['TRANSFORMER', '#ff7f0e']]
                #           }
            )
            fig1.update_layout(title="Pareto Front: Accuracy vs. Parameters")
            fig1.show()
            console.print("   ✅ Pareto Front plot generated.")

            # Plot 2: Parameter Importances (Can be complex/less reliable for multi-objective)
            # Plot importance specifically for the Accuracy objective
            try:
                fig2 = optuna.visualization.plot_param_importances(
                    study,
                    # Target is the first objective value (Accuracy)
                    target=lambda t: t.values[0] if t.state == optuna.trial.TrialState.COMPLETE and t.values else None,
                    target_name="Accuracy"
                )
                fig2.update_layout(title="Hyperparameter Importance for Accuracy")
                fig2.show()
                console.print("   ✅ Parameter Importance plot (for Accuracy) generated.")
            except Exception as e_imp1:
                 console.print(f"   ⚠️ Could not plot importance for Accuracy: {e_imp1}")

            # Plot importance specifically for the Parameters objective
            # try:
            #     fig3 = optuna.visualization.plot_param_importances(
            #         study,
            #         target=lambda t: t.values[1]/1e6 if t.state == optuna.trial.TrialState.COMPLETE and t.values else None,
            #         target_name="Parameters (M)"
            #     )
            #     fig3.update_layout(title="Hyperparameter Importance for Parameters (M)")
            #     fig3.show()
            #     console.print("   ✅ Parameter Importance plot (for Parameters) generated.")
            # except Exception as e_imp2:
            #      console.print(f"   ⚠️ Could not plot importance for Parameters: {e_imp2}")

        else:
            if not complete_trials:
                console.print("\n📈 Skipping plots: No trials completed successfully.")
            else:
                console.print("\n📈 Skipping plots: Plotly is not available. Install with: [code]pip install plotly kaleido[/code]")

    except Exception as e:
        console.print(f"\n[yellow]⚠️ Warning: Could not generate plots. Error:[/yellow] {e}")
        console.print("   Ensure plotly and kaleido are installed (`pip install plotly kaleido`).")

    # --- Final Message ---
    print("\n" + "=" * 80)
    console.print(" ✨ Optimization Complete. The Edge Revolution Marches On! ✨ ".center(80))
    print("=" * 80 + "\n")