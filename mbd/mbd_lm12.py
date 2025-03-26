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
              robust error handling, improved efficiency, comprehensive hyperparameter
              tuning for HydraScale, corrected Transformer compatibility checks,
              fixed Optuna dynamic categorical issue, and ✨ style ✨.
"""

# --- Core Imports ---
import gc
import math
import time
import warnings
import traceback # For detailed error logging
import random

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
# Ensure visualization tools are available or skip gracefully
try:
    from optuna.visualization import plot_pareto_front, plot_param_importances
    _plotly_available = True
except ImportError:
    _plotly_available = False
    print("⚠️ WARNING: Plotly not found. Optuna visualizations will be skipped.")
    print("   Install using: pip install plotly kaleido")

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
    else: # Other IPython shells or standard Python
        from tqdm import tqdm
        print("Using standard tqdm for progress bars.")
except NameError:
    from tqdm import tqdm # Fallback for standard Python environment
    print("Using standard tqdm for progress bars.")
except ImportError:
    print("⚠️ WARNING: tqdm not found. Progress bars will be basic.")
    # Dummy tqdm if not installed
    def tqdm(iterable, *args, **kwargs):
        # print("Info: tqdm not installed, progress bar disabled.")
        return iterable

# --- Dependency Check & Installation Hint ---
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("\n" + "="*60)
    print("❌ ERROR: Missing essential libraries!")
    print("Please install them using pip:")
    print("  pip install datasets transformers torch optuna rich tqdm")
    print("Optionally for plots: pip install plotly kaleido")
    print("="*60 + "\n")
    exit(1)

# --- Constants & Configuration ---
# Basic Setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42 # For reproducibility in sampling/initialization (where applicable)


# Optuna Study Configuration
OPTUNA_N_TRIALS = 30                # Increase trials for better search space coverage
NUM_DATA_LOAD_BATCHES = 800         # Batches to preload (reduce disk I/O during trials)
NUM_TRAIN_STEPS_PER_TRIAL = 500    # Steps per trial (balance speed & convergence signal)
NUM_EVAL_BATCHES_PER_TRIAL = NUM_TRAIN_STEPS_PER_TRIAL//16     # Batches for validation check during training

# Dataset & Tokenizer
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1" # Standard benchmark
TOKENIZER_NAME = "gpt2"             # Common & effective tokenizer

# Model/Training Hyperparameters (Tuneable Ranges defined in Optuna Objective)
SEQ_LEN = 64                        # Sequence length during training/eval
BATCH_SIZE = 64                     # Adjust based on GPU VRAM
GRAD_CLIP_NORM = 1.0                # Max norm for gradient clipping (prevents explosions)

default_sampling_steps = 10  # min(getattr(model, 'num_timesteps', 100) // 4, 25)
TEMPERATURE = 0.5
TOP_K = 50


# Text Generation Configuration
GENERATION_MAX_NEW = 50             # Max new tokens for sample generation
GENERATION_PROMPT = "Edge computing will revolutionize AI by" # Focused prompt

# Fixed list for Transformer head options (to avoid Optuna dynamic categorical error)
TRANSFORMER_FIXED_NHEAD_OPTIONS = [2, 4, 8, 12, 16]

pruning = False # Trial.should_prune is not supported for multi-objective optimization.
compile = False


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
        # Ensure text fields are strings, handle potential None values
        texts = [text if isinstance(text, str) else "" for text in examples.get("text", [])]
        # Tokenize without adding special tokens or truncating here
        output = tokenizer(texts, add_special_tokens=False, truncation=False)
        # Ensure output always contains 'input_ids', even if empty
        if 'input_ids' not in output:
            output['input_ids'] = [[] for _ in texts]
        return output

    try:
        # Adjust num_proc based on available CPU cores for faster mapping
        num_proc = min(max(1, torch.multiprocessing.cpu_count() // 2), 8)
    except NotImplementedError:
        num_proc = 4 # Sensible default

    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=[col for col in dataset.column_names if col != 'input_ids'], # Keep only input_ids
        desc=f"Tokenizing '{split}'"
    )

    # --- FIX: Explicitly keep only 'input_ids' ---
    # Since attention_mask isn't used later, discard it now to prevent mismatch errors during grouping.
    print(f"   ↳ Selecting only 'input_ids' column before chunking...")
    try:
        tokenized_datasets = tokenized_datasets.select_columns(['input_ids'])
    except ValueError as e:
         print(f"   ⚠️ Warning: Could not select 'input_ids' column (maybe it doesn't exist?). Error: {e}")
         print(f"      Available columns: {tokenized_datasets.column_names}")
         return [] # Cannot proceed without input_ids


    print(f"   ↳ Concatenating and chunking into sequences of length {seq_len}...")
    # --- MORE ROBUST group_texts function ---
    def group_texts(examples):
        # Concatenate all token IDs from the 'input_ids' list
        concatenated_ids = sum(examples.get('input_ids', []), [])
        total_length = len(concatenated_ids)

        # If no tokens, return empty structure
        if total_length == 0:
            return {'input_ids': []}

        # Drop the small remainder to ensure all sequences are exactly `seq_len` long
        total_length = (total_length // seq_len) * seq_len

        # Chunk the concatenated IDs into sequences of length `seq_len`
        result = {
            'input_ids': [concatenated_ids[i : i + seq_len] for i in range(0, total_length, seq_len)]
        }
        return result
    # --- End of ROBUST group_texts function ---

    lm_datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=num_proc,
        desc=f"Chunking '{split}'"
    )

    print(f"   ↳ Preparing {num_batches} batches of size {batch_size} for '{split}'...")
    data_batches: List[Tensor] = []
    total_sequences = len(lm_datasets)
    num_sequences_to_process = min(num_batches * batch_size, total_sequences)

    # Use tqdm for clear progress on batch creation
    indices = range(0, num_sequences_to_process, batch_size)
    pbar_batches = tqdm(indices, desc=f"Creating '{split}' batches", leave=False, unit="batch", total=len(indices))
    for i in pbar_batches:
        end_idx = min(i + batch_size, total_sequences)
        batch_ids = lm_datasets[i:end_idx]['input_ids']

        if not batch_ids: continue # Skip empty slices

        current_batch_size = len(batch_ids)
        batch_tensor = torch.tensor(batch_ids, dtype=torch.long)

        # Pad the last batch if it's smaller than the target batch_size
        if current_batch_size < batch_size:
            padding_needed = batch_size - current_batch_size
            # Ensure pad tensor dimensions match seq_len
            pad_tensor = torch.full((padding_needed, seq_len), tokenizer.pad_token_id, dtype=torch.long)
            batch_tensor = torch.cat([batch_tensor, pad_tensor], dim=0)

        # Final sanity check for sequence length
        if batch_tensor.shape != (batch_size, seq_len):
             print(f"   ⚠️ Warning: Skipping batch with incorrect shape {batch_tensor.shape} (expected ({batch_size}, {seq_len})). Check data processing pipeline.")
             continue

        # Store batches on CPU initially to conserve GPU memory
        data_batches.append(batch_tensor.cpu())

        if len(data_batches) >= num_batches:
            break # Stop once requested number of batches is reached

    pbar_batches.close()

    if not data_batches:
         print(f"   ⚠️ WARNING: No data batches were created for '{split}' split. Check dataset content and parameters (seq_len, batch_size).")
    else:
         print(f"   ✅ Loaded {len(data_batches)} batches ({len(data_batches) * batch_size} sequences target) for '{split}' split.")

    _data_cache[cache_key] = data_batches
    return data_batches

class SelectiveScan(nn.Module):
    def __init__(self, embed_dim: int, state_dim: int = 16, d_conv: int = 4, dt_rank: Union[str, int] = 'auto', bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.d_conv = d_conv
        self.dt_rank = math.ceil(embed_dim / 16) if dt_rank == 'auto' else dt_rank
        self.bias = bias

        self.in_proj_x = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_z = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_params = nn.Linear(embed_dim, self.dt_rank + 2 * self.state_dim, bias=False)

        self.conv1d = nn.Conv1d(embed_dim, embed_dim, bias=bias, kernel_size=d_conv, groups=embed_dim, padding=d_conv - 1)
        self.dt_proj = nn.Linear(self.dt_rank, embed_dim, bias=True)
        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).unsqueeze(0).repeat(embed_dim, 1))
        self.D = nn.Parameter(torch.ones(embed_dim))
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        with torch.no_grad():
            dt_init_std = self.dt_rank**-0.5 * 0.1
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            self.dt_proj.bias.data.fill_(math.log(math.expm1(0.001)))

    def _compute_A_tilde(self, dt: Tensor, A_log: Tensor) -> Tensor:
        A = -torch.exp(A_log.float())
        return torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(1))

    def _compute_B_tilde(self, dt: Tensor, A_log: Tensor, A_tilde: Optional[Tensor] = None) -> Tensor:
        A = -torch.exp(A_log.float())
        if A_tilde is None:
            A_tilde = self._compute_A_tilde(dt, A_log)
        A_unsqueezed = A.unsqueeze(0).unsqueeze(1)
        return torch.where(
            torch.abs(A_unsqueezed) < 1e-5, dt.unsqueeze(-1),
            (A_tilde - 1) / (A_unsqueezed + 1e-10)
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
            raise ValueError(f"Input dim {embed_dim} != Model dim {self.embed_dim}")

        z = self.in_proj_z(x)
        params = self.in_proj_params(x)
        dt_unproj, B_proj, C_proj = torch.split(params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        dt = F.softplus(self.dt_proj(dt_unproj))

        x_conv = self.conv1d(x.transpose(1, 2))[:, :, :seq_len].transpose(1, 2)
        u = F.silu(x_conv)

        A_tilde = self._compute_A_tilde(dt, self.A_log)  # [batch, seq_len, embed_dim, state_dim]
        B_tilde = self._compute_B_tilde(dt, self.A_log, A_tilde)  # [batch, seq_len, embed_dim, state_dim]

        y = self.fwd(A_tilde, B_tilde, C_proj, seq_len, u)

        y_ssm_out = y + u * self.D.unsqueeze(0).unsqueeze(1)
        y_gated = y_ssm_out * F.silu(z)
        return self.out_proj(y_gated)

    def fwd(self, A_tilde, B_tilde, C_proj, seq_len, u):
        #h = torch.zeros(batch, seq_len, self.embed_dim, self.state_dim, device=x.device, dtype=x.dtype, requires_grad=True)

        # Avoid in-place operations
        h_steps = [B_tilde[:, 0] * u[:, 0].unsqueeze(-1)]  # List to store each step
        for t in range(1, seq_len):
            h_t = A_tilde[:, t - 1] * h_steps[-1] + B_tilde[:, t - 1] * u[:, t - 1].unsqueeze(-1)
            h_steps.append(h_t)
        h = torch.stack(h_steps, dim=1)  # [batch, seq_len, embed_dim, state_dim]
        y = torch.einsum('bln,blen->ble', C_proj, h)  # [batch, seq_len, embed_dim]
        return y

    def step(self, x_step: Tensor, h_prev: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch, embed_dim = x_step.shape
        conv_input = torch.cat([conv_state, x_step.unsqueeze(2)], dim=2)
        conv_out = F.conv1d(conv_input, self.conv1d.weight, self.conv1d.bias, groups=self.embed_dim).squeeze(-1)
        new_conv_state = conv_input[:, :, 1:]
        u_step = F.silu(conv_out)

        z_step = self.in_proj_z(x_step)
        params = self.in_proj_params(x_step)
        dt_unproj, _, C_proj = torch.split(params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1)
        dt_step = F.softplus(self.dt_proj(dt_unproj))

        A_tilde_step = self._compute_A_tilde(dt_step.unsqueeze(1), self.A_log).squeeze(1)
        B_tilde_step = self._compute_B_tilde(dt_step.unsqueeze(1), self.A_log).squeeze(1)

        h_new = A_tilde_step * h_prev + B_tilde_step * u_step.unsqueeze(-1)
        y = torch.einsum('bn,bdn->bd', C_proj, h_new)
        y_ssm_out = y + u_step * self.D.unsqueeze(0)
        y_gated = y_ssm_out * F.silu(z_step)
        y_step = self.out_proj(y_gated)
        return y_step, h_new, new_conv_state

# --- HydraScale Components ---
class HydraBlock(nn.Module):
    """ A single block of the HydraScale model (Pre-LN). """
    def __init__(self, embed_dim: int, mlp_mult: int = 4, dropout_rate: float = 0.1, ssm_kwargs: Dict[str, Any] = {}):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ssm = SelectiveScan(embed_dim, **ssm_kwargs)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = embed_dim * mlp_mult
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim), nn.GELU(),
            nn.Dropout(dropout_rate), nn.Linear(mlp_hidden_dim, embed_dim)
        )
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x: Tensor) -> Tensor:
        # SSM Path
        x = x + self.dropout1(self.ssm(self.norm1(x)))
        # MLP Path
        x = x + self.dropout2(self.mlp(self.norm2(x)))
        return x

    def step(self, x_step: Tensor, ssm_state: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        # SSM Path (Step)
        residual = x_step
        x_norm1 = self.norm1(x_step)
        ssm_out_step, ssm_state_new, conv_state_new = self.ssm.step(x_norm1, ssm_state, conv_state)
        x = residual + self.dropout1(ssm_out_step)
        # MLP Path (Step)
        residual = x
        x_norm2 = self.norm2(x)
        mlp_out_step = self.mlp(x_norm2)
        y_step = residual + self.dropout2(mlp_out_step)
        return y_step, ssm_state_new, conv_state_new

# Sinusoidal Time Embedding Helper (with caching)
_sinusoidal_embedding_cache: Dict[Tuple[int, int, torch.device], Tensor] = {}

def sinusoidal_embedding(timesteps: Tensor, embedding_dim: int, max_period: int = 10000) -> Tensor:
    """ Creates sinusoidal time embeddings. Caches the frequency matrix. """
    if timesteps.ndim > 1: timesteps = timesteps.squeeze(-1)
    device = timesteps.device
    dtype = torch.float32
    key = (embedding_dim, max_period, device)
    if key not in _sinusoidal_embedding_cache:
        if embedding_dim % 2 != 0: raise ValueError("Sinusoidal embedding requires even dimension")
        half_dim = embedding_dim // 2
        exponent = -math.log(max_period) * torch.arange(half_dim, device=device, dtype=dtype) / half_dim
        inv_timescales = torch.exp(exponent)
        _sinusoidal_embedding_cache[key] = inv_timescales.unsqueeze(0)
    cached_freqs = _sinusoidal_embedding_cache[key].to(device)
    args = timesteps.to(dtype).unsqueeze(1) * cached_freqs
    embedding = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    return embedding

# --- HydraScale Language Model Definition ---
class HydraScaleLM(nn.Module):
    """ HydraScale Language Model: Text Diffusion using Selective Scan blocks. """
    def __init__(self, vocab_size: int, embed_dim: int = 512, depth: int = 6,
                 mlp_mult: int = 4, dropout_rate: float = 0.1,
                 num_diffusion_timesteps: int = 100, noise_schedule: str = 'cosine',
                 ssm_state_dim: int = 16, ssm_d_conv: int = 4, ssm_dt_rank: Union[str, int] = 'auto'):
        super().__init__()
        self.effective_vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_timesteps = num_diffusion_timesteps
        self.mask_token_id = self.effective_vocab_size
        self.ssm_d_conv = ssm_d_conv

        self.token_embedding = nn.Embedding(self.effective_vocab_size + 1, embed_dim)
        self.time_embedding_dim = embed_dim if embed_dim % 2 == 0 else embed_dim + 1
        if embed_dim % 2 != 0: print(f"   ⚠️ HydraScale Warning: embed_dim odd. Adjusting time_emb_dim to {self.time_embedding_dim}.")
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, embed_dim * 4), nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim)
        )
        ssm_kwargs = {'state_dim': ssm_state_dim, 'd_conv': ssm_d_conv, 'dt_rank': ssm_dt_rank}
        self.layers = nn.ModuleList([
            HydraBlock(embed_dim, mlp_mult=mlp_mult, dropout_rate=dropout_rate, ssm_kwargs=ssm_kwargs)
            for _ in range(depth)
        ])
        self.norm_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, self.effective_vocab_size)

        # --- Diffusion Schedule Buffers ---
        betas = self._calculate_betas(num_diffusion_timesteps, noise_schedule)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas', alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cumprod))
        # Add other precomputed values if needed for different sampling types later

        self.apply(self._init_weights)
        print(f"✨ HydraScaleLM Initialized ✨ (D={embed_dim}, Depth={depth}, SSM_N={ssm_state_dim}, ConvK={ssm_d_conv}, T={num_diffusion_timesteps}, Dropout={dropout_rate:.2f}). Vocab: {self.effective_vocab_size}.")
        # warnings.warn("Note: HydraScaleLM training uses a sequential SSM scan...", UserWarning)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def _calculate_betas(self, timesteps: int, schedule: str ='cosine', s: float = 0.008, beta_start: float = 0.0001, beta_end: float = 0.02) -> Tensor:
        if schedule == 'cosine':
            steps = timesteps + 1
            t = torch.linspace(0, timesteps, steps, dtype=torch.float64)
            alphas_cumprod = torch.cos(((t / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, min=1e-6, max=0.999).float()
        elif schedule == 'linear':
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        else: raise ValueError(f"Unknown noise schedule: {schedule}")

    def _get_mask_prob_from_time(self, t: Tensor) -> Tensor:
        t_clamped = torch.clamp(t, 0, self.num_timesteps - 1).float()
        # Cosine schedule for mask rate - higher prob earlier (lower t) is incorrect for diffusion!
        # Mask rate should INCREASE with t (more noise -> more masks)
        # Let's use linear or sqrt(1 - alpha_bar)
        # Linear: mask_prob = t_clamped / (self.num_timesteps -1) # Scale to [0, 1]
        # Sqrt(1-alpha_bar):
        alpha_bar_t = self.alphas_cumprod.gather(dim=0, index=t_clamped.long())
        mask_prob = torch.sqrt(1.0 - alpha_bar_t) # Higher t -> lower alpha_bar -> higher prob
        return torch.clamp(mask_prob, 0.0, 1.0)

    def _mask_tokens(self, x_0: Tensor, t: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, seq_len = x_0.shape
        device = x_0.device
        mask_prob = self._get_mask_prob_from_time(t)
        mask_prob_expanded = mask_prob.view(batch_size, 1).expand(-1, seq_len)
        rand_noise = torch.rand(batch_size, seq_len, device=device, dtype=torch.float32)
        is_padding = (x_0 == tokenizer.pad_token_id)
        mask = (rand_noise < mask_prob_expanded) & (~is_padding)
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t, mask

    def forward(self, x_t: Tensor, t: Tensor) -> Tensor:
        batch_size, seq_len = x_t.shape
        device = x_t.device
        max_id_in_batch = x_t.max().item()
        if max_id_in_batch >= self.token_embedding.num_embeddings:
             raise ValueError(f"Input ID {max_id_in_batch} >= emb size {self.token_embedding.num_embeddings}")

        token_emb = self.token_embedding(x_t)
        t_on_device = t.to(device)
        time_emb_sin = sinusoidal_embedding(t_on_device, self.time_embedding_dim)
        time_emb = self.time_mlp(time_emb_sin)
        h = token_emb + time_emb.unsqueeze(1)
        for layer in self.layers: h = layer(h)
        h = self.norm_out(h)
        logits = self.lm_head(h)
        return logits

    def compute_loss(self, x_0: Tensor, return_acc: bool = True) -> Union[Tensor, Tuple[Tensor, float]]:
        batch_size, seq_len = x_0.shape
        device = x_0.device
        if batch_size == 0:
            loss = torch.tensor(0.0, device=device, requires_grad=True) * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            return (loss, 1.0) if return_acc else loss

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=device).long()
        x_t, mask = self._mask_tokens(x_0, t)
        pred_logits = self.forward(x_t, t)

        masked_logits = pred_logits[mask]
        masked_targets = x_0[mask]

        if masked_targets.numel() > 0 and masked_targets.max() >= self.effective_vocab_size:
            print(f"⚠️ Target ID {masked_targets.max().item()} >= vocab {self.effective_vocab_size}")
            valid_target_mask = masked_targets < self.effective_vocab_size
            masked_logits = masked_logits[valid_target_mask]
            masked_targets = masked_targets[valid_target_mask]

        if masked_targets.numel() == 0:
            loss = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            accuracy = 1.0
        else:
            loss = F.cross_entropy(masked_logits, masked_targets)
            accuracy = 0.0
            if return_acc:
                with torch.no_grad():
                    correct_preds = (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
                    accuracy = correct_preds / masked_targets.numel()

        if not torch.isfinite(loss):
            warnings.warn(f"🤯 Loss is NaN/Inf ({loss.item()}) in HydraScale. Zeroing loss.", RuntimeWarning)
            loss = torch.tensor(0.0, device=pred_logits.device, dtype=pred_logits.dtype, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            accuracy = 0.0

        return (loss, accuracy) if return_acc else loss

    @torch.no_grad()
    def _predict_x0_from_logits(self, x_t: Tensor, logits: Tensor,
                                sampling_mode: str = 'argmax', temperature: float = 1.0, top_k: Optional[int] = None
                               ) -> Tensor:
        batch_size, seq_len, vocab_size = logits.shape
        device = logits.device
        if vocab_size != self.effective_vocab_size:
             raise ValueError(f"Logits vocab {vocab_size} != effective {self.effective_vocab_size}")

        if sampling_mode == 'argmax':
            pred_ids = torch.argmax(logits, dim=-1)
        elif sampling_mode in ['multinomial', 'topk']:
            logits_flat = logits.view(-1, vocab_size)
            if temperature > 0 and abs(temperature - 1.0) > 1e-6:
                logits_flat = logits_flat / temperature
            if top_k is not None and top_k > 0 and sampling_mode == 'topk':
                k = min(top_k, vocab_size)
                kth_vals, _ = torch.topk(logits_flat, k, dim=-1)
                kth_vals_min = kth_vals[..., -1, None]
                logits_flat = torch.where(logits_flat < kth_vals_min, torch.tensor(-float('Inf'), device=device, dtype=logits.dtype), logits_flat)

            probs = F.softmax(logits_flat, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            zero_prob_mask = probs.sum(dim=-1) <= 1e-9
            if zero_prob_mask.any():
                 uniform_dist = torch.full_like(probs[0], 1.0 / vocab_size)
                 probs[zero_prob_mask] = uniform_dist
                 probs /= probs.sum(dim=-1, keepdim=True)
            try:
                pred_ids_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
            except RuntimeError as e:
                 print(f"Multinomial error: {e}. Falling back to argmax.")
                 pred_ids_flat = torch.argmax(probs, dim=-1)
            pred_ids = pred_ids_flat.view(batch_size, seq_len)
        else: raise ValueError(f"Unknown sampling mode: {sampling_mode}")
        return pred_ids

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, num_tokens_to_generate: int,
                 num_sampling_steps: Optional[int] = None, temperature: float = 1.0,
                 top_k: Optional[int] = None, sampling_mode: str = 'topk') -> Tensor:
        self.eval()
        batch_size, prompt_len = prompt_ids.shape
        total_len = prompt_len + num_tokens_to_generate
        device = prompt_ids.device

        sampling_steps = num_sampling_steps if num_sampling_steps is not None else self.num_timesteps
        sampling_steps = min(sampling_steps, self.num_timesteps)
        if sampling_steps <= 0: sampling_steps = 1
        # print(f"   (Hydra Gen using {sampling_steps} diffusion steps)") # Moved to caller

        x_gen = torch.full((batch_size, total_len), self.mask_token_id, dtype=torch.long, device=device)
        x_gen[:, :prompt_len] = prompt_ids

        layer_states: List[Tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            ssm_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.state_dim, device=device, dtype=torch.float32)
            conv_state = torch.zeros(batch_size, self.embed_dim, self.ssm_d_conv - 1, device=device, dtype=torch.float32)
            layer_states.append((ssm_state, conv_state))

        time_indices = torch.linspace(self.num_timesteps - 1, 0, sampling_steps, device=device).long()

        for t_val in time_indices: # Use simple loop, tqdm can be added in caller if needed
            t_current = t_val.expand(batch_size)
            current_layer_states = [(s[0].clone(), s[1].clone()) for s in layer_states]
            token_emb = self.token_embedding(x_gen)
            time_emb_sin = sinusoidal_embedding(t_current, self.time_embedding_dim)
            time_emb = self.time_mlp(time_emb_sin)
            all_logits_step: List[Tensor] = []

            for token_idx in range(total_len):
                x_step = (token_emb[:, token_idx, :] + time_emb).to(torch.float32)
                for layer_idx, layer in enumerate(self.layers):
                    ssm_state, conv_state = current_layer_states[layer_idx]
                    x_step, ssm_state_new, conv_state_new = layer.step(x_step, ssm_state, conv_state)
                    current_layer_states[layer_idx] = (ssm_state_new, conv_state_new)
                h_final = self.norm_out(x_step)
                logits_token = self.lm_head(h_final)
                all_logits_step.append(logits_token)

            logits = torch.stack(all_logits_step, dim=1)
            predicted_x0_ids = self._predict_x0_from_logits(
                x_gen, logits, sampling_mode=sampling_mode, temperature=temperature, top_k=top_k
            )
            mask_for_update = (x_gen == self.mask_token_id)
            x_gen = torch.where(mask_for_update, predicted_x0_ids.long(), x_gen)

        self.train()
        return x_gen

# --- Comparison Baseline: Standard Transformer Language Model ---
class SimpleTransformerLM(nn.Module):
    """ Standard Decoder-only Transformer baseline. """
    def __init__(self, vocab_size: int, embed_dim: int, nhead: int, num_layers: int,
                 dim_feedforward: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        if embed_dim % nhead != 0:
            #raise ValueError(f"Transformer Config Error: embed_dim ({embed_dim}) not divisible by nhead ({nhead})")
            embed_dim = (embed_dim//nhead) * nhead

        self.effective_vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.nhead = nhead
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(self.effective_vocab_size, embed_dim)
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.normal_(self.pos_encoder, std=0.02)
        self.dropout = nn.Dropout(dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True
        )
        encoder_norm = nn.LayerNorm(embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers, norm=encoder_norm)
        self.lm_head = nn.Linear(embed_dim, self.effective_vocab_size)
        self.apply(self._init_weights)
        print(f"🔩 SimpleTransformerLM Initialized (D={embed_dim}, Layers={num_layers}, Head={nhead}, FFN={dim_feedforward}, MaxLen={max_seq_len}, Dropout={dropout:.2f}). Vocab: {self.effective_vocab_size}.")

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight, nonlinearity='relu')
            if module.bias is not None: nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            if hasattr(module, 'weight') and module.weight is not self.pos_encoder:
                 nn.init.normal_(module.weight, mean=0.0, std=0.02)

    _causal_mask_cache: Dict[Tuple[int, torch.device], Tensor] = {}
    def _generate_causal_mask(self, sz: int, device: torch.device) -> Tensor:
        key = (sz, device)
        if key not in self._causal_mask_cache:
            mask = torch.triu(torch.full((sz, sz), float('-inf'), device=device, dtype=torch.float32), diagonal=1)
            self._causal_mask_cache[key] = mask
        return self._causal_mask_cache[key]

    def forward(self, src: Tensor) -> Tensor:
        batch_size, seq_len = src.shape
        device = src.device
        if seq_len > self.max_seq_len:
            src = src[:, -self.max_seq_len:]
            seq_len = self.max_seq_len

        max_id_in_batch = src.max().item()
        if max_id_in_batch >= self.token_embedding.num_embeddings:
             raise ValueError(f"Input ID {max_id_in_batch} >= emb size {self.token_embedding.num_embeddings}")

        src_emb = self.token_embedding(src) * math.sqrt(self.embed_dim)
        pos_emb = self.pos_encoder[:, :seq_len, :].to(device=device, dtype=src_emb.dtype)
        src_combined = self.dropout(src_emb + pos_emb)
        causal_mask = self._generate_causal_mask(seq_len, device=device)
        padding_mask = (src == tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None
        output = self.transformer_encoder(src_combined, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.lm_head(output)
        return logits

    def compute_loss(self, x_0: Tensor, return_acc: bool = True) -> Union[Tensor, Tuple[Tensor, float]]:
        batch_size, seq_len = x_0.shape
        device = x_0.device
        if seq_len < 2:
            loss = torch.tensor(0.0, device=device, requires_grad=True) * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            return (loss, 1.0) if return_acc else loss

        inp = x_0[:, :-1].contiguous()
        tgt = x_0[:, 1:].contiguous()
        logits = self.forward(inp)
        ignore_idx = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
        loss = F.cross_entropy(logits.view(-1, self.effective_vocab_size), tgt.view(-1), ignore_index=ignore_idx)

        accuracy = 0.0
        if return_acc:
            with torch.no_grad():
                valid_targets_mask = (tgt.view(-1) != ignore_idx)
                num_valid_targets = valid_targets_mask.sum().item()
                if num_valid_targets > 0:
                    pred_ids = logits.view(-1, self.effective_vocab_size).argmax(dim=-1)
                    valid_preds = pred_ids[valid_targets_mask]
                    valid_targets = tgt.view(-1)[valid_targets_mask]
                    correct_preds = (valid_preds == valid_targets).sum().item()
                    accuracy = correct_preds / num_valid_targets
                else: accuracy = 1.0

        if not torch.isfinite(loss):
            warnings.warn(f"🤯 Loss is NaN/Inf ({loss.item()}) in Transformer. Zeroing loss.", RuntimeWarning)
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            accuracy = 0.0

        return (loss, accuracy) if return_acc else loss

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, num_tokens_to_generate: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> Tensor:
        self.eval()
        generated_ids = prompt_ids.to(DEVICE)
        batch_size = prompt_ids.shape[0]
        device = generated_ids.device
        # print("   (Transformer Gen using standard autoregressive sampling)") # Moved to caller

        for _ in range(num_tokens_to_generate): # Use simple loop
            context = generated_ids
            if context.shape[1] > self.max_seq_len: context = context[:, -self.max_seq_len:]
            use_amp = (device.type == 'cuda')
            with autocast(enabled=use_amp):
                logits = self.forward(context)
                next_token_logits = logits[:, -1, :]

            if temperature > 0 and abs(temperature - 1.0) > 1e-6:
                next_token_logits = next_token_logits / temperature
            if top_k is not None and top_k > 0:
                v = next_token_logits.size(-1)
                k = min(top_k, v)
                kth_vals, _ = torch.topk(next_token_logits, k, dim=-1)
                kth_vals_min = kth_vals[:, -1, None]
                next_token_logits = torch.where(
                    next_token_logits < kth_vals_min,
                    torch.tensor(-float('Inf'), device=device, dtype=next_token_logits.dtype),
                    next_token_logits
                )

            probs = F.softmax(next_token_logits, dim=-1)
            if temperature <= 0:
                next_token_id = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                probs = torch.nan_to_num(probs, nan=0.0)
                zero_prob_mask = probs.sum(dim=-1) <= 1e-9
                if zero_prob_mask.any():
                    uniform_dist = torch.full_like(probs[0], 1.0 / probs.shape[-1])
                    probs[zero_prob_mask] = uniform_dist
                    probs /= probs.sum(dim=-1, keepdim=True)
                try:
                    next_token_id = torch.multinomial(probs, num_samples=1)
                except RuntimeError as e:
                     print(f"Multinomial error (Transformer): {e}. Falling back to argmax.")
                     next_token_id = torch.argmax(probs, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        self.train()
        return generated_ids



@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    """ Counts trainable parameters. """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_sample_text(model: nn.Module, tokenizer: AutoTokenizer, prompt_text: str, device: torch.device, model_type: str, max_new: int = 48, gen_kwargs: Optional[Dict] = None) -> str:
    """Generates and decodes a text sample."""
    if gen_kwargs is None: gen_kwargs = {}
    model.eval()
    result_text = f"[{model_type.capitalize()} Gen Error]"

    try:
        prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
        if prompt_ids.shape[1] == 0:
             bos_token_id = getattr(tokenizer, 'bos_token_id', None)
             if bos_token_id is not None:
                  prompt_ids = torch.tensor([[bos_token_id]], dtype=torch.long, device=device)
                  prompt_text = tokenizer.decode([bos_token_id])
             else: return "[Error: Empty prompt, no BOS]"

        default_kwargs = {"temperature": TEMPERATURE, "top_k": TOP_K}
        current_gen_kwargs = default_kwargs.copy()
        gen_info = ""

        if model_type == "hydra":
            hydra_defaults = {"num_sampling_steps": default_sampling_steps, "sampling_mode": "topk"}
            current_gen_kwargs.update(hydra_defaults)
            gen_info = f"(T={current_gen_kwargs['num_sampling_steps']})"
        elif model_type == "transformer":
            gen_info = "(autoregressive)"
        else: return f"[Error: Invalid model_type '{model_type}']"

        current_gen_kwargs.update(gen_kwargs) # User kwargs override all
        gen_func = model.generate

        print(f"   Generating sample for {model_type} {gen_info}...", end="", flush=True)
        start_gen_time = time.time()
        use_amp = (device.type == 'cuda')
        with torch.inference_mode(), autocast(enabled=use_amp):
             output_ids = gen_func(prompt_ids, num_tokens_to_generate=max_new, **current_gen_kwargs)
        end_gen_time = time.time()
        print(f" done in {end_gen_time - start_gen_time:.2f}s")

        generated_part_ids = output_ids[0, prompt_ids.shape[1]:]
        generated_text = tokenizer.decode(generated_part_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        full_text = (prompt_text + generated_text).replace("\n", " ").replace("\r", "").strip()
        result_text = full_text

    except Exception as e:
        print(f"\n❌ Error during text generation for {model_type}:")
        traceback.print_exc(limit=1)
        result_text = f"[{model_type.capitalize()} Gen Error: {type(e).__name__}]"
    finally:
        model.train()
        return result_text


def train_eval_trial( trial: optuna.Trial, model: nn.Module, train_loader: List[Tensor],
    eval_loader: List[Tensor], optimizer: torch.optim.Optimizer, scaler: GradScaler,
    num_train_steps: int, num_eval_batches: int, device: torch.device, model_type: str
) -> Tuple[float, float]:
    """ Trains model, performs pruning checks, returns avg train loss & accuracy. """
    model.train()
    total_loss = total_accuracy = 0.0
    steps_done = 0
    initial_steps_skip_pruning = NUM_TRAIN_STEPS_PER_TRIAL // 5
    data_iter = iter(train_loader)
    train_pbar = tqdm(range(num_train_steps), desc=f"Trial {trial.number} Train {model_type[:5].upper()}", leave=False, unit="step")
    use_amp = (device.type == 'cuda')

    for step in train_pbar:
        try:
            try: batch_cpu = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch_cpu = next(data_iter)
            batch = batch_cpu.to(device, non_blocking=True)
        except StopIteration:
            print("\n   ⚠️ WARNING: Train data exhausted prematurely. Stopping training.")
            break
        if batch.nelement() == 0: continue

        optimizer.zero_grad(set_to_none=True)
        try:
            with autocast(enabled=use_amp):
                loss, accuracy = model.compute_loss(batch, return_acc=True)

            if not torch.isfinite(loss):
                print(f"\n   ⚠️ Invalid loss ({loss.item()}) at step {step}. Skipping update.")
                if step > initial_steps_skip_pruning: raise optuna.TrialPruned(f"Invalid loss ({loss.item()}) at step {step}")
                continue

            if use_amp: scaler.scale(loss).backward()
            else: loss.backward()

            if use_amp: scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)

            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else: optimizer.step()

            current_loss = loss.item()
            total_loss += current_loss
            total_accuracy += accuracy
            steps_done += 1

            if step % 50 == 0 or step == num_train_steps - 1:
                 log_dict = {"Loss": f"{current_loss:.3f}", "Acc": f"{accuracy:.3f}"}
                 if use_amp: log_dict["Scale"] = f"{scaler.get_scale():.1f}"
                 train_pbar.set_postfix(log_dict)

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"\n   ❌ CUDA OOM at step {step}, Trial {trial.number}.")
                print(f"      Model: {model_type}, Config: {getattr(model, 'embed_dim', 'N/A')}, {getattr(model, 'depth', 'N/A')}")
                del batch; del loss; optimizer.zero_grad(set_to_none=True); gc.collect(); torch.cuda.empty_cache()
                raise optuna.TrialPruned(f"OOM at step {step}") from e
            else:
                print(f"\n   ❌ Runtime Error at step {step}, Trial {trial.number}:")
                traceback.print_exc(limit=1)
                raise optuna.TrialPruned(f"Runtime error at step {step}: {e}") from e
        except Exception as e:
            print(f"\n   ❌ Unexpected Error at step {step}, Trial {trial.number}:")
            traceback.print_exc(limit=1)
            raise optuna.TrialPruned(f"Unexpected error at step {step}: {e}") from e

        # --- Intermediate Pruning Check ---
        pruning_check_interval = 150
        if step > initial_steps_skip_pruning and (step + 1) % pruning_check_interval == 0:
             if pruning and trial.should_prune():
                  del batch; gc.collect(); torch.cuda.empty_cache()
                  raise optuna.TrialPruned(f"Pruned at step {step+1} by pruner.")
             # Optional: Quick validation check here (no reporting)

    train_pbar.close()
    avg_loss = total_loss / steps_done if steps_done > 0 else float('inf')
    avg_accuracy = total_accuracy / steps_done if steps_done > 0 else 0.0
    return float(avg_loss), float(avg_accuracy)

# --- Optuna Objective Function ---
def objective(trial: optuna.Trial, train_data: List[Tensor], eval_data: List[Tensor], fixed_seq_len: int, device: torch.device) -> Tuple[float, float]:
    """ Optuna objective: Maximize Accuracy, Minimize Parameters. """
    print(f"\n--- 🚀 Starting Trial {trial.number} ---")
    model: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scaler: Optional[GradScaler] = None
    param_count = float('inf')
    final_accuracy = 0.0
    final_loss = float('inf')
    trial_status = "FAIL"
    fail_reason = "Unknown"
    model_type = "N/A" # Initialize model_type

    try:
        model_type = trial.suggest_categorical("model_type", ["hydra", "transformer"])
        print(f"   🔬 Model Type: {model_type.upper()}")

        lr = trial.suggest_float("lr", 1e-5, 8e-4, log=True)
        embed_dim_options = [d for d in range(32, 768 + 1, 16) if d % 8 == 0]
        if not embed_dim_options: raise ValueError("No valid embed_dim options.")
        embed_dim = trial.suggest_categorical("embed_dim", embed_dim_options)
        depth = trial.suggest_int("depth", 2, 8)
        dropout_rate = trial.suggest_float("dropout_rate", 0.05, 0.25)

        print(f"   ⚙️ Common Params: LR={lr:.2e}, EmbedDim={embed_dim}, Depth={depth}, Dropout={dropout_rate:.3f}")

        model_config: Dict[str, Any] = {"vocab_size": VOCAB_SIZE}

        if model_type == "hydra":
            mlp_mult = trial.suggest_categorical("mlp_mult", [2, 3, 4])
            ssm_state_dim = trial.suggest_categorical("ssm_state_dim", [8, 16, 24, 32])
            ssm_d_conv = trial.suggest_categorical("ssm_d_conv", [3, 4, 5])
            num_diffusion_timesteps = trial.suggest_categorical("num_diffusion_timesteps", [50, 100, 150, 200])
            #noise_schedule = "cosine" # Fixed
            noise_schedule = trial.suggest_categorical("noise_schedule", ["cosine", "linear"])
            ssm_dt_rank = 'auto'     # Fixed

            print(f"   ⚙️ Hydra Params: MLP_Mult={mlp_mult}, SSM_N={ssm_state_dim}, ConvK={ssm_d_conv}, T={num_diffusion_timesteps}")

            model_config.update({
                "embed_dim": embed_dim, "depth": depth, "mlp_mult": mlp_mult, "dropout_rate": dropout_rate,
                "num_diffusion_timesteps": num_diffusion_timesteps, "noise_schedule": noise_schedule,
                "ssm_state_dim": ssm_state_dim, "ssm_d_conv": ssm_d_conv, "ssm_dt_rank": ssm_dt_rank
            })
            model = HydraScaleLM(**model_config).to(device)

        elif model_type == "transformer":
            # --- FIXED: Suggest nhead from fixed list FIRST ---
            nhead = trial.suggest_categorical("nhead", TRANSFORMER_FIXED_NHEAD_OPTIONS)

            # --- FIXED: Check compatibility AFTER suggesting nhead ---
            if embed_dim % nhead != 0:
                #raise optuna.TrialPruned(f"Incompatible combo: embed_dim={embed_dim} not divisible by nhead={nhead}. Pruning.")
                embed_dim = (embed_dim // nhead) * nhead

            ffn_mult = trial.suggest_categorical("ffn_mult", [2, 3, 4])
            dim_feedforward = embed_dim * ffn_mult
            max_seq_len_mult = trial.suggest_categorical("max_seq_len_mult", [2, 4, 8])
            max_seq_len = fixed_seq_len * max_seq_len_mult

            print(f"   ⚙️ Transformer Params: Heads={nhead}, FFN_Mult={ffn_mult} ({dim_feedforward}), MaxLen={max_seq_len}")

            model_config.update({
                 "embed_dim": embed_dim, "nhead": nhead, "num_layers": depth,
                 "dim_feedforward": dim_feedforward, "max_seq_len": max_seq_len,
                 "dropout": dropout_rate
            })
            model = SimpleTransformerLM(**model_config).to(device)

        if compile:
            model = torch.compile(model)

        param_count = count_parameters(model)
        param_count_G = param_count / 1e9
        #param_count_M = param_count / 1e6
        trial.set_user_attr("param_count_M", param_count_G)
        trial.set_user_attr("config", model_config)
        print(f"   ✅ Model Instantiated ({model.__class__.__name__}). Parameters: {param_count_G:.3f} G ({param_count:,})")

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scaler = GradScaler(enabled=(device.type == 'cuda'))

        print(f"   ⏳ Starting training ({NUM_TRAIN_STEPS_PER_TRIAL} steps)...")
        start_time = time.time()
        avg_loss, avg_accuracy = train_eval_trial(
            trial, model, train_data, eval_data, optimizer, scaler,
            num_train_steps=NUM_TRAIN_STEPS_PER_TRIAL,
            num_eval_batches=NUM_EVAL_BATCHES_PER_TRIAL, device=device, model_type=model_type
        )
        end_time = time.time()
        print(f"   ⏱️ Training finished in {end_time - start_time:.2f} seconds.")
        print(f"   📊 Final Avg Train Loss: {avg_loss:.4f}, Final Avg Train Accuracy: {avg_accuracy:.4f}")

        final_accuracy = avg_accuracy
        final_loss = avg_loss
        trial.set_user_attr("final_accuracy", final_accuracy)
        trial.set_user_attr("final_loss", final_loss)
        #trial.set_user_attr("perplexity", final_ppl)

        if np.isfinite(final_loss) and final_accuracy > 0.02:
            gen_kwargs = {"temperature": 0.8, "top_k": 50}
            generated_text_sample = generate_sample_text(
                model, tokenizer, GENERATION_PROMPT, device, model_type,
                max_new=GENERATION_MAX_NEW, gen_kwargs=gen_kwargs
            )
            generated_text_sample = generated_text_sample.replace("\n", " ").strip()
            if len(generated_text_sample) > 150: generated_text_sample = generated_text_sample[:147] + "..."
            print(f"      Sample: {generated_text_sample}")
        else:
            generated_text_sample = "[Gen Skipped]"
            print(f"   📝 Skipping text generation (Loss: {final_loss:.4f}, Acc: {final_accuracy:.4f}).")

        trial.set_user_attr("generated_sample", generated_text_sample)

        if not np.isfinite(final_accuracy):
            print("   ⚠️ Final accuracy is NaN/Inf. Reporting 0.0.")
            final_accuracy = 0.0
            fail_reason = "Invalid Accuracy"
            trial_status = "FAIL"
        elif not np.isfinite(param_count):
             print("   ⚠️ Parameter count is Inf. Reporting float('inf').")
             param_count = float('inf')
             fail_reason = "Invalid Param Count"
             trial_status = "FAIL"
        else: trial_status = "COMPLETE"

        if trial_status == "COMPLETE": print(f"--- ✅ Trial {trial.number} Complete: Acc={final_accuracy:.4f}, Params={param_count_G:.3f}G ---")
        else: print(f"--- ❌ Trial {trial.number} Failed: Reason={fail_reason}. Reporting worst values. ---")

        return float(final_accuracy), float(param_count_G)

    except optuna.TrialPruned as e:
        trial_status = "PRUNED"
        fail_reason = str(e)
        print(f"--- ✂️ Trial {trial.number} Pruned: {fail_reason} ---")
        raise e
    except Exception as e:
        trial_status = "FAIL"
        fail_reason = f"{type(e).__name__}: {e}"
        print(f"\n!!!!!!!! 💥 TRIAL {trial.number} FAILED UNEXPECTEDLY ({model_type}) 💥 !!!!!!!!")
        traceback.print_exc()
        print(f"--- ❌ Trial {trial.number} Failed (Reason: {fail_reason}). Reporting worst objectives. ---")
        return float(0.0), float('inf') # Worst values

    finally:
        print(f"   🧹 Cleaning up for Trial {trial.number} (State: {trial_status})...")
        trial.set_user_attr("trial_status", trial_status)
        if trial_status != "COMPLETE": trial.set_user_attr("fail_reason", fail_reason)
        if model is not None: del model
        if optimizer is not None: del optimizer
        if scaler is not None: del scaler
        gc.collect()
        if device.type == 'cuda': torch.cuda.empty_cache()
        print(f"   -> Cleanup complete for Trial {trial.number}.")

# --- Live Dashboard Callback & Setup ---
dashboard_data: Dict[int, Dict[str, Any]] = {}
dashboard_lock = Lock()
console = Console(width=200)

def make_dashboard_table() -> Table:
    """Creates the Rich Table for the live dashboard."""
    table = Table(title=f"🤖 Optuna Live: HydraScale vs Transformer ({DATASET_NAME}@{DEVICE.type.upper()}) 🤖",
                  caption="Goal: Top-Left (High Accuracy, Low Params)", expand=True)
    table.add_column("Trial", justify="right", style="cyan", no_wrap=True, width=5)
    table.add_column("Model", style="magenta", width=11)
    table.add_column("State", style="yellow", width=10)
    table.add_column("Acc (%)", justify="right", style="green", width=8)  # Obj 1
    table.add_column("Params (M)", justify="right", style="blue", width=10) # Obj 2
    table.add_column("Loss", justify="right", style="yellow", width=8)
    table.add_column("LR", justify="right", style="dim cyan", width=9)
    table.add_column("Config", style="dim", max_width=35, overflow="fold")
    table.add_column("Sample / Status", style="white", min_width=40, overflow="fold")
    return table

class DashboardCallback:
    """ Optuna callback to update dashboard data. """
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        with dashboard_lock:
            trial_num = trial.number
            state = trial.state.name
            accuracy_obj, params_obj = None, None
            if trial.state == optuna.trial.TrialState.COMPLETE and trial.values:
                accuracy_obj, params_obj = trial.values[0], trial.values[1]
            elif trial.state in [optuna.trial.TrialState.FAIL, optuna.trial.TrialState.PRUNED]:
                accuracy_obj = trial.user_attrs.get("final_accuracy", 0.0)
                params_m_stored = trial.user_attrs.get("param_count_M")
                params_obj = params_m_stored * 1e6 if params_m_stored is not None else float('inf')
                if trial.values and len(trial.values) == 2: # Prefer Optuna's values if available
                     accuracy_obj = trial.values[0] if trial.values[0] is not None else accuracy_obj
                     params_obj = trial.values[1] if trial.values[1] is not None else params_obj

            final_loss = trial.user_attrs.get("final_loss", float('nan'))
            params_m_display = params_obj / 1e6 if params_obj is not None and np.isfinite(params_obj) else float('nan')
            accuracy_display = accuracy_obj * 100 if accuracy_obj is not None else float('nan')

            model_type = trial.params.get("model_type", "N/A")
            lr = trial.params.get("lr", float('nan'))
            embed_dim = trial.params.get("embed_dim", "?")
            depth = trial.params.get("depth", "?")
            dropout = trial.params.get("dropout_rate", float('nan'))

            config_str = f"D={embed_dim}, L={depth}, Drp={dropout:.2f}"
            if model_type == "hydra":
                ssm_n = trial.params.get("ssm_state_dim", "?"); t_steps = trial.params.get("num_diffusion_timesteps", "?")
                convk = trial.params.get("ssm_d_conv", "?"); mlp_m = trial.params.get("mlp_mult", "?")
                config_str += f", N={ssm_n}, K={convk}, T={t_steps}, Fx{mlp_m}"
            elif model_type == "transformer":
                nhead = trial.params.get("nhead", "?"); ffn_m = trial.params.get("ffn_mult", "?")
                max_len_m = trial.params.get("max_seq_len_mult", "?")
                config_str += f", H={nhead}, Fx{ffn_m}, MaxLM={max_len_m}"

            sample_or_status = trial.user_attrs.get("generated_sample", "")
            actual_state = trial.user_attrs.get("trial_status", state)
            if actual_state == "RUNNING" and not sample_or_status: sample_or_status = "[Training...]"
            elif actual_state == "FAIL": sample_or_status = f"[Failed: {trial.user_attrs.get('fail_reason', 'Unknown')}]"
            elif actual_state == "PRUNED": sample_or_status = f"[Pruned ✂️: {trial.user_attrs.get('fail_reason', 'Pruner')}]"
            elif actual_state == "COMPLETE" and not sample_or_status:
                 is_low_perf = not np.isfinite(final_loss) or (accuracy_obj is not None and accuracy_obj <= 0.02)
                 sample_or_status = "[Gen Skipped]" if is_low_perf else "[No Sample/Empty]"

            dashboard_data[trial_num] = {
                "model": model_type.upper(), "state": actual_state, "accuracy": accuracy_display,
                "params": params_m_display, "loss": final_loss, "lr": lr,
                "config": config_str, "sample": sample_or_status,
            }

# --- Main Execution Block ---
if __name__ == "__main__":
    print("\n" + "=" * 80)
    print(" ✨ Initializing Edge Revolution Optimization ✨ ".center(80, "="))
    print("=" * 80)
    print(f"▶ Device: {DEVICE.type.upper()}", f"▶ Dataset: {DATASET_NAME} ({DATASET_CONFIG})")
    if DEVICE.type == 'cuda': print(f"  ▶ CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"▶ Tokenizer: {TOKENIZER_NAME} (Vocab: {VOCAB_SIZE})")
    print(f"▶ Train Params: SeqLen={SEQ_LEN}, Batch={BATCH_SIZE}, GradClip={GRAD_CLIP_NORM}")
    print(f"▶ Data: Train Batches={NUM_DATA_LOAD_BATCHES}, Eval Batches={NUM_DATA_LOAD_BATCHES // 2}")
    print(f"▶ Optuna: Trials={OPTUNA_N_TRIALS}, Steps/Trial={NUM_TRAIN_STEPS_PER_TRIAL}")
    print(f"▶ Objectives: [Maximize Accuracy (Train Avg), Minimize Parameters]")
    print("-" * 80)

    print("⏳ Loading and preparing data...")
    start_data_time = time.time()
    train_data = prepare_data(DATASET_NAME, DATASET_CONFIG, tokenizer, SEQ_LEN, NUM_DATA_LOAD_BATCHES, BATCH_SIZE, split="train")
    eval_data = prepare_data(DATASET_NAME, DATASET_CONFIG, tokenizer, SEQ_LEN, NUM_DATA_LOAD_BATCHES // 2, BATCH_SIZE, split="validation")
    end_data_time = time.time()
    if not train_data or not eval_data:
        print("\n❌ FATAL: Failed to load data. Exiting.")
        exit(1)
    print(f"✅ Data loading complete in {end_data_time - start_data_time:.2f} seconds.")
    print("-" * 80)

    print("🛠️ Setting up Optuna study...")
    sampler = optuna.samplers.NSGAIISampler(population_size=30, mutation_prob=0.1, crossover_prob=0.9, seed=SEED)

    if pruning:
        pruning_interval = 150
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=NUM_TRAIN_STEPS_PER_TRIAL // 4, interval_steps=pruning_interval)
    else:
        pruner = None

    study_name = "Edge_Revolution_v3_5_Fixed" # Incremented version
    try:
        study = optuna.create_study(
            directions=["maximize", "minimize"], study_name=study_name,
            sampler=sampler,
            pruner=pruner,
            # storage=f"sqlite:///{study_name}.db", load_if_exists=True
        )
    except Exception as e:
        print(f"❌ Error creating/loading Optuna study '{study_name}': {e}")
        exit(1)

    callback = DashboardCallback()
    print(f"🏁 Starting Optuna optimization ({OPTUNA_N_TRIALS} trials)...")
    live_table = make_dashboard_table()
    try:
        with Live(live_table, console=console, refresh_per_second=1.0, vertical_overflow="visible") as live:
            objective_func = lambda trial: objective(trial, train_data, eval_data, SEQ_LEN, DEVICE)
            study.optimize( objective_func, n_trials=OPTUNA_N_TRIALS, callbacks=[callback],
                            gc_after_trial=True, show_progress_bar=False )

            final_table = make_dashboard_table()
            with dashboard_lock:
                 sorted_trial_nums = sorted(dashboard_data.keys())
                 for trial_num in sorted_trial_nums:
                      data = dashboard_data[trial_num]
                      state_color = {"COMPLETE": "green", "FAIL": "red", "PRUNED": "magenta"}.get(data.get('state'), "yellow")
                      acc_str = f"{data['accuracy']:.2f}" if data.get('accuracy') is not None and np.isfinite(data['accuracy']) else "N/A"
                      params_str = f"{data['params']:.3f}" if data.get('params') is not None and np.isfinite(data['params']) else "N/A"
                      loss_str = f"{data['loss']:.3f}" if data.get('loss') is not None and np.isfinite(data['loss']) else "N/A"
                      lr_str = f"{data['lr']:.1e}" if data.get('lr') and np.isfinite(data['lr']) else "N/A"
                      final_table.add_row(
                          str(trial_num), data.get('model', 'N/A'), f"[{state_color}]{data.get('state', 'UNK')}[/]",
                          acc_str, params_str, loss_str, lr_str, data.get('config', 'N/A'), data.get('sample', '')
                      )
            live.update(final_table)
            console.print("\n✨ Final dashboard. Pausing...")
            time.sleep(5)

    except KeyboardInterrupt: console.print("\n\n[bold yellow]🛑 Optimization interrupted by user.[/bold yellow]")
    except Exception as e: console.print(f"\n\n[bold red]💥 Optimization loop error:[/bold red] {e}"); traceback.print_exc()
    finally: console.print("\n✨ Live display stopped / Optimization loop finished.")

    print("\n" + "=" * 80); print("📊 Optimization Analysis 📊".center(80, "="))
    print(f"Study: {study.study_name}, Trials Run: {len(study.trials)}")
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    print(f"  ✅ Completed: {len(complete_trials)}, ✂️ Pruned: {len(pruned_trials)}, ❌ Failed: {len(failed_trials)}")

    if not complete_trials:
        print("\n[bold red]No trials completed successfully.[/bold red]")
        if failed_trials:
             print("\n[yellow]Failed trials:[/yellow]")
             for ft in failed_trials[:5]: print(f"  T{ft.number}: {ft.user_attrs.get('fail_reason', 'Unknown')}")
    else:
        print("\n" + "-" * 30 + " Pareto Front Trials " + "-" * 30)
        print("🌟 Best Accuracy vs. Parameter trade-offs:")
        try:
            pareto_trials = sorted(study.best_trials, key=lambda t: t.values[1])
        except ValueError as e:
             console.print(f"[yellow]Warning: Could not get best trials ({e}). Filtering manually.[/yellow]")
             pareto_trials = optuna.visualization._pareto_front._get_pareto_front_trials_by_trials(complete_trials, study.directions)
             pareto_trials = sorted(pareto_trials, key=lambda t: t.values[1]) if pareto_trials else []

        if not pareto_trials: console.print("[yellow]No Pareto optimal trials identified.[/yellow]")
        else:
            for trial in pareto_trials:
                data = dashboard_data.get(trial.number, {})
                acc_disp = data.get('accuracy', float('nan')); params_disp = data.get('params', float('nan'))
                params_raw = trial.values[1] if trial.values and len(trial.values) > 1 else float('nan')
                model_type_disp = data.get('model', 'N/A'); lr_disp = data.get('lr', float('nan'))
                config_disp = data.get('config', 'N/A'); sample_disp = data.get('sample', '[N/A]')
                console.print(f"\n[cyan bold]  Trial {trial.number}[/] ([magenta]{model_type_disp}[/])")
                console.print(f"    [green]Acc = {acc_disp:.3f}%[/], [blue]Params = {params_disp:.3f} M[/] ({int(params_raw):,})")
                console.print(f"    [dim]Config: LR={lr_disp:.2e}, {config_disp}[/]")
                console.print(f"    [white]Sample: {sample_disp}[/]")

    print("\n" + "-" * 80)
    try: # Visualizations
        if _plotly_available and complete_trials:
            console.print("\n📈 Generating plots (requires plotly)...")
            target_names = ["Accuracy (%)", "Parameters (M)"]
            def targets_format(t):
                 acc = t.values[0]*100 if t.values and t.values[0] is not None else float('nan')
                 p = t.values[1]/1e6 if t.values and t.values[1] is not None else float('nan')
                 return acc, p
            try: # Pareto Plot
                fig1 = plot_pareto_front(study, targets=targets_format, target_names=target_names)
                #    color_axis={"name": "Model Type", "values": [t.params.get("model_type","N/A").upper() for t in complete_trials],
                #                "colorscale": [['HYDRA', '#1f77b4'], ['TRANSFORMER', '#ff7f0e']]}
                fig1.update_layout(title="Pareto Front: Accuracy vs. Parameters")
                fig1.show(); console.print("   ✅ Pareto Front plot generated.")
                # try: fig1.write_image(f"{study_name}_pareto.png", scale=2) except Exception: pass
            except Exception as e_p: console.print(f"   ⚠️ Pareto plot failed: {e_p}"); traceback.print_exc(limit=1)
            try: # Importance Plot (Accuracy)
                fig2 = plot_param_importances(study, target=lambda t: t.values[0] if t.state == optuna.trial.TrialState.COMPLETE and t.values else None, target_name="Accuracy")
                fig2.update_layout(title="Hyperparameter Importance for Accuracy"); fig2.show(); console.print("   ✅ Importance plot (Accuracy) generated.")
                # try: fig2.write_image(f"{study_name}_importance_acc.png", scale=2) except Exception: pass
            except Exception as e_i1: console.print(f"   ⚠️ Importance (Acc) plot failed: {e_i1}")
            try: # Importance Plot (Params)
                 fig3 = plot_param_importances(study, target=lambda t: t.values[1]/1e6 if t.state == optuna.trial.TrialState.COMPLETE and t.values and t.values[1] is not None else None, target_name="Parameters (M)")
                 fig3.update_layout(title="Hyperparameter Importance for Parameters (M)"); fig3.show(); console.print("   ✅ Importance plot (Params) generated.")
                 # try: fig3.write_image(f"{study_name}_importance_params.png", scale=2) except Exception: pass
            except Exception as e_i2: console.print(f"   ⚠️ Importance (Params) plot failed: {e_i2}")
        else: console.print("\n📈 Skipping plots (Plotly unavailable or no completed trials).")
    except Exception as e: console.print(f"\n[yellow]⚠️ Plot generation error:[/yellow] {e}"); traceback.print_exc(limit=1)

    print("\n" + "=" * 80); console.print(" ✨ Optimization Complete. ✨ ".center(80)); print("=" * 80 + "\n")