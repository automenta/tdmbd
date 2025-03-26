#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✨ EDGE REVOLUTION: ThunderHub v1.0 vs Transformer ✨

Objective: Achieve Maximum Awesomeness. Leverage ThunderHub's design
           (inspired by RWKV/RetNet with dynamic noise masking)
           to challenge Transformers, parameter-for-parameter, focusing on Edge AI capabilities.

Methodology:
- ThunderHub Model: Uses a custom recurrent core (`ThunderCore`) with dynamic masking (`NoiseBlaster`).
    - Training: Mask-Predict Objective with dynamically predicted noise levels.
                The model learns to predict original tokens/embeddings given a dynamically masked sequence.
    - Inference: Efficient autoregressive generation.
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
NUM_EVAL_BATCHES_PER_TRIAL = 30     # Batches for validation check during training (mainly for pruning)

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

        return result
    # --- End of CORRECTED group_texts function ---

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


# --- ThunderHub Components ---
class ThunderCore(nn.Module):
    """ Simplified recurrent core, inspired by RWKV/RetNet concepts. """
    def __init__(self, embed_dim: int, state_dim: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        # Projects input embedding to combined embedding and state input
        self.input_proj = nn.Linear(embed_dim, embed_dim + state_dim)
        # Gate controlling state update vs. new state input
        self.gate = nn.Linear(embed_dim + state_dim, state_dim)
        # Projects combined embedding and final state to output embedding
        self.output_proj = nn.Linear(embed_dim + state_dim, embed_dim)

    def forward(self, x: torch.Tensor, state: Optional[Tensor] = None) -> Tuple[Tensor, Tensor]:
        """
        Processes a sequence recurrently.
        Can also run in single-step mode if state is provided.
        """
        batch, seq_len, _ = x.shape

        if state is None:
            # Initialize state if not provided (start of sequence processing)
            state = torch.zeros(batch, self.state_dim, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :] # Current token embedding [B, D]

            # Project input and split into embedding part and state input part
            combined = self.input_proj(x_t)  # [B, D + S]
            embed, state_in = combined.split([self.embed_dim, self.state_dim], dim=-1) # [B, D], [B, S]

            # Compute gate based on the combined projection
            gate = torch.sigmoid(self.gate(combined))  # [B, S]

            # Update state: gate * old_state + (1 - gate) * new_state_input
            state = gate * state + (1 - gate) * state_in # [B, S]

            # Compute output based on embedding part and current state
            out = self.output_proj(torch.cat([embed, state], dim=-1)) # [B, D]
            outputs.append(out)

        # Stack outputs along the sequence dimension
        output_seq = torch.stack(outputs, dim=1)  # [B, L, D]
        return output_seq, state # Return full output sequence and final state


class NoiseBlaster(nn.Module):
    """ Predicts a masking probability based on input embeddings. """
    def __init__(self, embed_dim: int):
        super().__init__()
        # Simple MLP to predict a single value (logit for sigmoid) per token
        self.noise_predictor = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ Predicts mask probability for each token in the sequence. """
        # Input x: [B, L, D] (token embeddings)
        mask_logits = self.noise_predictor(x) # [B, L, 1]
        # Apply sigmoid to get probability between 0 and 1
        mask_prob = torch.sigmoid(mask_logits)  # [B, L, 1]
        return mask_prob.squeeze(-1)  # [B, L]


class ThunderHubLM(nn.Module):
    """
    ThunderHub Language Model: Combines ThunderCore recurrence with NoiseBlaster dynamic masking.

    Architecture:
    1. Token Embedding
    2. NoiseBlaster (predicts mask probability from embeddings)
    3. Dynamic Masking
    4. Stack of ThunderCore Layers
    5. Final Layer Normalization
    6. Classification Head (predicts original token)
    7. Reconstruction Head (predicts original embedding - optional loss term)

    Training: Mask-Predict Objective with dynamic noise.
    Inference: Autoregressive Generation (standard sampling).
    """
    def __init__(self,
                 vocab_size: int,      # Original vocabulary size from tokenizer
                 embed_dim: int = 256, # Model's main embedding dimension (D)
                 depth: int = 4,       # Number of ThunderCore layers
                 core_state_dim: int = 8 # State dimension within each ThunderCore
                ):
        super().__init__()
        self.vocab_size = vocab_size # Store original vocab size
        self.embed_dim = embed_dim
        self.depth = depth
        # Define a unique ID for the [MASK] token (outside the regular vocab)
        self.mask_token_id = self.vocab_size

        # --- Model Layers ---
        # Embedding layer: Includes original vocab + 1 for the [MASK] token
        self.token_embedding = nn.Embedding(self.vocab_size + 1, embed_dim)
        # Use default initialization or apply custom:
        nn.init.normal_(self.token_embedding.weight, std=0.02)

        # ThunderCore stack
        # Each layer needs its own state during generation
        self.layers = nn.ModuleList([
            ThunderCore(embed_dim, state_dim=core_state_dim)
            for _ in range(depth)
        ])
        # Final normalization
        self.norm = nn.LayerNorm(embed_dim)

        # NoiseBlaster for dynamic masking probability prediction
        self.noise_blaster = NoiseBlaster(embed_dim)

        # Output Heads
        # Predicts the *original* token ID from the final hidden state
        self.class_head = nn.Linear(embed_dim, self.vocab_size)
        # Optional: Predicts the *original* token embedding (for reconstruction loss)
        self.recon_head = nn.Linear(embed_dim, embed_dim)

        # --- Initialization & Info ---
        print(f"⚡ ThunderHubLM Initialized ⚡ (D={embed_dim}, Depth={depth}, CoreS={core_state_dim}). Vocab: {self.vocab_size}.")

    def _mask_tokens(self, x_0: torch.Tensor, mask_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Creates the masked input x_t using predicted probabilities. """
        batch_size, seq_len = x_0.shape
        # mask_prob shape: [B, L]

        # Generate random noise in [0, 1) for masking decision
        noise = torch.rand(batch_size, seq_len, device=x_0.device)
        # Determine mask locations: where noise < probability AND token is not padding
        is_padding = (x_0 == tokenizer.pad_token_id)
        mask = (noise < mask_prob) & (~is_padding)

        # Replace masked locations in x_0 with the mask_token_id to create x_t
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t, mask # Return masked input and the boolean mask

    def forward(self, x_t: torch.Tensor, layer_states: Optional[List[Tensor]] = None) -> Tuple[torch.Tensor, torch.Tensor, List[Tensor]]:
        """
        Forward pass through the ThunderHub model.
        Can operate in sequence mode (layer_states=None) or single-step mode.

        Returns:
            logits: Classification logits [B, L, V] or [B, V]
            recon: Reconstruction embeddings [B, L, D] or [B, D]
            new_layer_states: List of final states for each layer [List[Tensor[B, S]]]
        """
        # 1. Embed the input tokens (which might include mask tokens)
        h = self.token_embedding(x_t) # Shape: [B, L, D] or [B, D] if L=1

        # Initialize states if not provided (sequence mode)
        if layer_states is None:
            # Need B and device info from h
            batch_size = h.shape[0]
            device = h.device
            dtype = h.dtype
            layer_states = [
                torch.zeros(batch_size, layer.state_dim, device=device, dtype=dtype)
                for layer in self.layers
            ]

        new_layer_states = []
        # 2. Pass through the stack of ThunderCore layers, updating states
        for i, layer in enumerate(self.layers):
            # Pass current hidden state 'h' and the state for this layer
            h, current_state = layer(h, state=layer_states[i])
            # Store the updated state for this layer
            new_layer_states.append(current_state)

        # 3. Final Layer Normalization
        h = self.norm(h) # Shape: [B, L, D] or [B, D]

        # 4. Compute output heads
        logits = self.class_head(h) # Shape: [B, L, V] or [B, V]
        recon = self.recon_head(h)  # Shape: [B, L, D] or [B, D]

        return logits, recon, new_layer_states

    def compute_loss(self, x_0: Tensor, return_acc: bool = True) -> Union[Tensor, Tuple[Tensor, float]]:
        """ Computes the combined training loss (CE + Reconstruction). """
        batch_size, seq_len = x_0.shape
        device = x_0.device

        # Handle empty batch case
        if batch_size == 0 or x_0.nelement() == 0:
            loss = torch.tensor(0.0, device=x_0.device, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            return (loss, 1.0) if return_acc else loss

        # --- Dynamic Masking ---
        # 1. Get original token embeddings
        with autocast(enabled=False): # Embeddings might be sensitive to low precision
             embed_0 = self.token_embedding(x_0).float() # Use float32 for noise prediction stability?

        # 2. Predict mask probability using NoiseBlaster
        # Detach embeddings before feeding to NoiseBlaster? Avoids direct gradient path
        # from loss back through noise predictor, making noise prediction more stable?
        # Or allow gradients? Let's allow gradients for now.
        mask_prob = self.noise_blaster(embed_0) # Shape [B, L]

        # 3. Create masked input x_t based on predicted probabilities
        x_t, mask = self._mask_tokens(x_0, mask_prob) # `mask` is True where x_0 was masked

        # --- Forward Pass & Loss Calculation ---
        # 4. Perform forward pass with the masked input x_t (sequence mode, no state needed)
        logits, recon, _ = self.forward(x_t, layer_states=None) # Shapes [B, L, V], [B, L, D]

        # 5. Select predictions and targets ONLY at the masked positions
        masked_logits = logits[mask]      # Shape [NumMasked, V]
        masked_targets = x_0[mask]        # Shape [NumMasked]
        masked_recon = recon[mask]        # Shape [NumMasked, D]
        masked_embed_targets = embed_0[mask] # Shape [NumMasked, D] (Original embeddings)

        # 6. Compute Losses
        if masked_targets.numel() == 0:
            # If no tokens were masked (e.g., noise predictor predicted 0 prob, or batch was all padding)
            loss = torch.tensor(0.0, device=device, dtype=logits.dtype, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            accuracy = 1.0 # Accuracy is 100% if no targets
        else:
            # Classification Loss (Cross-Entropy) on masked tokens
            ce_loss = F.cross_entropy(masked_logits, masked_targets)
            # Reconstruction Loss (MSE) between predicted embeddings and original embeddings at masked positions
            recon_loss = F.mse_loss(masked_recon, masked_embed_targets)
            # Combine losses (tune weights: 0.7 CE, 0.3 Recon is a starting point)
            loss = 0.7 * ce_loss + 0.3 * recon_loss

            # Calculate accuracy on masked tokens (optional)
            accuracy = 0.0
            if return_acc:
                with torch.no_grad():
                    correct_preds = (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
                    accuracy = correct_preds / masked_targets.numel()

        # Handle potential NaN/Inf loss
        if torch.isnan(loss) or torch.isinf(loss):
            warnings.warn("🤯 NaN or Inf loss detected in ThunderHub compute_loss. Replacing with zero loss.", RuntimeWarning)
            loss = torch.tensor(0.0, device=logits.device, dtype=logits.dtype, requires_grad=True)
            loss = loss * sum(p.sum() for p in self.parameters() if p.requires_grad) * 0.0
            accuracy = 0.0

        return (loss, accuracy) if return_acc else loss

    @torch.no_grad()
    def generate(self,
                 prompt_ids: torch.Tensor, # Input prompt token IDs, Shape: [B, L_prompt]
                 num_tokens_to_generate: int,
                 temperature: float = 0.8,
                 # top_k: Optional[int] = None # Add top_k later if needed
                ) -> torch.Tensor:
        """ Generates text autoregressively using the recurrent step capability. """
        self.eval() # Set model to evaluation mode
        batch_size, prompt_len = prompt_ids.shape
        generated_ids = prompt_ids.clone().to(DEVICE) # Start with the prompt

        # Initialize layer states based on the prompt
        layer_states: Optional[List[Tensor]] = None
        if prompt_len > 0:
             # Process the prompt through the model to get the initial states
             # We only care about the final states, not the logits/recon during this warm-up
             _, _, layer_states = self.forward(generated_ids, layer_states=None)
        else:
             # If prompt is empty, initialize zero states
             layer_states = [
                 torch.zeros(batch_size, layer.state_dim, device=DEVICE, dtype=self.token_embedding.weight.dtype)
                 for layer in self.layers
             ]

        # Autoregressive generation loop
        current_token_ids = generated_ids[:, -1:] # Start with the last token of the prompt

        gen_iterator = range(num_tokens_to_generate)
        # gen_iterator = tqdm(range(num_tokens_to_generate), desc="Thunder Gen Steps", leave=False, total=num_tokens_to_generate, unit="token")

        for _ in gen_iterator:
            # Use mixed precision for generation if on CUDA
            with autocast(enabled=(DEVICE.type == 'cuda')):
                # Forward pass for a single step (L=1) using current token and previous states
                # Input shape: [B, 1]
                logits_step, _, new_layer_states = self.forward(current_token_ids, layer_states=layer_states)
                # Logits shape: [B, 1, V] -> Squeeze to [B, V]
                next_token_logits = logits_step.squeeze(1)

            # Update layer states for the next step
            layer_states = new_layer_states

            # --- Sampling Logic ---
            # Apply temperature scaling
            if temperature > 0 and temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply Top-K or other sampling if needed here (e.g., using top_k parameter)
            # ...

            # Convert logits to probabilities via Softmax
            probs = F.softmax(next_token_logits, dim=-1) # Shape [B, V]

            # Sample the next token ID
            if temperature <= 0: # Greedy decoding
                next_token_id = torch.argmax(probs, dim=-1, keepdim=True) # Shape [B, 1]
            else:
                # Handle potential numerical issues (e.g., NaNs if temp is too high/low)
                probs = torch.nan_to_num(probs, nan=0.0)
                # Sample from the multinomial distribution
                try:
                     next_token_id = torch.multinomial(probs, num_samples=1) # Shape [B, 1]
                except RuntimeError as e:
                     print(f"⚠️ Multinomial sampling error: {e}. Probabilities sum: {probs.sum(-1)}. Using argmax.")
                     next_token_id = torch.argmax(probs, dim=-1, keepdim=True)


            # Append the predicted token ID to the generated sequence
            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

            # Update the current token for the next iteration
            current_token_ids = next_token_id

        self.train() # Return model to training mode
        return generated_ids


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
        if self.effective_vocab_size > 0 and src.max() >= self.token_embedding.num_embeddings:
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
        try:
            loss = F.cross_entropy(
                logits.view(-1, self.effective_vocab_size),
                tgt.view(-1),
                ignore_index=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
            )
        except IndexError as e:
             print(f"Error in CE Loss: {e}")
             print(f"Logits shape: {logits.shape}, Reshaped: {logits.view(-1, self.effective_vocab_size).shape}")
             print(f"Targets shape: {tgt.shape}, Reshaped: {tgt.view(-1).shape}")
             print(f"Vocab size: {self.effective_vocab_size}")
             print(f"Max target ID: {tgt.max().item()}")
             raise e


        # Calculate accuracy (optional)
        accuracy = 0.0
        if return_acc:
            with torch.no_grad():
                # Consider only non-padding target tokens for accuracy
                valid_targets_mask = (tgt.view(-1) != (tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100))
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
                # Check for rows where all probabilities are zero or near-zero
                zero_rows_mask = (row_sums <= 1e-9)
                # If a row sums to zero, replace with uniform distribution to allow sampling
                uniform_probs = torch.full_like(probs, 1.0 / self.effective_vocab_size) if self.effective_vocab_size > 0 else torch.zeros_like(probs)
                probs = torch.where(zero_rows_mask, uniform_probs, probs)
                # Re-normalize probabilities to ensure they sum to 1 after potential corrections
                probs_sum = probs.sum(dim=-1, keepdim=True)
                probs = torch.where(probs_sum > 1e-9, probs / (probs_sum + 1e-9), uniform_probs) # Avoid div by zero

                # Sample from the multinomial distribution
                try:
                     next_token_id = torch.multinomial(probs, num_samples=1) # Shape [B, 1]
                except RuntimeError as e:
                     print(f"⚠️ Multinomial sampling error: {e}. Probabilities sum: {probs.sum(-1)}. Using argmax.")
                     next_token_id = torch.argmax(probs, dim=-1, keepdim=True)


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
        default_kwargs = {"temperature": 0.8}
        # Add top_k default for models that use it (like Transformer)
        if model_type == "transformer":
            default_kwargs["top_k"] = 50

        current_gen_kwargs = default_kwargs.copy()

        # Apply user-provided kwargs, overriding defaults
        current_gen_kwargs.update(gen_kwargs)

        # Select the generation function
        gen_func = model.generate

        # Prepare arguments for the specific model's generate function
        # ThunderHub specific arguments (currently only temperature is used from defaults/kwargs)
        if model_type == "thunderhub":
            # Filter kwargs to only those accepted by ThunderHub.generate
            accepted_args = ["temperature"]
            call_kwargs = {k: v for k, v in current_gen_kwargs.items() if k in accepted_args}
        # Transformer specific arguments (temperature, top_k)
        elif model_type == "transformer":
            accepted_args = ["temperature", "top_k"]
            call_kwargs = {k: v for k, v in current_gen_kwargs.items() if k in accepted_args}
        else:
            return f"[Error: Invalid model_type '{model_type}']"


        # Generate using mixed precision on CUDA for speed
        with autocast(enabled=(device.type == 'cuda')):
            output_ids = gen_func(prompt_ids, num_tokens_to_generate=max_new, **call_kwargs)

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
                del batch
                if 'loss' in locals(): del loss # Check if loss exists before deleting
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

# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, train_data: List[Tensor], eval_data: List[Tensor], fixed_seq_len: int, device: torch.device) -> Tuple[float, float]:
    """
    Optuna objective function for optimizing ThunderHub vs Transformer.
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
        model_type = trial.suggest_categorical("model_type", ["thunderhub", "transformer"])
        print(f"   🔬 Model Type: {model_type.upper()}")

        # --- Suggest Common Hyperparameters ---
        lr = trial.suggest_float("lr", 1e-5, 8e-4, log=True) # Learning rate range
        embed_dim_options = [d for d in range(128, 768 + 1, 64) if d % 8 == 0] # Divisible by 8 for heads
        embed_dim = trial.suggest_categorical("embed_dim", embed_dim_options)
        depth = trial.suggest_int("depth", 2, 8) # Number of layers

        # --- Define FIXED choices for nhead (independent of embed_dim) ---
        # These are all the head counts we might *potentially* want to test
        nhead_choices = [2, 4, 8, 12, 16]

        print(f"   ⚙️ Common Params: LR={lr:.2e}, EmbedDim={embed_dim}, Depth={depth}")

        model_config = {"vocab_size": VOCAB_SIZE} # Start building config dict

        # --- Model Specific Hyperparameters & Instantiation ---
        if model_type == "thunderhub":
            core_state_dim = trial.suggest_categorical("core_state_dim", [4, 8, 12, 16]) # State size in ThunderCore
            print(f"   ⚙️ ThunderHub Params: CoreStateDim={core_state_dim}")
            model_config.update({
                "embed_dim": embed_dim,
                "depth": depth,
                "core_state_dim": core_state_dim,
            })
            model = ThunderHubLM(**model_config).to(device)

        elif model_type == "transformer":
            # --- Suggest nhead from the FIXED list ---
            nhead = trial.suggest_categorical("nhead", nhead_choices)

            # --- Validate the embed_dim / nhead combination and PRUNE if invalid ---
            if embed_dim % nhead != 0:
                raise optuna.TrialPruned(
                    f"Incompatible combo: embed_dim={embed_dim} not divisible by nhead={nhead}. Pruning."
                )
            # Optional: Add check if embed_dim is too small for nhead, though unlikely with current ranges
            if embed_dim < nhead:
                 raise optuna.TrialPruned(
                     f"Incompatible combo: embed_dim={embed_dim} < nhead={nhead}. Pruning."
                 )

            # --- Suggest other Transformer parameters ---
            ffn_mult = trial.suggest_categorical("ffn_mult", [2, 3, 4, 6])
            dim_feedforward = embed_dim * ffn_mult
            dropout = trial.suggest_float("dropout", 0.05, 0.25)
            max_seq_len = fixed_seq_len * 4 # Allow longer context than training length

            print(f"   ⚙️ Transformer Params: Heads={nhead}, FFN_Mult={ffn_mult} ({dim_feedforward}), Dropout={dropout:.3f}")

            model_config.update({
                 "embed_dim": embed_dim, "nhead": nhead, "num_layers": depth,
                 "dim_feedforward": dim_feedforward, "max_seq_len": max_seq_len,
                 "dropout": dropout
            })
            model = SimpleTransformerLM(**model_config).to(device)

        # --- Calculate Parameter Count (Objective 2) ---
        param_count = count_parameters(model) # Get raw parameter count
        param_count_M = param_count / 1e6
        trial.set_user_attr("param_count_M", param_count_M) # Store Millions for display
        trial.set_user_attr("config", model_config) # Store full config for review
        print(f"   ✅ Model Instantiated ({model.__class__.__name__}). Parameters: {param_count_M:.3f} M ({param_count:,})")

        # --- Training & Evaluation ---
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        # Corrected GradScaler initialization for newer PyTorch versions (use torch.amp)
        # Older: scaler = GradScaler(enabled=(device.type == 'cuda'))
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda')) # Stick with torch.cuda.amp if using older imports

        print(f"   ⏳ Starting training ({NUM_TRAIN_STEPS_PER_TRIAL} steps)...")
        start_time = time.time()
        avg_loss, avg_accuracy = train_eval_trial(
            trial, model, train_data, eval_data, optimizer, scaler,
            num_train_steps=NUM_TRAIN_STEPS_PER_TRIAL,
            num_eval_batches=NUM_EVAL_BATCHES_PER_TRIAL,
            device=device, model_type=model_type
        )
        end_time = time.time()
        print(f"   ⏱️ Training finished in {end_time - start_time:.2f} seconds.")
        print(f"   📊 Final Avg Train Loss: {avg_loss:.4f}, Final Avg Train Accuracy: {avg_accuracy:.4f}")

        # --- Final Accuracy (Objective 1) ---
        final_accuracy = avg_accuracy
        trial.set_user_attr("final_accuracy", final_accuracy)
        trial.set_user_attr("final_loss", avg_loss)

        # --- Generate Text Sample (Qualitative Check) ---
        generated_text_sample = "[Generation Skipped (Low Acc)]"
        if final_accuracy > 0.02 and not np.isnan(final_accuracy):
            print("   📝 Generating text sample...")
            gen_kwargs = {"temperature": 0.8}
            if model_type == "transformer":
                gen_kwargs["top_k"] = 50
            generated_text_sample = generate_sample_text(
                model, tokenizer, GENERATION_PROMPT, device, model_type,
                max_new=GENERATION_MAX_NEW, gen_kwargs=gen_kwargs
            )
            generated_text_sample = generated_text_sample.replace("\n", " ").strip()
            if len(generated_text_sample) > 150:
                generated_text_sample = generated_text_sample[:147] + "..."
            print(f"      Sample: {generated_text_sample}")
        else:
             print(f"   📝 Skipping text generation due to low/NaN accuracy ({final_accuracy:.4f}).")

        trial.set_user_attr("generated_sample", generated_text_sample)

        # --- Return Objectives: (Accuracy, Parameter Count) ---
        if np.isnan(final_accuracy):
            print("   ⚠️ Warning: Final accuracy is NaN. Reporting 0.0 to Optuna.")
            final_accuracy = 0.0

        print(f"--- ✅ Trial {trial.number} Complete: Acc={final_accuracy:.4f}, Params={param_count_M:.3f}M ---")
        return float(final_accuracy), float(param_count)

    # --- Error Handling & Pruning ---
    except optuna.TrialPruned as e:
        print(f"--- ✂️ Trial {trial.number} Pruned: {e} ---")
        # Store prune reason for dashboard
        trial.set_user_attr("fail_reason", str(e))
        raise e
    except Exception as e:
        print(f"\n!!!!!!!! 💥 TRIAL {trial.number} FAILED 💥 !!!!!!!!")
        traceback.print_exc()
        print(f"--- ❌ Trial {trial.number} Failed. Reporting worst objective values. ---")
        # Store fail reason for dashboard
        trial.set_user_attr("fail_reason", type(e).__name__)
        final_accuracy = 0.0
        param_count = float('inf')
        return float(final_accuracy), float(param_count)

    # --- Resource Cleanup (Ensures GPU memory is freed) ---
    finally:
        print(f"   🧹 Cleaning up resources for Trial {trial.number}...")
        del model, optimizer, scaler
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        print(f"   -> Cleanup complete for Trial {trial.number}.")


# --- Live Dashboard Callback & Setup ---
dashboard_data: Dict[int, Dict[str, Any]] = {} # Shared dict for dashboard state
dashboard_lock = Lock() # Mutex for thread-safe access to dashboard_data
# console = Console(width=200) # Rich console instance
console = Console() # Use automatic width detection

def make_dashboard_table() -> Table:
    """Creates the Rich Table object for the live dashboard display."""
    table = Table(title=f"🤖 Optuna Live: ThunderHub vs Transformer ({DATASET_NAME}@{DEVICE}) 🤖", # Updated title
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
            if model_type == "thunderhub": # Changed from "hydra"
                core_s = trial.params.get("core_state_dim", "?")
                config_str += f", CoreS={core_s}"
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
                 fail_reason = trial.user_attrs.get('fail_reason', 'Unknown') # Optuna doesn't automatically store this
                 # Check system message if available
                 sys_msg = next((log.message for log in study.get_trials(deepcopy=False) if log.number == trial.number and log.system_attrs.get('fail_reason')), None)
                 if sys_msg and not fail_reason: fail_reason = sys_msg

                 # Simple truncate if reason is long
                 if len(fail_reason) > 50: fail_reason = fail_reason[:47] + "..."
                 sample_or_status = f"[Failed: {fail_reason}]"

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
            study_name="Edge_Revolution_ThunderHub_v1", # Updated study name
            sampler=sampler,
            pruner=pruner,
            # Optional: Use persistent storage (SQLite DB) to resume study
            # storage="sqlite:///edge_revolution_thunder_optuna.db",
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
                      if trial_num not in dashboard_data: continue # Check if key exists
                      data = dashboard_data[trial_num]
                      # Format data for display
                      state_color = "green" if data['state'] == "COMPLETE" else ("red" if data['state'] in ["FAIL", "PRUNED"] else "yellow")
                      acc_str = f"{data['accuracy']:.2f}" if data['accuracy'] is not None else "N/A"
                      params_str = f"{data['params']:.3f}" if data['params'] is not None else "N/A"
                      lr_str = f"{data.get('lr'):.1e}" if data.get('lr') and not np.isnan(data['lr']) else "N/A"
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
            # Safely get config string from dashboard data (might not be populated if callback failed)
            config_str = dashboard_data.get(trial.number, {}).get('config', 'N/A')

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
                # Consider adding coloring by model type if Plotly version supports it well
                # color_axis = {'name': 'Model Type', ...} # Check Optuna docs for exact syntax
            )
            fig1.update_layout(title="Pareto Front: Accuracy vs. Parameters (ThunderHub vs Transformer)")
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
    console.print(" ✨ Optimization Complete. ThunderHub enters the ring! ✨ ".center(80))
    print("=" * 80 + "\n")