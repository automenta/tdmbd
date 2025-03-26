#!/usr/bin/env python3
"""
HydraScale V3.2 & Transformer Comparison with Optuna Optimization

Modularized code, uses WikiText-2 dataset with a GPT2 tokenizer,
and performs multi-objective Optuna optimization (accuracy vs. parameter count).
Includes a live updating dashboard with decoded text samples.
"""
import gc  # Garbage collector
import math
import time
import warnings
from threading import Lock
from typing import Optional, Tuple, List, Dict, Any, Union

import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from optuna.visualization import plot_pareto_front, plot_param_importances
from rich.console import Console
from rich.live import Live
from rich.table import Table
from torch import Tensor
from tqdm.notebook import tqdm  # Use notebook tqdm for better dashboard integration if in notebook

# --- Dependencies ---
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("Please install required libraries: pip install datasets transformers torch optuna rich tqdm")
    exit()

# --- Constants & Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEFAULT_L_PRIME = 0  # Default: Process full sequence (0 means no blocking)
DATASET_NAME = "wikitext"
DATASET_CONFIG = "wikitext-2-raw-v1"
TOKENIZER_NAME = "gpt2" # Using GPT2 tokenizer for real text
SEQ_LEN = 32        # Sequence length for training/eval
BATCH_SIZE = 128       # Adjust based on GPU memory
NUM_DATA_LOAD_BATCHES = 200 # Number of batches to preload from dataset
NUM_TRAIN_STEPS_PER_TRIAL = 1000 # Keep low for fast Optuna trials
NUM_EVAL_BATCHES_PER_TRIAL = 20 # Batches for intermediate eval during trial
OPTUNA_N_TRIALS = 16         # Total number of optimization trials
GENERATION_MAX_NEW = 48      # Max new tokens for sample generation
GENERATION_PROMPT = "The future of AI is" # Prompt for text generation samples

# --- Initialize Tokenizer ---
print(f"Loading tokenizer: {TOKENIZER_NAME}")
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token # Set pad token if not defined
VOCAB_SIZE = tokenizer.vocab_size
print(f"Tokenizer loaded. Vocab size: {VOCAB_SIZE}")

# --- Data Handling ---
def prepare_data(dataset_name: str, dataset_config: str, tokenizer: AutoTokenizer, seq_len: int, num_batches: int, batch_size: int, split="train") -> List[Tensor]:
    """Loads, tokenizes, and prepares data batches."""
    print(f"Loading dataset {dataset_name} ({dataset_config}) - split: {split}")
    try:
        dataset = load_dataset(dataset_name, dataset_config, split=split, trust_remote_code=True) # Added trust_remote_code=True
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Attempting to load without config...")
        try:
            dataset = load_dataset(dataset_name, split=split, trust_remote_code=True) # Added trust_remote_code=True
        except Exception as e2:
             print(f"Failed to load dataset even without config: {e2}")
             return []


    print("Tokenizing text...")
    def tokenize_function(examples):
        # Handle potential None values in 'text' field
        texts = [text if text is not None else "" for text in examples["text"]]
        return tokenizer(texts, add_special_tokens=False) # Don't add BOS/EOS here, handle later if needed

    tokenized_datasets = dataset.map(tokenize_function, batched=True, num_proc=4, remove_columns=dataset.column_names)

    print("Concatenating and chunking...")
    def group_texts(examples):
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        total_length = (total_length // seq_len) * seq_len
        result = {
            k: [t[i : i + seq_len] for i in range(0, total_length, seq_len)]
            for k, t in concatenated_examples.items()
        }
        # result["labels"] = result["input_ids"].copy() # For standard LM loss, not needed for Hydra
        return result

    lm_datasets = tokenized_datasets.map(group_texts, batched=True, num_proc=4)

    print(f"Preparing {num_batches} batches...")
    data_batches = []
    for i in range(min(num_batches * batch_size, len(lm_datasets))):
        if (i % batch_size) == 0:
            start_idx = i
            end_idx = min(i + batch_size, len(lm_datasets))
            batch_ids = lm_datasets[start_idx:end_idx]['input_ids']
            if len(batch_ids) > 0:
                 # Pad the last batch if necessary
                 if len(batch_ids) < batch_size:
                      padding_needed = batch_size - len(batch_ids)
                      pad_tensor = torch.full((padding_needed, seq_len), tokenizer.pad_token_id, dtype=torch.long)
                      batch_tensor = torch.tensor(batch_ids, dtype=torch.long)
                      batch_tensor = torch.cat([batch_tensor, pad_tensor], dim=0)
                 else:
                      batch_tensor = torch.tensor(batch_ids, dtype=torch.long)

                 if batch_tensor.shape[1] == seq_len: # Ensure correct seq len
                      data_batches.append(batch_tensor.cpu()) # Keep on CPU initially
                 else:
                      print(f"Warning: Skipping batch with incorrect sequence length {batch_tensor.shape[1]} != {seq_len}")


        if len(data_batches) >= num_batches:
             break

    print(f"Loaded {len(data_batches)} batches for {split} split.")
    return data_batches


# --- Self-Contained Selective Scan (S6) Implementation (UNCHANGED from original) ---
class SelectiveScan(nn.Module):
    def __init__(self, embed_dim: int, state_dim: int = 16, d_conv: int = 4, dt_rank: str | int = 'auto', bias=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.state_dim = state_dim
        self.d_conv = d_conv
        self.dt_rank = math.ceil(embed_dim / 16) if dt_rank == 'auto' else dt_rank
        self.bias = bias

        self.in_proj_x = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_z = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.in_proj_params = nn.Linear(embed_dim, self.dt_rank + 2 * self.state_dim, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=embed_dim, out_channels=embed_dim, bias=bias,
            kernel_size=d_conv, groups=embed_dim,
            padding=d_conv - 1,
        )

        self.dt_proj = nn.Linear(self.dt_rank, embed_dim, bias=True)

        self.A_log = nn.Parameter(torch.log(torch.arange(1, state_dim + 1, dtype=torch.float32)).unsqueeze(0).repeat(embed_dim, 1)) # Shape: [D, N]
        self.D = nn.Parameter(torch.ones(embed_dim)) # Shape: [D]

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        with torch.no_grad():
            dt_init_std = self.dt_rank**-0.5
            self.dt_proj.weight.uniform_(-dt_init_std, dt_init_std)
            inv_softplus_target = math.log(math.expm1(0.01))
            self.dt_proj.bias.data.fill_(inv_softplus_target)


    def _compute_A_tilde(self, dt: Tensor, A_log: Tensor) -> Tensor:
        A = -torch.exp(A_log.float())
        A_tilde = torch.exp(dt.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(1))
        return A_tilde

    def _compute_B_tilde(self, dt: Tensor, A_log: Tensor, A_tilde: Optional[Tensor] = None) -> Tensor:
        A = -torch.exp(A_log.float())
        if A_tilde is None:
             A_tilde = self._compute_A_tilde(dt, A_log)

        A_unsqueezed = A.unsqueeze(0).unsqueeze(1)
        is_zero_A = torch.abs(A_unsqueezed) < 1e-8
        B_tilde = torch.where(
            is_zero_A,
            dt.unsqueeze(-1),
            (A_tilde - 1) / (A_unsqueezed + 1e-10)
        )
        return B_tilde

    def forward(self, x: Tensor) -> Tensor:
        batch, seq_len, embed_dim = x.shape
        if embed_dim != self.embed_dim:
             raise ValueError(f"Input embed_dim ({embed_dim}) doesn't match model embed_dim ({self.embed_dim})")

        x_res = self.in_proj_x(x)
        z = self.in_proj_z(x)
        params = self.in_proj_params(x)
        dt_unproj, B_proj, C_proj = torch.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        dt = F.softplus(self.dt_proj(dt_unproj))

        x_conv = self.conv1d(x_res.transpose(1, 2))
        x_conv = x_conv[:, :, :seq_len]
        u = F.silu(x_conv.transpose(1, 2))

        A_tilde = self._compute_A_tilde(dt, self.A_log)
        B_tilde = self._compute_B_tilde(dt, self.A_log, A_tilde)

        h = torch.zeros(batch, self.embed_dim, self.state_dim, device=x.device, dtype=A_tilde.dtype)
        ys = []
        for t in range(seq_len):
            A_t = A_tilde[:, t, :, :]
            B_t = B_tilde[:, t, :, :]
            B_proj_t = B_proj[:, t, :]
            C_proj_t = C_proj[:, t, :]
            u_t = u[:, t, :]

            input_term = torch.einsum('bdn, bn, bd -> bdn', B_t, B_proj_t, u_t)
            h = A_t * h + input_term

            y_t = torch.einsum('bn, bdn -> bd', C_proj_t, h)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)

        y = y + u * self.D.unsqueeze(0).unsqueeze(1)
        y = y * F.silu(z)

        y_out = self.out_proj(y)
        return y_out

    def step(self, x_step: Tensor, h_prev: Tensor, conv_state: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        batch, embed_dim = x_step.shape

        conv_input = torch.cat([conv_state, x_step.unsqueeze(2)], dim=2)
        new_conv_state = conv_input[:, :, 1:]
        conv_out = F.conv1d(
            conv_input, weight=self.conv1d.weight, bias=self.conv1d.bias,
            groups=self.embed_dim, padding=0
        ).squeeze(-1)
        u_step = F.silu(conv_out)

        z_step = self.in_proj_z(x_step)
        params = self.in_proj_params(x_step)
        dt_unproj, B_proj, C_proj = torch.split(
            params, [self.dt_rank, self.state_dim, self.state_dim], dim=-1
        )
        dt_step = F.softplus(self.dt_proj(dt_unproj))

        dt_for_compute = dt_step.unsqueeze(1)
        A_tilde_step = self._compute_A_tilde(dt_for_compute, self.A_log).squeeze(1)
        B_tilde_step = self._compute_B_tilde(dt_for_compute, self.A_log, A_tilde_step.unsqueeze(1)).squeeze(1)

        input_term = torch.einsum('bdn, bn, bd -> bdn', B_tilde_step, B_proj, u_step)
        h = A_tilde_step * h_prev + input_term

        y = torch.einsum('bn, bdn -> bd', C_proj, h)
        y = y + u_step * self.D.unsqueeze(0)

        y = y * F.silu(z_step)
        y_step = self.out_proj(y)

        return y_step, h, new_conv_state

# --- HydraScale Components (UNCHANGED from original, except VOCAB_SIZE usage) ---
class HydraBlock(nn.Module):
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
        residual = x_step
        x_norm1 = self.norm1(x_step)
        ssm_out, ssm_state_new, conv_state_new = self.ssm.step(x_norm1, ssm_state, conv_state)
        x = residual + ssm_out
        residual = x
        x_norm2 = self.norm2(x)
        mlp_out = self.mlp(x_norm2)
        y_step = residual + mlp_out
        return y_step, ssm_state_new, conv_state_new

# Cache for sinusoidal embeddings to avoid recalculation per call
sinusoidal_embedding_cache = {}
def sinusoidal_embedding(timesteps: Tensor, embedding_dim: int) -> Tensor:
    if timesteps.ndim > 1: timesteps = timesteps.squeeze(-1)
    device = timesteps.device
    key = (embedding_dim, device)

    if key not in sinusoidal_embedding_cache:
        half_dim = embedding_dim // 2
        emb_log = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb_log)
        sinusoidal_embedding_cache[key] = emb.unsqueeze(0) # Shape [1, half_dim]
        # print(f"  (Re)Calculating sinusoidal embedding matrix cache for dim={embedding_dim} on {device}")

    pe = sinusoidal_embedding_cache[key]
    emb = timesteps.float().unsqueeze(1) * pe # [B, 1] * [1, H/2] -> [B, H/2]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1) # [B, H]
    if embedding_dim % 2 == 1: emb = F.pad(emb, (0, 1))
    return emb


class HydraScaleLM(nn.Module):
    def __init__(self,
                 vocab_size: int, # Now passed dynamically
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
        self.mask_token_id = vocab_size # Use vocab_size as the ID for the mask token
        self.ssm_d_conv = ssm_d_conv

        self.token_embedding = nn.Embedding(vocab_size + 1, embed_dim) # +1 for [MASK] token

        self.time_embedding_dim = embed_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_embedding_dim, self.time_embedding_dim * 4),
            nn.SiLU(),
            nn.Linear(self.time_embedding_dim * 4, self.time_embedding_dim),
        )

        ssm_kwargs = {'state_dim': ssm_state_dim, 'd_conv': ssm_d_conv, 'dt_rank': ssm_dt_rank}
        self.layers = nn.ModuleList([
            HydraBlock(embed_dim, mlp_mult=mlp_mult, ssm_kwargs=ssm_kwargs)
            for _ in range(depth)
        ])

        self.norm_out = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(embed_dim, vocab_size) # Output original vocab size

        betas = self._calculate_betas(num_diffusion_timesteps, noise_schedule)
        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        print(f"HydraScaleLM initialized (D={embed_dim}, Depth={depth}, SSM_N={ssm_state_dim}, T={num_diffusion_timesteps}).")
        warnings.warn(
            "HydraScaleLM training uses a sequential scan, potentially slow. Inference uses recurrent step."
        )

    def _calculate_betas(self, timesteps, schedule='cosine', s=0.008, beta_start=0.0001, beta_end=0.02):
        if schedule == 'cosine':
            steps = timesteps + 1
            x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
            alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            return torch.clip(betas, 0.0001, 0.9999).float()
        elif schedule == 'linear':
            return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        else: raise ValueError(f"Unknown noise schedule: {schedule}")

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
        # Ensure mask respects padding tokens (don't mask padding)
        is_padding = (x_0 == tokenizer.pad_token_id)
        rand_noise = torch.rand_like(x_0, dtype=torch.float32)
        mask = (rand_noise < mask_prob_expanded) & (~is_padding) # Don't mask padding
        x_t = torch.where(mask, self.mask_token_id, x_0)
        return x_t, mask

    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        batch_size, seq_len = x.shape
        token_emb = self.token_embedding(x)
        time_emb_sin = sinusoidal_embedding(t, self.time_embedding_dim)
        time_emb = self.time_mlp(time_emb_sin)
        h = token_emb + time_emb.unsqueeze(1)

        for layer in self.layers: h = layer(h)

        h = self.norm_out(h)
        logits = self.lm_head(h)
        return logits

    def compute_loss(self, x_0: Tensor, return_acc: bool = True) -> Union[Tensor, Tuple[Tensor, float]]:
        """ Computes the diffusion training loss and optionally accuracy on masked tokens. """
        batch_size, seq_len = x_0.shape
        if x_0.nelement() == 0:
             return torch.tensor(0.0, device=x_0.device, requires_grad=True), 0.0

        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device).long()
        x_t, mask = self._mask_tokens(x_0, t) # mask respects padding
        pred_logits = self.forward(x_t, t)

        # Ensure mask doesn't include positions beyond original sequence length if padding occurred
        valid_mask = mask & (x_0 != tokenizer.pad_token_id) # Double check padding isn't masked
        masked_logits = pred_logits[valid_mask]
        masked_targets = x_0[valid_mask]

        if masked_targets.numel() == 0:
            # Avoid NaN loss if no tokens were masked (e.g., t=0 or all padding)
            # Return 0 loss, but ensure it has a gradient node if needed.
            # The loss should be attached to the graph, so multiply by a parameter.
            loss = (self.lm_head.weight.sum() * 0.0).to(masked_logits.dtype) # Ensure grad flows
            accuracy = 1.0
        else:
            loss = F.cross_entropy(masked_logits, masked_targets)
            if return_acc:
                with torch.no_grad():
                    correct_preds = (masked_logits.argmax(dim=-1) == masked_targets).sum().item()
                    accuracy = correct_preds / masked_targets.numel()
            else:
                accuracy = 0.0 # Placeholder if not calculated

        return (loss, accuracy) if return_acc else loss

    @torch.no_grad()
    def _predict_x0_from_logits(self, x_t: Tensor, t: Tensor, logits: Tensor,
                                sampling_mode: str = 'argmax', temperature: float = 1.0, top_k: Optional[int] = None
                               ) -> Tensor:
        batch_size, seq_len, vocab_size = logits.shape

        if sampling_mode == 'argmax':
            pred_ids = torch.argmax(logits, dim=-1)
        elif sampling_mode in ['multinomial', 'topk']:
            logits_flat = logits.view(-1, vocab_size)
            if temperature > 0 and temperature != 1.0: logits_flat = logits_flat / temperature
            if top_k is not None and top_k > 0:
                k = min(top_k, vocab_size)
                kth_vals, _ = torch.topk(logits_flat, k, dim=-1)
                kth_vals_min = kth_vals[..., -1, None]
                logits_flat.masked_fill_(logits_flat < kth_vals_min, -float('Inf'))

            probs = F.softmax(logits_flat, dim=-1)
            probs = torch.nan_to_num(probs, nan=0.0)
            row_sums = probs.sum(dim=-1, keepdim=True)
            zero_rows = (row_sums <= 1e-9)
            uniform_probs = torch.full_like(probs, 1.0/vocab_size)
            probs = torch.where(zero_rows, uniform_probs, probs)
            probs /= probs.sum(dim=-1, keepdim=True) # Ensure normalization

            pred_ids_flat = torch.multinomial(probs, num_samples=1).squeeze(-1)
            pred_ids = pred_ids_flat.view(batch_size, seq_len)
        else: raise ValueError(f"Unknown sampling mode: {sampling_mode}")

        # Only replace the masked tokens
        mask = (x_t == self.mask_token_id)
        # Also ensure we don't overwrite original prompt tokens if generation starts from a prompt
        # (This logic might be better placed in the generate function itself)
        return torch.where(mask, pred_ids, x_t)

    @torch.no_grad()
    def generate(self,
                 prompt_ids: Tensor, # Expect token IDs [B, L_prompt]
                 num_tokens_to_generate: int,
                 num_sampling_steps: Optional[int] = None,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 sampling_mode: str = 'topk'
                 ) -> Tensor:
        self.eval()
        batch_size, prompt_len = prompt_ids.shape
        total_len = prompt_len + num_tokens_to_generate
        sampling_steps = num_sampling_steps if num_sampling_steps is not None else self.num_timesteps
        sampling_steps = min(sampling_steps, self.num_timesteps)

        x_gen = torch.full((batch_size, total_len), self.mask_token_id, dtype=torch.long, device=prompt_ids.device)
        x_gen[:, :prompt_len] = prompt_ids

        layer_states: List[Tuple[Tensor, Tensor]] = []
        for layer in self.layers:
            ssm_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.state_dim, device=prompt_ids.device, dtype=torch.float32)
            conv_state = torch.zeros(batch_size, self.embed_dim, layer.ssm.d_conv - 1, device=prompt_ids.device, dtype=torch.float32)
            layer_states.append((ssm_state, conv_state))

        time_indices = torch.linspace(self.num_timesteps - 1, 0, sampling_steps, device=prompt_ids.device).long()

        # Simple progress bar if not batch size 1
        gen_iterator = range(sampling_steps)
        # if batch_size > 1: gen_iterator = tqdm(gen_iterator, desc="Gen Steps (Hydra)", leave=False, total=sampling_steps)

        for i in gen_iterator:
            t_current_val = time_indices[i]
            t_current = t_current_val.expand(batch_size)

            current_layer_states = [(s[0].clone(), s[1].clone()) for s in layer_states] # Avoid inplace modification if using same states across steps? Does clone work?
            token_emb = self.token_embedding(x_gen)

            time_emb_sin = sinusoidal_embedding(t_current, self.time_embedding_dim)
            time_emb = self.time_mlp(time_emb_sin)

            all_logits_step = []
            # Use efficient step function - process sequence token by token
            for token_idx in range(total_len):
                x_step = token_emb[:, token_idx, :] + time_emb # Add time embedding at each step

                for layer_idx, layer in enumerate(self.layers):
                    ssm_state, conv_state = current_layer_states[layer_idx]
                    x_step, ssm_state_new, conv_state_new = layer.step(x_step, ssm_state, conv_state)
                    current_layer_states[layer_idx] = (ssm_state_new, conv_state_new)

                h_final = self.norm_out(x_step)
                logits_token = self.lm_head(h_final)
                all_logits_step.append(logits_token)

            logits = torch.stack(all_logits_step, dim=1) # [B, L_total, V]

            current_sampling_mode = sampling_mode
            # Heuristic: Use argmax for early steps (high noise)?
            # if t_current_val > self.num_timesteps // 2 : current_sampling_mode = 'argmax'

            predicted_x0 = self._predict_x0_from_logits(
                x_gen.clone(), # Pass a clone to avoid modifying x_gen used in next loop
                t_current, logits,
                sampling_mode=current_sampling_mode,
                temperature=temperature, top_k=top_k
            )

            # Crucial: Only update the *masked* parts. Don't overwrite prompt or already generated tokens.
            mask_for_update = (x_gen == self.mask_token_id)
            x_gen = torch.where(mask_for_update, predicted_x0, x_gen)

            # Alternative: Resample technique (less common for discrete diffusion?)
            # Could involve sampling x_{t-1} based on predicted x0 and current x_t.
            # For simplicity, we stick to predicting x0 and replacing masks.

        # Ensure prompt is intact
        x_gen[:, :prompt_len] = prompt_ids
        return x_gen


# --- Comparison Baseline: Transformer (Simplified) (UNCHANGED from original, except VOCAB_SIZE usage) ---
class SimpleTransformerLM(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, nhead: int, num_layers: int,
                 dim_feedforward: int, max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        # Make pos_encoder buffer instead of parameter if not learned? Or keep as parameter.
        self.pos_encoder = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.normal_(self.pos_encoder, std=0.02) # Initialize pos encodings

        self.dropout = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=nhead, dim_feedforward=dim_feedforward,
            dropout=dropout, activation=F.gelu, batch_first=True, norm_first=True # Use norm_first
        )
        encoder_norm = nn.LayerNorm(embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, norm=encoder_norm
        )
        self.lm_head = nn.Linear(embed_dim, vocab_size)
        print(f"SimpleTransformerLM initialized (D={embed_dim}, Layers={num_layers}, Head={nhead}, FFN={dim_feedforward}).")

    def _generate_causal_mask(self, sz: int, device: torch.device) -> Tensor:
        return torch.triu(torch.full((sz, sz), -float('inf'), device=device), diagonal=1)

    def forward(self, src: Tensor, attention_mask: Optional[Tensor] = None) -> Tensor:
        batch_size, seq_len = src.shape
        effective_seq_len = seq_len
        if seq_len > self.max_seq_len:
            src = src[:, -self.max_seq_len:]
            effective_seq_len = self.max_seq_len
            if attention_mask is not None:
                attention_mask = attention_mask[:,-self.max_seq_len:]


        src_emb = self.token_embedding(src) * math.sqrt(self.embed_dim)
        pos_emb = self.pos_encoder[:, :effective_seq_len, :]
        src_combined = src_emb + pos_emb.to(src_emb.device)
        src_combined = self.dropout(src_combined)

        causal_mask = self._generate_causal_mask(effective_seq_len, device=src.device)

        # Combine causal mask and padding mask if provided
        # `attention_mask` should be [B, L] where 0 means pad, 1 means token
        # Need to convert to attention mask format [B, N_heads, L, L] or [B, L, L] expected by TransformerEncoder
        # Or use src_key_padding_mask ( simpler: [B, L], True if masked/pad)
        padding_mask = (src == tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None


        # Note: TransformerEncoder expects mask [L, L] and src_key_padding_mask [B, L]
        output = self.transformer_encoder(src_combined, mask=causal_mask, src_key_padding_mask=padding_mask)
        logits = self.lm_head(output)
        return logits

    def compute_loss(self, x_0: Tensor, return_acc: bool = True) -> Union[Tensor, Tuple[Tensor, float]]:
        """ Computes standard Causal LM loss and optionally accuracy. """
        if x_0.shape[1] < 2:
             return torch.tensor(0.0, device=x_0.device, requires_grad=True), 0.0

        inp = x_0[:, :-1]
        tgt = x_0[:, 1:]
        attention_mask = (inp != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None

        logits = self.forward(inp, attention_mask=attention_mask) # [B, L-1, V]

        # Calculate loss only on non-padding tokens
        loss = F.cross_entropy(logits.reshape(-1, self.vocab_size), tgt.reshape(-1), ignore_index=tokenizer.pad_token_id)

        accuracy = 0.0
        if return_acc and tokenizer.pad_token_id is not None:
             with torch.no_grad():
                 valid_targets = (tgt != tokenizer.pad_token_id)
                 if valid_targets.sum() > 0:
                     valid_logits = logits[valid_targets]
                     valid_tgt_flat = tgt[valid_targets]
                     correct_preds = (valid_logits.argmax(dim=-1) == valid_tgt_flat).sum().item()
                     accuracy = correct_preds / valid_targets.sum().item()
                 else:
                      accuracy = 1.0 # Or 0.0 if no valid targets?

        elif return_acc: # No padding token handling
             with torch.no_grad():
                  correct_preds = (logits.argmax(dim=-1) == tgt).sum().item()
                  accuracy = correct_preds / tgt.numel()


        return (loss, accuracy) if return_acc else loss


    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, num_tokens_to_generate: int, temperature: float = 1.0,
                 top_k: Optional[int] = None) -> Tensor:
        self.eval()
        generated_ids = prompt_ids.to(DEVICE)
        batch_size = prompt_ids.shape[0]

        gen_iterator = range(num_tokens_to_generate)
        # if batch_size > 1: gen_iterator = tqdm(gen_iterator, desc="Gen Steps (TF)", leave=False, total=num_tokens_to_generate)

        for _ in gen_iterator:
            context = generated_ids
            # Truncate context if it exceeds max_seq_len
            if context.shape[1] > self.max_seq_len:
                context = context[:, -self.max_seq_len:]

            # Prepare attention mask for padding
            attention_mask = (context != tokenizer.pad_token_id).long() if tokenizer.pad_token_id is not None else None

            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                logits = self.forward(context, attention_mask=attention_mask)
                next_token_logits = logits[:, -1, :] # Logits for the very last token prediction

            if temperature == 0: # Greedy decoding
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            else: # Sampling
                if temperature != 1.0: next_token_logits = next_token_logits / temperature
                if top_k is not None and top_k > 0:
                    v = next_token_logits.size(-1)
                    k = min(top_k, v)
                    kth_vals, _ = torch.topk(next_token_logits, k, dim=-1)
                    kth_vals_min = kth_vals[:, -1, None]
                    indices_to_remove = next_token_logits < kth_vals_min
                    next_token_logits.masked_fill_(indices_to_remove, -float('Inf'))

                probs = F.softmax(next_token_logits, dim=-1)
                probs = torch.nan_to_num(probs, nan=0.0)
                zero_probs = (probs.sum(dim=-1, keepdim=True) < 1e-9)
                uniform_dist = torch.full_like(probs, 1.0 / probs.shape[-1])
                probs = torch.where(zero_probs, uniform_dist, probs)
                probs /= probs.sum(dim=-1, keepdim=True) # Re-normalize

                next_token_id = torch.multinomial(probs, num_samples=1)

            generated_ids = torch.cat([generated_ids, next_token_id], dim=1)

        return generated_ids


# --- Utility Functions ---
@torch.no_grad()
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def generate_sample_text(model: nn.Module, tokenizer: AutoTokenizer, prompt_text: str, device: torch.device, model_type: str, max_new: int = 48, gen_kwargs: Dict = {}) -> str:
    """Generates a text sample and decodes it."""
    model.eval()
    prompt_ids = tokenizer.encode(prompt_text, return_tensors="pt").to(device)
    try:
        with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
            if model_type == "hydra":
                 # Default Hydra generate kwargs if not provided
                 hydra_defaults = {"num_sampling_steps": model.num_timesteps // 2, "sampling_mode": "topk", "top_k": 50, "temperature": 0.8}
                 hydra_defaults.update(gen_kwargs) # Override with provided kwargs
                 output_ids = model.generate(prompt_ids, num_tokens_to_generate=max_new, **hydra_defaults)
            elif model_type == "transformer":
                 # Default Transformer generate kwargs if not provided
                 tf_defaults = {"top_k": 50, "temperature": 0.8}
                 tf_defaults.update(gen_kwargs) # Override with provided kwargs
                 output_ids = model.generate(prompt_ids, num_tokens_to_generate=max_new, **tf_defaults)
            else:
                 return "[Invalid model type for generation]"

        # Decode only the generated part (skip prompt)
        generated_text = tokenizer.decode(output_ids[0, prompt_ids.shape[1]:], skip_special_tokens=True)
        return prompt_text + generated_text
    except Exception as e:
        return f"[Generation Error: {e}]"

# --- Training & Evaluation Helpers for Optuna ---

def train_eval_trial(model: nn.Module, train_loader: List[Tensor], eval_loader: List[Tensor], optimizer: torch.optim.Optimizer, scaler: torch.amp.GradScaler, num_train_steps: int, num_eval_batches: int, device: torch.device, model_type: str) -> Tuple[float, float]:
    """ Trains model for num_train_steps and returns (avg_loss, avg_accuracy). """
    model.train()
    total_loss = total_accuracy = 0.0
    steps_done = batches_processed = 0
    train_iterator = tqdm(range(num_train_steps), desc=f"Trial Train {model_type[:5]}", leave=False)

    data_iter = iter(train_loader) # Use iterator to get batches

    for step in train_iterator:
        try:
            batch_cpu = next(data_iter)
        except StopIteration:
            # Reset iterator if dataset runs out before steps are done
            # print("Resetting training data iterator...")
            data_iter = iter(train_loader)
            try:
                 batch_cpu = next(data_iter)
            except StopIteration:
                 print("Warning: Training data loader is empty or too small.")
                 break # Not enough data

        batch = batch_cpu.to(device)
        if batch.nelement() == 0: continue

        optimizer.zero_grad()
        try:
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                loss, accuracy = model.compute_loss(batch, return_acc=True)

            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at step {step}. Skipping update.")
                optimizer.zero_grad()
                accuracy = 0.0 # Don't count accuracy for this step
                loss = torch.tensor(0.0) # Don't add NaN loss
                continue # Skip backward/step

            scaler.scale(loss).backward()
            # Grad clipping (optional but recommended)
            # scaler.unscale_(optimizer)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            total_accuracy += accuracy
            steps_done += 1

        except RuntimeError as e:
            if "out of memory" in str(e):
                print(f"WARNING: CUDA OOM during training step {step}. Skipping batch.")
                optimizer.zero_grad()
                gc.collect()
                torch.cuda.empty_cache()
                # Optionally raise Optuna Pruning or just continue
                raise optuna.TrialPruned(f"OOM during training step {step}")
            else:
                print(f"ERROR during training step {step}: {e}")
                raise e # Re-raise other runtime errors

        batches_processed += 1
        if steps_done >= num_train_steps:
             break # Stop if target steps reached

    avg_loss = total_loss / steps_done if steps_done > 0 else 0.0
    avg_accuracy = total_accuracy / steps_done if steps_done > 0 else 0.0

    # Simple validation check (optional, could add perplexity here too)
    # model.eval()
    # eval_loss = 0.0
    # eval_batches = 0
    # with torch.no_grad():
    #     for i, batch_cpu in enumerate(eval_loader):
    #         if i >= num_eval_batches: break
    #         batch = batch_cpu.to(device)
    #         with autocast(enabled=(device.type=='cuda')):
    #             loss, _ = model.compute_loss(batch, return_acc=False)
    #         if not torch.isnan(loss):
    #             eval_loss += loss.item()
    #             eval_batches += 1
    # avg_eval_loss = eval_loss / eval_batches if eval_batches > 0 else float('inf')
    # print(f"  Trial {model_type[:5]} Eval Loss (approx): {avg_eval_loss:.4f}")

    return avg_loss, avg_accuracy


# --- Optuna Objective Function ---

def objective(trial: optuna.Trial, train_data: List[Tensor], eval_data: List[Tensor], fixed_seq_len: int, device: torch.device) -> Tuple[float, float]:
    """ Optuna objective function for multi-objective optimization. """

    model_type = trial.suggest_categorical("model_type", ["hydra", "transformer"])

    # --- Hyperparameter Suggestions ---
    # Common HPs
    # embed_dim = trial.suggest_categorical("embed_dim", [128, 256, 384, 512])
    embed_dim = trial.suggest_int("embed_dim", 128, 512, step=64) # Wider range
    # depth = trial.suggest_int("depth", 2, 8)
    depth = trial.suggest_int("depth", 2, 6) # Reduce max depth slightly for faster trials

    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

    model = None
    param_count = 0
    generated_text_sample = "[Not Generated]"
    model_config = {}

    try:
        if model_type == "hydra":
            mlp_mult = trial.suggest_categorical("mlp_mult", [2, 4])
            ssm_state_dim = trial.suggest_categorical("ssm_state_dim", [8, 16, 32])
            ssm_d_conv = trial.suggest_categorical("ssm_d_conv", [3, 4])
            # dt_rank fixed to auto for simplicity, could be optimized:
            # dt_rank_opt = trial.suggest_categorical("ssm_dt_rank_opt", ['auto', 16])
            # dt_rank = math.ceil(embed_dim / 16) if dt_rank_opt == 'auto' else dt_rank_opt
            num_diffusion_timesteps = trial.suggest_categorical("num_diffusion_timesteps", [50, 100])

            model_config = {
                "vocab_size": VOCAB_SIZE, "embed_dim": embed_dim, "depth": depth,
                "mlp_mult": mlp_mult, "num_diffusion_timesteps": num_diffusion_timesteps,
                "ssm_state_dim": ssm_state_dim, "ssm_d_conv": ssm_d_conv, "ssm_dt_rank": 'auto'
            }
            model = HydraScaleLM(**model_config).to(device)

        elif model_type == "transformer":
            nhead = trial.suggest_categorical("nhead", [2, 4, 8])
            # Ensure embed_dim is divisible by nhead
            if embed_dim % nhead != 0:
                 # Adjust embed_dim to be divisible, or prune trial
                 # Option 1: Adjust embed_dim (might slightly violate suggestion)
                 # embed_dim = (embed_dim // nhead) * nhead
                 # if embed_dim == 0: embed_dim = nhead # handle case where embed_dim < nhead
                 # print(f"  Adjusted embed_dim to {embed_dim} for nhead={nhead}")
                 # Option 2: Prune trial
                 raise optuna.TrialPruned(f"embed_dim {embed_dim} not divisible by nhead {nhead}")

            # Feedforward dim: Link to embed_dim or independent? Let's make it independent but related.
            # ffn_mult = trial.suggest_categorical("ffn_mult", [2, 4])
            # dim_feedforward = embed_dim * ffn_mult
            dim_feedforward = trial.suggest_int("dim_feedforward", embed_dim * 2, embed_dim * 4, step=128)

            dropout = trial.suggest_float("dropout", 0.0, 0.2)
            # max_seq_len for Transformer - keep fixed for this study or optimize? Keep fixed.
            max_seq_len = fixed_seq_len * 2 # Allow longer context than training seq len

            model_config = {
                 "vocab_size": VOCAB_SIZE, "embed_dim": embed_dim, "nhead": nhead,
                 "num_layers": depth, "dim_feedforward": dim_feedforward,
                 "max_seq_len": max_seq_len, "dropout": dropout
            }
            model = SimpleTransformerLM(**model_config).to(device)

        # --- Calculate Parameter Count (Objective 2) ---
        param_count = count_parameters(model)
        trial.set_user_attr("param_count_M", param_count / 1e6)
        trial.set_user_attr("config", model_config) # Store config for inspection

        # --- Short Training and Evaluation ---
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

        avg_loss, avg_accuracy = train_eval_trial(
            model, train_data, eval_data, optimizer, scaler,
            num_train_steps=NUM_TRAIN_STEPS_PER_TRIAL,
            num_eval_batches=NUM_EVAL_BATCHES_PER_TRIAL,
            device=device, model_type=model_type
        )

        # --- Accuracy (Objective 1) ---
        # Use the average accuracy from the short training run
        final_accuracy = avg_accuracy
        trial.set_user_attr("final_accuracy", final_accuracy)
        trial.set_user_attr("final_loss", avg_loss)


        # --- Generate Text Sample ---
        # Generate only if training was successful (accuracy is not 0 or NaN)
        if final_accuracy > 0 and not np.isnan(final_accuracy):
             generated_text_sample = generate_sample_text(model, tokenizer, GENERATION_PROMPT, device, model_type, max_new=GENERATION_MAX_NEW)
             # Clean up sample for display
             generated_text_sample = generated_text_sample.replace("\n", " ").strip()

             print('  ',generated_text_sample)

             if len(generated_text_sample) > 150: # Truncate long samples
                 generated_text_sample = generated_text_sample[:147] + "..."

        else:
             generated_text_sample = "[Training Failed/No Accuracy]"

        trial.set_user_attr("generated_sample", generated_text_sample)

        # Clean up GPU memory
        del model
        del optimizer
        del scaler
        gc.collect()
        torch.cuda.empty_cache()

        # Return objectives: (Maximize Accuracy, Minimize Parameters)
        # Handle potential NaN accuracy
        if np.isnan(final_accuracy):
             final_accuracy = 0.0 # Penalize NaN results heavily

        return float(final_accuracy), float(param_count)

    except optuna.TrialPruned as e:
        print(f"Trial {trial.number} pruned: {e}")
        # Clean up GPU memory on prune
        if model is not None: del model
        if 'optimizer' in locals(): del optimizer
        if 'scaler' in locals(): del scaler
        gc.collect()
        torch.cuda.empty_cache()
        raise e # Re-raise prune exception
    except Exception as e:
        print(f"!!!!!!!! Trial {trial.number} FAILED: {e} !!!!!!!!!!")
        import traceback
        traceback.print_exc()
        # Clean up GPU memory on error
        if model is not None: del model
        if 'optimizer' in locals(): del optimizer
        if 'scaler' in locals(): del scaler
        gc.collect()
        torch.cuda.empty_cache()
        # Report failure to Optuna - return worst possible values or let it fail
        # Return low accuracy, high param count to penalize
        return 0.0, float('inf')


# --- Live Dashboard Callback ---
dashboard_data = {}
dashboard_lock = Lock()
console = Console(width=200)

def make_dashboard_table() -> Table:
    """Creates the Rich Table for the dashboard."""
    table = Table(title=f"Optuna Optimization Results (HydraScale vs Transformer) - {DATASET_NAME}")
    table.add_column("Trial", justify="right", style="cyan", no_wrap=True)
    table.add_column("Model", style="magenta")
    table.add_column("State", style="yellow")
    table.add_column("Accuracy (%)", justify="right", style="green")
    table.add_column("Params (M)", justify="right", style="blue")
    table.add_column("LR", justify="right", style="dim")
    table.add_column("Dims", justify="right", style="dim")
    # Add other key params based on model type later
    table.add_column("Sample Output", style="white", max_width=60, overflow="fold")
    return table

class DashboardCallback:
    def __call__(self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial):
        with dashboard_lock:
            accuracy, params = (None, None)
            if trial.values: # Objectives (Accuracy, Params)
                 accuracy = trial.values[0] * 100 if trial.values[0] is not None else None # Convert acc to %
                 params = trial.values[1] / 1e6 if trial.values[1] is not None else None # Convert params to M
            elif trial.state == optuna.trial.TrialState.FAIL:
                 # Fetch from user attrs if available, otherwise use placeholders
                 accuracy = trial.user_attrs.get("final_accuracy", 0.0) * 100
                 params = trial.user_attrs.get("param_count_M", float('inf'))

            model_type = trial.params.get("model_type", "N/A")
            lr = trial.params.get("lr", float('nan'))
            embed_dim = trial.params.get("embed_dim", -1)
            depth = trial.params.get("depth", -1)
            dims_str = f"D={embed_dim}, L={depth}"

            sample = trial.user_attrs.get("generated_sample", "")
            if not sample and trial.state != optuna.trial.TrialState.RUNNING:
                 sample = "[No Sample]"

            # Store data - use trial number as key
            dashboard_data[trial.number] = {
                "model": model_type,
                "state": trial.state.name,
                "accuracy": accuracy,
                "params": params,
                "lr": lr,
                "dims": dims_str,
                "sample": sample,
            }

# --- Main Execution ---
if __name__ == "__main__":
    print(f"Starting Optuna optimization on {DEVICE}")
    print(f"Dataset: {DATASET_NAME}/{DATASET_CONFIG}, Tokenizer: {TOKENIZER_NAME}")
    print(f"Seq Len: {SEQ_LEN}, Batch Size: {BATCH_SIZE}")
    print(f"Optuna Trials: {OPTUNA_N_TRIALS}, Steps/Trial: {NUM_TRAIN_STEPS_PER_TRIAL}")
    print("-" * 60)

    # --- Load Data ---
    # Load a bit more data than needed per trial to avoid reloading
    train_data = prepare_data(DATASET_NAME, DATASET_CONFIG, tokenizer, SEQ_LEN, NUM_DATA_LOAD_BATCHES, BATCH_SIZE, split="train")
    eval_data = prepare_data(DATASET_NAME, DATASET_CONFIG, tokenizer, SEQ_LEN, NUM_DATA_LOAD_BATCHES // 2, BATCH_SIZE, split="validation")

    if not train_data or not eval_data:
        print("Failed to load sufficient data. Exiting.")
        exit()

    # --- Setup Optuna Study ---
    # Use NSGA-II sampler for multi-objective optimization
    sampler = optuna.samplers.NSGAIISampler(
        population_size=20, # Population size for NSGA-II
        mutation_prob=None, # Use default
        crossover_prob=0.9, # Use default
        swapping_prob=0.5, # Use default
        seed=42
    )
    # Can also use TPESampler with multivariate=True
    # sampler = optuna.samplers.TPESampler(multivariate=True, group=True, seed=42)

    study = optuna.create_study(
        directions=["maximize", "minimize"], # Maximize Accuracy, Minimize Params
        study_name="HydraScale_vs_Transformer_Opt",
        sampler=sampler,
        # Optional: Use storage for persistence
        # storage="sqlite:///hydra_optuna.db",
        # load_if_exists=True
    )

    callback = DashboardCallback()
    study.optimize(
        lambda trial: objective(trial, train_data, eval_data, SEQ_LEN, DEVICE),
        n_trials=OPTUNA_N_TRIALS,
        callbacks=[callback],
        gc_after_trial=True, # Enable Optuna's garbage collection
        show_progress_bar=False # Disable default tqdm bar, use dashboard
    )

    # --- Live Dashboard Update Loop ---
    # Keep updating the table until study is complete
    table = make_dashboard_table()
    with Live(table, console=console, refresh_per_second=1, vertical_overflow="visible") as live:
        #while not study._is_running: # Wait for study to start if needed
        #      time.sleep(0.5)

        completed_trials = set()
        while len(completed_trials) < len(study.trials): # Keep updating until all trials are processed by callback
            live.console.clear_live() # Attempt to clear previous table state

            table = make_dashboard_table() # Recreate table to ensure clean state
            with dashboard_lock:
                # Sort trials by number for consistent display
                sorted_trial_nums = sorted(dashboard_data.keys())
                for trial_num in sorted_trial_nums:
                    data = dashboard_data[trial_num]
                    state_color = "green" if data['state'] == "COMPLETE" else "yellow" if data['state'] == "RUNNING" else "red"
                    acc_str = f"{data['accuracy']:.2f}" if data['accuracy'] is not None else "N/A"
                    params_str = f"{data['params']:.2f}" if data['params'] is not None else "N/A"
                    lr_str = f"{data['lr']:.1e}" if data.get('lr') and not np.isnan(data['lr']) else "N/A"

                    table.add_row(
                        str(trial_num),
                        data['model'],
                        f"[{state_color}]{data['state']}[/]",
                        acc_str,
                        params_str,
                        lr_str,
                        data.get('dims', 'N/A'),
                        data.get('sample', '')
                    )
                    if data['state'] != "RUNNING":
                         completed_trials.add(trial_num)

            live.update(table)
            time.sleep(1) # Update interval

    # --- Final Results ---
    print("\n" + "=" * 60)
    print("Optimization Finished!")
    print(f"Number of finished trials: {len(study.trials)}")

    print("Best trials (Pareto front):")
    for trial in study.best_trials:
        acc = trial.values[0] * 100 if trial.values else 0.0
        params = trial.values[1] / 1e6 if trial.values else float('inf')
        model_type = trial.params.get("model_type", "N/A")
        print(f"  Trial {trial.number} ({model_type}): Accuracy={acc:.2f}%, Params={params:.2f}M")
        print(f"    Params: {trial.params}")
        print(f"    Sample: {trial.user_attrs.get('generated_sample', '[N/A]')}")


    # --- Visualizations (Optional) ---
    try:
        # Show Pareto front
        fig1 = plot_pareto_front(study, target_names=["Accuracy", "Parameters"])
        fig1.show() # Might require installing plotly separately: pip install plotly

        # Show parameter importance
        # Note: Param importance might be less interpretable in multi-objective settings
        fig2 = plot_param_importances(
            study, target=lambda t: t.values[0], target_name="Accuracy" # Importance for Accuracy
        )
        fig2.show()
        fig3 = plot_param_importances(
            study, target=lambda t: t.values[1], target_name="Parameters" # Importance for Parameters
        )
        fig3.show()
    except Exception as e:
        print(f"\nCould not generate plots (requires plotly and possibly kaleido): {e}")

    print("\nAnalysis Complete.")