import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass
import math
import numpy as np
import unittest
import time
import psutil
import gc

# --- Configuration ---
@dataclass
class MambaDiffuserConfig:
    vocab_size: int = 50257  # Default GPT-2 vocab size
    d_model: int = 512  # Mamba model dimension
    n_layer: int = 8   # Number of Mamba layers
    dt_rank: str = "auto"  # "auto" or int
    d_state: int = 16
    expand: int = 2
    dt_min: float = 0.001
    dt_max: float = 0.1
    dt_init_floor: float = 1e-4
    conv_bias: bool = True
    bias: bool = False
    use_fast_path: bool = True # Fused CUDA kernels, more memory efficient
    max_seq_len: int = 1024
    diffusion_steps: int = 100  # Total diffusion steps
    noise_schedule: str = "cosine"  # "cosine", "linear"
    use_classifier_free_guidance: bool = True
    guidance_scale: float = 1.5  # For classifier-free guidance
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    tier: int = 2  # 0: Base, 1: Efficient, 2: Enhanced
    use_cache: bool = True # Caching


class MambaBlock(nn.Module):
    def __init__(self, dim, hidden_dim, dt_rank="auto", d_state=16, conv_bias=True, bias=False):
        super().__init__()

        self.dim = dim
        self.hidden_dim = hidden_dim
        self.dt_rank = dt_rank
        self.d_state = d_state
        self.conv_bias = conv_bias
        self.bias = bias

        self.in_proj = nn.Linear(dim, hidden_dim * 2, bias=bias)

        if dt_rank == "auto":
            self.dt_rank = math.ceil(dim / 16)
        self.conv1d = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            groups=hidden_dim,
            padding=1,
            bias=conv_bias,
        )
        # Mamba specific parameters
        self.x_proj = nn.Linear(hidden_dim, self.dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, hidden_dim, bias=True)

        # Initialize dt_proj
        nn.init.uniform_(self.dt_proj.weight, -1 / self.dt_rank, 1 / self.dt_rank)

        # Initialize dt_proj bias
        dt = torch.exp(
            torch.rand(hidden_dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
        )
        inv_dt = 1.0 / dt
        self.dt_proj.bias.data.copy_(-inv_dt)

        self.A_log = nn.Parameter(torch.arange(1, d_state + 1).float().mul(-1.0))
        self.D = nn.Parameter(torch.ones(hidden_dim))
        self.out_proj = nn.Linear(hidden_dim * 2, dim, bias=bias)

    def forward(self, x: torch.Tensor, kv_cache=None) -> Tuple[torch.Tensor, None]:
        """
        x: (batch, seq_len, dim)
        """
        batch, seq_len, _ = x.shape
        x_orig = x
        x, z = self.in_proj(x).chunk(2, dim=-1)  # (b, l, 2d) -> (b, l, d), (b, l, d)

        # Convolution
        x = x.transpose(1, 2)  # (b, d, l)
        x = self.conv1d(x)
        x = x.transpose(1, 2)  # (b, l, d)
        x = F.silu(x)

        # Mamba SSM logic
        x_dbl = self.x_proj(x)  # (b, l, d) -> (b, l, dt_rank + 2 * d_state)
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # (b, l, dt_rank) -> (b, l, d)

        # Discrete-time SSM
        dt = F.softplus(dt)
        B = B.transpose(1, 2)  # (b, l, d_state) -> (b, d_state, l)
        C = C.transpose(1, 2)  # (b, l, d_state) -> (b, d_state, l)
        A = -torch.exp(self.A_log.float())  # (d_state,)
        D = self.D.float()

        # Efficient SSM implementation
        y = self.ssm(x, dt, A, B, C, D)
        y = y * F.silu(z)
        y = self.out_proj(y)
        return x_orig + y, None

    def ssm(self, u, delta, A, B, C, D):
        batch, seq_len, hidden_dim = u.shape
        A_exp = torch.exp(delta[:, :, :, None] * A).transpose(2, 3)  # (b, l, d_state, d)
        B = B.unsqueeze(-1)  # (b, d_state, l, 1)
        C = C.unsqueeze(-2)  # (b, d_state, 1, l)

        y = torch.zeros_like(u)
        h = torch.zeros((batch, self.d_state, hidden_dim), device=u.device)

        for k in range(seq_len):
            h = A_exp[:, k] @ h + B[:, :, k] * u[:, k, None, :]
            y_k = C[:, :, :, k] @ h
            y[:, k, :] = y_k.squeeze(-2) + D * u[:, k, :]
        return y



# --- Transformer Block ---
class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim),
        )

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        attn_out, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False, key_value_states=kv_cache, attn_mask=None)
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        new_kv_cache = (self.norm1(x), self.norm1(x)) if self.training or kv_cache is None else None
        return x, new_kv_cache

# --- Timestep Embedding ---
class TimestepEmbedding(nn.Module):
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(1000, embed_dim)  # Assume max 1000 steps

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # Scale t to be within the embedding range
        t_scaled = (t * 999).long()  # Scale to 0-999
        return self.embedding(t_scaled)

# --- Noise Scheduling ---
def cosine_noise_schedule(timesteps: int) -> torch.Tensor:
    steps = torch.linspace(0, 1, timesteps + 1)
    alphas_cumprod = torch.cos(((steps + 0.008) / 1.008) * math.pi / 2) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

def linear_noise_schedule(timesteps: int) -> torch.Tensor:
    return torch.linspace(1e-4, 0.02, timesteps)  # Example values


# --- Main MambaDiffuser Model ---
class MambaDiffuser(nn.Module):
    def __init__(self, config: MambaDiffuserConfig):
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.config.vocab_size = self.tokenizer.vocab_size
        self.embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, config.max_seq_len, config.d_model) * 0.02)
        self.mamba_layers = nn.ModuleList([MambaBlock(config.d_model, config.d_model * config.expand, dt_rank=config.dt_rank, d_state=config.d_state) for _ in range(config.n_layer)])
        self.transformer_layers = nn.ModuleList([TransformerBlock(config.d_model, 8) for _ in range(config.n_layer)]) # num_heads hardcoded to 8

        if config.tier >= 2:
            self.timestep_embedding = TimestepEmbedding(config.d_model)

        self.output_layer = nn.Linear(config.d_model, config.vocab_size)
        self.noise_schedule_fn = {
            "cosine": cosine_noise_schedule,
            "linear": linear_noise_schedule,
        }[config.noise_schedule]

        self.to(config.device)

    def forward(self, input_ids: torch.Tensor, timestep: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None, use_cache: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:

        x = self.embedding(input_ids)
        # Correctly handle positional encoding addition
        if input_ids.size(1) > self.pos_encoding.size(1):
            raise ValueError("Input sequence length exceeds maximum sequence length.")
        x = x + self.pos_encoding[:, :input_ids.size(1)].to(x.device)


        if self.config.tier >= 2 and timestep is not None:
            t_emb = self.timestep_embedding(timestep).to(x.device)
            x = x + t_emb.unsqueeze(1)

        kv_caches = [None] * len(self.transformer_layers) if use_cache else None
        for i, (mamba_layer, transformer_layer) in enumerate(zip(self.mamba_layers, self.transformer_layers)):
            x, _ = mamba_layer(x)  # Mamba now returns a tuple
            x, new_kv_cache = transformer_layer(x, kv_caches[i] if use_cache else None)
            if use_cache:
              kv_caches[i] = new_kv_cache

        logits = self.output_layer(x)
        loss = None
        if labels is not None:
          loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), labels.view(-1))
        return logits, loss

    def diffuse(self, input_ids: torch.Tensor, steps: Optional[int] = None) -> torch.Tensor:
        x_t = input_ids.clone()
        if steps is None:
          steps = self.config.diffusion_steps
        betas = self.noise_schedule_fn(steps).to(self.config.device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)

        with torch.no_grad():
            for i in reversed(range(steps)):
                t = torch.tensor([i / steps] * x_t.size(0), device=self.config.device)  # Correct batch size for t
                if self.config.use_classifier_free_guidance and torch.rand(1) < 0.1:
                    input_ids_cond = torch.full_like(input_ids, self.tokenizer.pad_token_id)
                    logits_cond, _ = self.forward(input_ids_cond, t, use_cache=self.config.use_cache)
                    logits_uncond, _ = self.forward(x_t, t, use_cache=self.config.use_cache)
                    pred_logits = logits_uncond + self.config.guidance_scale * (logits_cond - logits_uncond)
                else:
                    pred_logits, _ = self.forward(x_t, t, use_cache=self.config.use_cache)

                predicted_noise = torch.randn_like(pred_logits, dtype=torch.float) # Noise is float
                alpha_t = alphas_cumprod[i]
                sigma_t = torch.sqrt(1-alpha_t)
                predicted_x0 = (x_t - sigma_t * predicted_noise) / torch.sqrt(alpha_t)
                # Correctly sample and reshape
                sampled_indices = torch.multinomial(F.softmax(pred_logits, dim=-1).view(-1, self.config.vocab_size), num_samples=1)
                x_t = sampled_indices.view(x_t.shape)
        return x_t

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int = 50) -> str:
        self.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=self.config.max_seq_len).input_ids.to(self.config.device)
        generated_ids = self.diffuse(input_ids)
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

# --- Training Loop ---
def train_mamba_diffuser(model: MambaDiffuser, dataset, epochs: int = 10, batch_size: int = 32, lr: float = 1e-4):

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs * len(dataset) // batch_size) # Cosine annealing
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(epochs):
        for i in range(0, len(dataset), batch_size):
            batch = dataset[i:i + batch_size]
            # Tokenize the entire batch at once
            encoding = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=model.config.max_seq_len)
            input_ids = encoding.input_ids.to(model.config.device)
            timesteps = torch.rand(input_ids.size(0), device=model.config.device)
            noise = torch.randn_like(input_ids,dtype=torch.float)
            betas = model.noise_schedule_fn(model.config.diffusion_steps).to(model.config.device)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, axis=0)
            sqrt_alpha_cumprod = torch.sqrt(alphas_cumprod)
            sqrt_one_minus_alpha_cumprod = torch.sqrt(1-alphas_cumprod)

            batch_alphas_cumprod = torch.gather(alphas_cumprod, 0, (timesteps * model.config.diffusion_steps).long())
            sqrt_alpha_cumprod_t = torch.sqrt(batch_alphas_cumprod).unsqueeze(-1)
            sqrt_one_minus_alpha_cumprod_t = torch.sqrt(1. - batch_alphas_cumprod).unsqueeze(-1)

            noised_ids = (sqrt_alpha_cumprod_t.unsqueeze(-1) * input_ids.float() + sqrt_one_minus_alpha_cumprod_t.unsqueeze(-1)  * noise).long()
            model.train()
            with torch.cuda.amp.autocast():
                logits, loss = model(noised_ids, timesteps, labels=input_ids, use_cache=model.config.use_cache) # Pass timesteps during training

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Gradient clipping
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if i % 100 == 0:
              print(f'Loss {loss.item()} at Epoch: {epoch} and batch: {i}')
        gc.collect() # Force garbage collection

# --- Unit Tests ---
class TestMambaDiffuser(unittest.TestCase):
    def setUp(self):
        self.config = MambaDiffuserConfig(tier=2, use_classifier_free_guidance=True, use_cache=True)
        self.model = MambaDiffuser(self.config)
        self.dataset = ["This is a test sentence.", "Another test sentence here.", "And a third one."] * 50
        gc.collect()

    def test_training(self):
        start_time = time.time()
        train_mamba_diffuser(self.model, self.dataset, epochs=1, batch_size=2) # Short training for test
        self.assertLess(time.time() - start_time, 60)  # Training shouldn't take too long (adjust as needed)

    def test_generation(self):
        prompt = "This is"
        start_time = time.time()
        generated_text = self.model.generate(prompt, max_new_tokens=10)
        gen_time = time.time() - start_time
        self.assertLess(gen_time, 10.0) # Should generate fairly quickly
        self.assertTrue(isinstance(generated_text, str))
        self.assertGreater(len(generated_text), len(prompt)) # Should generate *something*
        print(f"Generated text: {generated_text}")

    def test_mamba_block(self):
      batch_size = 4
      seq_len = 16
      dim = self.config.d_model

      mamba_block = MambaBlock(dim, dim * 2).to(self.config.device) # hidden_dim = dim * expand, and move to device
      x = torch.randn(batch_size, seq_len, dim).to(self.config.device) # Ensure x is on the correct device
      y, _ = mamba_block(x)
      self.assertEqual(y.shape, (batch_size, seq_len, dim))

    def test_shapes(self):
        batch_size = 2
        seq_length = 20
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length)).to(self.config.device)
        timesteps = torch.rand(batch_size).to(self.config.device)
        logits, loss = self.model(input_ids, timesteps, labels=input_ids)
        self.assertEqual(logits.shape, (batch_size, seq_length, self.config.vocab_size))
        self.assertIsNotNone(loss)


# --- Main Execution (for demonstration) ---
if __name__ == "__main__":
    if "unittest" in __import__("sys").modules:
        # Run unit tests
        unittest.main(argv=['first-arg-is-ignored'], exit=False)
    else:
        # Full training and generation example

        # Larger dataset for demonstration
        dataset = [
            "This is the first sentence of a longer example dataset.",
            "Here's another sentence to make it more interesting.",
            "The model should be able to learn from this data.",
            "And generate coherent and relevant text.",
            "We can also test the classifier-free guidance.",
            "Let's see how the model performs with more data.",
            "This is a crucial step for demonstrating its power."
        ] * 200 # Increased dataset size

        config = MambaDiffuserConfig(tier=2, use_classifier_free_guidance=True, use_cache = True)
        model = MambaDiffuser(config)
        print("Starting training...")
        train_mamba_diffuser(model, dataset, epochs=3, batch_size=8) # Increased batch_size
        prompt = "This is a"
        print(f"Generating text with prompt: '{prompt}'")
        generated_text = model.generate(prompt, max_new_tokens=30)
        print(f"Generated text: {generated_text}")
        process = psutil.Process()
        memory_usage = process.memory_info().rss / 1024 / 1024 # in MB
        print(f"Memory Usage: {memory_usage:.2f} MB")