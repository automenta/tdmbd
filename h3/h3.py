#!/usr/bin/env python3
"""TextMambaFusion: Enhanced with noise prediction and tool support."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer
from typing import Optional, Tuple, Dict
from dataclasses import dataclass
import math

# Constants
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VOCAB_SIZE = 50257  # GPT-2 vocab size


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    d_model: int = 128 #384
    n_layers: int = 3 #6
    d_state: int = 16
    d_inner: int = 128 #768
    max_seq_len: int = 32 #512
    diffusion_steps: int = 8 #50
    tier: str = "core"
    guidance_scale: float = 2.0
    device: str = DEVICE.type


class MambaVectorField(nn.Module):
    def __init__(self, d_model: int, d_inner: int, d_state: int):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner
        self.d_state = d_state

        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(d_inner, d_inner, kernel_size=4, padding=3, groups=d_inner)
        self.norm = nn.LayerNorm(d_inner)
        self.x_proj = nn.Linear(d_inner, d_state * 3, bias=False)
        self.dt_proj = nn.Linear(d_state, d_state, bias=True)

        self.A_log = nn.Parameter(torch.zeros(d_state))
        self.D = nn.Parameter(torch.ones(d_inner) * 0.1)
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

        nn.init.normal_(self.in_proj.weight, std=0.02)
        nn.init.normal_(self.out_proj.weight, std=0.02)
        nn.init.normal_(self.dt_proj.weight, std=0.01)
        dt = torch.rand(d_state) * 0.1
        self.dt_proj.bias.data = dt

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        x_and_res, gate = self.in_proj(x).split([self.d_inner, self.d_inner], dim=-1)
        if t_emb is not None:
            x_and_res = x_and_res + t_emb.unsqueeze(1).expand(-1, seq_len, -1)

        x_and_res = self.norm(F.silu(self.conv1d(x_and_res.transpose(1, 2))[:, :, :-3].transpose(1, 2)))

        x_proj = self.x_proj(x_and_res)
        B, C, dt = torch.split(x_proj, [self.d_state, self.d_state, self.d_state], dim=-1)
        dt = torch.tanh(self.dt_proj(dt)) * 0.01

        A = -torch.tanh(self.A_log)
        x_out = self._ssm_scan(x_and_res, dt, A, B, C)

        return self.out_proj(x_out * F.silu(gate))

    def _ssm_scan(self, x: torch.Tensor, dt: torch.Tensor, A: torch.Tensor, B: torch.Tensor,
                  C: torch.Tensor) -> torch.Tensor:
        batch, seq_len, d_inner = x.shape
        d_state = self.d_state

        A_discrete = 1 + dt * A[None, None, :]
        B_discrete = B * x.norm(dim=-1, keepdim=True).clamp(max=1.0)

        h = torch.zeros(batch, d_state, d_inner, device=x.device, dtype=x.dtype)
        y = torch.zeros_like(x)

        for t in range(seq_len):
            h = h * A_discrete[:, t, :, None] + B_discrete[:, t, :, None] * x[:, t, None, :]
            h = torch.clamp(h, min=-10, max=10)
            y[:, t] = torch.einsum('bd,bdh->bh', C[:, t], h) + self.D * x[:, t]

        return y


class FusionBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.mamba = MambaVectorField(cfg.d_model, cfg.d_inner, cfg.d_state)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.attn = nn.MultiheadAttention(cfg.d_model, 4, batch_first=True) if cfg.tier == "enhanced" else None

    def forward(self, x: torch.Tensor, t_emb: Optional[torch.Tensor] = None,
                kv_cache: Optional[Tuple] = None) -> Tuple[torch.Tensor, Optional[Tuple]]:
        h = self.mamba(x, t_emb)

        if self.attn:
            attn_in = self.norm(h)
            attn_out, _ = self.attn(attn_in, kv_cache[0], kv_cache[1]) if kv_cache else self.attn(attn_in, attn_in,
                                                                                                  attn_in)
            h = h + attn_out
            kv_cache = (attn_in, attn_in) if not kv_cache else kv_cache
        else:
            kv_cache = None

        return x + h, kv_cache


class TextMambaFusion(nn.Module):
    def __init__(self, cfg: Config = None):
        super().__init__()
        self.cfg = cfg or Config()
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, self.cfg.max_seq_len, self.cfg.d_model) * 0.01)
        self.t_emb = nn.Embedding(self.cfg.diffusion_steps, self.cfg.d_inner)
        self.layers = nn.ModuleList([FusionBlock(self.cfg) for _ in range(self.cfg.n_layers)])
        self.output = nn.Linear(self.cfg.d_model, self.cfg.vocab_size)

        nn.init.normal_(self.output.weight, std=0.02)
        self.tools = {"word_count": lambda x: len(x.split())} if self.cfg.tier in ["core", "enhanced"] else {}
        self.to(self.cfg.device)

    def _cosine_schedule(self, steps: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = torch.linspace(0, 1, steps + 1, device=self.cfg.device)
        alphas_cumprod = torch.cos((s + 0.008) / 1.008 * math.pi / 2) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.999), alphas_cumprod[:-1]

    def forward(self, x: torch.Tensor, t: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[
        torch.Tensor, Optional[float]]:
        if x.dim() > 2:
            x = x.view(-1, x.size(-1))
        batch_size, seq_len = x.shape

        x_emb = self.embedding(x)
        pos_enc = self.pos_encoding[:, :seq_len, :].expand(batch_size, -1, -1)
        x_emb = x_emb + pos_enc

        t_emb = self.t_emb(t)

        kv_cache = None
        for layer in self.layers:
            x_emb, kv_cache = layer(x_emb, t_emb, kv_cache)

        logits = self.output(x_emb)
        loss = F.cross_entropy(logits.view(-1, self.cfg.vocab_size), targets.view(-1),
                               ignore_index=self.tokenizer.pad_token_id) if targets is not None else None
        return logits, loss

    @torch.no_grad()
    def generate(self, prompt: str, max_len: int = 100, temperature: float = 1.0) -> str:
        self.eval()
        input_ids = self.tokenizer(prompt, return_tensors="pt", truncation=True,
                                   max_length=self.cfg.max_seq_len).input_ids.to(self.cfg.device)
        betas, alphas_cumprod = self._cosine_schedule(self.cfg.diffusion_steps)
        alphas = 1 - betas

        x_t = input_ids.float()  # Float for diffusion
        with torch.amp.autocast('cuda'):
            for step in reversed(range(self.cfg.diffusion_steps)):
                t = torch.full((x_t.size(0),), step, device=self.cfg.device)
                alpha_t = alphas_cumprod[step]
                sigma_t = (1 - alpha_t).sqrt()

                logits_cond, _ = self.forward(x_t.long(), t)
                if self.cfg.tier == "enhanced" and torch.rand(1) < 0.2:
                    logits_uncond, _ = self.forward(torch.full_like(x_t, self.tokenizer.pad_token_id).long(), t)
                    logits = logits_uncond + self.cfg.guidance_scale * (logits_cond - logits_uncond)
                else:
                    logits = logits_cond

                logits = torch.clamp(logits, min=-30, max=30)
                probs = F.softmax(logits / temperature, dim=-1)

                if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                    print(f"Warning: Invalid probs at step {step}")
                    probs = torch.nan_to_num(probs, nan=1e-10, posinf=1e-10, neginf=0)

                batch_size, seq_len, vocab_size = probs.shape
                probs = probs.view(batch_size * seq_len, vocab_size)
                x_t_next = torch.multinomial(probs, 1).view(batch_size, seq_len).float()

                noise_pred = (x_t - alpha_t.sqrt() * x_t_next) / sigma_t
                x_t = alpha_t.sqrt() * x_t_next + sigma_t * noise_pred.clamp(-1, 1)

                if x_t.size(-1) >= max_len or self.tokenizer.eos_token_id in x_t.long().flatten():
                    break

        x_t = x_t.clamp(0, self.cfg.vocab_size - 1).long()
        text = self.tokenizer.decode(x_t[0], skip_special_tokens=True)
        for name, fn in self.tools.items():
            if f"[TOOL:{name}]" in text:
                text = text.replace(f"[TOOL:{name}]", str(fn(text)))
        return text

    def add_tool(self, name: str, fn):
        if self.cfg.tier not in ["core", "enhanced"]:
            raise ValueError("Tools require 'core' or 'enhanced' tier")
        self.tools[name] = fn


def train_model(model: "TextMambaFusion", data: list, epochs: int = 5, batch_size: int = 16, lr: float = 1e-4):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    betas, alphas_cumprod = model._cosine_schedule(model.cfg.diffusion_steps)
    scaler = torch.amp.GradScaler('cuda')

    for epoch in range(epochs):
        for i in range(0, len(data), batch_size):
            batch = data[i:i + batch_size]
            inputs = model.tokenizer(batch, return_tensors="pt", padding=True, truncation=True,
                                     max_length=model.cfg.max_seq_len).input_ids.to(model.cfg.device)

            t = torch.randint(0, model.cfg.diffusion_steps, (inputs.size(0),), device=model.cfg.device)
            alpha_t = alphas_cumprod[t].view(-1, 1)
            noise = torch.randn_like(inputs.float()) * 0.1
            x_t = (alpha_t.sqrt() * inputs + (1 - alpha_t).sqrt() * noise).long().clamp(0, model.cfg.vocab_size - 1)

            with torch.amp.autocast('cuda'):
                logits, loss = model(x_t, t, inputs)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        print(f"Epoch {epoch + 1}, Loss: {loss.item():.4f}")


if __name__ == "__main__":
    cfg = Config(tier="enhanced")
    #cfg = Config()
    model = TextMambaFusion(cfg)

    model.add_tool("word_count", lambda x: len(x.split()))

    #data = ["Hello world this is a test", "Another example sentence here"] * 10
    data = [
               "Hello world this is a test",
               "Another example sentence here",
               "The quick brown fox jumps over the lazy dog",
               "Python is a great programming language",
               "I enjoy coding late at night",
               "This sentence has exactly seven words",
               "Rainy days are perfect for reading books",
               "Machine learning models require lots of data",
               "Coffee helps me stay awake during meetings",
               "The sun sets beautifully over the ocean",
               "Hello [TOOL:word_count] this is five",
               "Another [TOOL:word_count] example with six",
               "Short and sweet sentence here",
               "Long winding sentences can be quite challenging",
               "Data science is an exciting field today",
               "Cats prefer napping all day long",
               "Dogs love chasing after tennis balls",
               "Programming requires patience and practice",
               "This is a random test sentence",
               "Sunshine makes everything feel much better",
           ] * 4
    train_model(model, data, epochs=256)

    prompt = "Hello [TOOL:word_count]"
    print(model.generate(prompt, max_len=50))