#!/usr/bin/env python3
"""MBDTextDiffusion: Hybrid of MBDS and TextDiffusion."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from collections import deque
import numpy as np
from torch.cuda.amp import autocast

# Constants
VOCAB_SIZE = 50_000
DEFAULT_L_PRIME = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class Config:
    vocab_size: int = VOCAB_SIZE
    embed_dim: int = 256  # Must be divisible by num_heads when scaled
    num_heads: int = 4
    num_layers: int = 2
    hidden_dim: int = 512
    l_prime: int = DEFAULT_L_PRIME
    max_seq_len: int = 128
    buffer_size: int = 50
    device: str = DEVICE.type


def noise_schedule(t: float) -> Tuple[float, float]:
    """Noise schedule from TextDiffusion."""
    beta = 0.1 + 0.5 * t
    alpha = 1.0 - beta
    return alpha, beta


class MBDTransformerBlock(nn.Module):
    """Hybrid block combining Mamba and Transformer features."""

    def __init__(self, cfg: Config, tier: str):
        super().__init__()
        self.tier = tier.lower()
        self.cfg = cfg

        # Embedding shared across blocks
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)

        # Mamba-like MLP for simple tier
        if tier == "simple":
            self.mamba = nn.Sequential(
                nn.Linear(cfg.embed_dim * 2, cfg.hidden_dim),
                nn.ReLU(),
                nn.Linear(cfg.hidden_dim, cfg.embed_dim)
            )
        else:
            # Transformer components
            self.norm1 = nn.LayerNorm(cfg.embed_dim)
            self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
            self.norm2 = nn.LayerNorm(cfg.embed_dim)
            self.ffn = nn.Sequential(
                nn.Linear(cfg.embed_dim, cfg.hidden_dim),
                nn.GELU(),
                nn.Linear(cfg.hidden_dim, cfg.embed_dim)
            )

        self.denoise = nn.Linear(cfg.embed_dim, cfg.vocab_size)
        if tier in ["enhanced", "extreme", "simple"]:
            self._quantize_weights()

    def _quantize_weights(self):
        """Quantize weights to 4-bit."""
        for param in self.parameters():
            param.data = torch.round(param.data * 15) / 15

    def forward(self, x_t: torch.Tensor, x_prev: torch.Tensor, kv_cache: Optional[Tuple] = None) -> Tuple[
        torch.Tensor, Optional[Tuple]]:
        x_t_embed = self.embedding(x_t)
        x_prev_embed = self.embedding(x_prev).mean(dim=1, keepdim=True).expand_as(x_t_embed)
        x_input = torch.cat([x_t_embed, x_prev_embed], dim=-1) if self.tier == "simple" else x_t_embed

        if self.tier == "simple":
            h = self.mamba(x_input)
            kv_cache = None
        else:
            x = self.norm1(x_input)
            attn_out, _ = self.attn(x, kv_cache[0], kv_cache[1], need_weights=False) if kv_cache else self.attn(x, x, x,
                                                                                                                need_weights=False)
            x = x + attn_out
            h = x + self.ffn(self.norm2(x))
            kv_cache = (x, x) if not kv_cache else kv_cache

        logits = self.denoise(h)
        return logits, kv_cache


class MBDTextDiffusion(nn.Module):
    """Hybrid MBD and TextDiffusion model."""

    def __init__(self, tier: str = "core", vocab: Optional[Dict[str, int]] = None, config: Optional[Config] = None,
                 width: float = 1.0):
        super().__init__()
        self.tier = tier.lower()
        if self.tier not in ["simple", "core", "enhanced", "extreme"]:
            raise ValueError("Tier must be 'simple', 'core', 'enhanced', or 'extreme'")
        self.cfg = config or Config()
        self.vocab = vocab or {f"w{i}": i for i in range(self.cfg.vocab_size - 2)}
        self.vocab["<unk>"] = self.cfg.vocab_size - 1
        self.vocab["<eos>"] = self.cfg.vocab_size - 2
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.cfg.vocab_size = len(self.vocab)

        # Scale dimensions and ensure embed_dim is divisible by num_heads
        self.cfg.embed_dim = int(self.cfg.embed_dim * width)
        if self.cfg.embed_dim % self.cfg.num_heads != 0:
            self.cfg.embed_dim = ((self.cfg.embed_dim // self.cfg.num_heads) + 1) * self.cfg.num_heads
        self.cfg.hidden_dim = int(self.cfg.hidden_dim * width)

        self.pos_encoding = nn.Parameter(torch.randn(1, self.cfg.max_seq_len, self.cfg.embed_dim) * 0.02)
        self.layers = nn.ModuleList([MBDTransformerBlock(self.cfg, tier) for _ in range(self.cfg.num_layers)])
        self.buffer = deque(maxlen=self.cfg.buffer_size)

        # Tier-specific settings
        self.T = {"simple": 1, "core": 5, "enhanced": 3, "extreme": 1}[tier]
        self.tools = {"word_count": lambda x: len(x.split())} if tier in ["enhanced", "extreme"] else {}
        self.remask_threshold = 0.2 if tier == "extreme" else None
        self.to(self.cfg.device)

    def _split_blocks(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len = x.shape
        padded_len = ((seq_len + self.cfg.l_prime - 1) // self.cfg.l_prime) * self.cfg.l_prime
        x_padded = F.pad(x, (0, padded_len - seq_len), value=self.vocab["<unk>"])
        return x_padded.view(batch, -1, self.cfg.l_prime)

    def forward(self, x: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[float]]:
        blocks = self._split_blocks(x)
        batch, n_blocks, _ = blocks.shape
        logits = []
        kv_cache = None

        for b in range(n_blocks):
            x_prev = blocks[:, :b].flatten(1) if b > 0 else torch.full((batch, 1), self.vocab["<unk>"],
                                                                       device=self.cfg.device)
            x_b = blocks[:, b]
            t = np.random.random()
            alpha, beta = noise_schedule(t)
            x_t_b = (alpha * x_b + beta * torch.randint(0, self.cfg.vocab_size, x_b.shape,
                                                        device=self.cfg.device)).long()
            x_t_b_embed = self.layers[0].embedding(x_t_b) + self.pos_encoding[:, :x_t_b.size(1)]

            block_logits = []
            for layer in self.layers:
                layer_out, kv_cache = layer(x_t_b, x_prev, kv_cache if self.tier != "simple" else None)
                block_logits.append(layer_out)
                x_t_b = layer_out.argmax(-1) if self.tier == "simple" else x_t_b
            logits.append(torch.stack(block_logits, dim=0).mean(0))

        logits = torch.stack(logits, dim=1)
        loss = None
        if targets is not None:
            targets_blocks = self._split_blocks(targets)
            loss = F.cross_entropy(logits.flatten(0, 2), targets_blocks.flatten(), reduction="mean")
        return logits, loss

    def generate(self, prompt: str, max_blocks: int = 10, temperature: float = 0.7, top_k: int = 40) -> str:
        self.eval()
        prompt_tokens = torch.tensor([self.vocab.get(w, self.vocab["<unk>"]) for w in prompt.split()],
                                     dtype=torch.long, device=self.cfg.device).unsqueeze(0)
        blocks = self._split_blocks(prompt_tokens)
        batch = blocks.shape[0]
        generated = blocks.clone()
        kv_cache = None

        with torch.no_grad():
            for _ in range(max_blocks):
                if generated.flatten().tolist().count(self.vocab["<eos>"]) > 0:
                    break
                x_prev = generated[:, :-1].flatten(1) if generated.shape[1] > 1 else torch.full((batch, 1),
                                                                                                self.vocab["<unk>"],
                                                                                                device=self.cfg.device)
                x_m_b = torch.full((batch, self.cfg.l_prime), self.vocab["<unk>"], dtype=torch.long,
                                   device=self.cfg.device)

                for t in range(self.T, 0, -1):
                    with autocast():
                        block_logits = []
                        for layer in self.layers:
                            logits_b, kv_cache = layer(x_m_b, x_prev, kv_cache if self.tier != "simple" else None)
                            block_logits.append(logits_b)
                        logits_b = torch.stack(block_logits, dim=0).mean(0)

                        if t > 1 and self.tier in ["core", "extreme"]:
                            probs = F.softmax(logits_b / temperature, dim=-1)
                            top_k_probs, top_k_indices = probs.topk(top_k, dim=-1)
                            sampled_indices = torch.multinomial(top_k_probs.view(-1, top_k), 1).view(batch,
                                                                                                     self.cfg.l_prime)
                            x_m_b = top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)
                        else:
                            x_m_b = logits_b.argmax(-1)

                        if self.tier == "extreme" and t > 1:
                            confidence = F.softmax(logits_b, dim=-1).max(-1)[0]
                            mask = confidence < self.remask_threshold
                            if mask.any():
                                x_m_b[mask] = torch.randint(0, self.cfg.vocab_size, (mask.sum(),),
                                                            device=self.cfg.device)

                generated = torch.cat([generated, x_m_b.unsqueeze(1)], dim=1)

            # Handle tools
            output_words = [self.inv_vocab[t.item()] for t in generated.flatten()]
            if self.tier in ["enhanced", "extreme"]:
                for i, word in enumerate(output_words):
                    if word.startswith("[TOOL:"):
                        tool_name = word[6:-1]
                        if tool_name in self.tools:
                            clean_text = " ".join(w for w in output_words if not w.startswith("[TOOL:"))
                            output_words[i] = str(self.tools[tool_name](clean_text))
            return " ".join(output_words[:self.cfg.max_seq_len]).strip()

    def train_step(self, text: str, lr: float = 1e-4):
        """Single-step training."""
        x = torch.tensor([self.vocab.get(w, self.vocab["<unk>"]) for w in text.split()],
                         dtype=torch.long, device=self.cfg.device).unsqueeze(0)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr)
        logits, loss = self.forward(x, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        self.buffer.append((x.clone(), loss.item()))

    def add_tool(self, name: str, code: str):
        """Add a tool for enhanced/extreme tiers."""
        if self.tier not in ["enhanced", "extreme"]:
            raise ValueError("Tools require 'enhanced' or 'extreme' tier")
        try:
            tool_fn = eval(f"lambda x: {code}", {"len": len, "list": list}, {})
            self.tools[name] = tool_fn
            tool_token = f"[TOOL: {name}]"
            if tool_token not in self.vocab:
                self.vocab[tool_token] = len(self.vocab)
                self.inv_vocab[len(self.inv_vocab)] = tool_token
                self.cfg.vocab_size = len(self.vocab)
                for layer in self.layers:
                    layer.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.embed_dim).to(self.cfg.device)
        except Exception as e:
            raise ValueError(f"Invalid tool code: {e}")


def create_hybrid_model(tier: str, width: float = 1.0, vocab: Optional[Dict] = None) -> 'MBDTextDiffusion':
    return MBDTextDiffusion(tier=tier, vocab=vocab, width=width)


def train_model(model: 'MBDTextDiffusion', data: torch.Tensor, epochs: int = 1, lr: float = 1e-4):
    """Train model on dataset."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for _ in range(epochs):
        for batch in data.split(32, dim=0):  # Mini-batch size 32
            logits, loss = model(batch, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


if __name__ == "__main__":
    # Demonstration of power
    vocab = {"meeting": 0, "with": 1, "alex": 2, "at": 3, "pm": 4, "3": 5, "<unk>": 6, "<eos>": 7}
    model = create_hybrid_model("extreme", width=0.5, vocab=vocab)

    # Train on sample data
    training_texts = [
        "meeting with alex at 3 pm <eos>",
        "meeting with alex at pm <eos>",
    ]
    for text in training_texts:
        for _ in range(10):  # Train for 10 steps
            model.train_step(text)

    # Add a custom tool
    model.add_tool("upper_count", "len([w for w in x.split() if w[0].isupper()])")

    # Generate with tool
    prompt = "meeting with [TOOL: upper_count]"
    generated = model.generate(prompt, max_blocks=5)
    print(f"Prompt: {prompt}")
    print(f"Generated: {generated}")

    # Generate without tool
    prompt = "meeting with alex"
    generated = model.generate(prompt, max_blocks=5)
    print(f"\nPrompt: {prompt}")
    print(f"Generated: {generated}")