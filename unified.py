#!/usr/bin/env python3
"""TextDiffusion: A modular, tiered text diffusion system."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Callable, Optional, Tuple, List
from dataclasses import dataclass
import numpy as np
from collections import deque
import gc

@dataclass
class Config:
    vocab_size: int = 13
    embed_dim: int = 8
    num_heads: int = 2
    num_layers: int = 1
    block_size: int = 4
    max_seq_len: int = 8
    buffer_size: int = 50
    device: str = "cpu"

def noise_schedule(t: float) -> Tuple[float, float]:
    beta = 0.1 + 0.5 * t
    alpha = 1.0 - beta
    return alpha, beta

def tokenize(text: str, vocab: Dict[str, int]) -> torch.Tensor:
    tokens = [vocab.get(w, vocab["<unk>"]) for w in text.split()]
    return torch.tensor(tokens, dtype=torch.long).unsqueeze(0)

def detokenize(tokens: torch.Tensor, inv_vocab: Dict[int, str]) -> str:
    return " ".join(inv_vocab.get(t.item(), "<unk>") for t in tokens.squeeze(0))

class TransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, 4 * cfg.embed_dim),
            nn.GELU(),
            nn.Linear(4 * cfg.embed_dim, cfg.embed_dim),
            nn.Dropout(0.1)
        )

    def forward(self, x: torch.Tensor, kv_cache: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        x_norm = self.norm1(x)
        if kv_cache is not None:
            attn_out, _ = self.attn(x_norm, kv_cache[0], kv_cache[1], need_weights=False)
            k, v = kv_cache
        else:
            attn_out, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
            k, v = x_norm, x_norm
        x = x + attn_out
        x = x + self.ffn(self.norm2(x))
        return x, (k, v)

class DiffusionCore(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.embed_dim)
        self.pos_encoding = nn.Parameter(torch.randn(1, cfg.max_seq_len, cfg.embed_dim) * 0.02)
        self.layers = nn.ModuleList([TransformerBlock(cfg) for _ in range(cfg.num_layers)])
        self.output = nn.Linear(cfg.embed_dim, cfg.vocab_size)
        self.buffer = deque(maxlen=cfg.buffer_size)

    def forward(self, x: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        seq_len = min(x.size(1), self.cfg.max_seq_len)
        x = x[:, :seq_len]
        x = self.embedding(x) + self.pos_encoding[:, :seq_len]
        kv_cache = None
        for layer in self.layers:
            x, kv = layer(x, kv_cache if use_cache else None)
            if use_cache:
                kv_cache = kv
        return self.output(x)

    def add_to_buffer(self, x: torch.Tensor, loss: float):
        self.buffer.append((x.clone(), loss))

class TextDiffusion:
    def __init__(self, tier: int = 0, vocab: Optional[Dict[str, int]] = None, config: Optional[Config] = None):
        if not 0 <= tier <= 3:
            raise ValueError("Tier must be between 0 and 3")
        self.cfg = config or Config()
        self.tier = tier
        self.vocab = vocab or self._create_default_vocab()
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.cfg.vocab_size = len(self.vocab)
        self.model = DiffusionCore(self.cfg).to(self.cfg.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.use_cache = tier >= 1
        self.tools = {"word_count": lambda x: len(x.split())} if tier >= 2 else {}
        self.remask_threshold = 0.2 if tier >= 3 else None

    def _create_default_vocab(self) -> Dict[str, int]:
        vocab = {f"w{i}": i for i in range(self.cfg.vocab_size - 1)}
        vocab["<unk>"] = self.cfg.vocab_size - 1
        return vocab

    def _diffuse(self, x: torch.Tensor, steps: int = 5) -> torch.Tensor:
        x_t = x.clone()
        with torch.no_grad():
            for step in range(steps):
                t = step / steps
                alpha, beta = noise_schedule(t)
                pred_logits = self.model(x_t, use_cache=self.use_cache)
                x_t = (alpha * x_t + beta * pred_logits.argmax(-1)).long()
                if self.tier >= 1 and (x_t == x).float().mean() > 0.9:
                    break
                if self.tier >= 3 and step < steps - 1:
                    confidence = F.softmax(pred_logits, dim=-1).max(-1)[0]
                    mask = confidence < self.remask_threshold
                    if mask.any():
                        x_t[mask] = torch.randint(0, self.cfg.vocab_size, (mask.sum(),), device=self.cfg.device)
        return x_t

    def train_step(self, text: str):
        x = tokenize(text, self.vocab).to(self.cfg.device)[:, :self.cfg.max_seq_len]
        if x.size(1) == 0:
            x = torch.tensor([[self.vocab["<unk>"]]], device=self.cfg.device)
        t = np.random.random()
        alpha, beta = noise_schedule(t)

        with torch.no_grad():
            x_noised = (alpha * x + beta * torch.randint(0, self.cfg.vocab_size, x.shape, device=self.cfg.device)).long()
        pred_logits = self.model(x_noised, use_cache=self.use_cache)
        loss = F.cross_entropy(pred_logits.view(-1, self.cfg.vocab_size), x.view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.model.add_to_buffer(x[0], loss.item())
        gc.collect()

    def generate(self, prompt: str, max_len: int = 32) -> str:
        with torch.no_grad():
            max_len = min(max_len, self.cfg.max_seq_len)
            prompt_tokens = tokenize(prompt, self.vocab).to(self.cfg.device)[:, :max_len]
            x = prompt_tokens.clone()
            prompt_len = min(x.size(1), max_len)
            tool_pos = None

            # Handle tools for tiers >= 2
            if self.tier >= 2:
                prompt_words = prompt.split()
                for i, token in enumerate(prompt_words):
                    if token.startswith("[TOOL:"):
                        tool_name = token[6:-1]
                        if tool_name in self.tools:
                            clean_prompt = " ".join(w for w in prompt_words if not w.startswith("[TOOL:"))
                            tool_result = str(self.tools[tool_name](clean_prompt))
                            result_token = tokenize(tool_result, self.vocab).to(self.cfg.device)[0, 0]
                            if i < x.size(1):
                                x[:, i] = result_token
                                tool_pos = i
                            else:
                                x = torch.cat([x, result_token.unsqueeze(0).unsqueeze(0)], dim=1)[:, :max_len]
                                tool_pos = x.size(1) - 1
                            prompt_len = min(x.size(1), max_len)
                            break

            # Diffuse or generate
            if self.tier == 0:
                x = self._diffuse(x)[:, :max_len]
            elif prompt_len < max_len:
                # Only diffuse/generate after the prompt/tool portion
                start_pos = tool_pos + 1 if tool_pos is not None else prompt_len
                gen_part = x[:, start_pos:] if start_pos < x.size(1) else torch.tensor([], dtype=torch.long,
                                                                                       device=self.cfg.device).unsqueeze(
                    0)
                if gen_part.size(1) > 0:
                    gen_part = self._diffuse(gen_part)[:, :max_len - start_pos]
                    x = torch.cat([x[:, :start_pos], gen_part], dim=1)[:, :max_len]
                while x.size(1) < max_len:
                    pred_logits = self.model(x[:, -self.cfg.block_size:], use_cache=self.use_cache)
                    next_token = pred_logits[:, -1].argmax(-1).unsqueeze(-1)
                    x = torch.cat([x, next_token], dim=1)[:, :max_len]

            # Preserve prompt for tiers >= 1
            if self.tier >= 1 and prompt_len <= x.size(1):
                x[:, :prompt_len] = prompt_tokens[:, :prompt_len]

            # Final length check
            output = detokenize(x, self.inv_vocab)
            return " ".join(output.split()[:max_len])

    def add_tool(self, name: str, code: str):
        if self.tier < 2:
            raise ValueError("Tools require tier 2 or higher")
        try:
            tool_fn = eval(f"lambda x: {code}", {"len": len, "list": list}, {})
            self.tools[name] = tool_fn
            tool_token = f"[TOOL: {name}]"
            if tool_token not in self.vocab:
                self.vocab[tool_token] = len(self.vocab)
                self.inv_vocab[len(self.inv_vocab)] = tool_token
                self.cfg.vocab_size = len(self.vocab)
                self.model.embedding = nn.Embedding(self.cfg.vocab_size, self.cfg.embed_dim).to(self.cfg.device)
        except Exception as e:
            raise ValueError(f"Invalid tool code: {e}")

if __name__ == "__main__":
    vocab = {"meeting": 0, "with": 1, "alex": 2, "at": 3, "pm": 4, "3": 5, "<unk>": 6}
    diffuser = TextDiffusion(tier=2, vocab=vocab)
    diffuser.train_step("meeting with alex at pm")
    print(diffuser.generate("meeting with"))
    diffuser.add_tool("upper_count", "len([w for w in x.split() if w[0].isupper()])")
    print(diffuser.generate("meeting with [TOOL: upper_count]"))