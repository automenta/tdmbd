#!/usr/bin/env python3
"""MBD-S: Modular Mamba-Block-Diffusion Language Model Library."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch.cuda.amp import autocast
from torch import Tensor
import math

# Constants
VOCAB_SIZE = 50_000  # Default vocabulary size
DEFAULT_L_PRIME = 16  # Default block size
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MDBBlock(nn.Module):
    """Mamba-Diffusion Block: Core processing unit for all tiers."""

    def __init__(self, embed_dim: int, hidden_dim: int, vocab_size: int = VOCAB_SIZE, tier: str = "simple"):
        super().__init__()
        self.tier = tier.lower()
        self.embed_dim = embed_dim

        # Embedding layer for tokens
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Mamba Vector Field: Simplified MLP for state evolution
        layers = 1 if tier == "simple" else 2 if tier in ["core", "enhanced"] else 4
        self.mamba = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),  # Input: x_t + x_prev
            nn.ReLU(),
            *[nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()) for _ in range(layers - 1)],
            nn.Linear(hidden_dim, embed_dim)
        )

        # Denoising MLP: Predicts clean tokens
        self.denoise = nn.Linear(embed_dim, vocab_size)

        # Quantization for edge tiers
        if tier in ["enhanced", "extreme", "simple"]:
            self._quantize_weights()

    def _quantize_weights(self):
        """Quantize weights to 4-bit for edge efficiency."""
        for param in self.parameters():
            param.data = torch.round(param.data * 15) / 15  # 4-bit approximation

    def forward(self, x_t: Tensor, x_prev: Tensor) -> Tensor:
        """Process block: x_t (noisy), x_prev (context) -> logits."""
        # Embed tokens
        x_t_embed = self.embedding(x_t)  # [batch, seq_len, embed_dim]
        x_prev_embed = self.embedding(x_prev)  # [batch, prev_len, embed_dim]

        # Aggregate previous context
        x_prev_mean = x_prev_embed.mean(dim=1, keepdim=True).expand_as(x_t_embed)
        x_input = torch.cat([x_t_embed, x_prev_mean], dim=-1)  # [batch, seq_len, embed_dim * 2]

        # Mamba state evolution
        h = self.mamba(x_input)  # [batch, seq_len, embed_dim]
        return self.denoise(h)  # [batch, seq_len, vocab_size]


class MBDS(nn.Module):
    """MBD-S Language Model: Modular implementation across tiers."""

    def __init__(self,
                 vocab_size: int = VOCAB_SIZE,
                 embed_dim: int = 256,
                 hidden_dim: int = 512,
                 l_prime: int = DEFAULT_L_PRIME,
                 width: float = 1.0,
                 tier: str = "simple"):
        super().__init__()
        self.vocab_size = vocab_size
        self.l_prime = l_prime
        self.tier = tier.lower()
        self.width = width

        # Scale dimensions with width multiplier
        embed_dim = int(embed_dim * width)
        hidden_dim = int(hidden_dim * width)

        # Core MDB block
        self.mdb = MDBBlock(embed_dim, hidden_dim, vocab_size, tier)

        # Tier-specific parameters
        self.T = {"core": 10, "enhanced": 3, "extreme": 1, "simple": 1}[tier]
        self.beta, self.omega = (0.3, 0.8) if tier in ["enhanced", "extreme"] else (0, 1)
        self.stop_entropy = {"core": float("inf"), "enhanced": 4.0, "extreme": 3.5, "simple": 3.0}[tier]
        self.to(DEVICE)

    def _split_blocks(self, x: Tensor) -> List[Tensor]:
        """Split sequence into blocks of size l_prime."""
        batch, seq_len = x.shape
        padded_len = math.ceil(seq_len / self.l_prime) * self.l_prime
        x_padded = F.pad(x, (0, padded_len - seq_len), value=0)
        return x_padded.view(batch, -1, self.l_prime)

    def _mask_tokens(self, x: Tensor, mask_ratio: float) -> Tuple[Tensor, Tensor]:
        """Mask tokens with ratio, returning masked x and mask."""
        mask = torch.rand_like(x, dtype=torch.float) < mask_ratio
        return torch.where(mask, self.vocab_size - 1, x), mask  # Last token is [MASK]

    def forward(self, x: Tensor, targets: Optional[Tensor] = None) -> Tuple[Tensor, Optional[Tensor]]:
        """Forward pass: Compute logits and optional loss."""
        blocks = self._split_blocks(x)
        batch, n_blocks, _ = blocks.shape
        logits = []

        for b in range(n_blocks):
            x_prev = blocks[:, :b].flatten(1) if b > 0 else torch.zeros(batch, 1, dtype=torch.long, device=DEVICE)
            x_b = blocks[:, b]

            # Noise schedule based on tier
            t = torch.rand(1).item() * (self.omega - self.beta) + self.beta if self.tier in ["enhanced",
                                                                                             "extreme"] else 0.5
            x_t_b, mask = self._mask_tokens(x_b, t if self.tier != "simple" else 0.5)
            logits_b = self.mdb(x_t_b, x_prev)
            logits.append(logits_b)

        logits = torch.stack(logits, dim=1)  # [batch, n_blocks, l_prime, vocab_size]
        loss = None
        if targets is not None:
            targets_blocks = self._split_blocks(targets)
            loss = F.cross_entropy(logits.flatten(0, 2), targets_blocks.flatten(), reduction="mean")
        return logits, loss

    def generate(self, prompt: Tensor, max_blocks: int = 10, temperature: float = 0.7, top_k: int = 40) -> Tensor:
        """Generate sequence block-by-block with temperature and top-k sampling."""
        self.eval()
        blocks = self._split_blocks(prompt)
        batch = blocks.shape[0]
        generated = blocks.clone()
        min_blocks = 2  # Ensure at least 2 blocks

        with torch.no_grad():
            for _ in range(max_blocks):
                if generated.shape[1] >= min_blocks * self.l_prime:
                    entropy = -(F.softmax(logits_b, dim=-1) * F.log_softmax(logits_b, dim=-1)).sum(dim=-1).mean()
                    if x_m_b.eq(self.vocab_size - 2).any() or entropy < self.stop_entropy:
                        break

                x_prev = generated[:, :-1].flatten(1) if generated.shape[1] > 1 else torch.zeros(batch, 1,
                                                                                                 dtype=torch.long,
                                                                                                 device=DEVICE)
                x_m_b = torch.full((batch, self.l_prime), self.vocab_size - 1, dtype=torch.long,
                                   device=DEVICE)  # [MASK]

                for t in range(self.T, 0, -1):
                    with autocast():
                        logits_b = self.mdb(x_m_b, x_prev)  # [batch, seq_len, vocab_size]
                    if t > 1 and self.tier in ["core", "extreme"]:
                        probs = F.softmax(logits_b / temperature, dim=-1)  # [batch, seq_len, vocab_size]
                        top_k_probs, top_k_indices = probs.topk(top_k, dim=-1)  # [batch, seq_len, top_k]
                        # Reshape to 2D for multinomial
                        top_k_probs_2d = top_k_probs.view(-1, top_k)  # [batch * seq_len, top_k]
                        sampled_indices = torch.multinomial(top_k_probs_2d, 1).view(batch,
                                                                                    self.l_prime)  # [batch, seq_len]
                        x_m_b = top_k_indices.gather(-1, sampled_indices.unsqueeze(-1)).squeeze(-1)  # [batch, seq_len]
                    else:
                        x_m_b = torch.argmax(logits_b, dim=-1)

                generated = torch.cat([generated, x_m_b.unsqueeze(1)], dim=1)

        return generated.flatten(1)

    def train_online(self, x: Tensor, lr: float = 1e-6):
        """Single-step online training with EMA."""
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=0.1)
        logits, loss = self.forward(x, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # EMA update for simplified tier
        if self.tier == "simple":
            with torch.no_grad():
                for param in self.parameters():
                    param.data = 0.99 * param.data + 0.01 * param.grad.data if param.grad is not None else param.data


def create_model(tier: str, width: float = 1.0, l_prime: int = DEFAULT_L_PRIME, vocab_size: int = VOCAB_SIZE) -> MBDS:
    """Factory function to create MBD-S model for given tier."""
    return MBDS(vocab_size=vocab_size, embed_dim=256, hidden_dim=512, l_prime=l_prime, width=width, tier=tier)

def train_model(model: MBDS, data: Tensor, epochs: int = 1, lr: float = 1e-4):
    """Train model on dataset."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    for _ in range(epochs):
        for batch in data.split(128, dim=0):  # Mini-batch size 128
            logits, loss = model(batch, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def main():
    """Example usage across tiers."""
    # Dummy data: batch of sequences (batch_size=2, seq_len=32)
    data = torch.randint(0, VOCAB_SIZE - 2, (2, 32), device=DEVICE)
    prompt = data[:, :8]  # First 8 tokens as prompt

    for tier in ["core", "enhanced", "extreme", "simple"]:
        print(f"\nTesting {tier.upper()} tier:")

        # Small model for demo (width=0.1 ~ 10M params)
        model = create_model(tier, width=0.1, l_prime=8)

        # Train offline
        model = train_model(model, data, epochs=1, lr=1e-4)
        print(f"Trained {tier} model, loss: {model(data, data)[1].item():.4f}")

        # Generate
        generated = model.generate(prompt, max_blocks=4)
        print(f"Generated shape: {generated.shape}")

        # Online training (one step)
        model.train_online(data[0].unsqueeze(0), lr=1e-6)


if __name__ == "__main__":
    main()