#!/usr/bin/env python3
"""Comprehensive Pytest suite for the MBD-S language model library."""
import pytest
import torch
import torch.nn.functional as F
from mbd_lm import MBDS, create_model, train_model, VOCAB_SIZE, DEVICE
import time

# Mock Vocabulary
MOCK_VOCAB = ["[PAD]", "[MASK]", "[EOS]"] + [f"word{i}" for i in range(VOCAB_SIZE - 3)]
MOCK_VOCAB_MAP = {word: i for i, word in enumerate(MOCK_VOCAB)}


def tokens_to_text(tokens: torch.Tensor) -> list[str]:
    """Convert token tensor to readable text."""
    return [" ".join(MOCK_VOCAB[t] for t in seq if t != 0) for seq in tokens.tolist()]


def perplexity(model: MBDS, data: torch.Tensor) -> float:
    """Calculate perplexity on test data."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = model._split_blocks(data)
        nll = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), reduction="sum")
        return float(torch.exp(nll / data.numel()))


@pytest.fixture
def synthetic_data() -> torch.Tensor:
    """Generate synthetic data with strong patterns and small vocab."""
    batch_size, seq_len = 64, 128
    data = torch.zeros(batch_size, seq_len, dtype=torch.long, device=DEVICE)
    for i in range(0, seq_len, 4):
        base = torch.randint(3, 20, (batch_size,), device=DEVICE)  # Small vocab
        if i + 4 <= seq_len:
            data[:, i:i + 4] = base.unsqueeze(1).repeat(1, 4)
    data[:, -1] = 2  # [EOS]
    return data


@pytest.mark.parametrize("tier, width, l_prime, ppl_threshold", [
    ("core", 0.1, 8, 1000),
    ("enhanced", 0.01, 4, 2000),
    ("extreme", 1.0, 64, 100),
    ("simple", 0.05, 16, 1000),
])
def test_offline_training(tier: str, width: float, l_prime: int, ppl_threshold: float, synthetic_data: torch.Tensor):
    """Test offline training across tiers."""
    print(f"\nTesting {tier.upper()} offline training (width={width}, l_prime={l_prime})")
    model = create_model(tier, width=width, l_prime=l_prime)
    train_data = synthetic_data[:32]
    test_data = synthetic_data[32:]

    start_time = time.time()
    model = train_model(model, train_data, epochs=200, lr=5e-3)
    train_time = time.time() - start_time

    train_ppl = perplexity(model, train_data)
    test_ppl = perplexity(model, test_data)
    print(f"Train time: {train_time:.2f}s, Train PPL: {train_ppl:.2f}, Test PPL: {test_ppl:.2f}")
    assert train_ppl < ppl_threshold, f"{tier} training failed: PPL {train_ppl} > {ppl_threshold}"


def test_online_learning(synthetic_data: torch.Tensor):
    """Test online learning across tiers."""
    for tier in ["core", "enhanced", "extreme", "simple"]:
        print(f"\nTesting {tier.upper()} online learning")
        model = create_model(tier, width=0.1, l_prime=8)
        data = synthetic_data[:8]

        initial_ppl = perplexity(model, data)
        start_time = time.time()
        for _ in range(50):
            for seq in data:
                model.train_online(seq.unsqueeze(0), lr=1e-3)
        online_time = time.time() - start_time

        final_ppl = perplexity(model, data)
        print(f"Online time: {online_time:.2f}s, Initial PPL: {initial_ppl:.2f}, Final PPL: {final_ppl:.2f}")
        assert final_ppl < initial_ppl, f"{tier} online learning failed to improve PPL"


def test_generation(synthetic_data: torch.Tensor):
    """Test generation across tiers."""
    prompt = synthetic_data[:2, :8]
    for tier in ["core", "enhanced", "extreme", "simple"]:
        print(f"\nTesting {tier.upper()} generation")
        model = create_model(tier, width=0.1, l_prime=8)
        model = train_model(model, synthetic_data[:32], epochs=50)

        start_time = time.time()
        generated = model.generate(prompt, max_blocks=10)
        gen_time = time.time() - start_time

        text = tokens_to_text(generated)
        print(f"Generation time: {gen_time:.2f}s")
        for i, seq in enumerate(text):
            print(f"Generated {i}: {seq}")
        assert generated.shape[1] > prompt.shape[1] + model.l_prime, f"{tier} generation failed to extend prompt enough"


def test_edge_cases(synthetic_data: torch.Tensor):
    """Test edge cases: short, empty, large inputs."""
    print("\nTesting edge cases")

    # Short sequence
    model = create_model("simple", width=0.1, l_prime=4)
    short_data = synthetic_data[:4, :4]
    model = train_model(model, short_data, epochs=100, lr=5e-3)
    short_ppl = perplexity(model, short_data)
    print(f"Short sequence PPL: {short_ppl:.2f}")
    assert short_ppl < 10000, "Short sequence PPL too high"

    # Empty prompt
    empty_prompt = torch.zeros(2, 1, dtype=torch.long, device=DEVICE)
    generated = model.generate(empty_prompt, max_blocks=5)
    text = tokens_to_text(generated)
    print("Empty prompt generation:", text)
    assert generated.shape[1] > 1, "Empty prompt generation failed"

    # Large context
    model = create_model("extreme", width=1.0, l_prime=64)  # Increased width
    large_data = synthetic_data[:4].repeat(1, 8)  # 1024 tokens
    model = train_model(model, large_data, epochs=200)  # More epochs
    large_ppl = perplexity(model, large_data)
    print(f"Large context PPL: {large_ppl:.2f}")
    assert large_ppl < 100, "Large context PPL too high"


def test_adaptability(synthetic_data: torch.Tensor):
    """Test adaptability to data shift."""
    print("\nTesting adaptability")
    model = create_model("enhanced", width=0.1, l_prime=8)

    # Initial training
    initial_data = synthetic_data[:16]
    model = train_model(model, initial_data, epochs=50)
    initial_ppl = perplexity(model, initial_data)
    print(f"Initial PPL: {initial_ppl:.2f}")

    # Shifted data
    shifted_data = torch.randint(3, 20, (16, 128), device=DEVICE)
    for _ in range(20):
        for seq in shifted_data:
            model.train_online(seq.unsqueeze(0), lr=1e-3)
    shifted_ppl = perplexity(model, shifted_data)
    print(f"Shifted PPL: {shifted_ppl:.2f}")
    assert shifted_ppl < initial_ppl, "Adaptability failed: Shifted PPL not improved"


if __name__ == "__main__":
    pytest.main(["-v", __file__])