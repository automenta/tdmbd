#!/usr/bin/env python3
"""Test MBDS language model with real data (WikiText-2) using transformers for tokenization."""
import pytest
import torch
import torch.nn.functional as F
from mbd_lm import MBDS, create_model, train_model, DEVICE  # VOCAB_SIZE removed from import
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import numpy as np

# Constants
TOKENIZER_NAME = "gpt2"
MAX_STEPS = 1000  # Scaled down for demo
BATCH_SIZE = 8
#SEQ_LEN = 128
SEQ_LEN = 64


def tokens_to_text(tokens: torch.Tensor, tokenizer: AutoTokenizer) -> list[str]:
    """Convert token tensor to readable text using tokenizer."""
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def perplexity(model: MBDS, data: torch.Tensor) -> float:
    """Calculate perplexity on test data."""
    model.eval()
    with torch.no_grad():
        logits, _ = model(data)
        targets = model._split_blocks(data)
        nll = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), reduction="sum")
        return float(torch.exp(nll / data.numel()))


def prepare_data(dataset_name: str = "wikitext", config_name: str = "wikitext-2-raw-v1") -> tuple[
    torch.Tensor, torch.Tensor]:
    """Load and tokenize WikiText-2 dataset."""
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_dataset(dataset_name, config_name)
    train_texts = dataset["train"]["text"]
    test_texts = dataset["test"]["text"]

    # Tokenize and truncate/pad to fixed length
    def tokenize_and_process(texts, vocab_size):
        encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=SEQ_LEN, return_tensors="pt")
        input_ids = encodings["input_ids"].clamp(max=vocab_size - 1)  # Cap token IDs
        return input_ids.to(DEVICE)

    vocab_size = tokenizer.vocab_size
    train_data = tokenize_and_process([t for t in train_texts if t.strip()][:BATCH_SIZE * 100], vocab_size)
    test_data = tokenize_and_process([t for t in test_texts if t.strip()][:BATCH_SIZE * 10], vocab_size)

    return train_data, test_data, tokenizer


@pytest.fixture
def real_data():
    """Fixture for WikiText-2 data."""
    return prepare_data()


def test_real_data_training(real_data):
    """Test training MBDS on WikiText-2."""
    train_data, test_data, tokenizer = real_data
    vocab_size = tokenizer.vocab_size
    print(
        f"\nTesting MBDS with WikiText-2 (train size: {train_data.shape}, test size: {test_data.shape}, vocab size: {vocab_size})")

    # Initialize model with tokenizer's vocab size
    model = create_model("core", width=0.1, l_prime=16, vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-3)

    # Training loop
    start_time = time.time()
    for step in range(MAX_STEPS):
        batch_idx = np.random.randint(0, train_data.shape[0] - BATCH_SIZE)
        batch = train_data[batch_idx:batch_idx + BATCH_SIZE]

        model.train()
        logits, loss = model(batch, batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 100 == 0:
            print(f"Step {step}/{MAX_STEPS}, Loss: {loss.item():.4f}")

    train_time = time.time() - start_time
    train_ppl = perplexity(model, train_data[:BATCH_SIZE * 10])
    test_ppl = perplexity(model, test_data)
    print(f"Train time: {train_time:.2f}s, Train PPL: {train_ppl:.2f}, Test PPL: {test_ppl:.2f}")

    assert test_ppl < 1000, f"Training failed: Test PPL {test_ppl} too high"


def test_real_data_generation(real_data):
    """Test generation quality on WikiText-2."""
    train_data, _, tokenizer = real_data
    vocab_size = tokenizer.vocab_size
    model = create_model("core", width=0.1, l_prime=16, vocab_size=vocab_size)

    # Train briefly for generation
    model = train_model(model, train_data[:BATCH_SIZE * 50], epochs=10)

    # Generate from a prompt
    prompt = train_data[:2, :16]
    start_time = time.time()
    generated = model.generate(prompt, max_blocks=5)
    gen_time = time.time() - start_time

    text = tokens_to_text(generated, tokenizer)
    print(f"\nGeneration time: {gen_time:.2f}s")
    for i, seq in enumerate(text):
        print(f"Generated {i}: {seq}")
    assert generated.shape[1] > prompt.shape[1], "Generation failed to extend prompt"


if __name__ == "__main__":
    import sys

    if "pytest" not in sys.modules:
        print("Running standalone: Ensure transformers and datasets are installed!")
        print("pip install transformers datasets")
    pytest.main(["-v", __file__])