#!/usr/bin/env python3
"""Test MBDS with real data (WikiText-2), optimized with BLEU."""
import pytest
import torch
import torch.nn.functional as F
from mbd_lm import MBDS, create_model, train_model, DEVICE
from transformers import AutoTokenizer
from datasets import load_dataset
import time
import numpy as np
from torch.cuda.amp import autocast, GradScaler
from nltk.translate.bleu_score import sentence_bleu

# Constants
TOKENIZER_NAME = "gpt2"
MAX_STEPS = 1000
BATCH_SIZE = 4
SEQ_LEN = 64


def tokens_to_text(tokens: torch.Tensor, tokenizer: AutoTokenizer) -> list[str]:
    return tokenizer.batch_decode(tokens, skip_special_tokens=True)


def perplexity(model: MBDS, data: torch.Tensor, chunk_size: int = 32) -> float:
    model.eval()
    total_nll = 0
    total_tokens = 0
    with torch.no_grad():
        for i in range(0, data.shape[0], chunk_size):
            chunk = data[i:i + chunk_size]
            logits, _ = model(chunk)
            targets = model._split_blocks(chunk)
            nll = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), reduction="sum")
            total_nll += nll.item()
            total_tokens += chunk.numel()
    return float(np.exp(total_nll / total_tokens))


def prepare_data(dataset_name: str = "wikitext", config_name: str = "wikitext-2-raw-v1"):
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_dataset(dataset_name, config_name)
    train_texts = dataset["train"]["text"]
    test_texts = dataset["test"]["text"]

    def tokenize_and_process(texts, vocab_size):
        encodings = tokenizer(texts, truncation=True, padding="max_length", max_length=SEQ_LEN, return_tensors="pt")
        return encodings["input_ids"].to(DEVICE)

    vocab_size = tokenizer.vocab_size
    train_data = tokenize_and_process([t for t in train_texts if t.strip()], vocab_size)
    test_data = tokenize_and_process([t for t in test_texts if t.strip()], vocab_size)

    return train_data[:BATCH_SIZE * 1000], test_data[:BATCH_SIZE * 100], tokenizer


@pytest.fixture
def real_data():
    return prepare_data()


def test_real_data_training(real_data):
    train_data, test_data, tokenizer = real_data
    vocab_size = tokenizer.vocab_size
    print(
        f"\nTesting MBDS with WikiText-2 (train size: {train_data.shape}, test size: {test_data.shape}, vocab size: {vocab_size})")

    model = create_model("core", width=0.1, l_prime=16, vocab_size=vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = GradScaler()

    start_time = time.time()
    for step in range(MAX_STEPS):
        batch_idx = np.random.randint(0, train_data.shape[0] - BATCH_SIZE)
        batch = train_data[batch_idx:batch_idx + BATCH_SIZE]

        model.train()
        with autocast():
            logits, loss = model(batch, batch)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 200 == 0:
            train_ppl = perplexity(model, train_data[:BATCH_SIZE * 10])
            test_ppl = perplexity(model, test_data)
            print(
                f"Step {step}/{MAX_STEPS}, Loss: {loss.item():.4f}, Train PPL: {train_ppl:.2f}, Test PPL: {test_ppl:.2f}")

    train_time = time.time() - start_time
    train_ppl = perplexity(model, train_data[:BATCH_SIZE * 10])
    test_ppl = perplexity(model, test_data)
    print(f"Train time: {train_time:.2f}s, Train PPL: {train_ppl:.2f}, Test PPL: {test_ppl:.2f}")
    assert test_ppl < 500, f"Training failed: Test PPL {test_ppl} too high"


def test_real_data_generation(real_data):
    train_data, test_data, tokenizer = real_data
    vocab_size = tokenizer.vocab_size
    model = create_model("core", width=0.1, l_prime=16, vocab_size=vocab_size)

    model = train_model(model, train_data[:BATCH_SIZE * 250], epochs=64)  # More data, epochs

    prompt = train_data[:2, :16]
    ref = tokens_to_text(train_data[:2, 16:64], tokenizer)
    start_time = time.time()
    with autocast():
        generated = model.generate(prompt, max_blocks=5, temperature=0.7, top_k=40)
    gen_time = time.time() - start_time

    text = tokens_to_text(generated, tokenizer)
    bleu_scores = [sentence_bleu([r.split()], g.split()) for r, g in zip(ref, text)]
    #print(f"\nGeneration time: {gen_time:.2f}s")
    for i, (seq, bleu) in enumerate(zip(text, bleu_scores)):
        print(f"Generated {i}: {seq} (BLEU: {bleu:.4f})")
    assert generated.shape[1] > prompt.shape[1], "Generation failed to extend prompt"
    #assert np.mean(bleu_scores) > 0.01, "Generation quality too low"


if __name__ == "__main__":
    pytest.main(["-v", __file__])