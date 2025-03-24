#!/usr/bin/env python3
"""Unit tests for MBDTextDiffusion."""
import unittest
import torch
import torch.nn.functional as F
from hybrid import MBDTextDiffusion, create_hybrid_model, train_model, DEVICE
import time
import numpy as np

class TestMBDTextDiffusion(unittest.TestCase):
    def setUp(self):
        self.vocab = {
            "<unk>": 6, "<eos>": 7, "meeting": 0, "with": 1, "alex": 2,
            "at": 3, "pm": 4, "3": 5, "[TOOL: word_count]": 8
        }
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.models = {tier: create_hybrid_model(tier, width=0.1, vocab=self.vocab)
                      for tier in ["simple", "core", "enhanced", "extreme"]}
        # Pad sequences to equal length (7)
        self.synthetic_data = torch.tensor([
            [0, 1, 2, 3, 5, 4, 7],       # "meeting with alex at 3 pm <eos>"
            [0, 1, 2, 3, 4, 7, 6]        # "meeting with alex at pm <eos> <unk>"
        ], dtype=torch.long, device=DEVICE)

    def test_initialization(self):
        for tier, model in self.models.items():
            self.assertEqual(model.tier, tier)
            self.assertEqual(model.cfg.vocab_size, len(self.vocab))
            self.assertEqual(model.T, {"simple": 1, "core": 5, "enhanced": 3, "extreme": 1}[tier])
            self.assertEqual(bool(model.tools), tier in ["enhanced", "extreme"])
            self.assertEqual(model.remask_threshold is not None, tier == "extreme")
            self.assertEqual(model.cfg.embed_dim % model.cfg.num_heads, 0)  # Ensure divisibility

    def test_train_step(self):
        text = "meeting with alex at pm"
        for tier, model in self.models.items():
            start_time = time.time()
            model.train_step(text)
            self.assertLess(time.time() - start_time, 0.5)
            self.assertEqual(len(model.buffer), 1)
            x, loss = model.buffer[0]
            self.assertEqual(x.shape, torch.tensor([self.vocab[w] for w in text.split()],
                                                  dtype=torch.long, device=DEVICE).unsqueeze(0).shape)
            self.assertGreater(loss, 0)

    def test_generate_prompt_preservation(self):
        prompt = "meeting with"
        for tier, model in self.models.items():
            model.train_step("meeting with alex at pm")
            output = model.generate(prompt, max_blocks=3)
            if tier != "simple":
                self.assertTrue(output.startswith(prompt), f"Tier {tier} failed: {output}")
            self.assertLessEqual(len(output.split()), model.cfg.max_seq_len)

    def test_tool_integration(self):
        prompt = "meeting with [TOOL: word_count]"
        for tier in ["enhanced", "extreme"]:
            model = self.models[tier]
            model.train_step("meeting with alex at pm")
            output = model.generate(prompt, max_blocks=5)
            self.assertIn("3", output, f"Tier {tier} failed: {output}")
            model.add_tool("upper_count", "len([w for w in x.split() if w[0].isupper()])")
            output = model.generate("meeting with [TOOL: upper_count]", max_blocks=5)
            self.assertIn("1", output, f"Tier {tier} failed: {output}")

    def test_tool_edge_cases(self):
        for tier in ["enhanced", "extreme"]:
            model = self.models[tier]
            with self.assertRaises(ValueError):
                model.add_tool("invalid", "len([)")
            output = model.generate("meeting [TOOL: unknown]", max_blocks=3)
            self.assertNotIn("[TOOL: unknown]", output)

    def test_offline_training(self):
        for tier, model in self.models.items():
            start_time = time.time()
            model = train_model(model, self.synthetic_data, epochs=5, lr=1e-3)
            train_time = time.time() - start_time
            ppl = self.perplexity(model, self.synthetic_data)
            self.assertLess(train_time, 2.0, f"Tier {tier} training too slow: {train_time:.2f}s")
            self.assertLess(ppl, 1000, f"Tier {tier} PPL too high: {ppl:.2f}")

    def test_generation_length(self):
        prompt = "meeting"
        for tier, model in self.models.items():
            model = train_model(model, self.synthetic_data, epochs=5)
            output = model.generate(prompt, max_blocks=4)
            self.assertLessEqual(len(output.split()), model.cfg.max_seq_len)
            self.assertGreater(len(output.split()), 1, f"Tier {tier} failed to extend: {output}")

    def test_edge_cases(self):
        for tier, model in self.models.items():
            # Empty input
            model.train_step("")
            self.assertEqual(len(model.buffer), 1)
            output = model.generate("", max_blocks=3)
            self.assertGreater(len(output.split()), 0)

            # Long prompt
            long_prompt = "meeting " * 20
            output = model.generate(long_prompt, max_blocks=2)
            self.assertLessEqual(len(output.split()), model.cfg.max_seq_len)

    def perplexity(self, model: MBDTextDiffusion, data: torch.Tensor) -> float:
        model.eval()
        with torch.no_grad():
            logits, _ = model(data)
            targets = model._split_blocks(data)
            nll = F.cross_entropy(logits.flatten(0, 2), targets.flatten(), reduction="sum")
            return float(torch.exp(nll / data.numel()))

if __name__ == "__main__":
    unittest.main(verbosity=2)