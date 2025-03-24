#!/usr/bin/env python3
"""Comprehensive unit tests for TextDiffusion."""
import unittest
import time
from unified import TextDiffusion, Config, tokenize, detokenize
import psutil
import gc

class TestTextDiffusion(unittest.TestCase):
    def setUp(self):
        self.vocab = {
            "<unk>": 12, "meeting": 0, "with": 1, "alex": 2, "at": 3, "pm": 4,
            "3": 5, "1": 6, "[TOOL: word_count]": 7, "[TOOL: upper_count]": 8,
            "scheduled": 9, "for": 10, "today": 11
        }
        self.config = Config(vocab_size=len(self.vocab), embed_dim=8, num_heads=2, num_layers=1, max_seq_len=8, buffer_size=5)
        self.models = {tier: TextDiffusion(tier, self.vocab, self.config) for tier in range(4)}
        for model in self.models.values():
            model.model.buffer.clear()
            for param in model.model.parameters():
                if param.requires_grad:
                    param.data.uniform_(-0.1, 0.1)
        gc.collect()

    def test_initialization(self):
        for tier, model in self.models.items():
            self.assertEqual(model.tier, tier)
            self.assertEqual(model.cfg.vocab_size, len(self.vocab))
            self.assertTrue(model.model.embedding.num_embeddings >= len(self.vocab))
            self.assertEqual(model.use_cache, tier >= 1)
            self.assertEqual(bool(model.tools), tier >= 2)
            self.assertEqual(model.remask_threshold is not None, tier >= 3)

    def test_train_step(self):
        text = "meeting with alex at pm"
        for tier, model in self.models.items():
            start_time = time.time()
            model.train_step(text)
            self.assertLess(time.time() - start_time, 0.3)
            self.assertEqual(len(model.model.buffer), 1)
            x, loss = model.model.buffer[0]
            self.assertEqual(x.shape, tokenize(text, self.vocab)[0, :self.config.max_seq_len].shape)
            self.assertGreater(loss, 0)

    def test_generate_prompt_preservation(self):
        prompt = "meeting with"
        for tier, model in self.models.items():
            model.train_step("meeting with alex at pm")
            output = model.generate(prompt, max_len=6)
            if tier >= 1:
                self.assertTrue(output.startswith(prompt), f"Tier {tier} failed: {output}")
            self.assertLessEqual(len(output.split()), 6, f"Tier {tier} failed: {output}")

    def test_generate_length(self):
        prompt = "meeting"
        for tier, model in self.models.items():
            model.train_step("meeting with alex")
            output = model.generate(prompt, max_len=5)
            self.assertLessEqual(len(output.split()), 5, f"Tier {tier} failed: {output}")

    def test_tool_integration(self):
        prompt = "meeting with [TOOL: word_count]"
        for tier in range(2, 4):
            model = self.models[tier]
            model.train_step("meeting with alex at pm")
            output = model.generate(prompt, max_len=10)
            self.assertIn("3", output, f"Tier {tier} failed: {output}")
            model.add_tool("upper_count", "len([w for w in x.split() if w[0].isupper()])")
            output = model.generate("meeting with [TOOL: upper_count]", max_len=10)
            self.assertIn("1", output, f"Tier {tier} failed: {output}")

    def test_tool_edge_cases(self):
        for tier in range(2, 4):
            model = self.models[tier]
            with self.assertRaises(ValueError):
                model.add_tool("invalid", "len([)")
            output = model.generate("meeting [TOOL: unknown]", max_len=8)
            self.assertNotIn("[TOOL: unknown]", output)

    def test_tier_0_diffusion(self):
        model = self.models[0]
        model.train_step("meeting with alex")
        output = model.generate("meeting with", max_len=6)
        self.assertLessEqual(len(output.split()), 6)

    def test_buffer_management(self):
        text = "meeting with alex"
        for tier, model in self.models.items():
            for _ in range(model.cfg.buffer_size + 1):
                model.train_step(text)
            self.assertEqual(len(model.model.buffer), model.cfg.buffer_size)

    def test_performance(self):
        process = psutil.Process()
        for tier, model in self.models.items():
            start_time = time.time()
            model.train_step("meeting with alex at pm")
            train_time = time.time() - start_time
            output = model.generate("meeting with", max_len=8)
            gen_time = time.time() - start_time - train_time
            memory = process.memory_info().rss / 1024 ** 2
            self.assertLess(train_time, 0.5, f"Tier {tier} train too slow: {train_time:.3f}s")
            self.assertLess(gen_time, 0.3, f"Tier {tier} gen too slow: {gen_time:.3f}s")
            self.assertLess(memory, 550, f"Tier {tier} memory too high: {memory:.2f}MB")  # Adjusted to 550MB

    def test_edge_cases(self):
        for tier, model in self.models.items():
            model.train_step("")
            self.assertEqual(len(model.model.buffer), 1)
            long_prompt = " ".join(["meeting"] * 10)
            output = model.generate(long_prompt, max_len=5)
            self.assertLessEqual(len(output.split()), 5)

if __name__ == "__main__":
    unittest.main(verbosity=2)