import sys
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = ROOT_DIR / "fastapi_qlora"
if str(PKG_DIR) not in sys.path:
	sys.path.insert(0, str(PKG_DIR))

from config import FineTuningConfig
from dataset import (
	CompletionDataCollator,
	PackedConstantLengthDataset,
	_pack_samples,
	build_chatml_messages,
	stream_chunks_from_sqlite,
)
from trainer import install_fused_masked_chunked_lm_head


class _DummyBackboneOutput:
	def __init__(self, hidden_states: torch.Tensor):
		self.hidden_states = hidden_states
		self.past_key_values = None
		self.attentions = None

	def __getitem__(self, idx: int) -> torch.Tensor:
		if idx == 0:
			return self.hidden_states
		raise IndexError(idx)


class _DummyBackbone(nn.Module):
	def __init__(self, vocab_size: int = 64, hidden_size: int = 16):
		super().__init__()
		self.embed = nn.Embedding(vocab_size, hidden_size)
		self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

	def forward(self, input_ids=None, **kwargs):
		h = self.proj(self.embed(input_ids))
		return _DummyBackboneOutput(h)


class _DummyCausalLM(nn.Module):
	def __init__(self, vocab_size: int = 64, hidden_size: int = 16):
		super().__init__()
		self.model = _DummyBackbone(vocab_size, hidden_size)
		self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

	def forward(self, input_ids=None, labels=None, **kwargs):
		out = self.model(input_ids=input_ids)
		logits = self.lm_head(out[0])
		loss = None
		if labels is not None:
			shift_logits = logits[..., :-1, :].contiguous()
			shift_labels = labels[..., 1:].contiguous()
			loss = nn.functional.cross_entropy(
				shift_logits.view(-1, shift_logits.size(-1)).float(),
				shift_labels.view(-1),
				ignore_index=-100,
			)
		return type("Out", (), {"loss": loss, "logits": logits})()


class TestFineTuningPipeline(unittest.TestCase):
	"""Unit tests for Qwen2.5-0.5B RAM/VRAM/Speed optimizations and FastAPI dataset pipeline."""

	def test_sqlite_streaming_excludes_embeddings(self):
		"""Ensure SQLite streaming yields FastAPI chunks without loading 130MB embeddings."""
		cfg = FineTuningConfig()
		if not cfg.sqlite_db_path.exists():
			self.skipTest("data/rag.db not present")
		gen = stream_chunks_from_sqlite(cfg.sqlite_db_path, languages=("en",), batch_size=8)
		first_chunk = next(gen)
		self.assertIn("text", first_chunk)
		self.assertIn("breadcrumb", first_chunk)
		self.assertNotIn("embedding", first_chunk)
		self.assertEqual(first_chunk["language"], "en")

	def test_build_chatml_messages_generates_pairs(self):
		"""Ensure FastAPI chunk with code yields both direct Q&A and RAG grounded ChatML pairs."""
		sample_chunk = {
			"language": "en",
			"section_title": "APIRouter",
			"breadcrumb": "FastAPI > APIRouter",
			"section_url": "https://fastapi.tiangolo.com/reference/apirouter/#fastapi.APIRouter",
			"symbol_type": "class",
			"text": "[FastAPI > APIRouter]\n## APIRouter\n```python\nrouter = APIRouter()\n```\nRouter class for grouping path operations.",
			"metadata": {"doc_title": "APIRouter", "signature": "APIRouter(prefix='')"},
		}
		pairs = build_chatml_messages(sample_chunk, include_rag_pair=True)
		self.assertEqual(len(pairs), 2)
		prompt, completion = pairs[0]
		self.assertIn("<|im_start|>system", prompt)
		self.assertIn("<|im_start|>user", prompt)
		self.assertTrue(prompt.endswith("<|im_start|>assistant\n"))
		self.assertIn("```python", completion)
		self.assertTrue(completion.endswith("<|im_end|>"))

	def test_sequence_packing_eliminates_padding_and_masks_boundaries(self):
		"""Ensure constant-length sequence packing produces exact block shapes and masks boundaries."""
		samples = [
			(
				np.array([1, 2, 3, 4, 5, 6], dtype=np.int32),
				np.array([-100, -100, 3, 4, 5, 6], dtype=np.int32),
			),
			(
				np.array([10, 11, 12, 13, 14, 15], dtype=np.int32),
				np.array([-100, 11, 12, 13, 14, 15], dtype=np.int32),
			),
		]
		packed: PackedConstantLengthDataset = _pack_samples(samples, seq_len=8)
		self.assertEqual(len(packed), 1)
		item = packed[0]
		self.assertEqual(tuple(item["input_ids"].shape), (8,))
		self.assertEqual(tuple(item["attention_mask"].shape), (8,))
		self.assertTrue(torch.all(item["attention_mask"] == 1))
		self.assertEqual(int(item["labels"][6].item()), -100)

	def test_completion_collator_pads_to_multiple_of_8(self):
		"""Ensure dynamic collator aligns sequence lengths to multiples of 8 for Tensor Cores."""
		collator = CompletionDataCollator(pad_token_id=0, pad_to_multiple_of=8)
		batch = collator([
			{"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]},
			{"input_ids": [1, 2, 3, 4, 5, 6, 7, 8, 9], "labels": [-100, 2, 3, 4, 5, 6, 7, 8, 9]},
		])
		self.assertEqual(tuple(batch["input_ids"].shape), (2, 16))
		self.assertEqual(tuple(batch["labels"].shape), (2, 16))
		self.assertTrue(torch.all(batch["labels"][0, 5:] == -100))

	def test_fused_masked_chunked_lm_head_matches_exact_loss_and_grads(self):
		"""Ensure masked chunked LM-head forward matches standard CrossEntropy loss and gradients."""
		torch.manual_seed(42)
		ref_model = _DummyCausalLM(vocab_size=64, hidden_size=16)
		opt_model = _DummyCausalLM(vocab_size=64, hidden_size=16)
		opt_model.load_state_dict(ref_model.state_dict())
		install_fused_masked_chunked_lm_head(opt_model, chunk_size=4)

		input_ids = torch.randint(0, 64, (2, 12))
		labels = input_ids.clone()
		labels[:, :5] = -100  # mask prompt tokens

		ref_out = ref_model(input_ids=input_ids, labels=labels)
		ref_out.loss.backward()

		opt_out = opt_model(input_ids=input_ids, labels=labels)
		opt_out.loss.backward()

		self.assertAlmostEqual(float(ref_out.loss.item()), float(opt_out.loss.item()), places=5)
		self.assertTrue(
			torch.allclose(
				ref_model.lm_head.weight.grad,
				opt_model.lm_head.weight.grad,
				atol=1e-5,
			)
		)

	def test_console_db_retrieval_and_readonly_sql(self):
		"""Ensure FastSQLiteRAGConsole retrieves chunks from data/rag.db and enforces read-only SQL."""
		from cli_chat import FastSQLiteRAGConsole, detect_query_language

		self.assertEqual(detect_query_language("Как работает Depends?", "auto"), "ru")
		self.assertEqual(detect_query_language("How does Depends work?", "auto"), "en")

		console = FastSQLiteRAGConsole(mode="fast", language="auto", top_k=2)
		if not console.db_path.exists():
			self.skipTest("data/rag.db not present")

		stats = console.get_db_stats()
		self.assertGreater(stats["total_chunks"], 1000)

		hits, lang = console.retrieve_from_db("APIRouter prefix tags")
		self.assertEqual(lang, "en")
		self.assertEqual(len(hits), 2)
		self.assertIn("APIRouter", hits[0]["text"])

		with self.assertRaises(ValueError):
			console.execute_readonly_sql("DELETE FROM documents")


if __name__ == "__main__":
	unittest.main()
