from __future__ import annotations

import gc
import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

try:
	from .config import FineTuningConfig
except ImportError:
	from config import FineTuningConfig

SYSTEM_PROMPT_EN = (
	"You are an expert FastAPI documentation assistant. Answer questions accurately "
	"using official FastAPI Reference signatures, parameters, and Python code examples."
)

SYSTEM_PROMPT_RU = (
	"Вы — экспертный ассистент по документации FastAPI. Точно отвечайте на вопросы, "
	"используя официальные сигнатуры, параметры и примеры кода Python из справочника FastAPI."
)


def clean_chunk_body(raw_text: str) -> str:
	"""Remove internal indexing breadcrumb headers and source-code footers from chunk text."""
	lines = raw_text.strip().splitlines()
	cleaned: List[str] = []
	for idx, line in enumerate(lines):
		stripped = line.strip()
		if stripped.startswith("[") and stripped.endswith("]") and (idx == 0 or " > " in stripped):
			continue
		if stripped.startswith("[Source code in ") and stripped.endswith("]"):
			continue
		cleaned.append(line)
	return "\n".join(cleaned).strip()


def stream_chunks_from_sqlite(
	db_path: Path,
	languages: Optional[Sequence[str]] = None,
	batch_size: int = 256,
) -> Iterator[Dict[str, Any]]:
	"""Stream FastAPI chunks from SQLite without loading the 130MB embedding column into RAM."""
	conn = sqlite3.connect(str(db_path))
	conn.row_factory = sqlite3.Row
	try:
		cur = conn.cursor()
		if languages and "all" not in [lang.lower() for lang in languages]:
			placeholders = ",".join("?" for _ in languages)
			query = (
				"SELECT id, document_id, chunk_index, language, section_title, "
				"anchor, section_url, breadcrumb, symbol_type, text, metadata_json "
				f"FROM document_chunks WHERE language IN ({placeholders}) "
				"ORDER BY document_id, chunk_index"
			)
			cur.execute(query, tuple(lang.lower() for lang in languages))
		else:
			query = (
				"SELECT id, document_id, chunk_index, language, section_title, "
				"anchor, section_url, breadcrumb, symbol_type, text, metadata_json "
				"FROM document_chunks ORDER BY document_id, chunk_index"
			)
			cur.execute(query)

		while True:
			rows = cur.fetchmany(batch_size)
			if not rows:
				break
			for row in rows:
				meta_raw = row["metadata_json"]
				meta = json.loads(meta_raw) if isinstance(meta_raw, str) and meta_raw else (meta_raw or {})
				yield {
					"id": row["id"],
					"document_id": row["document_id"],
					"chunk_index": row["chunk_index"],
					"language": row["language"] or "en",
					"section_title": row["section_title"] or "",
					"anchor": row["anchor"] or "",
					"section_url": row["section_url"] or "",
					"breadcrumb": row["breadcrumb"] or "",
					"symbol_type": row["symbol_type"] or "section",
					"text": row["text"] or "",
					"metadata": meta,
				}
	finally:
		conn.close()


def build_chatml_messages(
	chunk: Dict[str, Any],
	include_rag_pair: bool = True,
) -> List[Tuple[str, str]]:
	"""Convert a FastAPI documentation chunk into (prompt_prefix, completion_suffix) ChatML pairs."""
	lang = (chunk.get("language") or "en").lower()
	meta = chunk.get("metadata") or {}
	doc_title = meta.get("doc_title") or chunk.get("section_title") or "FastAPI"
	section_title = chunk.get("section_title") or doc_title
	breadcrumb = chunk.get("breadcrumb") or section_title
	section_url = chunk.get("section_url") or meta.get("doc_url") or ""
	symbol_type = chunk.get("symbol_type") or meta.get("symbol_type") or "section"
	signature = meta.get("signature") or ""
	body = clean_chunk_body(chunk.get("text", ""))

	if not body or len(body) < 40:
		return []

	is_ru = lang in ("ru", "uk")
	sys_prompt = SYSTEM_PROMPT_RU if is_ru else SYSTEM_PROMPT_EN

	# Format structured reference answer with URL citation
	answer_lines = [f"### {breadcrumb}"]
	if section_url:
		answer_lines.append(f"Reference URL: {section_url}")
	answer_lines.append("")
	answer_lines.append(body)
	assistant_answer = "\n".join(answer_lines).strip()

	pairs: List[Tuple[str, str]] = []

	# 1. Direct API / Documentation Question -> Answer
	if is_ru:
		if symbol_type in ("class", "function", "method"):
			user_q = (
				f"Как устроен и используется {symbol_type} `{section_title}` "
				f"({breadcrumb}) в FastAPI? Приведи сигнатуру, параметры и пример кода."
			)
		else:
			user_q = f"Объясни раздел документации FastAPI «{breadcrumb}» ({doc_title}) с примерами."
	else:
		if symbol_type in ("class", "function", "method"):
			user_q = (
				f"How do I use the `{section_title}` {symbol_type} ({breadcrumb}) in FastAPI? "
				f"Provide its signature, parameters, and code example."
			)
		else:
			user_q = f"Explain the FastAPI documentation section '{breadcrumb}' ({doc_title}) with examples."

	prompt_1 = (
		f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
		f"<|im_start|>user\n{user_q}<|im_end|>\n"
		f"<|im_start|>assistant\n"
	)
	completion_1 = f"{assistant_answer}<|im_end|>"
	pairs.append((prompt_1, completion_1))

	# 2. Grounded RAG Context -> Answer (trains the model to act as the RAG generator)
	if include_rag_pair and ("```python" in body or signature or symbol_type in ("class", "function")):
		if is_ru:
			rag_q = f"Как работает `{section_title}` в FastAPI?"
		else:
			rag_q = f"How does `{section_title}` work in FastAPI?"

		rag_context = (
			f"--- Source [1] (lang={lang}) ---\n"
			f"Section: {breadcrumb} ({symbol_type})\n"
			f"URL: {section_url}\n"
			f"Content:\n{body}"
		)
		rag_user = (
			f"Question: {rag_q}\n\n"
			f"FastAPI Reference Context:\n{rag_context}\n\n"
			"Answer the question strictly using the provided FastAPI Reference Context and include code blocks."
		)
		prompt_2 = (
			f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
			f"<|im_start|>user\n{rag_user}<|im_end|>\n"
			f"<|im_start|>assistant\n"
		)
		completion_2 = f"{assistant_answer}<|im_end|>"
		pairs.append((prompt_2, completion_2))

	return pairs


class PackedConstantLengthDataset(Dataset):
	"""Constant-length packed dataset with completion-only loss masking.

	Why this maximizes speed and minimizes RAM/VRAM:
	1. Stores all packed sequences in two contiguous numpy/torch int32 arrays (`input_ids`, `labels`),
	   avoiding Python object overhead in system RAM (< 5 MB total RAM).
	2. Eliminates 100% of padding tokens: every GPU Tensor Core operation runs on real tokens.
	3. Masks prompt tokens (`system` + `user`) and sequence boundary tokens with `-100` so gradients
	   are driven exclusively by assistant completions.
	"""

	def __init__(self, input_blocks: np.ndarray, label_blocks: np.ndarray):
		self.input_blocks = input_blocks
		self.label_blocks = label_blocks

	def __len__(self) -> int:
		return int(self.input_blocks.shape[0])

	def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
		input_ids = torch.from_numpy(self.input_blocks[idx].astype(np.int64, copy=False))
		labels = torch.from_numpy(self.label_blocks[idx].astype(np.int64, copy=False))
		return {
			"input_ids": input_ids,
			"attention_mask": torch.ones_like(input_ids),
			"labels": labels,
		}


class DynamicCompletionDataset(Dataset):
	"""Pre-tokenized variable-length dataset with completion-only labels for dynamic batch padding."""

	def __init__(self, samples: List[Tuple[np.ndarray, np.ndarray]]):
		self.samples = samples

	def __len__(self) -> int:
		return len(self.samples)

	def __getitem__(self, idx: int) -> Dict[str, Any]:
		inp, lbl = self.samples[idx]
		return {
			"input_ids": inp.astype(np.int64, copy=False).tolist(),
			"attention_mask": [1] * len(inp),
			"labels": lbl.astype(np.int64, copy=False).tolist(),
		}


class CompletionDataCollator:
	"""Dynamic padding collator aligned to multiples of 8 for NVIDIA Tensor Cores."""

	def __init__(self, pad_token_id: int, pad_to_multiple_of: int = 8):
		self.pad_token_id = pad_token_id
		self.pad_to_multiple_of = pad_to_multiple_of

	def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
		max_len = max(len(f["input_ids"]) for f in features)
		if self.pad_to_multiple_of > 1 and max_len % self.pad_to_multiple_of != 0:
			max_len = ((max_len // self.pad_to_multiple_of) + 1) * self.pad_to_multiple_of

		batch_size = len(features)
		input_ids = torch.full((batch_size, max_len), self.pad_token_id, dtype=torch.long)
		attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
		labels = torch.full((batch_size, max_len), -100, dtype=torch.long)

		for i, f in enumerate(features):
			seq_len = len(f["input_ids"])
			input_ids[i, :seq_len] = torch.tensor(f["input_ids"], dtype=torch.long)
			attention_mask[i, :seq_len] = 1
			labels[i, :seq_len] = torch.tensor(f["labels"], dtype=torch.long)

		return {
			"input_ids": input_ids,
			"attention_mask": attention_mask,
			"labels": labels,
		}


def prepare_fastapi_datasets(
	tokenizer: Any,
	cfg: FineTuningConfig,
) -> Tuple[Dataset, Dataset, Dict[str, Any]]:
	"""Stream FastAPI chunks, tokenize on-the-fly, and build RAM-compact train/eval datasets."""
	if not cfg.sqlite_db_path.exists():
		raise FileNotFoundError(
			f"SQLite database not found at '{cfg.sqlite_db_path}'. "
			"Clone or link data/rag.db from https://github.com/Squ1dyyy/FastAPI-RAG "
			"or set FASTAPI_RAG_DB_PATH environment variable."
		)
	chunk_iter = stream_chunks_from_sqlite(cfg.sqlite_db_path, languages=cfg.languages)

	eos_id = tokenizer.eos_token_id
	tokenized_samples: List[Tuple[np.ndarray, np.ndarray]] = []
	total_raw_chunks = 0
	total_kept_chunks = 0
	total_tokens = 0

	for chunk in chunk_iter:
		total_raw_chunks += 1
		sym_type = chunk.get("symbol_type") or ""
		raw_txt = chunk.get("text") or ""

		if cfg.exclude_trivial_attributes:
			is_short_attr = (
				sym_type == "attribute" or "instance-attribute" in raw_txt[:180]
			) and len(raw_txt) < 320 and "```" not in raw_txt
			if is_short_attr:
				continue

		pairs = build_chatml_messages(
			chunk,
			include_rag_pair=cfg.include_rag_qa_samples,
		)
		if not pairs:
			continue

		total_kept_chunks += 1
		for prompt_str, completion_str in pairs:
			prompt_ids = tokenizer.encode(prompt_str, add_special_tokens=False)
			comp_ids = tokenizer.encode(completion_str, add_special_tokens=False)
			if not comp_ids:
				continue
			if comp_ids[-1] != eos_id and eos_id is not None:
				comp_ids.append(eos_id)

			full_ids = prompt_ids + comp_ids
			full_labels = ([-100] * len(prompt_ids)) + comp_ids

			if not cfg.sequence_packing and len(full_ids) > cfg.max_seq_length:
				full_ids = full_ids[: cfg.max_seq_length]
				full_labels = full_labels[: cfg.max_seq_length]
				if all(x == -100 for x in full_labels):
					continue

			inp_arr = np.asarray(full_ids, dtype=np.int32)
			lbl_arr = np.asarray(full_labels, dtype=np.int32)
			tokenized_samples.append((inp_arr, lbl_arr))
			total_tokens += len(inp_arr)

			if cfg.max_samples > 0 and len(tokenized_samples) >= cfg.max_samples:
				break
		if cfg.max_samples > 0 and len(tokenized_samples) >= cfg.max_samples:
			break

	# Deterministic shuffle before train/val split
	rng = np.random.default_rng(cfg.seed)
	indices = rng.permutation(len(tokenized_samples))
	val_count = max(1, int(len(tokenized_samples) * cfg.val_split_ratio))
	val_idx = set(indices[:val_count].tolist())

	train_samples = [tokenized_samples[i] for i in range(len(tokenized_samples)) if i not in val_idx]
	val_samples = [tokenized_samples[i] for i in range(len(tokenized_samples)) if i in val_idx]

	del tokenized_samples
	gc.collect()

	if cfg.sequence_packing:
		train_ds = _pack_samples(train_samples, cfg.max_seq_length)
		val_ds = _pack_samples(val_samples, cfg.max_seq_length)
	else:
		train_samples.sort(key=lambda pair: len(pair[0]), reverse=True)
		val_samples.sort(key=lambda pair: len(pair[0]), reverse=True)
		train_ds = DynamicCompletionDataset(train_samples)
		val_ds = DynamicCompletionDataset(val_samples)

	gc.collect()

	stats = {
		"raw_chunks_scanned": total_raw_chunks,
		"qualifying_chunks": total_kept_chunks,
		"total_tokens": total_tokens,
		"train_items": len(train_ds),
		"val_items": len(val_ds),
		"sequence_packing": cfg.sequence_packing,
		"max_seq_length": cfg.max_seq_length,
	}
	return train_ds, val_ds, stats


def _pack_samples(
	samples: List[Tuple[np.ndarray, np.ndarray]],
	seq_len: int,
) -> PackedConstantLengthDataset:
	"""Concatenate token & label arrays and slice into dense fixed-length blocks."""
	if not samples:
		empty = np.zeros((0, seq_len), dtype=np.int32)
		return PackedConstantLengthDataset(empty, empty)

	input_stream: List[np.ndarray] = []
	label_stream: List[np.ndarray] = []

	for inp, lbl in samples:
		if len(inp) == 0:
			continue
		lbl_copy = lbl.copy()
		lbl_copy[0] = -100
		input_stream.append(inp)
		label_stream.append(lbl_copy)

	all_inputs = np.concatenate(input_stream, axis=0)
	all_labels = np.concatenate(label_stream, axis=0)

	num_blocks = len(all_inputs) // seq_len
	if num_blocks == 0:
		pad_len = seq_len - len(all_inputs)
		padded_inp = np.pad(all_inputs, (0, pad_len), constant_values=all_inputs[-1])
		padded_lbl = np.pad(all_labels, (0, pad_len), constant_values=-100)
		return PackedConstantLengthDataset(
			padded_inp.reshape(1, seq_len),
			padded_lbl.reshape(1, seq_len),
		)

	cutoff = num_blocks * seq_len
	input_blocks = np.ascontiguousarray(all_inputs[:cutoff].reshape(num_blocks, seq_len))
	label_blocks = np.ascontiguousarray(all_labels[:cutoff].reshape(num_blocks, seq_len))

	valid_mask = np.any(label_blocks != -100, axis=1)
	return PackedConstantLengthDataset(input_blocks[valid_mask], label_blocks[valid_mask])
