from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _resolve_default_db_path() -> Path:
	"""Resolve SQLite database path from env, local data/rag.db, or sibling FastAPI-RAG repo."""
	env_db = os.environ.get("FASTAPI_RAG_DB_PATH")
	if env_db:
		return Path(env_db)
	local_db = PROJECT_ROOT / "data" / "rag.db"
	if local_db.exists():
		return local_db
	for sibling_name in ("RAGFastAPI", "FastAPI-RAG"):
		sibling_db = PROJECT_ROOT.parent / sibling_name / "data" / "rag.db"
		if sibling_db.exists():
			return sibling_db
	return local_db


def _resolve_default_html_dir() -> Path:
	"""Resolve HTML data directory from local data/ or sibling FastAPI-RAG repo."""
	local_dir = PROJECT_ROOT / "data"
	if local_dir.exists():
		return local_dir
	for sibling_name in ("RAGFastAPI", "FastAPI-RAG"):
		sibling_dir = PROJECT_ROOT.parent / sibling_name / "data"
		if sibling_dir.exists():
			return sibling_dir
	return local_dir


@dataclass
class FineTuningConfig:
	"""Configuration for educational, low-RAM/VRAM 4-bit QLoRA fine-tuning of Qwen/Qwen2.5-0.5B."""

	# Model & Paths (resolved relative to project root or sibling FastAPI-RAG repository)
	model_name: str = "Qwen/Qwen2.5-0.5B"
	sqlite_db_path: Path = _resolve_default_db_path()
	html_data_dir: Path = _resolve_default_html_dir()
	output_dir: Path = PROJECT_ROOT / "fastapi_qlora" / "output" / "qwen2.5-0.5b-fastapi-lora"
	cache_dir: Path = PROJECT_ROOT / "fastapi_qlora" / "cache"

	# Dataset & Sequence Packing (Speed + RAM optimization)
	languages: Tuple[str, ...] = ("en", "ru")
	include_rag_qa_samples: bool = True
	include_api_doc_samples: bool = True
	exclude_trivial_attributes: bool = True
	max_seq_length: int = 512
	sequence_packing: bool = True
	val_split_ratio: float = 0.05
	max_samples: int = 0  # 0 = use all qualifying FastAPI samples

	# 4-bit QLoRA (bitsandbytes NF4 + Double Quantization for minimal VRAM)
	use_4bit: bool = True
	bnb_4bit_quant_type: str = "nf4"
	bnb_4bit_use_double_quant: bool = True
	low_cpu_mem_usage: bool = True

	# LoRA / RSLoRA Configuration (lora_dropout=0.0 uses nn.Identity for faster forward/backward)
	lora_r: int = 16
	lora_alpha: int = 32
	lora_dropout: float = 0.0
	use_rslora: bool = True
	target_modules: Tuple[str, ...] = (
		"q_proj",
		"k_proj",
		"v_proj",
		"o_proj",
		"gate_proj",
		"up_proj",
		"down_proj",
	)

	# Attention, Precision & Memory Checkpointing
	attn_implementation: str = "sdpa"
	use_tf32: bool = True
	use_gradient_checkpointing: bool = True
	gradient_checkpointing_use_reentrant: bool = False

	# Optimizer & Training Hyperparameters (Maximum throughput on 8GB VRAM)
	optim: str = "paged_adamw_8bit"
	per_device_train_batch_size: int = 4
	per_device_eval_batch_size: int = 4
	gradient_accumulation_steps: int = 4
	num_train_epochs: float = 1.0
	max_steps: int = -1  # -1 = train for num_train_epochs
	learning_rate: float = 3e-4
	weight_decay: float = 0.01
	warmup_ratio: float = 0.05
	lr_scheduler_type: str = "cosine"
	max_grad_norm: float = 1.0

	# Batching, DataLoader & Host RAM Optimizations
	group_by_length: bool = True
	pad_to_multiple_of: int = 8
	dataloader_num_workers: int = 0  # 0 on Windows prevents multi-GB process duplication in RAM
	dataloader_pin_memory: bool = True
	torch_empty_cache_steps: int = 25
	logging_steps: int = 10
	save_strategy: str = "no"
	seed: int = 42

	@property
	def use_bf16(self) -> bool:
		"""Return True when CUDA GPU natively supports bfloat16 Tensor Cores."""
		return torch.cuda.is_available() and torch.cuda.is_bf16_supported()

	@property
	def use_fp16(self) -> bool:
		"""Return True when CUDA is available but bfloat16 is not supported."""
		return torch.cuda.is_available() and not torch.cuda.is_bf16_supported()

	@property
	def compute_dtype(self) -> torch.dtype:
		"""Select optimal compute dtype (bfloat16 on Ampere/Ada+, float16 otherwise)."""
		if self.use_bf16:
			return torch.bfloat16
		if torch.cuda.is_available():
			return torch.float16
		return torch.float32
