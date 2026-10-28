from __future__ import annotations

import gc
import json
import os
import time
import types
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import psutil
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.checkpoint import checkpoint
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	Trainer,
	TrainerCallback,
	TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

try:
	from .config import FineTuningConfig
	from .dataset import (
		CompletionDataCollator,
		prepare_fastapi_datasets,
	)
except ImportError:
	from config import FineTuningConfig
	from dataset import (
		CompletionDataCollator,
		prepare_fastapi_datasets,
	)


def get_process_ram_mb() -> float:
	"""Return current process Resident Set Size (RSS) in MB."""
	proc = psutil.Process(os.getpid())
	return proc.memory_info().rss / (1024 * 1024)


def get_gpu_vram_stats_mb() -> Dict[str, float]:
	"""Return current allocated, reserved, and peak allocated CUDA VRAM in MB."""
	if not torch.cuda.is_available():
		return {"allocated_mb": 0.0, "reserved_mb": 0.0, "peak_allocated_mb": 0.0}
	return {
		"allocated_mb": round(torch.cuda.memory_allocated() / (1024 * 1024), 2),
		"reserved_mb": round(torch.cuda.memory_reserved() / (1024 * 1024), 2),
		"peak_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2),
	}


def enable_hardware_optimizations(cfg: FineTuningConfig) -> None:
	"""Enable CUDA TensorFloat-32, CuDNN benchmark, and Flash/MemEfficient SDPA kernels."""
	os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
	os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
	if os.name != "nt":
		os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

	if torch.cuda.is_available():
		if cfg.use_tf32:
			torch.backends.cuda.matmul.allow_tf32 = True
			torch.backends.cudnn.allow_tf32 = True
			torch.set_float32_matmul_precision("high")
		torch.backends.cudnn.benchmark = True
		if hasattr(torch.backends.cuda, "enable_flash_sdp"):
			torch.backends.cuda.enable_flash_sdp(True)
		if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
			torch.backends.cuda.enable_mem_efficient_sdp(True)


def install_fused_masked_chunked_lm_head(model: nn.Module, chunk_size: int = 512) -> None:
	"""Patch CausalLM forward to skip masked prompt tokens (-100) and chunk LM-head CE loss.

	Why this is critical for Qwen2.5-0.5B:
	- Qwen2.5 has a huge vocabulary (151,936 tokens) vs small hidden dimension (896).
	- Standard CausalLM forward projects ALL tokens `[B, S, 896] -> [B, S, 151936]` and upcasts
	  the entire tensor to FP32 (`> 1.2 GB` VRAM for B=4, S=512).
	- By filtering `shift_labels != -100` BEFORE `lm_head`, we completely skip the 151,936-way
	  projection on system/user prompt tokens (saving ~40% FLOPs).
	- By computing `lm_head + cross_entropy` in 512-token micro-chunks with checkpointing,
	  the full `[B, S, 151936]` logits tensor is never materialized in VRAM (saving >1.5 GB VRAM).
	"""
	base_causal_lm = model.base_model.model if hasattr(model, "base_model") else model
	original_forward = base_causal_lm.forward

	def _chunk_ce_loss(
		hidden_slice: torch.Tensor,
		label_slice: torch.Tensor,
		lm_head_weight: torch.Tensor,
		lm_head_bias: Optional[torch.Tensor],
	) -> torch.Tensor:
		logits = F.linear(hidden_slice, lm_head_weight, lm_head_bias)
		return F.cross_entropy(logits.float(), label_slice, reduction="sum")

	def fast_memory_efficient_forward(
		self: Any,
		input_ids: Optional[torch.LongTensor] = None,
		attention_mask: Optional[torch.Tensor] = None,
		position_ids: Optional[torch.LongTensor] = None,
		past_key_values: Optional[Any] = None,
		inputs_embeds: Optional[torch.FloatTensor] = None,
		labels: Optional[torch.LongTensor] = None,
		use_cache: Optional[bool] = None,
		output_attentions: Optional[bool] = None,
		output_hidden_states: Optional[bool] = None,
		return_dict: Optional[bool] = None,
		**kwargs: Any,
	) -> Any:
		if labels is None:
			return original_forward(
				input_ids=input_ids,
				attention_mask=attention_mask,
				position_ids=position_ids,
				past_key_values=past_key_values,
				inputs_embeds=inputs_embeds,
				labels=None,
				use_cache=use_cache,
				output_attentions=output_attentions,
				output_hidden_states=output_hidden_states,
				return_dict=return_dict,
				**kwargs,
			)

		outputs = self.model(
			input_ids=input_ids,
			attention_mask=attention_mask,
			position_ids=position_ids,
			past_key_values=past_key_values,
			inputs_embeds=inputs_embeds,
			use_cache=False,
			output_attentions=output_attentions,
			output_hidden_states=output_hidden_states,
			return_dict=True,
		)
		hidden_states = outputs[0]

		# Shift for autoregressive next-token prediction
		shift_hidden = hidden_states[..., :-1, :].contiguous().view(-1, hidden_states.size(-1))
		shift_labels = labels[..., 1:].contiguous().view(-1)

		active_mask = shift_labels != -100
		active_count = active_mask.sum()

		if active_count.item() == 0:
			zero_loss = (shift_hidden.sum() * 0.0).to(torch.float32)
			return CausalLMOutputWithPast(loss=zero_loss, logits=None)

		active_hidden = shift_hidden[active_mask]
		active_labels = shift_labels[active_mask]

		lm_weight = self.lm_head.weight
		lm_bias = self.lm_head.bias
		total_loss = torch.zeros((), device=active_hidden.device, dtype=torch.float32)
		num_active = active_hidden.size(0)

		for start_idx in range(0, num_active, chunk_size):
			end_idx = min(start_idx + chunk_size, num_active)
			h_chunk = active_hidden[start_idx:end_idx]
			l_chunk = active_labels[start_idx:end_idx]
			if torch.is_grad_enabled() and h_chunk.requires_grad:
				chunk_loss = checkpoint(
					_chunk_ce_loss,
					h_chunk,
					l_chunk,
					lm_weight,
					lm_bias,
					use_reentrant=False,
				)
			else:
				chunk_loss = _chunk_ce_loss(h_chunk, l_chunk, lm_weight, lm_bias)
			total_loss = total_loss + chunk_loss

		loss = total_loss / active_count.to(torch.float32)
		return CausalLMOutputWithPast(
			loss=loss,
			logits=None,
			past_key_values=outputs.past_key_values,
			hidden_states=outputs.hidden_states,
			attentions=outputs.attentions,
		)

	base_causal_lm.forward = types.MethodType(fast_memory_efficient_forward, base_causal_lm)


class MemoryAndSpeedCallback(TrainerCallback):
	"""Periodically clears unused cached CUDA blocks and records throughput metrics."""

	def __init__(self, empty_cache_steps: int = 25):
		self.empty_cache_steps = empty_cache_steps

	def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
		if (
			self.empty_cache_steps > 0
			and state.global_step > 0
			and state.global_step % self.empty_cache_steps == 0
			and torch.cuda.is_available()
		):
			torch.cuda.empty_cache()


def load_model_and_tokenizer(
	cfg: FineTuningConfig,
) -> Tuple[Any, Any, Dict[str, Any]]:
	"""Load Qwen/Qwen2.5-0.5B with 4-bit NF4 double quantization, SDPA, and RSLoRA."""
	enable_hardware_optimizations(cfg)
	if torch.cuda.is_available():
		torch.cuda.empty_cache()
		torch.cuda.reset_peak_memory_stats()

	ram_before = get_process_ram_mb()

	tokenizer = AutoTokenizer.from_pretrained(
		cfg.model_name,
		use_fast=True,
		trust_remote_code=False,
	)
	im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
	if isinstance(im_end_id, int) and im_end_id != tokenizer.unk_token_id:
		tokenizer.eos_token = "<|im_end|>"
		tokenizer.eos_token_id = im_end_id
	if tokenizer.pad_token is None:
		tokenizer.pad_token = "<|endoftext|>"
		tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
	tokenizer.padding_side = "right"

	quant_config = None
	if cfg.use_4bit and torch.cuda.is_available():
		quant_config = BitsAndBytesConfig(
			load_in_4bit=True,
			bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
			bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
			bnb_4bit_compute_dtype=cfg.compute_dtype,
		)

	model = AutoModelForCausalLM.from_pretrained(
		cfg.model_name,
		quantization_config=quant_config,
		dtype=cfg.compute_dtype,
		attn_implementation=cfg.attn_implementation if torch.cuda.is_available() else "eager",
		device_map={"": 0} if torch.cuda.is_available() else "cpu",
		low_cpu_mem_usage=cfg.low_cpu_mem_usage,
	)

	model.config.use_cache = False

	for param in model.parameters():
		param.requires_grad = False

	if cfg.use_gradient_checkpointing:
		model.gradient_checkpointing_enable(
			gradient_checkpointing_kwargs={
				"use_reentrant": cfg.gradient_checkpointing_use_reentrant
			}
		)
		model.enable_input_require_grads()

	lora_config = LoraConfig(
		r=cfg.lora_r,
		lora_alpha=cfg.lora_alpha,
		lora_dropout=cfg.lora_dropout,
		use_rslora=cfg.use_rslora,
		bias="none",
		task_type=TaskType.CAUSAL_LM,
		target_modules=list(cfg.target_modules),
	)
	model = get_peft_model(model, lora_config)

	install_fused_masked_chunked_lm_head(model, chunk_size=512)

	trainable_params, all_params = model.get_nb_trainable_parameters()
	vram_after_load = get_gpu_vram_stats_mb()
	ram_after_load = get_process_ram_mb()

	model_info = {
		"model_name": cfg.model_name,
		"compute_dtype": str(cfg.compute_dtype),
		"attn_implementation": cfg.attn_implementation,
		"use_4bit_nf4": cfg.use_4bit,
		"double_quant": cfg.bnb_4bit_use_double_quant,
		"use_rslora": cfg.use_rslora,
		"lora_r": cfg.lora_r,
		"lora_alpha": cfg.lora_alpha,
		"lora_dropout": cfg.lora_dropout,
		"trainable_params": trainable_params,
		"all_params": all_params,
		"trainable_pct": round(100.0 * trainable_params / max(all_params, 1), 4),
		"ram_before_mb": round(ram_before, 2),
		"ram_after_load_mb": round(ram_after_load, 2),
		"vram_after_load": vram_after_load,
	}
	return model, tokenizer, model_info


def run_fine_tuning(cfg: Optional[FineTuningConfig] = None) -> Dict[str, Any]:
	"""Execute the end-to-end optimized QLoRA fine-tuning pipeline and save adapter weights."""
	cfg = cfg or FineTuningConfig()
	cfg.output_dir.mkdir(parents=True, exist_ok=True)
	cfg.cache_dir.mkdir(parents=True, exist_ok=True)

	start_wall = time.perf_counter()
	model, tokenizer, model_info = load_model_and_tokenizer(cfg)

	train_dataset, eval_dataset, dataset_stats = prepare_fastapi_datasets(tokenizer, cfg)

	data_collator = (
		None
		if cfg.sequence_packing
		else CompletionDataCollator(
			pad_token_id=tokenizer.pad_token_id,
			pad_to_multiple_of=cfg.pad_to_multiple_of,
		)
	)

	optim_name = cfg.optim if torch.cuda.is_available() else "adamw_torch"
	training_args = TrainingArguments(
		output_dir=str(cfg.output_dir),
		per_device_train_batch_size=cfg.per_device_train_batch_size,
		per_device_eval_batch_size=cfg.per_device_eval_batch_size,
		gradient_accumulation_steps=cfg.gradient_accumulation_steps,
		num_train_epochs=cfg.num_train_epochs,
		max_steps=cfg.max_steps,
		learning_rate=cfg.learning_rate,
		weight_decay=cfg.weight_decay,
		warmup_ratio=cfg.warmup_ratio,
		lr_scheduler_type=cfg.lr_scheduler_type,
		max_grad_norm=cfg.max_grad_norm,
		optim=optim_name,
		bf16=cfg.use_bf16,
		fp16=cfg.use_fp16,
		tf32=cfg.use_tf32 and torch.cuda.is_available(),
		gradient_checkpointing=cfg.use_gradient_checkpointing,
		gradient_checkpointing_kwargs={
			"use_reentrant": cfg.gradient_checkpointing_use_reentrant
		},
		group_by_length=(not cfg.sequence_packing) and cfg.group_by_length,
		dataloader_num_workers=cfg.dataloader_num_workers,
		dataloader_pin_memory=cfg.dataloader_pin_memory and torch.cuda.is_available(),
		logging_steps=cfg.logging_steps,
		save_strategy=cfg.save_strategy,
		report_to="none",
		seed=cfg.seed,
		remove_unused_columns=False,
		prediction_loss_only=True,
	)

	trainer = Trainer(
		model=model,
		args=training_args,
		train_dataset=train_dataset,
		eval_dataset=eval_dataset,
		data_collator=data_collator,
		callbacks=[MemoryAndSpeedCallback(empty_cache_steps=cfg.torch_empty_cache_steps)],
	)

	train_start = time.perf_counter()
	train_result = trainer.train()
	train_elapsed = time.perf_counter() - train_start

	eval_metrics = trainer.evaluate() if len(eval_dataset) > 0 else {}

	model.save_pretrained(str(cfg.output_dir))
	tokenizer.save_pretrained(str(cfg.output_dir))

	peak_vram = get_gpu_vram_stats_mb()
	peak_ram_mb = round(get_process_ram_mb(), 2)
	total_wall = time.perf_counter() - start_wall

	trained_tokens = (
		len(train_dataset) * cfg.max_seq_length * cfg.num_train_epochs
		if cfg.sequence_packing
		else dataset_stats["total_tokens"] * cfg.num_train_epochs
	)
	if cfg.max_steps > 0:
		effective_batch = cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps
		trained_tokens = cfg.max_steps * effective_batch * cfg.max_seq_length

	tokens_per_sec = round(trained_tokens / max(train_elapsed, 1e-6), 2)

	summary = {
		"model_info": model_info,
		"dataset_stats": dataset_stats,
		"train_metrics": {
			**train_result.metrics,
			"eval_loss": eval_metrics.get("eval_loss"),
			"train_runtime_sec": round(train_elapsed, 2),
			"total_wall_sec": round(total_wall, 2),
			"effective_tokens_per_sec": tokens_per_sec,
		},
		"memory_stats": {
			"peak_vram": peak_vram,
			"process_ram_mb": peak_ram_mb,
		},
		"adapter_output_dir": str(cfg.output_dir.resolve()),
	}

	summary_path = cfg.output_dir / "training_summary.json"
	summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()

	return summary
