from __future__ import annotations

import asyncio
import re
from pathlib import Path
from threading import Thread
from typing import Any, Dict, Iterator, List, Optional

import torch
from peft import PeftModel
from transformers import (
	AutoModelForCausalLM,
	AutoTokenizer,
	BitsAndBytesConfig,
	StoppingCriteria,
	StoppingCriteriaList,
	TextIteratorStreamer,
)

try:
	from .config import FineTuningConfig
	from .dataset import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_RU, clean_chunk_body
	from .trainer import enable_hardware_optimizations
except ImportError:
	from config import FineTuningConfig
	from dataset import SYSTEM_PROMPT_EN, SYSTEM_PROMPT_RU, clean_chunk_body
	from trainer import enable_hardware_optimizations


def extract_code_blocks(context: List[Dict[str, Any]]) -> List[str]:
	"""Extract fenced Python code blocks from retrieved chunks, prioritizing multi-line examples."""
	blocks: List[str] = []
	seen: set[str] = set()
	pattern = re.compile(r"```(?:python)?\n([\s\S]*?)```")

	for chunk in context:
		raw = chunk.get("text", "")
		for match in pattern.finditer(raw):
			code = match.group(1).strip()
			if not code or code in seen:
				continue
			if "=" in code and "\n" not in code:
				parts = [p.strip() for p in code.split("=", 1)]
				if len(parts) == 2 and parts[0] == parts[1]:
					continue
			seen.add(code)
			blocks.append(code)

	blocks.sort(
		key=lambda c: (
			1 if ("from fastapi" in c or "@app" in c or "def " in c) else 0,
			c.count("\n"),
			len(c),
		),
		reverse=True,
	)
	return blocks


def ensure_code_snippet_in_answer(
	answer: str,
	context: List[Dict[str, Any]],
) -> str:
	"""Guarantee that a code block from the retrieved chunks is present in the answer if available."""
	if not answer:
		return answer

	available_codes = extract_code_blocks(context)
	if not available_codes:
		return answer

	best_code = available_codes[0]
	if "```" not in answer:
		return f"{answer.rstrip()}\n\n```python\n{best_code}\n```"

	if best_code.count("\n") >= 2 and best_code not in answer:
		existing_blocks = re.findall(r"```(?:python)?\n([\s\S]*?)```", answer)
		max_existing_lines = max((b.strip().count("\n") for b in existing_blocks), default=0)
		if max_existing_lines < 2:
			return f"{answer.rstrip()}\n\n```python\n{best_code}\n```"

	return answer


class _ChatMLStopCriteria(StoppingCriteria):
	"""Stop generation cleanly on ChatML boundaries, untrained token artifacts, or repeated headers."""

	def __init__(self, tokenizer: Any, prompt_len: int):
		super().__init__()
		self.tokenizer = tokenizer
		self.prompt_len = prompt_len
		self.stop_markers = (
			"<|im_end|>",
			"<|im_start|>",
			"<|endoftext|>",
			"鲕",
			"Вы — экспертный ассистент",
			"You are an expert FastAPI",
			"\nQuestion:",
			"\nuser\n",
			"\nsystem\n",
		)

	def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs: Any) -> bool:
		gen_len = input_ids.shape[1] - self.prompt_len
		if gen_len <= 0 or gen_len % 4 != 0:
			return False
		gen_ids = input_ids[0, self.prompt_len :]
		tail_text = self.tokenizer.decode(gen_ids[-16:], skip_special_tokens=False)
		for marker in self.stop_markers:
			if marker in tail_text:
				return True
		if "Reference URL:" in tail_text and gen_len > 48:
			full_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
			if full_text.count("Reference URL:") >= 2:
				return True
		return False


class FineTunedQwenGenerator:
	"""Local GPU generator powered by fine-tuned Qwen/Qwen2.5-0.5B + merged QLoRA adapter.

	By default (`use_4bit=False` at inference), merges the trained LoRA adapter directly into
	the bfloat16 base weights (`merge_and_unload()`), using ~980 MB VRAM and accelerating
	token generation by ~35x compared to unmerged 4-bit dequantization per token.
	"""

	def __init__(
		self,
		adapter_dir: Optional[Path] = None,
		base_model_name: str = "Qwen/Qwen2.5-0.5B",
		use_4bit: bool = False,
		max_new_tokens: int = 320,
		temperature: float = 0.0,
	):
		self.cfg = FineTuningConfig(
			model_name=base_model_name,
			use_4bit=use_4bit,
		)
		self.adapter_dir = Path(adapter_dir) if adapter_dir else self.cfg.output_dir
		self.max_new_tokens = max_new_tokens
		self.temperature = temperature
		self._model: Optional[Any] = None
		self._tokenizer: Optional[Any] = None

	def _ensure_loaded(self) -> None:
		"""Lazy-load base model and merge/attach fine-tuned LoRA adapter."""
		if self._model is not None and self._tokenizer is not None:
			return

		enable_hardware_optimizations(self.cfg)
		source_dir = (
			str(self.adapter_dir)
			if (self.adapter_dir / "tokenizer_config.json").exists()
			and (self.adapter_dir / "tokenizer.json").exists()
			else self.cfg.model_name
		)
		self._tokenizer = AutoTokenizer.from_pretrained(source_dir, use_fast=True)
		if self._tokenizer.pad_token is None:
			self._tokenizer.pad_token = "<|endoftext|>"
			self._tokenizer.pad_token_id = self._tokenizer.convert_tokens_to_ids("<|endoftext|>")

		quant_config = None
		if self.cfg.use_4bit and torch.cuda.is_available():
			quant_config = BitsAndBytesConfig(
				load_in_4bit=True,
				bnb_4bit_quant_type="nf4",
				bnb_4bit_use_double_quant=True,
				bnb_4bit_compute_dtype=self.cfg.compute_dtype,
			)

		base_model = AutoModelForCausalLM.from_pretrained(
			self.cfg.model_name,
			quantization_config=quant_config,
			dtype=self.cfg.compute_dtype,
			attn_implementation=self.cfg.attn_implementation if torch.cuda.is_available() else "eager",
			device_map={"": 0} if torch.cuda.is_available() else "cpu",
			low_cpu_mem_usage=True,
		)
		base_model.config.use_cache = True

		has_adapter_weights = (
			(self.adapter_dir / "adapter_config.json").exists()
			and (
				(self.adapter_dir / "adapter_model.safetensors").exists()
				or (self.adapter_dir / "adapter_model.bin").exists()
			)
		)
		if has_adapter_weights:
			peft_model = PeftModel.from_pretrained(base_model, str(self.adapter_dir))
			if not self.cfg.use_4bit:
				self._model = peft_model.merge_and_unload()
			else:
				self._model = peft_model
		else:
			self._model = base_model
		self._model.eval()

	def _build_chatml_prompt(
		self,
		query: str,
		context: Optional[List[Dict[str, Any]]] = None,
		language: Optional[str] = None,
	) -> str:
		is_ru = (language or "").lower() in ("ru", "uk") or bool(
			any("а" <= ch.lower() <= "я" for ch in query)
		)
		sys_prompt = SYSTEM_PROMPT_RU if is_ru else SYSTEM_PROMPT_EN

		if context:
			top_chunk = context[0]
			lang = top_chunk.get("language") or ("ru" if is_ru else "en")
			breadcrumb = top_chunk.get("breadcrumb") or top_chunk.get("section_title") or "FastAPI"
			sym_type = top_chunk.get("symbol_type") or "section"
			sec_url = top_chunk.get("section_url") or ""
			body = clean_chunk_body(top_chunk.get("text", ""))[:1500]
			rag_context = (
				f"--- Source [1] (lang={lang}) ---\n"
				f"Section: {breadcrumb} ({sym_type})\n"
				f"URL: {sec_url}\n"
				f"Content:\n{body}"
			)
			user_text = (
				f"Question: {query}\n\n"
				f"FastAPI Reference Context:\n{rag_context}\n\n"
				"Answer the question strictly using the provided FastAPI Reference Context and include code blocks."
			)
		else:
			user_text = query

		return (
			f"<|im_start|>system\n{sys_prompt}<|im_end|>\n"
			f"<|im_start|>user\n{user_text}<|im_end|>\n"
			f"<|im_start|>assistant\n"
		)

	@staticmethod
	def _clean_generated_text(text: str) -> str:
		"""Trim any trailing stop markers or partial duplicate headers."""
		for marker in (
			"<|im_end|>",
			"<|im_start|>",
			"<|endoftext|>",
			"鲕",
			"Вы — экспертный ассистент",
			"You are an expert FastAPI",
			"\nQuestion:",
		):
			if marker in text:
				text = text.split(marker, 1)[0]
		if text.count("Reference URL:") >= 2:
			first_pos = text.find("Reference URL:")
			second_pos = text.find("Reference URL:", first_pos + 14)
			prefix = text[:second_pos]
			last_heading = prefix.rfind("\n#")
			if last_heading != -1 and last_heading > first_pos:
				text = prefix[:last_heading]
			else:
				text = prefix
		return text.strip()

	def generate_sync(
		self,
		query: str,
		context: Optional[List[Dict[str, Any]]] = None,
		language: Optional[str] = None,
	) -> str:
		"""Synchronously generate a FastAPI answer with KV-cache and torch.inference_mode."""
		self._ensure_loaded()
		assert self._model is not None and self._tokenizer is not None

		chatml_prompt = self._build_chatml_prompt(query, context, language)
		inputs = self._tokenizer(chatml_prompt, return_tensors="pt")
		device = next(self._model.parameters()).device
		inputs = {k: v.to(device) for k, v in inputs.items()}
		input_len = inputs["input_ids"].shape[1]

		eos_ids = [
			tid
			for tid in (
				self._tokenizer.eos_token_id,
				self._tokenizer.convert_tokens_to_ids("<|im_end|>"),
				self._tokenizer.convert_tokens_to_ids("<|endoftext|>"),
			)
			if isinstance(tid, int) and tid >= 0
		]
		stop_criteria = StoppingCriteriaList([_ChatMLStopCriteria(self._tokenizer, input_len)])

		with torch.inference_mode():
			output_ids = self._model.generate(
				**inputs,
				max_new_tokens=self.max_new_tokens,
				do_sample=self.temperature > 0.0,
				temperature=self.temperature if self.temperature > 0.0 else None,
				top_p=0.9 if self.temperature > 0.0 else None,
				repetition_penalty=1.12,
				eos_token_id=eos_ids,
				pad_token_id=self._tokenizer.pad_token_id,
				stopping_criteria=stop_criteria,
				use_cache=True,
			)

		gen_ids = output_ids[0, input_len:]
		raw_answer = self._tokenizer.decode(gen_ids, skip_special_tokens=True)
		answer = self._clean_generated_text(raw_answer)
		if context:
			answer = ensure_code_snippet_in_answer(answer, context)
		return answer

	def stream_generate_sync(
		self,
		query: str,
		context: Optional[List[Dict[str, Any]]] = None,
		language: Optional[str] = None,
	) -> Iterator[str]:
		"""Stream generated tokens in real time for console UI responsiveness."""
		self._ensure_loaded()
		assert self._model is not None and self._tokenizer is not None

		chatml_prompt = self._build_chatml_prompt(query, context, language)
		inputs = self._tokenizer(chatml_prompt, return_tensors="pt")
		device = next(self._model.parameters()).device
		inputs = {k: v.to(device) for k, v in inputs.items()}
		input_len = inputs["input_ids"].shape[1]

		eos_ids = [
			tid
			for tid in (
				self._tokenizer.eos_token_id,
				self._tokenizer.convert_tokens_to_ids("<|im_end|>"),
				self._tokenizer.convert_tokens_to_ids("<|endoftext|>"),
			)
			if isinstance(tid, int) and tid >= 0
		]
		stop_criteria = StoppingCriteriaList([_ChatMLStopCriteria(self._tokenizer, input_len)])

		streamer = TextIteratorStreamer(
			self._tokenizer,
			skip_prompt=True,
			skip_special_tokens=True,
		)
		gen_kwargs = {
			**inputs,
			"streamer": streamer,
			"max_new_tokens": self.max_new_tokens,
			"do_sample": self.temperature > 0.0,
			"temperature": self.temperature if self.temperature > 0.0 else None,
			"top_p": 0.9 if self.temperature > 0.0 else None,
			"repetition_penalty": 1.12,
			"eos_token_id": eos_ids,
			"pad_token_id": self._tokenizer.pad_token_id,
			"stopping_criteria": stop_criteria,
			"use_cache": True,
		}

		def _worker():
			with torch.inference_mode():
				self._model.generate(**gen_kwargs)

		thread = Thread(target=_worker, daemon=True)
		thread.start()
		accum = ""
		for chunk_text in streamer:
			accum += chunk_text
			cleaned_chunk = chunk_text.replace("鲕", "")
			if any(
				m in accum
				for m in ("Вы — экспертный ассистент", "You are an expert FastAPI", "\nQuestion:")
			):
				break
			if accum.count("Reference URL:") >= 2:
				break
			if cleaned_chunk:
				yield cleaned_chunk
		thread.join()

	async def generate(self, query: str, context: List[Dict[str, Any]]) -> str:
		"""Asynchronously generate a response given query and retrieved context."""
		return await asyncio.to_thread(self.generate_sync, query, context, None)

	async def generate_with_history(
		self,
		query: str,
		context: List[Dict[str, Any]],
		history: Optional[List[Dict[str, str]]] = None,
		language: Optional[str] = None,
	) -> str:
		"""Asynchronously generate a response compatible with HybridRAGPipeline."""
		return await asyncio.to_thread(self.generate_sync, query, context, language)
