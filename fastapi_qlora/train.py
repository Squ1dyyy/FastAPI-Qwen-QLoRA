from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))
if str(PKG_DIR) not in sys.path:
	sys.path.insert(0, str(PKG_DIR))

from config import FineTuningConfig
from inference import FineTunedQwenGenerator
from trainer import run_fine_tuning


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Educational low-RAM/VRAM 4-bit QLoRA fine-tuning for Qwen/Qwen2.5-0.5B on FastAPI docs."
	)
	parser.add_argument("--model-name", type=str, default="Qwen/Qwen2.5-0.5B")
	parser.add_argument("--languages", type=str, default="en,ru", help="Comma-separated languages or 'all'")
	parser.add_argument("--max-seq-length", type=int, default=512)
	parser.add_argument("--batch-size", type=int, default=4)
	parser.add_argument("--grad-accum", type=int, default=4)
	parser.add_argument("--epochs", type=float, default=1.0)
	parser.add_argument("--max-steps", type=int, default=-1)
	parser.add_argument("--lr", type=float, default=3e-4)
	parser.add_argument("--lora-r", type=int, default=16)
	parser.add_argument("--lora-alpha", type=int, default=32)
	parser.add_argument("--no-packing", action="store_true", help="Disable constant-length sequence packing")
	parser.add_argument("--max-samples", type=int, default=0, help="Limit number of training pairs (0 = all)")
	parser.add_argument("--output-dir", type=str, default="fastapi_qlora/output/qwen2.5-0.5b-fastapi-lora")
	parser.add_argument("--verify-generation", action="store_true", help="Run sample generation after training")
	return parser.parse_args()


def main() -> None:
	args = parse_args()
	langs = tuple(l.strip() for l in args.languages.split(",") if l.strip())

	cfg = FineTuningConfig(
		model_name=args.model_name,
		languages=langs,
		max_seq_length=args.max_seq_length,
		per_device_train_batch_size=args.batch_size,
		per_device_eval_batch_size=args.batch_size,
		gradient_accumulation_steps=args.grad_accum,
		num_train_epochs=args.epochs,
		max_steps=args.max_steps,
		learning_rate=args.lr,
		lora_r=args.lora_r,
		lora_alpha=args.lora_alpha,
		sequence_packing=not args.no_packing,
		max_samples=args.max_samples,
		output_dir=Path(args.output_dir),
	)

	print("=" * 72)
	print("Starting Qwen/Qwen2.5-0.5B FastAPI QLoRA Fine-Tuning (Educational)")
	print(f"Languages: {cfg.languages} | Packing: {cfg.sequence_packing} | MaxSeqLen: {cfg.max_seq_length}")
	print(f"BatchSize: {cfg.per_device_train_batch_size} x GradAccum: {cfg.gradient_accumulation_steps}")
	print("=" * 72)

	summary = run_fine_tuning(cfg)
	print("\n" + "=" * 72)
	print("FINE-TUNING SUMMARY")
	print("=" * 72)
	print(json.dumps(summary, indent=2, ensure_ascii=False))

	if args.verify_generation:
		print("\n" + "=" * 72)
		print("POST-TRAINING VERIFICATION GENERATION")
		print("=" * 72)
		gen = FineTunedQwenGenerator(
			adapter_dir=cfg.output_dir,
			base_model_name=cfg.model_name,
			max_new_tokens=200,
			temperature=0.1,
		)
		sample_q = "How do I use the `APIRouter` class in FastAPI? Provide its signature, parameters, and code example."
		ans = gen.generate_sync(sample_q, language="en")
		print(f"Query: {sample_q}\n\nResponse:\n{ans}")


if __name__ == "__main__":
	main()
