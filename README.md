# FastAPI-Qwen-QLoRA (Educational SFT / QLoRA Pipeline)

> **Educational Project Notice / Учебный проект:**
> This repository is an **educational and experimental implementation** demonstrating how to fine-tune a small language model (`Qwen/Qwen2.5-0.5B`) on structured technical documentation using **4-bit NF4 QLoRA + RSLoRA**, constant-length sequence packing, and a custom fused masked/chunked `lm_head` cross-entropy kernel designed for consumer GPUs (`~1.5 GB` peak VRAM).

---

## Data Source (`FastAPI-RAG`)

All training and evaluation data in this project is sourced directly from the companion **FastAPI Hybrid RAG** repository:

- **Source Repository**: [**Squ1dyyy/FastAPI-RAG**](https://github.com/Squ1dyyy/FastAPI-RAG)

In [**FastAPI-RAG**](https://github.com/Squ1dyyy/FastAPI-RAG), the official **FastAPI Reference** documentation (`https://fastapi.tiangolo.com/reference/`) was scraped across **13 languages** (`299` HTML documents) and parsed along the DOM tree (`h1 -> h2 -> h3`) into **14,534 structural chunks** stored in `data/rag.db` (SQLite / PostgreSQL).

`FastAPI-Qwen-QLoRA` streams the English (`en`) and Russian (`ru`) structural documentation chunks directly from `data/rag.db` (excluding the 130 MB dense embedding column to keep host RAM usage minimal) and converts each documentation section into **ChatML (`<|im_start|>` / `<|im_end|>`)** instruction pairs:
1. **Direct API Reference Q&A**: Asking for signatures, parameter tables, and usage examples of FastAPI classes/functions (`FastAPI`, `APIRouter`, `Depends`, `BackgroundTasks`, `HTTPException`, etc.).
2. **Context-Grounded RAG Synthesis**: Training the model to synthesize accurate, code-complete Markdown answers strictly from retrieved documentation chunks with official reference URLs.

---

## Key Educational & Hardware Optimizations

1. **4-Bit NF4 Double Quantization + RSLoRA (`peft` + `bitsandbytes`)**:
   - Base `Qwen/Qwen2.5-0.5B` weights are quantized to 4-bit NormalFloat (`nf4`) with double quantization and `bfloat16` compute.
   - Rank-Stabilized LoRA (`r=16`, `lora_alpha=32`, `use_rslora=True`, `lora_dropout=0.0`) targets all attention and MLP projections (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` — `8.8M` trainable parameters, `1.75%` of total).
2. **Fused Masked & Chunked `lm_head` Cross-Entropy (`install_fused_masked_chunked_lm_head`)**:
   - `Qwen2.5-0.5B` has a massive vocabulary (`151,936` tokens) relative to its hidden dimension (`896`). Standard `CausalLM` forward passes materialize a full `[B, S, 151936]` FP32 logits tensor (`>1.2 GB` VRAM).
   - Our custom forward patch filters out masked prompt tokens (`shift_labels != -100`) *before* `lm_head` projection (saving ~40% FLOPs) and computes `lm_head + cross_entropy` in `512`-token gradient-checkpointed micro-chunks.
3. **Constant-Length Sequence Packing (`PackedConstantLengthDataset`)**:
   - Concatenates tokenized ChatML pairs into dense `512`-token `int32` NumPy blocks with `0%` padding waste and masks sequence boundaries with `-100` so gradients only flow through assistant completions.
4. **Zero-Overhead Merged BFloat16 Inference (`FineTunedQwenGenerator`)**:
   - At inference time, calls `merge_and_unload()` to fuse the trained LoRA adapter directly into `bfloat16` weights (`~980 MB` VRAM) for fast streaming token generation in the CLI console.

---

## Training Benchmarks (`training_summary.json`)

| Metric | Value |
| :--- | :--- |
| **Base Model** | `Qwen/Qwen2.5-0.5B` |
| **Method** | `4-bit NF4 QLoRA + RSLoRA (r=16, alpha=32)` |
| **Trainable Parameters** | `8,798,208 / 502,830,976 (1.75%)` |
| **Source Chunks Scanned (`en`, `ru`)** | `2,236` (`2,094` qualifying chunks, `1,117,481` tokens) |
| **Packed Sequences (`seq_len=512`)** | `2,081` train / `101` eval |
| **Effective Throughput** | `2,135.92 tokens/sec` (`~8.3 min` for 1 full epoch) |
| **Final Train / Eval Loss** | `1.6294` / `0.2740` |
| **Peak GPU VRAM** | `1,524.01 MB` (`~503 MB` allocated steady-state) |
| **Process Host RAM** | `1,834.56 MB` |

---

## Project Structure

```text
FastAPI-Qwen-QLoRA/
├── fastapi_qlora/
│   ├── output/
│   │   └── qwen2.5-0.5b-fastapi-lora/
│   │       ├── adapter_config.json      # PEFT LoRA / RSLoRA configuration
│   │       └── training_summary.json    # Recorded training & VRAM/RAM metrics
│   ├── __init__.py
│   ├── cli_chat.py                      # Interactive SQLite BM25 + Qwen2.5 console assistant
│   ├── config.py                        # FineTuningConfig dataclass & DB path resolver
│   ├── dataset.py                       # SQLite chunk streamer, ChatML builder & sequence packer
│   ├── inference.py                     # Merged bfloat16 generator with real-time token streaming
│   ├── train.py                         # CLI entry point for QLoRA fine-tuning
│   └── trainer.py                       # Fused chunked lm_head patch & HuggingFace Trainer loop
├── tests/
│   └── test_finetuning.py               # Unit tests for dataset packing, loss equivalence & CLI retrieval
├── ask_db.bat                           # Windows launcher for the interactive CLI assistant
├── ask_db.py                            # Python entry point for the interactive CLI assistant
├── requirements.txt                     # PyTorch CUDA 12.6, Transformers, PEFT, bitsandbytes, bm25s
└── README.md
```

---

## Quickstart

### 1. Install Dependencies

```powershell
python -m venv venv
.\venv\Scripts\pip install -r requirements.txt
```

### 2. Link or Copy `data/rag.db` from `FastAPI-RAG`

Clone [**Squ1dyyy/FastAPI-RAG**](https://github.com/Squ1dyyy/FastAPI-RAG) alongside this repository (or copy its `data/rag.db` into `data/rag.db`, or set `FASTAPI_RAG_DB_PATH`):

```powershell
git clone https://github.com/Squ1dyyy/FastAPI-RAG.git ../FastAPI-RAG
```

*(Note: `fastapi_qlora/config.py` automatically detects `../FastAPI-RAG/data/rag.db` or `./data/rag.db`).*

### 3. Run QLoRA Fine-Tuning

```powershell
.\venv\Scripts\python fastapi_qlora/train.py --languages en,ru --epochs 1.0 --verify-generation
```

### 4. Interactive Console RAG Assistant (`ask_db.py`)

Query the `data/rag.db` SQLite database in Russian or English using the fine-tuned `Qwen2.5-0.5B` model:

```powershell
# Interactive REPL mode
.\venv\Scripts\python ask_db.py

# Single question from CLI
.\venv\Scripts\python ask_db.py "Как работает Depends в FastAPI? Приведи пример кода"
```

### 5. Run Unit Tests

```powershell
.\venv\Scripts\python -m unittest discover tests -v
```
