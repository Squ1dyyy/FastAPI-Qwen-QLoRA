from __future__ import annotations

import argparse
import json
import math
import re
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
PKG_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
	sys.path.insert(0, str(ROOT_DIR))
if str(PKG_DIR) not in sys.path:
	sys.path.insert(0, str(PKG_DIR))

from config import _resolve_default_db_path
from dataset import stream_chunks_from_sqlite
from inference import FineTunedQwenGenerator, ensure_code_snippet_in_answer
from trainer import get_gpu_vram_stats_mb, get_process_ram_mb

try:
	import bm25s

	HAS_BM25S = True
except ImportError:
	HAS_BM25S = False


class StandaloneBM25Retriever:
	"""Self-contained BM25-Fast lexical retriever with snake_case and CamelCase identifier splitting."""

	def __init__(self, k1: float = 1.5, b: float = 0.75, use_bm25s: bool = True):
		self.k1 = k1
		self.b = b
		self.use_bm25s = use_bm25s and HAS_BM25S
		self.corpus: List[Dict[str, Any]] = []
		self._bm25s_model = None
		self.doc_lengths: List[int] = []
		self.avg_doc_len: float = 0.0
		self.doc_count: int = 0
		self.inverted_index: Dict[str, List[tuple[int, int]]] = defaultdict(list)
		self.idf: Dict[str, float] = {}

	@staticmethod
	def tokenize(text: str) -> List[str]:
		"""Tokenize text into lowercase words, splitting snake_case and CamelCase identifiers."""
		raw_tokens = re.findall(r"\b\w+\b", text)
		tokens: List[str] = []
		for tok in raw_tokens:
			low = tok.lower()
			tokens.append(low)
			if "_" in low:
				parts = [p for p in low.split("_") if p]
				if len(parts) > 1:
					tokens.extend(parts)
			camel_parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\b)|[A-Z]?[a-z]+|[0-9]+", tok)
			if len(camel_parts) > 1:
				tokens.extend(p.lower() for p in camel_parts if p)
		return tokens

	def index(self, corpus: List[Dict[str, Any]]) -> None:
		self.corpus = corpus
		self.doc_count = len(corpus)
		self.doc_lengths = []
		self.inverted_index.clear()
		self.idf.clear()
		self._bm25s_model = None
		if not corpus:
			return

		tokenized_corpus: List[List[str]] = []
		total_len = 0
		doc_freqs: Dict[str, int] = Counter()

		for doc_idx, doc in enumerate(corpus):
			parts = [
				str(doc.get("section_title", "")),
				str(doc.get("anchor", "")),
				str(doc.get("breadcrumb", "")),
				str(doc.get("text", "")),
			]
			combined_text = " ".join(p for p in parts if p)
			tokens = self.tokenize(combined_text) or ["_empty_"]
			tokenized_corpus.append(tokens)
			doc_len = len(tokens)
			self.doc_lengths.append(doc_len)
			total_len += doc_len
			counts = Counter(tokens)
			for token, count in counts.items():
				self.inverted_index[token].append((doc_idx, count))
				doc_freqs[token] += 1

		self.avg_doc_len = total_len / self.doc_count if self.doc_count > 0 else 0.0
		for token, df in doc_freqs.items():
			self.idf[token] = math.log((self.doc_count - df + 0.5) / (df + 0.5) + 1.0)

		if self.use_bm25s:
			self._bm25s_model = bm25s.BM25(k1=self.k1, b=self.b)
			self._bm25s_model.index(tokenized_corpus, show_progress=False)

	def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
		if not self.corpus:
			return []
		tokens = self.tokenize(query)
		if not tokens:
			return []
		k = min(top_k, len(self.corpus))

		if self.use_bm25s and self._bm25s_model is not None:
			doc_indices, scores = self._bm25s_model.retrieve([tokens], k=k, show_progress=False)
			results = []
			for idx_val, score_val in zip(doc_indices[0], scores[0]):
				score = float(score_val)
				if score <= 0.0:
					continue
				doc = dict(self.corpus[int(idx_val)])
				doc["score"] = score
				doc["retriever"] = "bm25_fast"
				results.append(doc)
			if results:
				return results

		scores_map: Dict[int, float] = defaultdict(float)
		for token in tokens:
			if token not in self.inverted_index:
				continue
			idf_val = self.idf.get(token, 0.0)
			for doc_idx, tf in self.inverted_index[token]:
				doc_len = self.doc_lengths[doc_idx]
				num = tf * (self.k1 + 1.0)
				den = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
				scores_map[doc_idx] += idf_val * (num / den)

		sorted_docs = sorted(scores_map.items(), key=lambda x: x[1], reverse=True)[:k]
		return [
			{**self.corpus[doc_idx], "score": float(score), "retriever": "bm25_fast"}
			for doc_idx, score in sorted_docs
		]


RU_FASTAPI_TERM_MAP: List[tuple[str, str]] = [
	("зависимост", "Depends dependency"),
	("роутер", "APIRouter router include_router"),
	("маршрут", "APIRouter api_route route"),
	("фонов", "BackgroundTasks background add_task"),
	("запрос", "Request HTTPConnection"),
	("ответ", "Response JSONResponse HTMLResponse StreamingResponse FileResponse RedirectResponse"),
	("исключен", "HTTPException WebSocketException RequestValidationError"),
	("ошибк", "HTTPException WebSocketException RequestValidationError"),
	("файл", "UploadFile File FileResponse"),
	("загрузк", "UploadFile File"),
	("параметр", "Query Path Body Cookie Header Form parameters"),
	("заголов", "Header headers"),
	("кук", "Cookie cookies"),
	("форм", "Form"),
	("вебсокет", "WebSocket WebSocketDisconnect websockets"),
	("безопасност", "Security OAuth2PasswordBearer HTTPBearer APIKeyHeader security"),
	("авторизац", "Security OAuth2PasswordBearer HTTPBearer"),
	("аутентификац", "Security OAuth2PasswordBearer HTTPBasic"),
	("токен", "OAuth2PasswordBearer HTTPBearer Security"),
	("мидлвар", "Middleware CORSMiddleware GZipMiddleware TrustedHostMiddleware"),
	("посредник", "Middleware CORSMiddleware"),
	("корс", "CORSMiddleware"),
	("тест", "TestClient testclient"),
	("клиент", "TestClient"),
	("шаблон", "Jinja2Templates templating"),
	("кодиров", "jsonable_encoder encoders"),
	("статус", "status HTTP_200_OK HTTP_404_NOT_FOUND"),
	("статик", "StaticFiles staticfiles"),
	("стрим", "StreamingResponse EventSourceResponse sse"),
	("событ", "EventSourceResponse ServerSentEvent sse"),
	("приложен", "FastAPI fastapi"),
	("документац", "FastAPI get_openapi get_swagger_ui_html"),
]


def expand_multilingual_fastapi_query(query: str) -> str:
	"""Expand Russian FastAPI terminology into official FastAPI symbol identifiers for BM25."""
	low = query.lower()
	extra_terms: List[str] = []
	for ru_stem, en_symbols in RU_FASTAPI_TERM_MAP:
		if ru_stem in low:
			extra_terms.append(en_symbols)
	if extra_terms:
		return f"{query} {' '.join(extra_terms)}"
	return query


def detect_query_language(query: str, configured_lang: str = "auto") -> str:
	"""Detect 'ru' or 'en' from query characters when language is set to 'auto'."""
	if configured_lang and configured_lang.lower() != "auto":
		return configured_lang.lower()
	has_cyrillic = any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in query)
	return "ru" if has_cyrillic else "en"


class FastSQLiteRAGConsole:
	"""Memory-efficient console RAG engine over data/rag.db + fine-tuned Qwen2.5-0.5B."""

	def __init__(
		self,
		db_path: Optional[Path] = None,
		adapter_dir: Path = ROOT_DIR / "fastapi_qlora" / "output" / "qwen2.5-0.5b-fastapi-lora",
		mode: str = "fast",
		language: str = "auto",
		top_k: int = 3,
		max_new_tokens: int = 384,
		temperature: float = 0.1,
	):
		resolved_db = db_path if db_path is not None else _resolve_default_db_path()
		self.db_path = Path(resolved_db) if Path(resolved_db).is_absolute() else (
			Path(resolved_db) if Path(resolved_db).exists() else (ROOT_DIR / resolved_db)
		)
		if not self.db_path.exists():
			fallback = _resolve_default_db_path()
			if fallback.exists():
				self.db_path = fallback

		self.adapter_dir = Path(adapter_dir) if Path(adapter_dir).is_absolute() else (
			Path(adapter_dir) if Path(adapter_dir).exists() else (ROOT_DIR / adapter_dir)
		)
		self.mode = mode.lower()
		self.language = language.lower()
		self.top_k = top_k
		self.generator = FineTunedQwenGenerator(
			adapter_dir=self.adapter_dir,
			max_new_tokens=max_new_tokens,
			temperature=temperature,
		)
		self._bm25_by_lang: Dict[str, StandaloneBM25Retriever] = {}
		self._corpora_by_lang: Dict[str, List[Dict[str, Any]]] = {}
		self._parent_h2_by_lang: Dict[str, Dict[tuple[str, str], Dict[str, Any]]] = {}

	def _ensure_lang_bm25(self, lang: str) -> StandaloneBM25Retriever:
		"""Lazy-load chunks for `lang` from SQLite without 130MB embeddings and build BM25-Fast index."""
		lang = lang.lower()
		if lang in self._bm25_by_lang:
			return self._bm25_by_lang[lang]

		chunks = list(stream_chunks_from_sqlite(self.db_path, languages=(lang,)))
		if not chunks and lang != "en":
			chunks = list(stream_chunks_from_sqlite(self.db_path, languages=("en",)))

		parent_map: Dict[tuple[str, str], Dict[str, Any]] = {}
		for ch in chunks:
			meta = ch.get("metadata", {})
			doc_url = meta.get("doc_url") or ch.get("section_url", "").split("#")[0]
			crumb = ch.get("breadcrumb") or ""
			parts = [p.strip() for p in crumb.split(" > ") if p.strip()]
			if meta.get("heading_level") == 2 or len(parts) == 2:
				h2_crumb = " > ".join(parts[:2])
				if doc_url and h2_crumb and (doc_url, h2_crumb) not in parent_map:
					parent_map[(doc_url, h2_crumb)] = ch

		retr = StandaloneBM25Retriever()
		retr.index(chunks)
		self._corpora_by_lang[lang] = chunks
		self._parent_h2_by_lang[lang] = parent_map
		self._bm25_by_lang[lang] = retr
		return retr

	def _expand_h2_context(self, results: List[Dict[str, Any]], lang: str) -> List[Dict[str, Any]]:
		"""Enrich child h3 attributes/methods with parent h2 class/function signatures & examples."""
		parent_map = self._parent_h2_by_lang.get(lang, {})
		if not parent_map:
			return results

		expanded: List[Dict[str, Any]] = []
		for item in results:
			meta = item.get("metadata", {})
			doc_url = meta.get("doc_url") or item.get("section_url", "").split("#")[0]
			crumb = item.get("breadcrumb") or ""
			parts = [p.strip() for p in crumb.split(" > ") if p.strip()]
			if len(parts) >= 3:
				h2_crumb = " > ".join(parts[:2])
				parent = parent_map.get((doc_url, h2_crumb))
				if parent and parent.get("id") != item.get("id"):
					merged = dict(item)
					p_txt = parent.get("text", "").strip()
					c_txt = item.get("text", "").strip()
					if c_txt not in p_txt:
						merged["text"] = f"{p_txt}\n\n{c_txt}"
					else:
						merged["text"] = p_txt
					expanded.append(merged)
					continue
			expanded.append(item)
		return expanded

	def retrieve_from_db(self, query: str) -> tuple[List[Dict[str, Any]], str]:
		"""Retrieve top-k relevant FastAPI chunks from data/rag.db."""
		active_lang = detect_query_language(query, self.language)

		if self.mode == "direct" or not self.db_path.exists():
			return [], active_lang

		retr = self._ensure_lang_bm25(active_lang)
		expanded_query = expand_multilingual_fastapi_query(query)
		raw_hits = retr.retrieve(expanded_query, top_k=self.top_k * 3)
		if not raw_hits and active_lang != "en":
			en_retr = self._ensure_lang_bm25("en")
			raw_hits = en_retr.retrieve(expanded_query, top_k=self.top_k * 3)

		raw_hits.sort(
			key=lambda item: (
				0 if item.get("symbol_type") == "overview" else 1,
				1 if "```python" in item.get("text", "") else 0,
				float(item.get("score", 0.0)),
			),
			reverse=True,
		)
		expanded_hits = self._expand_h2_context(raw_hits, active_lang)
		deduped: List[Dict[str, Any]] = []
		seen_keys: set[str] = set()
		for hit in expanded_hits:
			key = f"{hit.get('section_url', '')}|{hit.get('breadcrumb', '')}"
			if key not in seen_keys:
				seen_keys.add(key)
				deduped.append(hit)
			if len(deduped) >= self.top_k:
				break
		return deduped, active_lang

	def get_db_stats(self) -> Dict[str, Any]:
		"""Query SQLite data/rag.db statistics."""
		if not self.db_path.exists():
			return {"error": f"Database not found at {self.db_path}"}
		conn = sqlite3.connect(str(self.db_path))
		try:
			cur = conn.cursor()
			total_docs = cur.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
			total_chunks = cur.execute("SELECT COUNT(*) FROM document_chunks").fetchone()[0]
			langs = cur.execute(
				"SELECT language, COUNT(*) FROM document_chunks GROUP BY language ORDER BY language"
			).fetchall()
			symbols = cur.execute(
				"SELECT symbol_type, COUNT(*) FROM document_chunks GROUP BY symbol_type ORDER BY COUNT(*) DESC"
			).fetchall()
			return {
				"db_path": str(self.db_path.resolve()),
				"db_size_mb": round(self.db_path.stat().st_size / (1024 * 1024), 2),
				"total_documents": total_docs,
				"total_chunks": total_chunks,
				"chunks_by_language": dict(langs),
				"chunks_by_symbol_type": dict(symbols),
				"process_ram_mb": round(get_process_ram_mb(), 2),
				"gpu_vram": get_gpu_vram_stats_mb(),
			}
		finally:
			conn.close()

	def execute_readonly_sql(self, sql_query: str) -> List[Any]:
		"""Execute a read-only SELECT query on data/rag.db for inspection."""
		cleaned = sql_query.strip()
		if not cleaned.lower().startswith("select"):
			raise ValueError("Only SELECT queries are allowed in /sql.")
		conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
		try:
			cur = conn.cursor()
			return cur.execute(cleaned).fetchmany(20)
		finally:
			conn.close()

	def ask(self, query: str, stream: bool = True) -> str:
		"""Retrieve context from data/rag.db and generate/stream answer with fine-tuned Qwen2.5-0.5B."""
		t0 = time.perf_counter()
		context, active_lang = self.retrieve_from_db(query)
		t_retr = time.perf_counter() - t0

		if context:
			print(f"\n[Найдено в БД ({active_lang}, {t_retr*1000:.1f} мс)]:")
			for idx, ch in enumerate(context, start=1):
				crumb = ch.get("breadcrumb") or ch.get("section_title") or "FastAPI"
				sym = ch.get("symbol_type") or "section"
				url = ch.get("section_url") or ""
				score = ch.get("rerank_score") or ch.get("score") or 0.0
				print(f"  {idx}. [{sym}] {crumb} (score={float(score):.3f})")
				if url:
					print(f"     URL: {url}")
		else:
			print(f"\n[Режим: {self.mode} | Язык: {active_lang}]")

		print("\n[Ответ Qwen2.5-0.5B-FastAPI]:")
		t_gen_start = time.perf_counter()

		if stream:
			pieces: List[str] = []
			for token_str in self.generator.stream_generate_sync(
				query=query,
				context=context if context else None,
				language=active_lang,
			):
				sys.stdout.write(token_str)
				sys.stdout.flush()
				pieces.append(token_str)
			print()
			full_answer = "".join(pieces).strip()
			if context:
				enriched = ensure_code_snippet_in_answer(full_answer, context)
				if enriched != full_answer:
					added_part = enriched[len(full_answer) :]
					print(added_part)
					full_answer = enriched
		else:
			full_answer = self.generator.generate_sync(
				query=query,
				context=context if context else None,
				language=active_lang,
			)
			print(full_answer)

		t_gen = time.perf_counter() - t_gen_start
		vram = get_gpu_vram_stats_mb()
		ram_mb = get_process_ram_mb()
		print(
			f"\n--- Время: поиск {t_retr*1000:.0f} мс | генерация {t_gen:.2f} с | "
			f"VRAM: {vram['allocated_mb']:.0f} МБ | RAM: {ram_mb:.0f} МБ ---"
		)
		return full_answer


def print_help() -> None:
	print(
		"\nДоступные команды консоли:\n"
		"  <любой текст>          — Задать вопрос по базе документации FastAPI (data/rag.db)\n"
		"  /search <запрос>       — Показать найденные чанки в БД без генерации ответа\n"
		"  /stats                 — Показать статистику БД data/rag.db и памяти (RAM/VRAM)\n"
		"  /sql <SELECT ...>      — Выполнить SQL SELECT запрос к data/rag.db (макс. 20 строк)\n"
		"  /lang <auto|ru|en|...> — Переключить язык поиска в БД (по умолчанию: auto)\n"
		"  /mode <fast|direct>    — Режим: fast (BM25+LCA), direct (только LLM)\n"
		"  /top <число>           — Количество используемых чанков из БД (по умолчанию: 3)\n"
		"  /help                  — Показать эту справку\n"
		"  /exit или /quit        — Выйти из программы\n"
	)


def run_repl(console: FastSQLiteRAGConsole) -> None:
	print("=" * 76)
	print(" FastAPI QLoRA Console — Qwen/Qwen2.5-0.5B (4-bit NF4 QLoRA) + SQLite DB")
	print(f" База данных : {console.db_path} ({'найдена' if console.db_path.exists() else 'не найдена'})")
	print(f" Адаптер LoRA: {console.adapter_dir}")
	print(f" Режим       : {console.mode} | Язык: {console.language} | Top-K: {console.top_k}")
	print(" Введите вопрос или /help для списка команд (/exit для выхода).")
	print("=" * 76)

	print("\nЗагрузка модели Qwen/Qwen2.5-0.5B + LoRA адаптера в VRAM...")
	console.generator._ensure_loaded()
	vram = get_gpu_vram_stats_mb()
	print(
		f"Готово! VRAM: {vram['allocated_mb']:.1f} МБ | "
		f"RAM: {get_process_ram_mb():.1f} МБ\n"
	)

	while True:
		try:
			user_input = input("\nFastAPI-DB> ").strip()
		except (EOFError, KeyboardInterrupt):
			print("\nВыход.")
			break

		if not user_input:
			continue

		if user_input.lower() in ("/exit", "/quit", "exit", "quit", ":q"):
			print("До свидания!")
			break

		if user_input.lower() == "/help":
			print_help()
			continue

		if user_input.lower() == "/stats":
			stats = console.get_db_stats()
			print(json.dumps(stats, indent=2, ensure_ascii=False))
			continue

		if user_input.lower().startswith("/lang "):
			new_lang = user_input.split(maxsplit=1)[1].strip().lower()
			console.language = new_lang
			print(f"Язык поиска установлен: {console.language}")
			continue

		if user_input.lower().startswith("/mode "):
			new_mode = user_input.split(maxsplit=1)[1].strip().lower()
			if new_mode not in ("fast", "direct"):
				print("Допустимые режимы: fast, direct")
			else:
				console.mode = new_mode
				print(f"Режим установлен: {console.mode}")
			continue

		if user_input.lower().startswith("/top "):
			try:
				console.top_k = max(1, int(user_input.split(maxsplit=1)[1].strip()))
				print(f"Top-K установлен: {console.top_k}")
			except ValueError:
				print("Укажите целое число, например: /top 3")
			continue

		if user_input.lower().startswith("/search "):
			q = user_input.split(maxsplit=1)[1].strip()
			hits, lang = console.retrieve_from_db(q)
			print(f"\nНайдено чанков ({lang}): {len(hits)}")
			for i, h in enumerate(hits, start=1):
				print(f"\n--- Чанк [{i}] {h.get('breadcrumb')} ({h.get('symbol_type')}) ---")
				print(f"URL: {h.get('section_url')}")
				print(h.get("text", "")[:600])
			continue

		if user_input.lower().startswith("/sql "):
			sql_q = user_input.split(maxsplit=1)[1].strip()
			try:
				rows = console.execute_readonly_sql(sql_q)
				for r in rows:
					print(r)
			except Exception as exc:
				print(f"Ошибка SQL: {exc}")
			continue

		console.ask(user_input, stream=True)


def main() -> None:
	if hasattr(sys.stdout, "reconfigure"):
		sys.stdout.reconfigure(encoding="utf-8", errors="replace")
	if hasattr(sys.stderr, "reconfigure"):
		sys.stderr.reconfigure(encoding="utf-8", errors="replace")

	parser = argparse.ArgumentParser(
		description="Консольный ассистент по БД документации FastAPI (data/rag.db) на базе дообученной Qwen/Qwen2.5-0.5B."
	)
	parser.add_argument("query", nargs="*", help="Вопрос к базе данных (если не указан — запускается интерактивный режим)")
	parser.add_argument("-q", "--question", type=str, default="", help="Одиночный вопрос к БД")
	parser.add_argument("--db", type=str, default="", help="Путь к SQLite БД (по умолчанию: data/rag.db)")
	parser.add_argument(
		"--adapter",
		type=str,
		default="fastapi_qlora/output/qwen2.5-0.5b-fastapi-lora",
		help="Путь к дообученному LoRA адаптеру",
	)
	parser.add_argument(
		"--mode",
		type=str,
		default="fast",
		choices=["fast", "direct"],
		help="Режим поиска: fast (BM25+LCA), direct (без контекста)",
	)
	parser.add_argument("--lang", type=str, default="auto", help="Язык документации (auto, ru, en, de, ...)")
	parser.add_argument("--top-k", type=int, default=3, help="Количество чанков из БД")
	parser.add_argument("--max-tokens", type=int, default=384, help="Макс. новых токенов ответа")
	parser.add_argument("--stats", action="store_true", help="Показать статистику БД и выйти")
	args = parser.parse_args()

	console = FastSQLiteRAGConsole(
		db_path=Path(args.db) if args.db else None,
		adapter_dir=Path(args.adapter),
		mode=args.mode,
		language=args.lang,
		top_k=args.top_k,
		max_new_tokens=args.max_tokens,
	)

	if args.stats:
		print(json.dumps(console.get_db_stats(), indent=2, ensure_ascii=False))
		return

	one_shot_query = args.question.strip() or " ".join(args.query).strip()
	if one_shot_query:
		console.ask(one_shot_query, stream=True)
	else:
		run_repl(console)


if __name__ == "__main__":
	main()
