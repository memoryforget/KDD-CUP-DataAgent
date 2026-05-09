from __future__ import annotations

import math
import os
import re
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover - optional vector dependency
    np = None

MARKDOWN_SUFFIXES = {".md", ".markdown", ".mdown", ".txt"}
DEFAULT_CHUNK_LINES = 90
DEFAULT_CHUNK_OVERLAP = 15
MAX_SNIPPET_CHARS = 1800
MAX_LINE_RANGE = 260
DEFAULT_MAX_CHUNKS = 8000
DEFAULT_MAX_INDEX_CHARS = 12_000_000
DEFAULT_EMBEDDING_TEXT_MAX_CHARS = 1200
BM25_K1 = 1.5
BM25_B = 0.75
DEFAULT_VECTOR_WEIGHT = 0.30
TOKEN_RE = re.compile(r"[a-zA-Z0-9][\w.-]*")
WHITESPACE_RE = re.compile(r"\s+")
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
SECTION_BOUNDARY_RE = re.compile(r"\b(record|section|client|account|transaction|patient|table|item)\b", re.I)
EMBEDDING_MODEL_CACHE: dict[str, Any] = {}
STOPWORDS = {
    "the", "and", "for", "that", "with", "from", "this", "are", "was", "were", "have", "has",
    "all", "any", "not", "but", "you", "task", "list", "what", "which", "whose", "among", "them",
}


def clamp_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def clamp_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, min(maximum, parsed))


def normalize_query(query: str) -> str:
    return WHITESPACE_RE.sub(" ", query.strip().lower())


def tokenize_search_text(text: str) -> list[str]:
    tokens = TOKEN_RE.findall(text.lower())
    return [token for token in tokens if len(token) >= 2 and token not in STOPWORDS]


def compact_snippet(text: str, terms: list[str], max_chars: int) -> str:
    collapsed = WHITESPACE_RE.sub(" ", text).strip()
    if len(collapsed) <= max_chars:
        return collapsed

    lowered = collapsed.lower()
    positions = [lowered.find(term) for term in terms if term and lowered.find(term) >= 0]
    center = min(positions) if positions else 0
    start = max(0, center - max_chars // 3)
    end = min(len(collapsed), start + max_chars)
    start = max(0, end - max_chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(collapsed) else ""
    return f"{prefix}{collapsed[start:end]}{suffix}"


def current_heading_stack(lines: list[str], line_index: int) -> list[str]:
    stack: list[tuple[int, str]] = []
    for line in lines[: line_index + 1]:
        match = HEADING_RE.match(line.strip())
        if not match:
            continue
        level = len(match.group(1))
        title = match.group(2).strip()
        stack = [(old_level, old_title) for old_level, old_title in stack if old_level < level]
        stack.append((level, title))
    return [title for _, title in stack[-6:]]


def split_markdown_chunks(lines: list[str], chunk_lines: int = DEFAULT_CHUNK_LINES, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[tuple[int, int]]:
    if not lines:
        return []

    boundaries = {0, len(lines)}
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("#") or SECTION_BOUNDARY_RE.search(stripped):
            boundaries.add(index)
        if not stripped and index + 1 < len(lines):
            boundaries.add(index + 1)

    chunks: list[tuple[int, int]] = []
    start = 0
    while start < len(lines):
        target_end = min(len(lines), start + chunk_lines)
        candidates = [point for point in boundaries if start + 20 <= point <= target_end]
        end = max(candidates) if candidates else target_end
        if end <= start:
            end = target_end
        chunks.append((start, end))
        if end >= len(lines):
            break
        start = max(end - overlap, start + 1)
    return chunks


def minmax_normalize(scores: list[float]) -> list[float]:
    if not scores:
        return []
    minimum = min(scores)
    maximum = max(scores)
    if math.isclose(maximum, minimum):
        return [0.0] * len(scores)
    return [(score - minimum) / (maximum - minimum) for score in scores]


class OptionalEmbeddingBackend:
    def __init__(self, model_path: str | None, enabled: bool):
        self.model_path = model_path
        self.enabled = enabled and bool(model_path)
        self.model: Any | None = None
        self.error: str | None = None
        if np is None:
            self.error = "numpy is not available; vector retrieval disabled"
            self.enabled = False
            return
        if not self.enabled:
            return
        cache_key = str(Path(model_path).expanduser().resolve())
        cached_model = EMBEDDING_MODEL_CACHE.get(cache_key)
        if cached_model is not None:
            self.model = cached_model
            self.model_path = cache_key
            return
        try:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(cache_key, device="cpu")
            EMBEDDING_MODEL_CACHE[cache_key] = self.model
            self.model_path = cache_key
        except Exception as exc:  # pragma: no cover - depends on optional model files
            self.model = None
            self.error = str(exc)
            self.enabled = False

    @property
    def available(self) -> bool:
        return self.enabled and self.model is not None

    def encode(self, texts: list[str]) -> Any | None:
        if not self.available or np is None:
            return None
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return np.asarray(embeddings, dtype=np.float32)
        except Exception as exc:  # pragma: no cover - depends on optional model/tokenizer
            self.error = str(exc)
            self.enabled = False
            return None


class MarkdownRagIndex:
    def __init__(self, task_dir: Path):
        self.task_dir = task_dir
        self.context_dir = task_dir / "context"
        self.docs: dict[str, dict[str, Any]] = {}
        self.chunks: list[dict[str, Any]] = []
        self.doc_freq: dict[str, int] = {}
        self.avg_doc_len = 1.0
        self.max_chunks = clamp_int(os.environ.get("EVAL_RAG_MAX_CHUNKS"), DEFAULT_MAX_CHUNKS, 100, 100_000)
        self.max_index_chars = clamp_int(os.environ.get("EVAL_RAG_MAX_INDEX_CHARS"), DEFAULT_MAX_INDEX_CHARS, 100_000, 200_000_000)
        self.embedding_text_max_chars = clamp_int(
            os.environ.get("EVAL_RAG_EMBEDDING_TEXT_MAX_CHARS"),
            DEFAULT_EMBEDDING_TEXT_MAX_CHARS,
            256,
            8000,
        )
        self.indexed_chars = 0
        self.index_truncated = False
        self.vector_weight = clamp_float(os.environ.get("EVAL_RAG_VECTOR_WEIGHT"), DEFAULT_VECTOR_WEIGHT, 0.0, 0.8)
        self.embedding_backend = OptionalEmbeddingBackend(
            model_path=os.environ.get("EVAL_RAG_EMBEDDING_MODEL"),
            enabled=os.environ.get("EVAL_RAG_ENABLE_VECTOR", "1") != "0",
        )
        self.chunk_embeddings: Any | None = None
        self._build()
        self._build_embeddings()

    def _build(self) -> None:
        if not self.context_dir.exists():
            return

        stop_indexing = False
        for path in sorted(self.context_dir.rglob("*")):
            if stop_indexing:
                break
            if not path.is_file() or path.suffix.lower() not in MARKDOWN_SUFFIXES:
                continue
            try:
                raw_text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            lines = raw_text.splitlines()
            if not lines:
                continue

            rel_path = path.relative_to(self.task_dir).as_posix()
            headings = [line.strip() for line in lines if line.lstrip().startswith("#")][:40]
            doc_chunks: list[dict[str, Any]] = []
            for chunk_id, (start_index, end_index) in enumerate(split_markdown_chunks(lines)):
                if len(self.chunks) >= self.max_chunks:
                    self.index_truncated = True
                    stop_indexing = True
                    break
                chunk_lines = lines[start_index:end_index]
                chunk_text = "\n".join(chunk_lines).strip()
                if not chunk_text:
                    continue
                if self.indexed_chars + len(chunk_text) > self.max_index_chars:
                    self.index_truncated = True
                    stop_indexing = True
                    break
                headings_stack = current_heading_stack(lines, start_index)
                searchable_text = chunk_text + " " + " ".join(headings_stack) + " " + rel_path
                tokens = tokenize_search_text(searchable_text)
                term_freq: dict[str, int] = {}
                for token in tokens:
                    term_freq[token] = term_freq.get(token, 0) + 1
                token_set = set(term_freq)
                chunk = {
                    "path": rel_path,
                    "chunk_id": chunk_id,
                    "line_start": start_index + 1,
                    "line_end": end_index,
                    "text": chunk_text,
                    "headings": headings_stack,
                    "term_freq": term_freq,
                    "doc_len": max(1, len(tokens)),
                    "char_count": len(chunk_text),
                }
                doc_chunks.append(chunk)
                self.chunks.append(chunk)
                self.indexed_chars += len(chunk_text)
                for token in token_set:
                    self.doc_freq[token] = self.doc_freq.get(token, 0) + 1

            self.docs[rel_path] = {
                "path": rel_path,
                "line_count": len(lines),
                "chunk_count": len(doc_chunks),
                "headings": headings,
            }

        if self.chunks:
            self.avg_doc_len = sum(chunk["doc_len"] for chunk in self.chunks) / len(self.chunks)

    def _build_embeddings(self) -> None:
        if not self.embedding_backend.available or not self.chunks:
            return
        texts = [self._embedding_text(chunk) for chunk in self.chunks]
        self.chunk_embeddings = self.embedding_backend.encode(texts)

    def _embedding_text(self, chunk: dict[str, Any]) -> str:
        heading_text = " > ".join(chunk["headings"])
        text = f"{chunk['path']}\n{heading_text}\n{chunk['text']}"
        if len(text) <= self.embedding_text_max_chars:
            return text
        return text[: self.embedding_text_max_chars]

    def list_docs(self, max_docs: int = 200) -> dict[str, Any]:
        docs = list(self.docs.values())[:max_docs]
        return {
            "doc_count": len(self.docs),
            "chunk_count": len(self.chunks),
            "index_truncated": self.index_truncated,
            "retrieval": self.retrieval_status(),
            "docs": docs,
        }

    def retrieval_status(self) -> dict[str, Any]:
        return {
            "bm25_enabled": True,
            "vector_enabled": self.chunk_embeddings is not None,
            "vector_weight": self.vector_weight if self.chunk_embeddings is not None else 0.0,
            "embedding_model": self.embedding_backend.model_path if self.chunk_embeddings is not None else None,
            "embedding_error": self.embedding_backend.error,
            "indexed_chars": self.indexed_chars,
            "max_chunks": self.max_chunks,
            "max_index_chars": self.max_index_chars,
            "index_truncated": self.index_truncated,
        }

    def _bm25_score(self, query_terms: list[str], chunk: dict[str, Any]) -> tuple[float, list[str]]:
        total_chunks = max(1, len(self.chunks))
        term_freq = chunk["term_freq"]
        doc_len = chunk["doc_len"]
        score = 0.0
        matched_terms: list[str] = []
        for term in query_terms:
            tf = term_freq.get(term, 0)
            if tf <= 0:
                continue
            df = self.doc_freq.get(term, 0)
            idf = math.log(1 + (total_chunks - df + 0.5) / (df + 0.5))
            denominator = tf + BM25_K1 * (1 - BM25_B + BM25_B * doc_len / self.avg_doc_len)
            score += idf * (tf * (BM25_K1 + 1)) / denominator
            matched_terms.append(term)
        return score, matched_terms

    def search(self, query_text: str, limit: int = 8, max_chars: int = MAX_SNIPPET_CHARS) -> dict[str, Any]:
        query_text = query_text.strip()
        query_terms = tokenize_search_text(query_text)
        if not query_text or not query_terms:
            return {"query": query_text, "matches": [], "message": "search_markdown.query is required."}

        phrase = normalize_query(query_text)
        bm25_scores: list[float] = []
        matched_terms_by_chunk: list[list[str]] = []
        phrase_boosts: list[float] = []
        for chunk in self.chunks:
            bm25, matched_terms = self._bm25_score(query_terms, chunk)
            lowered = normalize_query(chunk["text"])
            heading_text = normalize_query(" ".join(chunk["headings"]))
            path_text = chunk["path"].lower()
            phrase_boost = 0.0
            if phrase and phrase in lowered:
                phrase_boost += 3.0
            if phrase and phrase in heading_text:
                phrase_boost += 2.0
            for term in query_terms:
                if not term or term in STOPWORDS:
                    continue
                if term in heading_text:
                    phrase_boost += 0.8
                if term in path_text:
                    phrase_boost += 0.5
            bm25_scores.append(bm25)
            matched_terms_by_chunk.append(matched_terms)
            phrase_boosts.append(phrase_boost)

        vector_scores = [0.0] * len(self.chunks)
        if self.chunk_embeddings is not None and self.embedding_backend.available and np is not None:
            query_embedding = self.embedding_backend.encode([query_text])
            if query_embedding is not None:
                similarities = np.matmul(self.chunk_embeddings, query_embedding[0])
                vector_scores = similarities.astype(float).tolist()

        lexical_raw = [bm25 + boost for bm25, boost in zip(bm25_scores, phrase_boosts)]
        lexical_norm = minmax_normalize(lexical_raw)
        vector_norm = minmax_normalize(vector_scores) if self.chunk_embeddings is not None else [0.0] * len(self.chunks)
        vector_weight = self.vector_weight if self.chunk_embeddings is not None else 0.0
        lexical_weight = 1.0 - vector_weight

        scored: list[tuple[float, dict[str, Any], float, float]] = []
        for index, chunk in enumerate(self.chunks):
            score = lexical_weight * lexical_norm[index] + vector_weight * vector_norm[index]
            if score <= 0:
                continue
            chunk_with_matches = {**chunk, "matched_terms": sorted(set(matched_terms_by_chunk[index]))}
            scored.append((score, chunk_with_matches, lexical_raw[index], vector_scores[index]))

        scored.sort(key=lambda item: (-item[0], item[1]["path"], item[1]["line_start"]))
        matches = []
        for score, chunk, lexical_score, vector_score in scored[:limit]:
            matches.append({
                "path": chunk["path"],
                "chunk_id": chunk["chunk_id"],
                "line_start": chunk["line_start"],
                "line_end": chunk["line_end"],
                "score": round(score, 4),
                "bm25_score": round(lexical_score, 4),
                "vector_score": round(vector_score, 4),
                "matched_terms": chunk["matched_terms"][:12],
                "headings": chunk["headings"],
                "snippet": compact_snippet(chunk["text"], query_terms, max_chars),
            })
        return {
            "query": query_text,
            "retrieval": self.retrieval_status(),
            "match_count": len(scored),
            "matches": matches,
        }

    def read_chunk(
        self,
        rel_path: str,
        chunk_id: int | None = None,
        line_start: int | None = None,
        line_end: int | None = None,
        context_lines: int = 0,
    ) -> dict[str, Any]:
        rel_path = rel_path.strip().lstrip("/")
        if rel_path not in self.docs:
            return {"is_error": True, "message": f"Markdown document not found: {rel_path}"}

        if chunk_id is not None:
            for chunk in self.chunks:
                if chunk["path"] == rel_path and chunk["chunk_id"] == chunk_id:
                    line_start = chunk["line_start"]
                    line_end = chunk["line_end"]
                    break
            else:
                return {"is_error": True, "message": f"Chunk {chunk_id} not found for {rel_path}"}

        path = self.task_dir / rel_path
        try:
            lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError as exc:
            return {"is_error": True, "message": str(exc)}

        start = max(1, (line_start or 1) - max(0, context_lines))
        end = min(len(lines), (line_end or min(len(lines), start + DEFAULT_CHUNK_LINES - 1)) + max(0, context_lines))
        if end < start:
            return {"is_error": True, "message": "line_end must be greater than or equal to line_start."}
        if end - start + 1 > MAX_LINE_RANGE:
            end = start + MAX_LINE_RANGE - 1
        return {
            "path": rel_path,
            "line_start": start,
            "line_end": end,
            "headings": current_heading_stack(lines, start - 1),
            "text": "\n".join(lines[start - 1:end]),
        }
