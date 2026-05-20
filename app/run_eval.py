#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.markdown_rag import MAX_SNIPPET_CHARS, MarkdownRagIndex, clamp_int
from app.structured_tools import (
    inspect_data,
    pandas_query,
    read_docx_full,
    read_pdf_pages,
    sqlite_query,
)

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    SystemMessage,
    create_sdk_mcp_server,
    query,
    tool,
)

INPUT_ROOT = Path(os.environ.get("EVAL_INPUT_ROOT", "/input"))
OUTPUT_ROOT = Path(os.environ.get("EVAL_OUTPUT_ROOT", "/output"))
LOG_ROOT = Path(os.environ.get("EVAL_LOG_ROOT", "/logs"))
WORK_ROOT = Path(os.environ.get("EVAL_WORK_ROOT", "/tmp/claude_eval_workspace"))
ROUTER_BASE_URL = os.environ.get("CLAUDE_ROUTER_BASE_URL", os.environ.get("CLAUDE_CC_SWITCH_BASE_URL", "http://127.0.0.1:3456")).rstrip("/")
MODEL_NAME = os.environ["MODEL_NAME"]
MODEL_API_KEY = os.environ.get("MODEL_API_KEY", "EMPTY")
MAX_TASKS = int(os.environ.get("EVAL_MAX_TASKS", "0"))
TASK_IDS_FILTER = [item.strip() for item in os.environ.get("EVAL_TASK_IDS", "").split(",") if item.strip()]
MAX_TURNS = int(os.environ.get("CLAUDE_EVAL_MAX_TURNS", "50"))
MAX_WORKERS = max(1, int(os.environ.get("EVAL_MAX_WORKERS", "4")))
PERMISSION_MODE = os.environ.get("CLAUDE_PERMISSION_MODE", "bypassPermissions")
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")
DEBUG_TO_STDERR = os.environ.get("CLAUDE_DEBUG_TO_STDERR", "0") == "1"
SETTING_SOURCES = [
    s.strip() for s in os.environ.get("CLAUDE_SETTING_SOURCES", "project,local").split(",") if s.strip()
]
LOG_MODE = os.environ.get("EVAL_LOG_MODE", "submission").strip().lower()
VERBOSE_LOGS = os.environ.get("EVAL_VERBOSE_LOGS", "0") == "1"
TASK_LOG_DIR = LOG_ROOT / "tasks"
NULL_TOKENS = {"", "null", "none", "nan", "nat", "<na>"}

# Per-task wall-clock budget (seconds). 0 means no timeout.
TASK_TIMEOUT_SEC = int(os.environ.get("EVAL_TASK_TIMEOUT_SEC", "600"))
# How many extra retry attempts after the first run if the task fails.
TASK_MAX_RETRIES = int(os.environ.get("EVAL_TASK_MAX_RETRIES", "1"))
# When True, write a placeholder prediction.csv if everything fails so the task is not 0-file.
TASK_FALLBACK_CSV = os.environ.get("EVAL_TASK_FALLBACK_CSV", "1") == "1"

SYSTEM_PROMPT_APPEND = "\n".join([
    "You are a data-analysis agent solving benchmark tasks one at a time.",
    "",
    "## General rules",
    "- Only access files belonging to the current task. Never read other tasks or any gold.csv.",
    "- Never write to the input directory.",
    "- Do not spend turns diagnosing paths, environment variables, symlinks, or network connectivity.",
    "- If context/knowledge.md exists for the current task, read it first — it defines domain-specific terms, code mappings, and categorical meanings that override general knowledge.",
    "",
    "## Data analysis strategy",
    "- Start by reading task.json to understand the question.",
    "- Call `inspect_data` BEFORE writing any csv/json parser. It returns column names, dtypes, sample values, and full enumerations of low-cardinality categorical columns. This is the cheapest way to understand the data and decide what categorical codes mean.",
    "- For SQLite databases: prefer `sqlite_query` (read-only SELECT) over writing your own sqlite3 Python scripts.",
    "- For CSVs: prefer `pandas_query` over writing your own csv parser. The `df` variable is the loaded DataFrame; you can pass extra_csvs to load joined tables in one call.",
    "- For PDFs use `read_pdf_pages`; for DOCX use `read_docx_full`.",
    "- For unstructured data, especially Markdown/text files, use the MCP tools `list_markdown_docs`, `search_markdown`, and `read_markdown_chunk` before any broad Read/Grep/Bash parsing.",
    "- If a Markdown/text file is longer than about 200 lines, do not read or parse the whole file first. Search it with `search_markdown`, then read only relevant chunks with `read_markdown_chunk`.",
    "- For questions over long narrative Markdown datasets, run multiple focused `search_markdown` queries for the requested fields/entities before writing Python parsers.",
    "- Do not guess the meaning of categorical codes or integer labels from their numeric order. Rely on documentation (knowledge.md) or `inspect_data`'s enumerated values.",
    "- Falling back to Bash + Python is allowed but should be a last resort, not the first tool you reach for.",
    "",
    "## How the grader scores you (read carefully)",
    "- The grader IGNORES column names and IGNORES row order.",
    "- For each column it sorts the values and matches against gold columns by value-set equality.",
    "- Per-task score = recall × redundancy where:",
    "  - recall = matched_columns / gold_columns",
    "  - redundancy = matched_columns / max(matched_columns, your_columns)",
    "- Adding ANY extra column you were not asked for directly REDUCES your score.",
    "- The naming of columns and the row order are FREE — focus on (a) which columns to include and (b) per-cell value accuracy.",
    "",
    "## Answer submission rules",
    "- Submit the final result exclusively through the MCP tool named `answer`. Do not write prediction.csv directly with Write, Edit, or Bash.",
    "- Before calling `answer`, ALWAYS call `preview_answer` first to see the column-level signature summary, and use it to decide whether to drop columns.",
    "- Call `answer` exactly once with a JSON object containing `columns` and `rows`.",
    "  - Prefer native arrays, not quoted JSON strings; quoted JSON strings are accepted only as a fallback.",
    "  - Every row must have exactly the same number of cells as columns.",
    "",
    "## Choosing the columns",
    "- Default to ONE column — the entity or metric the question literally asks for. Do not add 'helpful' context columns.",
    "- The phrasing of the question is the strongest signal. Map it as follows:",
    "  - 'list/which/what {X}' / 'name of …' / 'who …' → return ONLY the identifier or name column for X (one column).",
    "  - 'when did …' / 'on what date …' → return ONLY the date column.",
    "  - 'how many …' / 'count of …' → return ONE row, ONE column with the integer count.",
    "  - 'what is the average/sum/min/max …' → return ONE row, ONE column with the aggregate value (do NOT round; keep full precision).",
    "  - 'list {entities} and their {attribute}' (an explicit two-field request) → return TWO columns: the entity and the attribute.",
    "- Never add date/amount/category columns unless the question literally asks for them.",
    "- If unsure between including or excluding a column, EXCLUDE it. Adding a wrong column is strictly worse than omitting it.",
    "",
    "## Per-cell value accuracy",
    "- Do NOT round numeric values. Preserve the full precision returned by SQL aggregates or pandas (e.g., 60.77956989247312, not 60.78).",
    "- Do NOT format numbers with thousands separators or currency symbols unless the question asks.",
    "- Dates: keep the exact format used by the source data unless the question specifies a format.",
    "- Strings: keep the exact spelling and case from the source data.",
    "",
    "## Multi-answer awareness",
    "- 'lowest/highest/maximum/minimum X' may have ties. After computing the extremum, check whether multiple rows share that extremum value and include ALL tied rows.",
    "- 'all …' / 'list all …' means return every matching row, not a single example.",
    "",
    "## Workflow",
    "- Read task.json → list context/ → read knowledge.md if present.",
    "- For SQLite, prefer one well-formed SELECT over many trial queries.",
    "- After computing your candidate result, call `preview_answer` to see column signatures and trim redundant columns.",
    "- Then call `answer` exactly once. After a successful answer, briefly summarize and stop.",
])


def normalize_task_id(task_id: str) -> str:
    task_id = task_id.strip()
    if not task_id:
        return task_id
    if task_id.startswith("task_"):
        return task_id
    if task_id.isdigit():
        return f"task_{task_id}"
    return task_id


def emit(task_log_path: Path, message: str) -> None:
    print(message)
    task_log_path.parent.mkdir(parents=True, exist_ok=True)
    with task_log_path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def make_task_logger(task_log_path: Path):
    def _log(message: str) -> None:
        emit(task_log_path, message)

    return _log


def log_verbose(task_log, message: str) -> None:
    if VERBOSE_LOGS:
        task_log(message)


def log_claude_stderr_factory(task_log_path: Path):
    def _log(line: str) -> None:
        if LOG_MODE == "debug":
            emit(task_log_path, f"[claude stderr] {line}")

    return _log


def write_prediction_csv(output_csv: Path, columns: list[str], rows: list[list[str]]) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def normalize_cell(value: object) -> str:
    """Normalize a cell value into the string form the grader will see.

    Strategy:
      * None and NaN → ""
      * bool → "true" / "false"
      * int → str(int)
      * float → drop `.0` for integer-valued floats (60.0 → "60"); otherwise `repr(value)` to
        keep full precision (e.g. 60.77956989247312, NOT str() rounding).
      * everything else → str(value).strip(), with NULL tokens and \r normalized.

    Agents often pass numeric types via JSON, so handling them explicitly here is more reliable
    than relying on str() alone — `60.0` vs `60` is the most common silent mismatch.
    """
    if value is None:
        return ""

    if isinstance(value, bool):
        # Defensive: bool is a subclass of int in Python, so this branch must come first.
        return "true" if value else "false"

    if isinstance(value, float):
        if value != value:  # NaN
            return ""
        if value.is_integer():
            return str(int(value))
        # repr is round-trip exact for floats in Python 3 and avoids str()'s short-form rounding.
        return repr(value)

    if isinstance(value, int):
        return str(value)

    text = str(value).strip().replace("\r\n", "\n").replace("\r", "\n")
    if text.lower() in NULL_TOKENS:
        return ""
    return text


def validate_answer_submission(columns: list[str], rows: list[list[Any]]) -> tuple[list[str], list[list[str]]]:
    normalized_columns: list[str] = []
    seen_columns: set[str] = set()
    for column in columns:
        normalized_column = column.strip()
        if not normalized_column:
            raise ValueError("answer.columns cannot contain empty strings.")
        lowered = normalized_column.lower()
        if lowered in seen_columns:
            raise ValueError(f"Duplicate answer column name: {normalized_column}")
        seen_columns.add(lowered)
        normalized_columns.append(normalized_column)

    normalized_rows: list[list[str]] = []
    for index, row in enumerate(rows, start=1):
        if len(row) != len(normalized_columns):
            raise ValueError(f"Row {index} does not match the number of columns.")
        normalized_rows.append([normalize_cell(value) for value in row])

    return normalized_columns, normalized_rows


# When enabled, auto-drop any column that is (a) entirely empty, or (b) all-same-value
# while having more than 1 row. Both shapes never match a gold column under the
# value-set metric, so removing them strictly improves the redundancy ratio.
ANSWER_AUTO_TRIM = os.environ.get("EVAL_ANSWER_AUTO_TRIM", "1") == "1"


def auto_trim_answer(
    columns: list[str], rows: list[list[str]]
) -> tuple[list[str], list[list[str]], list[dict[str, Any]]]:
    """Drop garbage columns. Returns (kept_columns, kept_rows, drop_log).

    A column is dropped iff:
      * every cell in it is empty, OR
      * every cell shares one identical non-empty value AND len(rows) > 1
    Never drops the LAST column — if everything would be dropped we keep it.
    """
    if not ANSWER_AUTO_TRIM or not columns or not rows:
        return columns, rows, []

    n_cols = len(columns)
    keep_idx: list[int] = []
    drops: list[dict[str, Any]] = []
    for idx in range(n_cols):
        cells = [r[idx] for r in rows]
        non_empty = [v for v in cells if v != ""]
        if not non_empty:
            drops.append({"index": idx, "name": columns[idx], "reason": "all_empty"})
            continue
        if len(rows) > 1 and len(set(non_empty)) == 1 and len(non_empty) == len(rows):
            drops.append({"index": idx, "name": columns[idx], "reason": "all_same_value", "value": non_empty[0]})
            continue
        keep_idx.append(idx)

    if not keep_idx:
        # everything would be dropped — keep original to avoid producing an empty csv
        return columns, rows, []

    new_cols = [columns[i] for i in keep_idx]
    new_rows = [[r[i] for i in keep_idx] for r in rows]
    return new_cols, new_rows, drops


def build_answer_mcp_server(task_id: str, output_csv: Path, task_log_path: Path, task_dir: Path):
    task_log = make_task_logger(task_log_path)
    markdown_index = MarkdownRagIndex(task_dir)

    @tool(
        name="list_markdown_docs",
        description=(
            "List task-local Markdown/text documents indexed for focused retrieval. "
            "Use this before reading long Markdown files directly."
        ),
        input_schema={
            "max_docs": Any,
        },
    )
    async def list_markdown_docs(args: dict[str, Any]) -> dict[str, Any]:
        max_docs = clamp_int(args.get("max_docs"), default=200, minimum=1, maximum=500)
        result = markdown_index.list_docs(max_docs=max_docs)
        rendered = json.dumps(result, ensure_ascii=False)
        log_verbose(task_log, f"[{task_id}][list_markdown_docs] docs={result['doc_count']}")
        if LOG_MODE == "debug":
            task_log(f"[{task_id}][list_markdown_docs][debug_result] {rendered[:12000]}")
        return {"content": [{"type": "text", "text": rendered}]}

    @tool(
        name="search_markdown",
        description=(
            "Search task-local Markdown/text documents and return ranked snippets with file paths, chunk ids, and line ranges. "
            "Use this for long documents instead of broad Read/Grep. Search for entity ids, field names, abbreviations, and domain terms."
        ),
        input_schema={
            "query": str,
            "limit": Any,
            "max_chars": Any,
        },
    )
    async def search_markdown(args: dict[str, Any]) -> dict[str, Any]:
        query_text = str(args.get("query", "")).strip()
        limit = clamp_int(args.get("limit"), default=8, minimum=1, maximum=25)
        max_chars = clamp_int(args.get("max_chars"), default=MAX_SNIPPET_CHARS, minimum=300, maximum=5000)
        result = markdown_index.search(query_text=query_text, limit=limit, max_chars=max_chars)
        rendered = json.dumps(result, ensure_ascii=False)
        log_verbose(task_log, f"[{task_id}][search_markdown] query={query_text[:200]!r} matches={result.get('match_count')}")
        if LOG_MODE == "debug":
            task_log(f"[{task_id}][search_markdown][debug_result] {rendered[:20000]}")
        return {"content": [{"type": "text", "text": rendered}]}

    @tool(
        name="read_markdown_chunk",
        description=(
            "Read one focused chunk or line range from a task-local Markdown/text document. "
            "Pass path plus either chunk_id from search_markdown or line_start/line_end."
        ),
        input_schema={
            "path": str,
            "chunk_id": Any,
            "line_start": Any,
            "line_end": Any,
            "context_lines": Any,
        },
    )
    async def read_markdown_chunk(args: dict[str, Any]) -> dict[str, Any]:
        rel_path = str(args.get("path", "")).strip()
        chunk_id = args.get("chunk_id")
        line_start = args.get("line_start")
        line_end = args.get("line_end")
        context_lines = clamp_int(args.get("context_lines"), default=0, minimum=0, maximum=40)
        parsed_chunk_id = None if chunk_id in (None, "") else clamp_int(chunk_id, default=-1, minimum=-1, maximum=1_000_000)
        parsed_line_start = None if line_start in (None, "") else clamp_int(line_start, default=1, minimum=1, maximum=10_000_000)
        parsed_line_end = None if line_end in (None, "") else clamp_int(line_end, default=1, minimum=1, maximum=10_000_000)
        result = markdown_index.read_chunk(
            rel_path=rel_path,
            chunk_id=parsed_chunk_id,
            line_start=parsed_line_start,
            line_end=parsed_line_end,
            context_lines=context_lines,
        )
        rendered = json.dumps(result, ensure_ascii=False)
        log_verbose(
            task_log,
            f"[{task_id}][read_markdown_chunk] path={rel_path} chunk_id={parsed_chunk_id} "
            f"lines={parsed_line_start}-{parsed_line_end}",
        )
        if LOG_MODE == "debug":
            task_log(f"[{task_id}][read_markdown_chunk][debug_result] {rendered[:20000]}")
        return {"content": [{"type": "text", "text": rendered}], "is_error": bool(result.get("is_error"))}

    @tool(
        name="inspect_data",
        description=(
            "Scan the current task's context/ directory and return a compact schema summary of every "
            "structured file (csv, json, sqlite, pdf, docx). For each tabular file you get column names, "
            "non-empty counts, dtype guesses, sample values, and full enumerated value lists for any "
            "low-cardinality column (≤50 unique values). USE THIS FIRST before writing any csv/json parser "
            "or guessing what categorical codes mean — the enumerated values often answer 'is this what the "
            "question is asking about'."
        ),
        input_schema={},
    )
    async def inspect_data_tool(args: dict[str, Any]) -> dict[str, Any]:
        result = inspect_data(task_dir)
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        log_verbose(task_log, f"[{task_id}][inspect_data] csv={len(result.get('csv',[]))} json={len(result.get('json',[]))} sqlite={len(result.get('sqlite',[]))} pdf={len(result.get('pdf',[]))} docx={len(result.get('docx',[]))}")
        if LOG_MODE == "debug":
            task_log(f"[{task_id}][inspect_data][debug_result] {rendered[:20000]}")
        return {"content": [{"type": "text", "text": rendered}]}

    @tool(
        name="sqlite_query",
        description=(
            "Run a read-only SELECT (or WITH) against a task-local SQLite database file. "
            "Reject any non-SELECT/WITH statement. Pass `db_path` (path under context/) and `sql`. "
            "Up to 200 rows are returned; the response includes `truncated` if more are available. "
            "Use this instead of writing your own sqlite3 Python scripts when a SQLite file is present."
        ),
        input_schema={
            "db_path": str,
            "sql": str,
            "max_rows": Any,
        },
    )
    async def sqlite_query_tool(args: dict[str, Any]) -> dict[str, Any]:
        rel_path = str(args.get("db_path", "")).strip().lstrip("/")
        sql = str(args.get("sql", ""))
        max_rows = clamp_int(args.get("max_rows"), default=200, minimum=1, maximum=2000)
        # Resolve under the task dir, defending against directory traversal
        candidate = (task_dir / rel_path).resolve()
        try:
            candidate.relative_to(task_dir.resolve())
        except ValueError:
            return {"content": [{"type": "text", "text": "db_path must be inside the task directory."}], "is_error": True}
        if not candidate.exists():
            return {"content": [{"type": "text", "text": f"db not found: {rel_path}"}], "is_error": True}
        result = sqlite_query(candidate, sql, max_rows=max_rows)
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        log_verbose(task_log, f"[{task_id}][sqlite_query] db={rel_path} rows={result.get('row_count')} err={result.get('is_error')}")
        if LOG_MODE == "debug":
            task_log(f"[{task_id}][sqlite_query][debug_result] {rendered[:20000]}")
        return {"content": [{"type": "text", "text": rendered}], "is_error": bool(result.get("is_error"))}

    @tool(
        name="pandas_query",
        description=(
            "Evaluate a pandas expression on a CSV file in the task. The expression operates on a "
            "DataFrame named `df` (loaded from `csv_path`). You may also pass `extra_csvs` as a "
            "JSON object mapping variable names to CSV paths to load additional DataFrames. "
            "Up to 200 rows are returned. Allowed names in the expression: pd, len, min, max, sum, "
            "abs, round, sorted, set, list, dict, tuple, range. Imports and dunder access are blocked. "
            "Examples: `df[df['client_id']==3356]['trans_id']`, `df.groupby('category')['cost'].sum().idxmin()`."
        ),
        input_schema={
            "csv_path": str,
            "expression": str,
            "extra_csvs": Any,
            "max_rows": Any,
        },
    )
    async def pandas_query_tool(args: dict[str, Any]) -> dict[str, Any]:
        rel_path = str(args.get("csv_path", "")).strip().lstrip("/")
        expression = str(args.get("expression", ""))
        max_rows = clamp_int(args.get("max_rows"), default=200, minimum=1, maximum=2000)
        candidate = (task_dir / rel_path).resolve()
        try:
            candidate.relative_to(task_dir.resolve())
        except ValueError:
            return {"content": [{"type": "text", "text": "csv_path must be inside the task directory."}], "is_error": True}
        if not candidate.exists():
            return {"content": [{"type": "text", "text": f"csv not found: {rel_path}"}], "is_error": True}

        extra_csvs_arg = args.get("extra_csvs")
        if isinstance(extra_csvs_arg, str):
            try:
                extra_csvs_arg = json.loads(extra_csvs_arg)
            except json.JSONDecodeError as exc:
                return {"content": [{"type": "text", "text": f"extra_csvs is a string but not valid JSON: {exc}"}], "is_error": True}
        extra_csvs: dict[str, Any] = {}
        if isinstance(extra_csvs_arg, dict):
            for name, p in extra_csvs_arg.items():
                if not isinstance(p, str):
                    continue
                ec = (task_dir / p.lstrip("/")).resolve()
                try:
                    ec.relative_to(task_dir.resolve())
                except ValueError:
                    return {"content": [{"type": "text", "text": f"extra_csvs[{name}] is outside the task dir."}], "is_error": True}
                extra_csvs[str(name)] = ec

        result = pandas_query(candidate, expression, max_rows=max_rows, extra_csvs=extra_csvs or None)
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        log_verbose(
            task_log,
            f"[{task_id}][pandas_query] csv={rel_path} kind={result.get('kind')} err={result.get('is_error')}",
        )
        if LOG_MODE == "debug":
            task_log(f"[{task_id}][pandas_query][debug_result] {rendered[:20000]}")
        return {"content": [{"type": "text", "text": rendered}], "is_error": bool(result.get("is_error"))}

    @tool(
        name="read_pdf_pages",
        description=(
            "Read text from a range of pages in a task-local PDF. Useful when the data package contains "
            "narrative PDFs. Defaults to page 1 only."
        ),
        input_schema={
            "pdf_path": str,
            "page_start": Any,
            "page_end": Any,
        },
    )
    async def read_pdf_pages_tool(args: dict[str, Any]) -> dict[str, Any]:
        rel_path = str(args.get("pdf_path", "")).strip().lstrip("/")
        page_start = clamp_int(args.get("page_start"), default=1, minimum=1, maximum=10_000)
        page_end = clamp_int(args.get("page_end"), default=page_start, minimum=1, maximum=10_000)
        candidate = (task_dir / rel_path).resolve()
        try:
            candidate.relative_to(task_dir.resolve())
        except ValueError:
            return {"content": [{"type": "text", "text": "pdf_path must be inside the task directory."}], "is_error": True}
        if not candidate.exists():
            return {"content": [{"type": "text", "text": f"pdf not found: {rel_path}"}], "is_error": True}
        result = read_pdf_pages(candidate, page_start=page_start, page_end=page_end)
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        log_verbose(task_log, f"[{task_id}][read_pdf_pages] {rel_path} p={page_start}-{page_end}")
        return {"content": [{"type": "text", "text": rendered}], "is_error": bool(result.get("is_error"))}

    @tool(
        name="read_docx_full",
        description="Read paragraphs (and table count) from a task-local DOCX file.",
        input_schema={
            "docx_path": str,
            "max_paragraphs": Any,
        },
    )
    async def read_docx_full_tool(args: dict[str, Any]) -> dict[str, Any]:
        rel_path = str(args.get("docx_path", "")).strip().lstrip("/")
        max_paragraphs = clamp_int(args.get("max_paragraphs"), default=200, minimum=1, maximum=2000)
        candidate = (task_dir / rel_path).resolve()
        try:
            candidate.relative_to(task_dir.resolve())
        except ValueError:
            return {"content": [{"type": "text", "text": "docx_path must be inside the task directory."}], "is_error": True}
        if not candidate.exists():
            return {"content": [{"type": "text", "text": f"docx not found: {rel_path}"}], "is_error": True}
        result = read_docx_full(candidate, max_paragraphs=max_paragraphs)
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        log_verbose(task_log, f"[{task_id}][read_docx_full] {rel_path} paragraphs={result.get('paragraph_count')}")
        return {"content": [{"type": "text", "text": rendered}], "is_error": bool(result.get("is_error"))}

    @tool(
        name="preview_answer",
        description=(
            "Preview the column-level signature summary of a candidate answer table BEFORE submitting it. "
            "Returns per-column unique-value count, sample values, dtype guess, and a redundancy hint based on the question. "
            "Use this every time before calling `answer` to decide whether to drop columns. "
            "Input shape is the same as `answer`: an object with columns and rows."
        ),
        input_schema={
            "columns": Any,
            "rows": Any,
        },
    )
    async def preview_answer(args: dict[str, Any]) -> dict[str, Any]:
        columns = args.get("columns")
        rows = args.get("rows")

        if isinstance(columns, str):
            try:
                columns = json.loads(columns)
            except json.JSONDecodeError as exc:
                return {
                    "content": [{"type": "text", "text": f"preview_answer.columns is a string but not valid JSON: {exc}"}],
                    "is_error": True,
                }
        if isinstance(rows, str):
            try:
                rows = json.loads(rows)
            except json.JSONDecodeError as exc:
                return {
                    "content": [{"type": "text", "text": f"preview_answer.rows is a string but not valid JSON: {exc}"}],
                    "is_error": True,
                }
        if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
            return {
                "content": [{"type": "text", "text": "preview_answer.columns must be a non-empty list of strings."}],
                "is_error": True,
            }
        if not isinstance(rows, list):
            return {
                "content": [{"type": "text", "text": "preview_answer.rows must be a list of rows."}],
                "is_error": True,
            }

        column_summaries: list[dict[str, Any]] = []
        for col_idx, col_name in enumerate(columns):
            cells = []
            for row in rows:
                if isinstance(row, list) and col_idx < len(row):
                    cells.append(row[col_idx])
                else:
                    cells.append(None)
            normalized = [normalize_cell(c) for c in cells]
            non_empty = [v for v in normalized if v != ""]
            unique_values = sorted(set(non_empty))
            dtype_guess = "string"
            try:
                if non_empty and all(v.lstrip("-").isdigit() for v in non_empty):
                    dtype_guess = "int"
                elif non_empty:
                    [float(v) for v in non_empty]
                    dtype_guess = "float"
            except ValueError:
                dtype_guess = "string"
            column_summaries.append({
                "column_name": col_name,
                "dtype_guess": dtype_guess,
                "row_count": len(normalized),
                "non_empty_count": len(non_empty),
                "unique_count": len(unique_values),
                "sample_values": unique_values[:8],
                "all_empty": len(non_empty) == 0,
                "all_same_value": len(unique_values) == 1,
            })

        # Heuristic warnings tied to the grader's column-signature metric.
        warnings: list[str] = []
        for s in column_summaries:
            if s["all_empty"]:
                warnings.append(f"column '{s['column_name']}' is entirely empty — likely junk, drop it.")
            if s["all_same_value"] and s["row_count"] > 1:
                warnings.append(
                    f"column '{s['column_name']}' has only one distinct value ({s['sample_values']}); "
                    f"this column rarely matches a gold column unless the question demanded it."
                )
        if len(columns) > 1:
            warnings.append(
                "You have more than 1 column. The grader penalizes extra columns. "
                "Re-read the question: if it asks for a single field, drop everything except that field."
            )

        result = {
            "column_count": len(columns),
            "row_count": len(rows),
            "columns": column_summaries,
            "warnings": warnings,
            "reminder": (
                "Grader ignores column NAMES and row order. It sorts each column's values and matches "
                "by value-set equality. Per-task score = recall × redundancy. Adding an extra column "
                "ALWAYS reduces the score unless it happens to match a gold column."
            ),
        }
        rendered = json.dumps(result, ensure_ascii=False)
        log_verbose(
            task_log,
            f"[{task_id}][preview_answer] cols={len(columns)} rows={len(rows)} warnings={len(warnings)}",
        )
        if LOG_MODE == "debug":
            task_log(f"[{task_id}][preview_answer][debug_result] {rendered[:8000]}")
        return {"content": [{"type": "text", "text": rendered}]}

    @tool(
        name="answer",
        description=(
            "Submit the final benchmark answer as a compact table. "
            "Use only the minimum required columns and rows for the question. "
            "Do not include explanatory or intermediate columns unless the question explicitly asks for them. "
            "Input must be an object with columns and rows. "
            "columns should be a JSON array of strings, for example [\"id\", \"amount\"]. "
            "rows should be a JSON array of row arrays, for example [[\"1\", \"100\"], [\"2\", \"200\"]]. "
            "Every row must have the same number of cells as columns. "
            "Prefer passing native arrays, not quoted JSON strings; quoted JSON strings are accepted only as a fallback."
        ),
        input_schema={
            "columns": Any,
            "rows": Any,
        },
    )
    async def submit_answer(args: dict[str, Any]) -> dict[str, Any]:
        columns = args.get("columns")
        rows = args.get("rows")
        log_verbose(task_log, f"[{task_id}][answer tool] received columns={columns!r}")

        if isinstance(columns, str):
            try:
                columns = json.loads(columns)
            except json.JSONDecodeError as exc:
                return {
                    "content": [{"type": "text", "text": f"answer.columns is a string but not valid JSON: {exc}"}],
                    "is_error": True,
                }

        if isinstance(rows, str):
            try:
                rows = json.loads(rows)
            except json.JSONDecodeError as exc:
                return {
                    "content": [{"type": "text", "text": f"answer.rows is a string but not valid JSON: {exc}"}],
                    "is_error": True,
                }

        if not isinstance(columns, list) or not columns or not all(isinstance(item, str) for item in columns):
            return {
                "content": [{"type": "text", "text": "answer.columns must be a non-empty list of strings."}],
                "is_error": True,
            }

        if not isinstance(rows, list):
            return {
                "content": [{"type": "text", "text": "answer.rows must be a list of rows."}],
                "is_error": True,
            }

        normalized_rows: list[list[Any]] = []
        for row in rows:
            if not isinstance(row, list):
                return {
                    "content": [{"type": "text", "text": "Each answer row must be a list."}],
                    "is_error": True,
                }
            normalized_rows.append(list(row))

        try:
            validated_columns, validated_rows = validate_answer_submission(columns=list(columns), rows=normalized_rows)
        except ValueError as exc:
            log_verbose(task_log, f"[{task_id}][answer tool] rejected submission: {exc}")
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "is_error": True,
            }

        # Auto-trim garbage columns (all-empty or all-same-value when n_rows > 1)
        trimmed_columns, trimmed_rows, drop_log = auto_trim_answer(validated_columns, validated_rows)
        if drop_log:
            task_log(
                f"[{task_id}][answer tool] auto-trimmed columns: "
                f"{json.dumps(drop_log, ensure_ascii=False)}"
            )

        write_prediction_csv(output_csv=output_csv, columns=trimmed_columns, rows=trimmed_rows)
        task_log(
            f"[{task_id}][answer tool] submitted rows={len(trimmed_rows)} cols={len(trimmed_columns)}"
            f" original_cols={len(validated_columns)} path={output_csv}"
        )
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Submitted final answer for {task_id}. "
                        f"columns={len(trimmed_columns)} rows={len(trimmed_rows)}"
                        + (f" (auto-trimmed {len(drop_log)} columns)" if drop_log else "")
                        + f" path={output_csv}"
                    ),
                }
            ]
        }

    return create_sdk_mcp_server(
        name=f"{task_id}_answer_server",
        tools=[
            list_markdown_docs,
            search_markdown,
            read_markdown_chunk,
            inspect_data_tool,
            sqlite_query_tool,
            pandas_query_tool,
            read_pdf_pages_tool,
            read_docx_full_tool,
            preview_answer,
            submit_answer,
        ],
    )


def _summarize_schema_for_prompt(schema: dict[str, Any], char_budget: int = 16000) -> str:
    """Render `inspect_data` output as a compact prompt-ready chunk.

    We aggressively trim per-column samples and skip files that have errors.
    The character budget bounds prompt growth on huge datasets. 16k chars (~4.5k tokens)
    is comfortable on a 256k-context model and keeps complete schema for typical tasks.
    """
    lines: list[str] = []
    used = 0

    def push(line: str) -> bool:
        nonlocal used
        line = line + "\n"
        if used + len(line) > char_budget:
            return False
        lines.append(line)
        used += len(line)
        return True

    for kind in ("csv", "json", "sqlite", "pdf", "docx"):
        items = schema.get(kind) or []
        if not items:
            continue
        if not push(f"### {kind.upper()} files ({len(items)})"):
            return "".join(lines)
        for f in items:
            if f.get("is_error"):
                push(f"- {f.get('path')}: ERROR {f.get('message','')[:120]}")
                continue
            if kind in ("csv", "json"):
                cols = f.get("columns") or []
                push(f"- {f.get('path')}  rows={f.get('row_count','?')}  cols={len(cols)}")
                for c in cols:
                    sample = ",".join(str(s) for s in (c.get("samples") or [])[:5])
                    enum = c.get("values")
                    suffix = (
                        f"  values=[{','.join(str(v) for v in enum)}]"
                        if enum and len(enum) <= 12
                        else ""
                    )
                    pushed = push(
                        f"    - {c['name']} ({c.get('dtype_guess','')})"
                        f"  unique={c.get('unique_count_capped_1000','?')}"
                        f"  ne={c.get('non_empty_count','?')}"
                        f"  ex=[{sample}]{suffix}"
                    )
                    if not pushed:
                        return "".join(lines)
            elif kind == "sqlite":
                push(f"- {f.get('path')}")
                for t in f.get("tables", []):
                    push(f"    table {t['name']} ({t.get('row_count','?')} rows): "
                         + ", ".join(f"{c['name']}:{c.get('dtype','')}" for c in t.get('columns', [])[:30]))
            elif kind == "pdf":
                push(f"- {f.get('path')}  pages={f.get('page_count','?')}")
                excerpt = f.get("page1_excerpt")
                if excerpt:
                    push(f"    page1: {excerpt[:200]}")
            elif kind == "docx":
                push(f"- {f.get('path')}  paragraphs={f.get('paragraph_count','?')}")
                for p in (f.get("first_paragraphs") or [])[:3]:
                    push(f"    {p[:200]}")
    return "".join(lines).rstrip()


def _read_knowledge_md(task_dir: Path, char_budget: int = 12000) -> str | None:
    kp = task_dir / "context" / "knowledge.md"
    if not kp.exists():
        return None
    try:
        text = kp.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    if len(text) > char_budget:
        text = text[:char_budget] + "\n…[truncated]"
    return text


def build_task_prompt(task_id: str, question: str) -> str:
    input_task_dir = INPUT_ROOT / task_id
    work_task_dir = WORK_ROOT / task_id
    return "\n".join([
        f"## Task {task_id}",
        "",
        f"**Question:** {question}",
        "",
        f"**Task directory:** {input_task_dir}",
        f"  - task.json: {input_task_dir / 'task.json'}",
        f"  - context/: {input_task_dir / 'context'}",
        f"**Working directory:** {work_task_dir}",
        "  - Use this directory for temporary scripts, scratch files, and intermediate outputs.",
        "  - Do not write to the task input directory.",
        "",
        "**Steps:**",
        f"1. The schema summary of context/ is provided below — use it to pick the right files. Skip blind file listing.",
        "2. Read knowledge.md (provided below if present); it overrides general knowledge.",
        "3. Choose the right query tool: `sqlite_query` for .sqlite/.db, `pandas_query` for CSV joins/aggregations, `search_markdown`+`read_markdown_chunk` for long markdown.",
        "4. Compute the candidate answer. Preserve full numeric precision (no rounding).",
        "5. Re-read the question and decide the MINIMUM column set. Default to one column unless the question literally asks for multiple fields.",
        "6. Call `preview_answer` with your candidate columns/rows. Drop any column not directly demanded by the question.",
        "7. Call `answer` exactly once with the trimmed result.",
    ])



async def run_task(task_id: str, attempt: int = 1, prior_failure_note: str | None = None) -> None:
    task_dir = INPUT_ROOT / task_id
    task_json = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    question = task_json["question"]
    output_csv = OUTPUT_ROOT / task_id / "prediction.csv"
    task_work_dir = WORK_ROOT / task_id
    task_work_dir.mkdir(parents=True, exist_ok=True)
    task_log_path = TASK_LOG_DIR / f"{task_id}.log"
    task_log = make_task_logger(task_log_path)

    task_log(f"\n[task] {task_id} attempt={attempt}")
    log_verbose(task_log, f"[task] question={question}")
    task_log(f"[{task_id}][turn] started")

    extra_args: dict[str, str | None] = {}
    if DEBUG_TO_STDERR:
        extra_args["debug-to-stderr"] = None

    answer_server = build_answer_mcp_server(task_id=task_id, output_csv=output_csv, task_log_path=task_log_path, task_dir=task_dir)
    # Trim the Claude Code preset tool set down to what we actually need for offline data
    # analysis. The full preset injects ~33 tools (~22.6K tokens) per request including ones
    # that make no sense at evaluation time (AskUserQuestion, WebFetch, WebSearch, Cron*,
    # Plan/Worktree mode, Skill, TodoWrite, Task subagent). Keeping only the file/text/exec
    # primitives reduces request size by ~30% and narrows the model's choice space.
    BUILTIN_TOOLS = ["Bash", "Read", "Write", "Edit", "Glob", "Grep"]
    options = ClaudeAgentOptions(
        tools=BUILTIN_TOOLS,
        system_prompt={"type": "preset", "preset": "claude_code", "append": SYSTEM_PROMPT_APPEND},
        mcp_servers={"answer_server": answer_server},
        permission_mode=PERMISSION_MODE,
        model=MODEL_NAME,
        max_turns=MAX_TURNS,
        cwd=task_work_dir,
        add_dirs=[str(INPUT_ROOT), str(OUTPUT_ROOT), str(LOG_ROOT)],
        cli_path=CLAUDE_CLI_PATH,
        setting_sources=SETTING_SOURCES,
        extra_args=extra_args,
        stderr=log_claude_stderr_factory(task_log_path),
        env={
            **os.environ,
            "ANTHROPIC_API_KEY": MODEL_API_KEY,
            "ANTHROPIC_BASE_URL": ROUTER_BASE_URL,
            "HTTP_PROXY": "",
            "HTTPS_PROXY": "",
            "ALL_PROXY": "",
            "http_proxy": "",
            "https_proxy": "",
            "all_proxy": "",
            "NO_PROXY": "127.0.0.1,localhost",
        },
    )

    final_texts: list[str] = []

    async def run_agent_turn(turn_prompt: str, turn_options: ClaudeAgentOptions, label: str) -> tuple[bool, str | None, str | None, bool]:
        result_subtype: str | None = None
        session_id: str | None = None
        turn_failed = False
        turn_started = False

        try:
            async for message in query(prompt=turn_prompt, options=turn_options):
                if isinstance(message, AssistantMessage):
                    rendered_blocks = 0
                    text_parts: list[str] = []
                    for block in message.content:
                        block_text = getattr(block, "text", None)
                        if block_text:
                            rendered_blocks += 1
                            text_parts.append(block_text)
                            log_verbose(task_log, f"\n[{task_id}][{label}][assistant][text]\n{block_text}")
                            continue

                        thinking = getattr(block, "thinking", None)
                        if thinking:
                            rendered_blocks += 1
                            log_verbose(task_log, f"\n[{task_id}][{label}][assistant][thinking]\n{thinking}")
                            continue

                        tool_name = getattr(block, "name", None)
                        tool_input = getattr(block, "input", None)
                        if tool_name is not None and tool_input is not None:
                            rendered_blocks += 1
                            log_verbose(
                                task_log,
                                f"\n[{task_id}][{label}][assistant][tool_use] name={tool_name} input="
                                f"{json.dumps(tool_input, ensure_ascii=False, sort_keys=True)[:4000]}"
                            )
                            continue

                        tool_use_id = getattr(block, "tool_use_id", None)
                        if tool_use_id is not None:
                            rendered_blocks += 1
                            tool_result = getattr(block, "content", None)
                            tool_is_error = getattr(block, "is_error", None)
                            if isinstance(tool_result, list):
                                rendered_result = json.dumps(tool_result, ensure_ascii=False, sort_keys=True)[:4000]
                            else:
                                rendered_result = str(tool_result)[:4000]
                            log_verbose(
                                task_log,
                                f"\n[{task_id}][{label}][assistant][tool_result] tool_use_id={tool_use_id} "
                                f"is_error={tool_is_error} content={rendered_result}"
                            )
                            continue

                    if text_parts:
                        final_texts.append("\n".join(text_parts))
                    if rendered_blocks == 0:
                        log_verbose(task_log, f"\n[{task_id}][{label}][assistant][empty_message]")
                elif isinstance(message, SystemMessage):
                    turn_started = True
                    subtype = getattr(message, "subtype", "")
                    data = getattr(message, "data", {})
                    if isinstance(data, dict) and data.get("session_id"):
                        session_id = str(data["session_id"])
                    task_log(f"\n[{task_id}][{label}][system] subtype={subtype}")
                    if VERBOSE_LOGS and data:
                        task_log(json.dumps(data, ensure_ascii=False, default=str)[:4000])
                elif isinstance(message, ResultMessage):
                    result_subtype = message.subtype
                    if getattr(message, "session_id", None):
                        session_id = message.session_id
                    task_log(f"[{task_id}][{label}][result] {message.subtype}")
                    if VERBOSE_LOGS and getattr(message, "usage", None):
                        task_log(f"[{task_id}][{label}][result][usage] {json.dumps(message.usage, ensure_ascii=False, default=str)[:2000]}")
                    if VERBOSE_LOGS and getattr(message, "result", None):
                        task_log(f"[{task_id}][{label}][result][text] {str(message.result)[:4000]}")
                    if VERBOSE_LOGS and getattr(message, "errors", None):
                        task_log(f"[{task_id}][{label}][result][errors] {json.dumps(message.errors, ensure_ascii=False, default=str)[:4000]}")
                    break
        except asyncio.CancelledError as exc:
            turn_failed = True
            task_log(f"[{task_id}][{label}][turn cancelled] {exc}")
            task_log(f"[{task_id}][{label}][turn cancelled repr] {exc!r}")
            cause = getattr(exc, "__cause__", None)
            if cause is not None:
                task_log(f"[{task_id}][{label}][turn cancelled cause] {cause!r}")
            context = getattr(exc, "__context__", None)
            if context is not None:
                task_log(f"[{task_id}][{label}][turn cancelled context] {context!r}")
            task_log(f"[{task_id}][{label}][turn cancelled traceback]\n{traceback.format_exc()}")
        except Exception as exc:
            turn_failed = True
            task_log(f"[{task_id}][{label}][turn failed] {exc}")
            task_log(f"[{task_id}][{label}][turn failed repr] {exc!r}")
            cause = getattr(exc, "__cause__", None)
            if cause is not None:
                task_log(f"[{task_id}][{label}][turn failed cause] {cause!r}")
            context = getattr(exc, "__context__", None)
            if context is not None:
                task_log(f"[{task_id}][{label}][turn failed context] {context!r}")
            for attr in ("stderr", "stdout", "message", "args"):
                value = getattr(exc, attr, None)
                if value:
                    task_log(f"[{task_id}][{label}][turn failed {attr}] {value}")
            task_log(f"[{task_id}][{label}][turn failed traceback]\n{traceback.format_exc()}")

        return turn_failed, result_subtype, session_id, turn_started

    initial_prompt = build_task_prompt(task_id, question)

    # Pre-compute and inject the schema summary so the agent doesn't burn turns on
    # discovery. Same for knowledge.md — it's usually critical and tiny.
    schema_section = ""
    try:
        schema = inspect_data(task_dir)
        rendered = _summarize_schema_for_prompt(schema)
        if rendered:
            schema_section = "\n\n## Schema summary of context/ (auto-generated, no tool call needed)\n" + rendered
    except Exception as exc:  # pragma: no cover - defensive
        task_log(f"[{task_id}][prompt-inject] inspect_data failed: {exc!r}")

    knowledge_section = ""
    knowledge_text = _read_knowledge_md(task_dir)
    if knowledge_text:
        knowledge_section = "\n\n## context/knowledge.md (auto-included)\n" + knowledge_text

    initial_prompt = initial_prompt + schema_section + knowledge_section

    if prior_failure_note:
        initial_prompt = (
            initial_prompt
            + "\n\n## Retry context\n"
            + prior_failure_note
            + "\nThis is retry attempt #" + str(attempt) + ". Be efficient: read task.json, list context, "
            + "compute the answer, call `preview_answer` then `answer`. Do not spend turns debugging shells or paths."
        )
    failed, result_subtype, session_id, _initial_started = await run_agent_turn(
        turn_prompt=initial_prompt,
        turn_options=options,
        label="initial",
    )

    has_output = output_csv.exists()
    if has_output:
        task_log(f"[{task_id}] wrote {output_csv} ({output_csv.stat().st_size} bytes)")
    else:
        task_log(f"[{task_id}] prediction.csv missing after agent run")

    if VERBOSE_LOGS:
        task_log(f"\n[{task_id}][final_response]")
        task_log(final_texts[-1] if final_texts else "")

    if result_subtype:
        task_log(f"[{task_id}][turn] completed subtype={result_subtype}")

    if failed or not has_output:
        raise RuntimeError(f"task {task_id} failed: failed={failed} output_exists={has_output}")


async def amain() -> int:
    if not INPUT_ROOT.exists():
        raise RuntimeError(f"input root not found: {INPUT_ROOT}")

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_ROOT.mkdir(parents=True, exist_ok=True)
    TASK_LOG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[info] inputRoot={INPUT_ROOT}")
    print(f"[info] outputRoot={OUTPUT_ROOT}")
    print(f"[info] logRoot={LOG_ROOT}")
    print(f"[info] workRoot={WORK_ROOT}")
    print(f"[info] routerBaseUrl={ROUTER_BASE_URL}")
    print(f"[info] model={MODEL_NAME}")
    print(f"[info] permissionMode={PERMISSION_MODE}")
    print(f"[info] maxTurns={MAX_TURNS}")
    print(f"[info] maxWorkers={MAX_WORKERS}")
    print(f"[info] logMode={LOG_MODE}")
    print(f"[info] settingSources={SETTING_SOURCES}")
    print(f"[info] taskTimeoutSec={TASK_TIMEOUT_SEC}")
    print(f"[info] taskMaxRetries={TASK_MAX_RETRIES}")
    print(f"[info] taskFallbackCsv={TASK_FALLBACK_CSV}")

    discovered_task_ids = sorted(
        (p.name for p in INPUT_ROOT.iterdir() if p.is_dir() and p.name.startswith("task_")),
        key=lambda name: int(name.split("_", 1)[1]),
    )
    print(f"[info] discoveredTotal={len(discovered_task_ids)} tasks")

    task_ids = discovered_task_ids
    if TASK_IDS_FILTER:
        requested_task_ids = [normalize_task_id(task_id) for task_id in TASK_IDS_FILTER]
        requested_task_id_set = set(requested_task_ids)
        missing_requested = [task_id for task_id in requested_task_ids if task_id not in discovered_task_ids]
        if missing_requested:
            print(f"[warn] requestedTaskIdsMissing={missing_requested}")
        task_ids = [task_id for task_id in discovered_task_ids if task_id in requested_task_id_set]
        print(f"[info] taskIdsFilter={requested_task_ids}")
    if MAX_TASKS > 0:
        task_ids = task_ids[:MAX_TASKS]

    print(f"[info] selectedTasks={len(task_ids)}")

    semaphore = asyncio.Semaphore(MAX_WORKERS)

    def write_fallback_csv(task_id: str, reason: str) -> None:
        """Last-ditch placeholder so the task is not file-missing.
        The grader still scores 0 for content but at least we don't crash a downstream check."""
        output_csv = OUTPUT_ROOT / task_id / "prediction.csv"
        if output_csv.exists():
            return
        if not TASK_FALLBACK_CSV:
            return
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["result"])
            writer.writerow(["unknown"])
        task_log_path = TASK_LOG_DIR / f"{task_id}.log"
        try:
            with task_log_path.open("a", encoding="utf-8") as h:
                h.write(f"[{task_id}][fallback] wrote placeholder prediction.csv reason={reason}\n")
        except OSError:
            pass

    async def run_task_with_retry(task_id: str) -> None:
        prior_note: str | None = None
        last_exc: str | None = None
        for attempt in range(1, max(1, TASK_MAX_RETRIES + 1) + 1):
            try:
                if TASK_TIMEOUT_SEC > 0:
                    await asyncio.wait_for(
                        run_task(task_id, attempt=attempt, prior_failure_note=prior_note),
                        timeout=TASK_TIMEOUT_SEC,
                    )
                else:
                    await run_task(task_id, attempt=attempt, prior_failure_note=prior_note)
                # success
                return
            except asyncio.TimeoutError:
                last_exc = f"timeout after {TASK_TIMEOUT_SEC}s"
                prior_note = (
                    f"Previous attempt timed out after {TASK_TIMEOUT_SEC}s. "
                    "Skip exploration; go straight to a minimal solution and call `answer` quickly."
                )
                print(f"[task timeout] {task_id} attempt={attempt} after {TASK_TIMEOUT_SEC}s")
            except RuntimeError as exc:
                # raised by run_task when prediction.csv missing or turn failed
                last_exc = str(exc)
                prior_note = (
                    f"Previous attempt failed: {exc}. "
                    "The most likely fix is to reduce turns: skip diagnostic Bash commands, "
                    "open the smallest sufficient slice of data, compute the answer in a single Python script, "
                    "then call `preview_answer` and `answer`."
                )
                print(f"[task retryable failure] {task_id} attempt={attempt}: {exc}")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_exc = repr(exc)
                prior_note = f"Previous attempt raised {exc!r}."
                print(f"[task error] {task_id} attempt={attempt}: {exc!r}")
        # all attempts exhausted
        write_fallback_csv(task_id, reason=last_exc or "unknown")
        print(f"[task failure-final] {task_id}: {last_exc}")

    async def run_task_with_limit(task_id: str) -> None:
        async with semaphore:
            try:
                await run_task_with_retry(task_id)
            except asyncio.CancelledError as exc:
                print(f"[task cancelled] {task_id}: {exc!r}")
            except Exception as exc:
                print(f"[task scheduler error] {task_id}: {exc}")
                write_fallback_csv(task_id, reason=f"scheduler-error: {exc!r}")

    await asyncio.gather(*(run_task_with_limit(task_id) for task_id in task_ids), return_exceptions=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
