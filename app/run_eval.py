#!/usr/bin/env python3
from __future__ import annotations

import asyncio
import csv
import json
import os
import re
import traceback
from pathlib import Path
from typing import Any

from app.data_query_tools import ContextQueryWorkspace
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
MAX_TURNS = int(os.environ.get("CLAUDE_EVAL_MAX_TURNS", "40"))
MAX_WORKERS = max(1, int(os.environ.get("EVAL_MAX_WORKERS", "8")))
PERMISSION_MODE = os.environ.get("CLAUDE_PERMISSION_MODE", "bypassPermissions")
CLAUDE_CLI_PATH = os.environ.get("CLAUDE_CLI_PATH", "claude")
DEBUG_TO_STDERR = os.environ.get("CLAUDE_DEBUG_TO_STDERR", "0") == "1"
SETTING_SOURCES = [
    s.strip() for s in os.environ.get("CLAUDE_SETTING_SOURCES", "project,local").split(",") if s.strip()
]
LOG_MODE = os.environ.get("EVAL_LOG_MODE", "submission").strip().lower()
VERBOSE_LOGS = os.environ.get("EVAL_VERBOSE_LOGS", "0") == "1"
SYSTEM_PROMPT_APPEND = "\n".join([
    "You are solving one benchmark task at a time.",
    "Only read files inside the current task directory. Do not read files from other tasks, shared benchmark outputs, or any gold.csv file.",
    "If context/knowledge.md exists, read it before deciding any business meaning such as severe, mild, stage, level, abnormal, positive, or status.",
    "Do not guess the meaning of integer codes from numeric order. Use documented mappings from the current task evidence when available.",
    "Use describe_query_workspace first.",
    "If the workspace already contains the key data needed for the question, use query_data for filtering, joins, aggregation, sorting, and ranking.",
    "If the workspace is missing a key table but the current task has relevant context/doc/*.md files, treat those docs as the primary source instead of continuing to search for more tables.",
    "For repeated template-style documents, use Grep and Read to extract structured facts such as identifiers, measurements, categories, or affiliations.",
    "Once current-task docs reveal the fields needed to solve the question, stop searching for more tables or files and finish from current-task evidence only.",
    "Return only the columns explicitly requested by the question.",
    "Submit the final result with the answer tool only. Do not write prediction.csv with Write, Edit, or Bash.",
    'Pass answer.columns as a JSON array of strings and answer.rows as a JSON array of row arrays.',
    "If a required field is still missing after choosing the correct source data, exclude that row unless the question explicitly allows missing values.",
])
TASK_LOG_DIR = LOG_ROOT / "tasks"
NULL_TOKENS = {"", "null", "none", "nan", "nat", "<na>"}


def normalize_task_id(task_id: str) -> str:
    """Normalize task identifiers so filters accept both numeric ids and task_<id> values."""
    task_id = task_id.strip()
    if not task_id:
        return task_id
    if task_id.startswith("task_"):
        return task_id
    if task_id.isdigit():
        return f"task_{task_id}"
    return task_id


def emit(task_log_path: Path, message: str) -> None:
    """Write a log line to stdout and append the same line to the task log file."""
    print(message)
    task_log_path.parent.mkdir(parents=True, exist_ok=True)
    with task_log_path.open("a", encoding="utf-8") as handle:
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def make_task_logger(task_log_path: Path):
    """Create a small task-scoped logger closure bound to one log file."""
    def _log(message: str) -> None:
        emit(task_log_path, message)

    return _log


def log_verbose(task_log, message: str) -> None:
    """Emit a log message only when verbose evaluation logging is enabled."""
    if VERBOSE_LOGS:
        task_log(message)


def log_claude_stderr_factory(task_log_path: Path):
    """Create a stderr sink for Claude Code subprocess output."""
    def _log(line: str) -> None:
        if LOG_MODE == "debug":
            emit(task_log_path, f"[claude stderr] {line}")

    return _log


def write_prediction_csv(output_csv: Path, columns: list[str], rows: list[list[str]]) -> None:
    """Write the final prediction table to the required benchmark CSV path."""
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def normalize_cell(value: object) -> str:
    """Normalize one output cell to the benchmark CSV text representation."""
    if value is None:
        return ""

    text = str(value).strip().replace("\r\n", "\n").replace("\r", "\n")
    if text.lower() in NULL_TOKENS:
        return ""
    return text


def normalize_query_limit(raw_limit: Any, *, default: int = 200, minimum: int = 1, maximum: int = 1000) -> int:
    """Validate and clamp the query_data limit argument from agent tool input."""
    if raw_limit is None or raw_limit == "":
        return default
    if isinstance(raw_limit, bool):
        raise ValueError("query_data.limit must be an integer, not a boolean.")
    if isinstance(raw_limit, int):
        parsed = raw_limit
    elif isinstance(raw_limit, float):
        if not raw_limit.is_integer():
            raise ValueError(f"query_data.limit must be an integer, got non-integer float {raw_limit!r}.")
        parsed = int(raw_limit)
    elif isinstance(raw_limit, str):
        text = raw_limit.strip()
        if not text:
            return default
        if not re.fullmatch(r"[+-]?\d+", text):
            raise ValueError(
                f"query_data.limit must be an integer or numeric string, got {raw_limit!r}. "
                'Retry with {"limit": 100, "sql": "..."} or omit limit.'
            )
        parsed = int(text)
    else:
        raise ValueError(
            f"query_data.limit must be an integer or numeric string, got {type(raw_limit).__name__}. "
            'Retry with {"limit": 100, "sql": "..."} or omit limit.'
        )

    if parsed < minimum:
        return minimum
    if parsed > maximum:
        return maximum
    return parsed


def validate_answer_submission(columns: list[str], rows: list[list[Any]]) -> tuple[list[str], list[list[str]]]:
    """Validate answer tool payloads and normalize them into CSV-ready strings."""
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


def build_task_mcp_server(task_id: str, output_csv: Path, task_log_path: Path, query_workspace: ContextQueryWorkspace):
    """Construct the task-local MCP server exposed to one agent session."""
    task_log = make_task_logger(task_log_path)

    @tool(
        name="list_context_files",
        description="List files and directories available under the current task context.",
        input_schema={
            "max_depth": int,
        },
    )
    async def list_context_files(args: dict[str, Any]) -> dict[str, Any]:
        max_depth = max(1, int(args.get("max_depth", 4)))
        result = query_workspace.list_context_files(max_depth=max_depth)
        log_verbose(task_log, f"[{task_id}][list_context_files] entries={len(result['entries'])} max_depth={max_depth}")
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}

    @tool(
        name="describe_query_workspace",
        description=(
            "Describe the unified query workspace for this task. "
            "CSV and JSON files are preloaded into a temporary SQLite workspace, and sqlite/db files are mirrored into it."
        ),
        input_schema={
            "max_tables": int,
            "sample_rows": int,
        },
    )
    async def describe_query_workspace(args: dict[str, Any]) -> dict[str, Any]:
        max_tables = max(1, int(args.get("max_tables", 200)))
        sample_rows = max(1, int(args.get("sample_rows", 3)))
        result = query_workspace.describe_query_workspace(max_tables=max_tables, sample_rows=sample_rows)
        log_verbose(task_log, f"[{task_id}][describe_query_workspace] tables={result['table_count']}")
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}

    @tool(
        name="query_data",
        description=(
            "Run a read-only SQL query against the task's unified temporary SQLite workspace. "
            "Use describe_query_workspace first to discover available table names. "
            "Only SELECT, WITH, and PRAGMA statements are allowed. "
            "limit is optional and accepts either an integer or a numeric string."
        ),
        input_schema={
            "sql": str,
            "limit": Any,
        },
    )
    async def query_data(args: dict[str, Any]) -> dict[str, Any]:
        sql = str(args.get("sql", "")).strip()
        if not sql:
            return {
                "content": [{"type": "text", "text": "query_data.sql is required."}],
                "is_error": True,
            }
        try:
            limit = normalize_query_limit(args.get("limit", 200))
            result = query_workspace.execute_query(sql, limit=limit)
        except Exception as exc:
            return {
                "content": [{"type": "text", "text": str(exc)}],
                "is_error": True,
            }
        log_verbose(task_log, f"[{task_id}][query_data] limit={limit} sql={sql[:4000]}")
        return {"content": [{"type": "text", "text": json.dumps(result, ensure_ascii=False)}]}


    @tool(
        name="answer",
        description=(
            "Submit the final benchmark answer as a compact table. "
            "Use only the minimum required columns and rows for the question. "
            "Do not include explanatory or intermediate columns unless the question explicitly asks for them."
        ),
        input_schema={
            "columns": list[str],
            "rows": list[list[Any]],
        },
    )
    async def submit_answer(args: dict[str, Any]) -> dict[str, Any]:
        columns = args.get("columns")
        rows = args.get("rows")
        log_verbose(task_log, f"[{task_id}][answer tool] received columns={columns!r}")

        if isinstance(columns, str):
            try:
                columns = json.loads(columns)
            except json.JSONDecodeError:
                pass

        if isinstance(rows, str):
            try:
                rows = json.loads(rows)
            except json.JSONDecodeError:
                pass

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

        write_prediction_csv(output_csv=output_csv, columns=validated_columns, rows=validated_rows)
        task_log(
            f"[{task_id}][answer tool] submitted rows={len(validated_rows)} cols={len(validated_columns)} path={output_csv}"
        )
        return {
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Submitted final answer for {task_id}. "
                        f"columns={len(validated_columns)} rows={len(validated_rows)} path={output_csv}"
                    ),
                }
            ]
        }

    return create_sdk_mcp_server(
        name=f"{task_id}_task_server",
        tools=[list_context_files, describe_query_workspace, query_data, submit_answer],
    )


def build_task_prompt(task_id: str, question: str) -> str:
    """Build the compact task-specific prompt passed to Claude Code."""
    input_task_dir = INPUT_ROOT / task_id
    output_task_dir = OUTPUT_ROOT / task_id
    return "\n".join(
        [
            f"You are solving benchmark task {task_id}.",
            f"Question: {question}",
            "Paths:",
            f"- task.json: {input_task_dir / 'task.json'}",
            f"- context dir: {input_task_dir / 'context'}",
            f"- output csv: {output_task_dir / 'prediction.csv'}",
            "Execution:",
            "1. Read task.json.",
            "2. Read context/knowledge.md when it exists.",
            "3. Inspect the available task-local data sources.",
            "4. Produce one well-supported final result.",
            "5. Call answer exactly once, then stop.",
        ]
    )



async def run_task(task_id: str) -> None:
    """Run one benchmark task end-to-end and require a prediction.csv artifact on success."""
    task_dir = INPUT_ROOT / task_id
    task_json = json.loads((task_dir / "task.json").read_text(encoding="utf-8"))
    question = task_json["question"]
    output_csv = OUTPUT_ROOT / task_id / "prediction.csv"
    task_work_dir = WORK_ROOT / task_id
    task_work_dir.mkdir(parents=True, exist_ok=True)
    task_log_path = TASK_LOG_DIR / f"{task_id}.log"
    task_log = make_task_logger(task_log_path)

    task_log(f"\n[task] {task_id}")
    log_verbose(task_log, f"[task] question={question}")
    task_log(f"[{task_id}][turn] started")

    extra_args: dict[str, str | None] = {}
    if DEBUG_TO_STDERR:
        extra_args["debug-to-stderr"] = None

    query_workspace = ContextQueryWorkspace(context_dir=task_dir / "context", work_dir=task_work_dir / "query_workspace")
    task_server = build_task_mcp_server(
        task_id=task_id,
        output_csv=output_csv,
        task_log_path=task_log_path,
        query_workspace=query_workspace,
    )
    options = ClaudeAgentOptions(
        tools={"type": "preset", "preset": "claude_code"},
        system_prompt={"type": "preset", "preset": "claude_code", "append": SYSTEM_PROMPT_APPEND},
        mcp_servers={"task_server": task_server},
        disallowed_tools=["Bash", "Write", "Edit", "NotebookEdit"],
        permission_mode=PERMISSION_MODE,
        model=MODEL_NAME,
        max_turns=MAX_TURNS,
        cwd=task_work_dir,
        add_dirs=[str(task_dir), str(output_csv.parent), str(task_log_path.parent)],
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
    result_subtype: str | None = None
    failed = False

    try:
        async for message in query(prompt=build_task_prompt(task_id, question), options=options):
            if isinstance(message, AssistantMessage):
                rendered_blocks = 0
                text_parts: list[str] = []
                for block in message.content:
                    block_text = getattr(block, "text", None)
                    if block_text:
                        rendered_blocks += 1
                        text_parts.append(block_text)
                        log_verbose(task_log, f"\n[{task_id}][assistant][text]\n{block_text}")
                        continue

                    thinking = getattr(block, "thinking", None)
                    if thinking:
                        rendered_blocks += 1
                        log_verbose(task_log, f"\n[{task_id}][assistant][thinking]\n{thinking}")
                        continue

                    tool_name = getattr(block, "name", None)
                    tool_input = getattr(block, "input", None)
                    if tool_name is not None and tool_input is not None:
                        rendered_blocks += 1
                        log_verbose(
                            task_log,
                            f"\n[{task_id}][assistant][tool_use] name={tool_name} input="
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
                            f"\n[{task_id}][assistant][tool_result] tool_use_id={tool_use_id} "
                            f"is_error={tool_is_error} content={rendered_result}"
                        )
                        continue

                if text_parts:
                    final_texts.append("\n".join(text_parts))
                if rendered_blocks == 0:
                    log_verbose(task_log, f"\n[{task_id}][assistant][empty_message]")
            elif isinstance(message, SystemMessage):
                subtype = getattr(message, "subtype", "")
                data = getattr(message, "data", {})
                task_log(f"\n[{task_id}][system] subtype={subtype}")
                if VERBOSE_LOGS and data:
                    task_log(json.dumps(data, ensure_ascii=False, default=str)[:4000])
            elif isinstance(message, ResultMessage):
                result_subtype = message.subtype
                task_log(f"[{task_id}][result] {message.subtype}")
                if VERBOSE_LOGS and getattr(message, "usage", None):
                    task_log(f"[{task_id}][result][usage] {json.dumps(message.usage, ensure_ascii=False, default=str)[:2000]}")
                if VERBOSE_LOGS and getattr(message, "result", None):
                    task_log(f"[{task_id}][result][text] {str(message.result)[:4000]}")
                if VERBOSE_LOGS and getattr(message, "errors", None):
                    task_log(f"[{task_id}][result][errors] {json.dumps(message.errors, ensure_ascii=False, default=str)[:4000]}")
    except Exception as exc:
        failed = True
        task_log(f"[{task_id}][turn failed] {exc}")
        task_log(f"[{task_id}][turn failed repr] {exc!r}")
        cause = getattr(exc, "__cause__", None)
        if cause is not None:
            task_log(f"[{task_id}][turn failed cause] {cause!r}")
        context = getattr(exc, "__context__", None)
        if context is not None:
            task_log(f"[{task_id}][turn failed context] {context!r}")
        for attr in ("stderr", "stdout", "message", "args"):
            value = getattr(exc, attr, None)
            if value:
                task_log(f"[{task_id}][turn failed {attr}] {value}")
        task_log(f"[{task_id}][turn failed traceback]\n{traceback.format_exc()}")

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
    """Discover tasks, apply runtime filters, and execute evaluation with bounded concurrency."""
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

    async def run_task_with_limit(task_id: str) -> None:
        async with semaphore:
            try:
                await run_task(task_id)
            except Exception as exc:
                print(f"[task failure] {task_id}: {exc}")

    await asyncio.gather(*(run_task_with_limit(task_id) for task_id in task_ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
