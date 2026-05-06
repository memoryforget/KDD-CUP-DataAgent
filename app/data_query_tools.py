from __future__ import annotations

import csv
import json
import re
import sqlite3
from pathlib import Path
from typing import Any

STRUCTURED_SUFFIXES = {".csv", ".json", ".db", ".sqlite", ".sqlite3"}
READ_ONLY_SQL_PREFIXES = ("select", "with", "pragma")
DOC_SUFFIXES = {".md", ".txt",}
SUMMARY_COLUMN_LIMIT = 12


def quote_ident(value: str) -> str:
    """Return a safely quoted SQLite identifier."""
    return '"' + value.replace('"', '""') + '"'


def normalize_sql_name(value: str, *, prefix: str = "table") -> str:
    """Normalize arbitrary text into a stable SQLite table or column name."""
    text = re.sub(r"[^0-9A-Za-z_]+", "_", value.strip()).strip("_").lower()
    if not text:
        text = prefix
    if text[0].isdigit():
        text = f"{prefix}_{text}"
    return text


def normalize_scalar(value: Any) -> Any:
    """Convert complex Python values into SQLite-friendly scalar representations."""
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, bool):
        return int(value)
    return value


def classify_scalar(value: Any) -> str:
    """Infer a coarse logical type for a scalar value before SQLite import."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int) and not isinstance(value, bool):
        return "int"
    if isinstance(value, float):
        return "float"
    if not isinstance(value, str):
        return "text"

    text = value.strip()
    if text == "":
        return "null"
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return "bool"
    try:
        int(text)
    except ValueError:
        pass
    else:
        return "int"
    try:
        float(text)
    except ValueError:
        return "text"
    return "float"


def sqlite_type_for_values(values: list[Any]) -> str:
    """Choose an appropriate SQLite column type from observed row values."""
    kinds = {classify_scalar(value) for value in values if value is not None}
    kinds.discard("null")
    if not kinds:
        return "TEXT"
    if kinds <= {"bool"}:
        return "INTEGER"
    if kinds <= {"bool", "int"}:
        return "INTEGER"
    if kinds <= {"bool", "int", "float"}:
        return "REAL"
    return "TEXT"


def cast_value_for_sqlite(value: Any, declared_type: str) -> Any:
    """Cast a normalized value to the declared SQLite column type."""
    normalized = normalize_scalar(value)
    if normalized is None:
        return None
    if declared_type == "INTEGER":
        if isinstance(normalized, str):
            text = normalized.strip()
            if text == "":
                return None
            lowered = text.lower()
            if lowered == "true":
                return 1
            if lowered == "false":
                return 0
            return int(text)
        return int(normalized)
    if declared_type == "REAL":
        if isinstance(normalized, str):
            text = normalized.strip()
            if text == "":
                return None
            lowered = text.lower()
            if lowered == "true":
                return 1.0
            if lowered == "false":
                return 0.0
            return float(text)
        return float(normalized)
    return normalized


def build_column_types(columns: list[str], rows: list[list[Any]]) -> list[str]:
    """Infer SQLite types column-by-column from a rectangular row set."""
    inferred_types: list[str] = []
    for index in range(len(columns)):
        values = [row[index] for row in rows if index < len(row)]
        inferred_types.append(sqlite_type_for_values(values))
    return inferred_types


def summarize_columns(columns: list[str], column_types: list[str], *, limit: int = SUMMARY_COLUMN_LIMIT) -> dict[str, Any]:
    """Build a compact column summary for workspace metadata responses."""
    preview = []
    for name, declared_type in list(zip(columns, column_types))[:limit]:
        preview.append({"name": name, "type": declared_type})
    return {
        "preview": preview,
        "total": len(columns),
        "omitted": max(0, len(columns) - limit),
    }




class ContextQueryWorkspace:
    def __init__(self, context_dir: Path, work_dir: Path) -> None:
        """Initialize a task-local query workspace and materialize its structured sources."""
        self.context_dir = context_dir.resolve()
        self.work_dir = work_dir.resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.work_dir / "context_workspace.sqlite"
        self.source_metadata: dict[str, dict[str, Any]] = {}
        self.workspace_tables: list[dict[str, Any]] = []
        self.excluded_tables: list[dict[str, Any]] = []
        self._table_name_counts: dict[str, int] = {}
        self._build_workspace()

    def list_context_files(self, *, max_depth: int = 4) -> dict[str, Any]:
        """Return a bounded recursive listing of files available to the current task."""
        entries: list[dict[str, Any]] = []

        def walk(path: Path, depth: int) -> None:
            if depth > max_depth:
                return
            for child in sorted(path.iterdir(), key=lambda item: (item.is_file(), item.name)):
                rel_path = child.relative_to(self.context_dir).as_posix()
                entries.append(
                    {
                        "path": rel_path,
                        "kind": "dir" if child.is_dir() else "file",
                        "size": child.stat().st_size if child.is_file() else None,
                    }
                )
                if child.is_dir():
                    walk(child, depth + 1)

        walk(self.context_dir, 1)
        return {"root": str(self.context_dir), "entries": entries}

    def describe_query_workspace(self, *, max_tables: int = 200, sample_rows: int = 3) -> dict[str, Any]:
        """Summarize the tables currently available in the temporary SQLite workspace."""
        tables = []
        for item in self.workspace_tables[:max_tables]:
            sample = [row[: len(item["columns"])] for row in item["sample_rows"][:sample_rows]]
            tables.append(
                {
                    "table_name": item["table_name"],
                    "source_path": item["source_path"],
                    "source_kind": item["source_kind"],
                    "row_count": item["row_count"],
                    "columns": summarize_columns(item["columns"], item["column_types"]),
                    "sample_rows": sample,
                }
            )
        return {
            "db_path": str(self.db_path),
            "table_count": len(self.workspace_tables),
            "tables": tables,
            "truncated": len(self.workspace_tables) > max_tables,
            "excluded_tables": self.excluded_tables[:max_tables],
            "excluded_truncated": len(self.excluded_tables) > max_tables,
        }


    def execute_query(self, sql: str, *, limit: int = 200) -> dict[str, Any]:
        """Execute a read-only SQL statement against the temporary task workspace."""
        normalized_sql = sql.lstrip().lower()
        if not normalized_sql.startswith(READ_ONLY_SQL_PREFIXES):
            raise ValueError("Only read-only SQL statements are allowed.")

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(sql)
            column_names = [item[0] for item in cursor.description or []]
            rows = cursor.fetchmany(limit + 1)

        truncated = len(rows) > limit
        limited_rows = rows[:limit]
        return {
            "columns": column_names,
            "rows": [list(row) for row in limited_rows],
            "row_count": len(limited_rows),
            "truncated": truncated,
        }


    def _build_workspace(self) -> None:
        """Scan task context files and import supported structured sources into SQLite."""
        if self.db_path.exists():
            self.db_path.unlink()

        with sqlite3.connect(self.db_path) as conn:
            for path in sorted(self.context_dir.rglob("*")):
                if not path.is_file():
                    continue
                rel_path = path.relative_to(self.context_dir).as_posix()
                suffix = path.suffix.lower()
                if suffix not in STRUCTURED_SUFFIXES:
                    continue
                try:
                    if suffix == ".csv":
                        self._import_csv_source(conn, path, rel_path)
                    elif suffix == ".json":
                        self._import_json_source(conn, path, rel_path)
                    else:
                        self._import_sqlite_source(conn, path, rel_path)
                except Exception as exc:
                    self.source_metadata[rel_path] = {
                        "path": rel_path,
                        "source_kind": suffix.lstrip(".") or "file",
                        "tables": [],
                        "excluded_tables": [],
                        "error": str(exc),
                    }

    def _next_table_name(self, preferred_name: str) -> str:
        """Generate a unique normalized table name within the current workspace."""
        base_name = normalize_sql_name(preferred_name)
        count = self._table_name_counts.get(base_name, 0)
        if count == 0:
            self._table_name_counts[base_name] = 1
            return base_name
        self._table_name_counts[base_name] = count + 1
        return f"{base_name}_{count + 1}"

    def _record_table(
        self,
        *,
        source_path: str,
        source_kind: str,
        table_name: str,
        columns: list[str],
        column_types: list[str],
        row_count: int,
        sample_rows: list[list[Any]],
    ) -> None:
        """Record imported table metadata for later workspace description calls."""
        table_info = {
            "table_name": table_name,
            "columns": columns,
            "column_types": column_types,
            "row_count": row_count,
            "sample_rows": sample_rows,
        }
        source_info = self.source_metadata.setdefault(
            source_path,
            {
                "path": source_path,
                "source_kind": source_kind,
                "tables": [],
                "excluded_tables": [],
                "error": None,
            },
        )
        source_info["tables"].append(table_info)
        self.workspace_tables.append(
            {
                "table_name": table_name,
                "source_path": source_path,
                "source_kind": source_kind,
                "columns": columns,
                "column_types": column_types,
                "row_count": row_count,
                "sample_rows": sample_rows,
            }
        )

    def _exclude_table(
        self,
        *,
        source_path: str,
        source_kind: str,
        table_name: str,
        reason: str,
        columns: list[str],
    ) -> None:
        """Record a source table that was skipped so exclusions remain visible to the agent."""
        excluded = {
            "table_name": table_name,
            "source_path": source_path,
            "source_kind": source_kind,
            "reason": reason,
            "columns": columns,
        }
        source_info = self.source_metadata.setdefault(
            source_path,
            {
                "path": source_path,
                "source_kind": source_kind,
                "tables": [],
                "excluded_tables": [],
                "error": None,
            },
        )
        source_info["excluded_tables"].append(excluded)
        self.excluded_tables.append(excluded)

    def _create_table(
        self,
        conn: sqlite3.Connection,
        table_name: str,
        columns: list[str],
        declared_types: list[str] | None = None,
    ) -> None:
        """Create a SQLite table using normalized column definitions."""
        column_defs = []
        for index, column in enumerate(columns):
            declared_type = ""
            if declared_types and index < len(declared_types) and declared_types[index]:
                declared_type = f" {declared_types[index]}"
            column_defs.append(f"{quote_ident(column)}{declared_type}")
        conn.execute(f"CREATE TABLE {quote_ident(table_name)} ({', '.join(column_defs)})")

    def _insert_rows(self, conn: sqlite3.Connection, table_name: str, columns: list[str], rows: list[list[Any]]) -> None:
        """Insert a batch of rows into a SQLite table if rows are present."""
        if not columns or not rows:
            return
        placeholders = ", ".join("?" for _ in columns)
        insert_sql = f"INSERT INTO {quote_ident(table_name)} ({', '.join(quote_ident(c) for c in columns)}) VALUES ({placeholders})"
        conn.executemany(insert_sql, rows)

    def _import_csv_source(self, conn: sqlite3.Connection, path: Path, relative_path: str) -> None:
        """Import a CSV file into the temporary workspace with inferred column types."""
        with path.open(newline="", encoding="utf-8", errors="replace") as handle:
            reader = csv.reader(handle)
            all_rows = list(reader)

        if not all_rows:
            columns = ["value"]
            raw_rows: list[list[Any]] = []
        else:
            columns = [column.strip() or f"column_{index + 1}" for index, column in enumerate(all_rows[0])]
            raw_rows = [list(row) for row in all_rows[1:]]

        normalized_rows = [row + [None] * (len(columns) - len(row)) for row in raw_rows]
        normalized_rows = [row[: len(columns)] for row in normalized_rows]
        column_types = build_column_types(columns, normalized_rows)
        converted_rows = [
            [cast_value_for_sqlite(row[index], column_types[index]) for index in range(len(columns))]
            for row in normalized_rows
        ]

        table_name = self._next_table_name(f"csv_{path.stem}")
        self._create_table(conn, table_name, columns, column_types)
        self._insert_rows(conn, table_name, columns, converted_rows)
        row_count = len(converted_rows)
        if row_count == 0:
            conn.execute(f"DROP TABLE {quote_ident(table_name)}")
            self._exclude_table(
                source_path=relative_path,
                source_kind="csv",
                table_name=table_name,
                reason="empty_table",
                columns=columns,
            )
            return
        self._record_table(
            source_path=relative_path,
            source_kind="csv",
            table_name=table_name,
            columns=columns,
            column_types=column_types,
            row_count=row_count,
            sample_rows=converted_rows[:5],
        )

    def _import_json_source(self, conn: sqlite3.Connection, path: Path, relative_path: str) -> None:
        """Import a JSON payload into the temporary workspace as a flat table."""
        payload = json.loads(path.read_text(encoding="utf-8"))
        preferred_name = path.stem

        if isinstance(payload, dict) and isinstance(payload.get("records"), list):
            preferred_name = str(payload.get("table") or preferred_name)
            records = payload["records"]
        elif isinstance(payload, list):
            records = payload
        else:
            records = [payload]

        normalized_records: list[dict[str, Any]] = []
        for item in records:
            if isinstance(item, dict):
                normalized_records.append(item)
            else:
                normalized_records.append({"value": item})

        columns: list[str] = []
        seen_columns: set[str] = set()
        for record in normalized_records:
            for key in record:
                column_name = str(key)
                if column_name not in seen_columns:
                    seen_columns.add(column_name)
                    columns.append(column_name)
        if not columns:
            columns = ["value"]

        raw_rows = [[normalize_scalar(record.get(column)) for column in columns] for record in normalized_records]
        column_types = build_column_types(columns, raw_rows)
        converted_rows = [
            [cast_value_for_sqlite(row[index], column_types[index]) for index in range(len(columns))]
            for row in raw_rows
        ]

        table_name = self._next_table_name(f"json_{preferred_name}")
        self._create_table(conn, table_name, columns, column_types)
        self._insert_rows(conn, table_name, columns, converted_rows)
        row_count = len(converted_rows)
        if row_count == 0:
            conn.execute(f"DROP TABLE {quote_ident(table_name)}")
            self._exclude_table(
                source_path=relative_path,
                source_kind="json",
                table_name=table_name,
                reason="empty_table",
                columns=columns,
            )
            return
        self._record_table(
            source_path=relative_path,
            source_kind="json",
            table_name=table_name,
            columns=columns,
            column_types=column_types,
            row_count=row_count,
            sample_rows=converted_rows[:5],
        )

    def _import_sqlite_source(self, conn: sqlite3.Connection, path: Path, relative_path: str) -> None:
        """Mirror tables from an existing SQLite database into the temporary workspace."""
        uri = f"file:{path.resolve().as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as source_conn:
            source_tables = source_conn.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            ).fetchall()
            for (source_table_name,) in source_tables:
                column_info = source_conn.execute(f"PRAGMA table_info({quote_ident(source_table_name)})").fetchall()
                columns = [str(item[1]) for item in column_info]
                declared_types = [str(item[2] or "TEXT") for item in column_info]
                preferred_name = f"db_{path.stem}_{source_table_name}"
                table_name = self._next_table_name(preferred_name)
                self._create_table(conn, table_name, columns, declared_types)

                rows_cursor = source_conn.execute(f"SELECT * FROM {quote_ident(source_table_name)}")
                batch: list[list[Any]] = []
                sample_rows: list[list[Any]] = []
                row_count = 0
                for row in rows_cursor:
                    row_list = [normalize_scalar(value) for value in row]
                    if len(sample_rows) < 5:
                        sample_rows.append(row_list)
                    batch.append(row_list)
                    row_count += 1
                    if len(batch) >= 1000:
                        self._insert_rows(conn, table_name, columns, batch)
                        batch.clear()
                if batch:
                    self._insert_rows(conn, table_name, columns, batch)

                if row_count == 0:
                    conn.execute(f"DROP TABLE {quote_ident(table_name)}")
                    self._exclude_table(
                        source_path=relative_path,
                        source_kind="sqlite",
                        table_name=table_name,
                        reason="empty_table",
                        columns=columns,
                    )
                    continue

                self._record_table(
                    source_path=relative_path,
                    source_kind="sqlite",
                    table_name=table_name,
                    columns=columns,
                    column_types=declared_types,
                    row_count=row_count,
                    sample_rows=sample_rows,
                )
