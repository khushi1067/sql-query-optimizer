from __future__ import annotations
import io
import re
import uuid
from dataclasses import dataclass
from decimal import Decimal
from itertools import combinations
import pandas as pd
from pglast import keywords
from psycopg import sql
from .config import MAX_COLUMNS, MAX_FILE_BYTES, MAX_ROWS, SCHEMA

MAX_TABLE_NAME_LENGTH = 40
MANAGED_TABLE_NAME = re.compile(r"ds_[a-f0-9]{32}")
UNAVAILABLE_TABLE_NAMES = (
    keywords.RESERVED_KEYWORDS | keywords.COL_NAME_KEYWORDS | keywords.TYPE_FUNC_NAME_KEYWORDS
    | {"public", "information_schema", SCHEMA}
)


class DatasetError(ValueError):
    pass


def validate_table_name(name: str) -> str:
    if not re.fullmatch(rf"[a-z][a-z0-9_]{{0,{MAX_TABLE_NAME_LENGTH - 1}}}", name):
        raise DatasetError("Use 1 to 40 lowercase letters, digits, or underscores, starting with a letter.")
    if name in UNAVAILABLE_TABLE_NAMES or name.startswith("pg_") or MANAGED_TABLE_NAME.fullmatch(name):
        raise DatasetError("This table name is reserved. Choose another name.")
    return name


def default_table_name(filename: str) -> str:
    stem = filename.rsplit(".", 1)[0].lower()
    name = re.sub(r"[^a-z0-9_]+", "_", stem).strip("_")
    if not name or name[0].isdigit():
        name = f"dataset_{name}" if name else "dataset"
    name = name[:MAX_TABLE_NAME_LENGTH].rstrip("_")
    try:
        return validate_table_name(name)
    except DatasetError:
        return f"data_{name}"[:MAX_TABLE_NAME_LENGTH].rstrip("_")


@dataclass
class UploadPreview:
    dataframe: pd.DataFrame
    pg_types: dict[str, str]


def dataset_sql_name(dataset: dict) -> str:
    return dataset.get("friendly_name") or dataset["qualified"]


def suggested_relationships(datasets: dict[str, dict]) -> list[dict]:
    relationships = []
    for (left, a), (right, b) in combinations(datasets.items(), 2):
        right_columns = dict(b["columns"])
        for column, kind in a["columns"]:
            if right_columns.get(column) == kind:
                relationships.append({
                    "left_table": left, "right_table": right, "column": column, "type": kind,
                })
    return relationships


def normalize_headers(headers: list[str]) -> list[str]:
    result = []
    used: set[str] = set()
    for i, raw in enumerate(headers):
        name = re.sub(r"[^a-zA-Z0-9_]", "_", str(raw).strip()).lower().strip("_")
        if not name:
            name = f"column_{i+1}"
        if name[0].isdigit():
            name = "col_" + name
        name = name[:55]
        candidate = name
        index = 2
        while candidate in used:
            candidate = (name[:50] + "_" + str(index))
            index += 1
        used.add(candidate)
        result.append(candidate)
    return result


def infer_types(frame: pd.DataFrame) -> dict[str, str]:
    result = {}
    for col in frame.columns:
        values = frame[col].dropna().astype(str).str.strip()
        values = values[values != ""]
        if values.empty:
            result[col] = "TEXT"
            continue
        # Protect leading-zero identifiers from conversion to numbers.
        if values.str.fullmatch(r"[+-]?0\d+").any():
            result[col] = "TEXT"
            continue
        if values.str.fullmatch(r"[+-]?\d+").all():
            try:
                ints = [int(v) for v in values]
                if all(-(2**63) <= n < 2**63 for n in ints):
                    result[col] = "BIGINT"
                    continue
            except (ValueError, OverflowError):
                pass
        if values.str.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)").all():
            try:
                if all(Decimal(v).is_finite() for v in values):
                    result[col] = "NUMERIC"
                    continue
            except Exception:
                pass
        if values.str.fullmatch(r"\d{4}-\d{2}-\d{2}").all():
            dates = pd.to_datetime(values, format="%Y-%m-%d", errors="coerce")
            if dates.notna().all():
                result[col] = "DATE"
                continue
        result[col] = "TEXT"
    return result


def inspect_csv(content: bytes, filename: str) -> UploadPreview:
    if not filename.lower().endswith(".csv"):
        raise DatasetError("Upload a .csv file.")
    if not content or len(content) > MAX_FILE_BYTES:
        raise DatasetError("CSV must be nonempty and 10 MB or smaller.")
    try:
        text = content.decode("utf-8-sig", errors="strict")
        if "\x00" in text:
            raise DatasetError("CSV contains invalid NUL characters.")
        frame = pd.read_csv(io.StringIO(text), dtype=str, keep_default_na=False,
                            on_bad_lines="error", encoding="utf-8")
    except (UnicodeError, pd.errors.ParserError, pd.errors.EmptyDataError, ValueError) as exc:
        raise DatasetError("Invalid UTF-8 CSV or malformed rows.") from exc
    if frame.empty or len(frame.columns) == 0:
        raise DatasetError("CSV must have a header and at least one data row.")
    if len(frame) > MAX_ROWS or len(frame.columns) > MAX_COLUMNS:
        raise DatasetError(f"Maximum {MAX_ROWS:,} rows and {MAX_COLUMNS} columns.")
    if any(str(c).startswith("Unnamed:") for c in frame.columns):
        raise DatasetError("All columns need nonempty headers.")
    # Pandas automatically mangles duplicates; reject instead of quietly inventing columns.
    import csv
    header = next(csv.reader(io.StringIO(text)))
    if len(header) != len(frame.columns) or len(set(header)) != len(header):
        raise DatasetError("Duplicate CSV headers are not allowed.")
    if any(not str(h).strip() for h in header):
        raise DatasetError("CSV headers cannot be blank.")
    frame.columns = normalize_headers(list(frame.columns))
    frame = frame.replace("", None)
    return UploadPreview(frame, infer_types(frame))


def create_dataset(preview: UploadPreview, display_name: str, friendly_name: str | None = None) -> dict:
    """Create a private-to-session table name; isolation is NOT a security boundary."""
    from .db import upload_connection
    name = validate_table_name(default_table_name(display_name) if friendly_name is None else friendly_name)
    table = "ds_" + uuid.uuid4().hex
    columns = list(preview.dataframe.columns)
    col_defs = sql.SQL(", ").join(
        sql.SQL("{} {}").format(sql.Identifier(col), sql.SQL(preview.pg_types[col]))
        for col in columns
    )
    ident = sql.Identifier(SCHEMA, table)
    from psycopg import sql as psql
    with upload_connection() as conn:
        with conn.transaction():
            conn.execute(psql.SQL("CREATE TABLE {} ({})").format(ident, col_defs))
            # psycopg COPY row protocol correctly escapes embedded tabs, newlines,
            # backslashes and NULLs (including literal CSV values like \N).
            copy_stmt = psql.SQL("COPY {} ({}) FROM STDIN").format(
                ident, psql.SQL(",").join(map(psql.Identifier, columns))
            )
            with conn.cursor().copy(copy_stmt) as copy:
                for row in preview.dataframe.itertuples(index=False, name=None):
                    copy.write_row(row)
            # Ensure reader has SELECT, independent of default-privilege setup.
            conn.execute(psql.SQL("GRANT SELECT ON {} TO optimizer_reader").format(ident))
    return {"table": table, "qualified": f'{SCHEMA}.{table}', "friendly_name": name,
            "display_name": display_name[:100], "rows": len(preview.dataframe),
            "columns": list(preview.pg_types.items()), "preview": preview.dataframe.head(10).copy()}


def remove_dataset(table: str) -> None:
    if not MANAGED_TABLE_NAME.fullmatch(table):
        raise DatasetError("Invalid managed dataset identifier.")
    from .db import upload_connection
    with upload_connection() as conn:
        conn.execute(sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(SCHEMA, table)))
