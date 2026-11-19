from __future__ import annotations
from contextlib import contextmanager
import psycopg
from psycopg.conninfo import conninfo_to_dict
from .config import SCHEMA, reader_url, uploader_url, STATEMENT_TIMEOUT_MS


class ReaderConfigurationError(ValueError):
    pass


@contextmanager
def read_connection():
    url = reader_url().strip()
    if not url:
        raise ReaderConfigurationError("Set READONLY_DATABASE_URL in .env to the optimizer_reader connection URL.")
    try:
        settings = conninfo_to_dict(url)
    except psycopg.ProgrammingError as exc:
        raise ReaderConfigurationError("READONLY_DATABASE_URL in .env is not a valid PostgreSQL connection string.") from exc
    if settings.get("host", "").lower() in {"host", "example.com"}:
        raise ReaderConfigurationError(
            "READONLY_DATABASE_URL still points to the example host. Set it in .env "
            "to the optimizer_reader URL for the database where you imported the CSV, then restart Streamlit."
        )
    with psycopg.connect(url, connect_timeout=8, autocommit=False, prepare_threshold=None) as conn:
        with conn.transaction():
            conn.execute("SET TRANSACTION READ ONLY")
            conn.execute("SET LOCAL search_path TO pg_catalog")
            conn.execute("SELECT set_config('statement_timeout', %s, true)",
                         (str(STATEMENT_TIMEOUT_MS),))
            yield conn


def workspace_indexes(tables: dict[str, str]) -> dict[str, list[dict]]:
    """Read usable index metadata for the workspace without exposing physical identifiers."""
    names = {physical: friendly for friendly, physical in tables.items()}
    indexes = {friendly: [] for friendly in tables}
    with read_connection() as conn:
        rows = conn.execute("""
            SELECT t.relname, am.amname, i.indisunique, i.indpred IS NOT NULL,
                   ARRAY(SELECT a.attname
                         FROM unnest(i.indkey) WITH ORDINALITY AS k(number, position)
                         LEFT JOIN pg_attribute a ON a.attrelid = t.oid AND a.attnum = k.number
                         WHERE k.position <= i.indnkeyatts ORDER BY k.position)
            FROM pg_index i
            JOIN pg_class t ON t.oid = i.indrelid
            JOIN pg_class idx ON idx.oid = i.indexrelid
            JOIN pg_namespace n ON n.oid = t.relnamespace
            JOIN pg_am am ON am.oid = idx.relam
            WHERE n.nspname = %s AND t.relname = ANY(%s::name[]) AND i.indisvalid
              AND i.indisready AND i.indislive
        """, (SCHEMA, list(names))).fetchall()
    for physical, method, unique, partial, columns in rows:
        indexes[names[physical]].append({
            "method": method, "unique": unique, "partial": partial,
            "key_columns": [column or "(expression)" for column in columns],
        })
    return indexes


@contextmanager
def upload_connection():
    url = uploader_url()
    if not url:
        raise RuntimeError("UPLOAD_DATABASE_URL is not configured. Run provisioning first.")
    with psycopg.connect(url, connect_timeout=8, prepare_threshold=None) as conn:
        conn.execute("SET statement_timeout = '15000ms'")
        yield conn
