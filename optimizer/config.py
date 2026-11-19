from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv


def load_project_environment(path: Path | None = None) -> None:
    load_dotenv(path or Path(__file__).resolve().parents[1] / ".env", override=False)


load_project_environment()

SCHEMA = "optimizer_data"
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_ROWS = 50_000
MAX_COLUMNS = 50
MAX_QUERIES = 5
BENCHMARK_RUNS = 5
STATEMENT_TIMEOUT_MS = 5_000


def reader_url() -> str:
    return os.getenv("READONLY_DATABASE_URL", "")


def uploader_url() -> str:
    return os.getenv("UPLOAD_DATABASE_URL", "")
