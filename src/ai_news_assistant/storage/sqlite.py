import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from ai_news_assistant.models import Article

# Plain SQL so the Postgres version is a near copy.
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS articles (
    id TEXT PRIMARY KEY,
    url TEXT NOT NULL,
    title TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at TEXT,  -- ISO 8601 in UTC, so text order is time order
    summary TEXT,
    text TEXT,
    tags TEXT NOT NULL,  -- JSON array
    extraction_successful INTEGER,  -- NULL: not tried, 0: failed, 1: succeeded
    extracted_at TEXT  -- ISO 8601 in UTC, set only on success
)
"""

# Columns added after the first schema. Databases created before them get them via ALTER TABLE,
# so existing rows are kept, with NULL meaning "extraction not tried".
_ADDED_COLUMNS = {
    "extraction_successful": "INTEGER",
    "extracted_at": "TEXT",
}

# One column list for every query, in the order `_to_row` produces values.
_COLUMNS = """
    id, url, title, source, published_at, summary, text, tags,
    extraction_successful, extracted_at
"""

_INSERT = f"""
INSERT INTO articles ({_COLUMNS})
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (id) DO NOTHING
"""

_SELECT = f"""
SELECT {_COLUMNS}
FROM articles
WHERE id = ?
"""

# Newest first, since pages change and disappear; undated last; id makes ties deterministic.
_SELECT_UNEXTRACTED = f"""
SELECT {_COLUMNS}
FROM articles
WHERE extraction_successful IS NULL
ORDER BY published_at DESC NULLS LAST, id
LIMIT ?
"""

# Only never-tried rows match, so each article records exactly one extraction attempt
# and overlapping extraction runs cannot overwrite each other.
_UPDATE_EXTRACTION = """
UPDATE articles
SET text = ?, extraction_successful = ?, extracted_at = ?
WHERE id = ? AND extraction_successful IS NULL
"""


class SQLiteStore:
    """Article store backed by one SQLite database file.

    Pass ":memory:" for a throwaway database in tests. The table is created on startup.
    """

    def __init__(self, path: str | Path) -> None:
        # One connection for the store's lifetime, so ":memory:" keeps its data.
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.execute(_CREATE_TABLE)
            self._migrate()

    def save(self, articles: Iterable[Article]) -> int:
        """Insert articles not stored yet and return how many were new.

        Existing ids are left unchanged, including repeats within the same batch.
        """
        rows = [_to_row(article) for article in articles]
        if not rows:
            return 0
        # Article fields are untrusted, so values only ever go in as parameters.
        with self._conn:
            cursor = self._conn.executemany(_INSERT, rows)
        return cursor.rowcount

    def get(self, article_id: str) -> Article | None:
        row = self._conn.execute(_SELECT, (article_id,)).fetchone()
        return None if row is None else _from_row(row)

    def get_unextracted(self, limit: int | None = None) -> list[Article]:
        """Return articles whose extraction was never tried, newest first, undated last."""
        if limit is not None and limit < 0:
            raise ValueError(f"limit must be >= 0, got {limit}")
        # -1 is SQLite's "no limit"; Postgres would use LIMIT ALL.
        rows = self._conn.execute(_SELECT_UNEXTRACTED, (-1 if limit is None else limit,))
        return [_from_row(row) for row in rows]

    def save_extraction(self, article_id: str, text: str | None, extracted_at: datetime) -> bool:
        """Record the one extraction attempt of an article.

        Non-empty `text` means success; None or blank text means failure. Returns False,
        changing nothing, if the id is unknown or an attempt was already recorded.
        """
        if extracted_at.utcoffset() is None:
            raise ValueError("extracted_at must be timezone-aware")
        params: tuple[Any, ...]
        if text is None or not text.strip():
            params = (None, 0, None, article_id)
        else:
            params = (text, 1, _to_utc_text(extracted_at), article_id)
        # Extracted text is untrusted, so it only ever goes in as a parameter.
        with self._conn:
            cursor = self._conn.execute(_UPDATE_EXTRACTION, params)
        return cursor.rowcount == 1

    def _migrate(self) -> None:
        """Add columns missing from a database created by an older schema."""
        existing = {row["name"] for row in self._conn.execute("PRAGMA table_info(articles)")}
        for name, sql_type in _ADDED_COLUMNS.items():
            if name not in existing:
                # Names and types come from the constant above, never from input.
                self._conn.execute(f"ALTER TABLE articles ADD COLUMN {name} {sql_type}")

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


def _to_row(article: Article) -> tuple[Any, ...]:
    successful = article.extraction_successful
    return (
        article.id,
        str(article.url),
        article.title,
        article.source,
        _to_utc_text(article.published_at),
        article.summary,
        article.text,
        json.dumps(article.tags),
        None if successful is None else int(successful),
        _to_utc_text(article.extracted_at),
    )


def _from_row(row: sqlite3.Row) -> Article:
    successful = row["extraction_successful"]
    # Validate on the way out too, so a bad row fails loudly instead of leaking through.
    return Article.model_validate(
        {
            "id": row["id"],
            "url": row["url"],
            "title": row["title"],
            "source": row["source"],
            "published_at": _from_text(row["published_at"]),
            "summary": row["summary"],
            "text": row["text"],
            "tags": json.loads(row["tags"]),
            "extraction_successful": None if successful is None else bool(successful),
            "extracted_at": _from_text(row["extracted_at"]),
        }
    )


def _to_utc_text(value: datetime | None) -> str | None:
    return value.astimezone(UTC).isoformat() if value else None


def _from_text(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None
