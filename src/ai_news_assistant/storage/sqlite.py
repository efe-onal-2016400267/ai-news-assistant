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
    tags TEXT NOT NULL  -- JSON array
)
"""

_INSERT = """
INSERT INTO articles (id, url, title, source, published_at, summary, text, tags)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
ON CONFLICT (id) DO NOTHING
"""

_SELECT = """
SELECT id, url, title, source, published_at, summary, text, tags
FROM articles
WHERE id = ?
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
    published_at = article.published_at
    return (
        article.id,
        str(article.url),
        article.title,
        article.source,
        published_at.astimezone(UTC).isoformat() if published_at else None,
        article.summary,
        article.text,
        json.dumps(article.tags),
    )


def _from_row(row: sqlite3.Row) -> Article:
    published_at = row["published_at"]
    # Validate on the way out too, so a bad row fails loudly instead of leaking through.
    return Article.model_validate(
        {
            "id": row["id"],
            "url": row["url"],
            "title": row["title"],
            "source": row["source"],
            "published_at": datetime.fromisoformat(published_at) if published_at else None,
            "summary": row["summary"],
            "text": row["text"],
            "tags": json.loads(row["tags"]),
        }
    )
