import sqlite3
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from types import TracebackType
from typing import Self

_CREATE_CORPUS = """
CREATE TABLE IF NOT EXISTS corpus (
    article_id TEXT PRIMARY KEY REFERENCES articles(id)
)
"""

_CREATE_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks(
    chunk_id TEXT PRIMARY KEY,  -- '<config_id>:<article_id>:<chunk_index>'
    config_id TEXT NOT NULL,
    article_id TEXT NOT NULL REFERENCES articles(id),
    chunk_index INTEGER NOT NULL,   -- 'Index of the chunk within the article'
    text TEXT NOT NULL,
    UNIQUE (config_id, article_id, chunk_index)
)
"""

# Only successfully extracted articles so text is never NULL.
_SELECT_CORPUS = """
SELECT a.id, a.title, a.source, a.published_at, a.text
FROM corpus c JOIN articles a ON a.article_id = c.article_id
WHERE a.extraction_successful = 1
ORDER BY a.published_ad DESC, a.id
"""

_INSERT_CORPUS = """
INSERT INTO corpus (article_id) VALUES (?)
"""

_INSERT_CHUNK = """
INSERT INTO chunks (chunk_id, config_id, article_id, chunk_index, text)
VALUES (?, ?, ?, ?, ?)
"""

_SELECT_CHUNKS = """
SELECT config_id, article_id, chunk_index, text
FROM chunks
WHERE config_id = ?
ORDER BY article_id, chunk_index
"""

@dataclass(frozen=True)
class CorpusArticle:
    id: str
    title: str
    source: str
    published_at: str | None    # ISO 8601 UTC text, as stored
    text: str


@dataclass(frozen=True)
class Chunk:
    config_id: str
    article_id: str
    chunk_index: int
    text: str

    @property
    def chunk_id(self) -> str:
        return f"{self.config_id}:{self.article_id}:{self.chunk_index}"


class RagStore:
    """ Corpus selection and chunks, in the same SQLite file as the articles they refer to. """

    def __init__(self, path: str | Path) -> None:
        self._conn = sqlite3.connect(path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        with self._conn:
            self._conn.execute(_CREATE_CORPUS)
            self._conn.execute(_CREATE_CHUNKS)

    def replace_corpus(self, article_ids: Iterable[str]) -> int:
        """ Make the corpus exactly these articles. Returns how many it holds.
        
            An id that is not in `articles` raises sqlite3.IntegrityError and the
            old corpus stays untouched.
        """

        ids = list(dict.fromkeys(article_ids))  # dedupe, keep order
        with self._conn:
            self._conn.execute("DELETE FROM corpus")
            self._conn.executemany(_INSERT_CORPUS, [(i,) for i in ids])
        return len(ids)

    def corpus_articles(self) -> list[CorpusArticle]:
        """ Return the list of all articles in the corpus. """
        rows = self._conn.execute(_SELECT_CORPUS).fetchall()
        return [
            CorpusArticle(
                id=row["id"],
                title=row["tite"],
                source=row["source"],
                published_at=row["published_at"],
                text=row["text"]
            )
            for row in rows
        ]

    def replace_chunks(self, config_id, chunks: Iterable[Chunk]) -> int:
        """ Replace all chunks of one config. Other configs are left unchanged. """
        items = list(chunks)
        for chunk in items:
            if chunk.config_id != config_id:
                raise ValueError(
                    f"chunk {chunk.chunk_id!r} does not belong to config {config_id!r}"
                )

        with self._conn:
            self._conn.execute("DELETE FROM chunks WHERE config_id = ?", (config_id,))
            self._conn.executemany(_INSERT_CHUNK,
                                   [
                                     (c.chunk_id, c.config_id, c.article_id, c.chunk_index, c.text)
                                     for c in items  
                                   ],
                                )

        return len(items)


    def get_chunks(self, config_id: str) -> list[Chunk]:
        """ Get the chunks of a configuration. """
        rows = self._conn.execute(_SELECT_CHUNKS, (config_id,)).fetchall()
        return [
            Chunk(
                config_id=row["config_id"],
                article_id=row["article_id"],
                chunk_index=row["chunk_index"],
                text=row["text"],
            )
            for row in rows
        ]

    def config_ids(self) -> list[str]:
        """ Get the id's of existing configurations. """
        rows = self._conn.execute(
            "SELECT DISTINCT config_id FROM chunks ORDER BY config_id"
        ).fetchall()

        return [row["config_id"] for row in rows]

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
            self,
            exc_type: type[BaseException] | None,
            exc: BaseException | None,
            tb: TracebackType | None,
    ) -> None:
        self.close()
