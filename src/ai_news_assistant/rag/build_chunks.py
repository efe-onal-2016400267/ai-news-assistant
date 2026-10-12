import argparse
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path

from ai_news_assistant.rag.chunkers import chunk_recursive, chunk_words, tokenizer
from ai_news_assistant.rag.store import Chunk, RagStore

DEFAULT_DB = Path("data/snapshots/news-2026-10-12.db")
SIZE = 256
OVERLAP = 32

# config_id -> chunker. The name carries the parameters
# recursive-tok256-32, words-tok256-32
CONFIGS: dict[str, Callable[[str, int, int], list[str]]] = {
    f"words-tok{SIZE}-{OVERLAP}": chunk_words,
    f"recursive-tok{SIZE}-{OVERLAP}": chunk_recursive,
}


def build(db: Path, config_id: str) -> None:
    """Build the chunks based on the configuration."""
    chunker = CONFIGS[config_id]
    with RagStore(db) as store:
        chunks = []
        for article in store.corpus_articles():
            for i, chunk in enumerate(chunker(article.text, SIZE, OVERLAP)):
                chunks.append(
                    Chunk(config_id=config_id, article_id=article.id, chunk_index=i, text=chunk)
                )
        store.replace_chunks(config_id=config_id, chunks=chunks)
    print_stats(config_id=config_id, chunks=chunks)


def print_stats(config_id: str, chunks: list[Chunk]) -> None:
    """Print the stats of the chunking operation."""
    print(f"{config_id}: {len(chunks)} chunks.")
    if not chunks:
        return

    # Print mean, min, max number of tokens in each chunk
    tokens = [len(tokenizer.encode(c.text, add_special_tokens=False)) for c in chunks]
    print(f"    tokens: mean {statistics.mean(tokens):.1f}, min {min(tokens)}, max {max(tokens)}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Chunk the corpus and store the chunks.")
    parser.add_argument("--config", choices=sorted(CONFIGS), required=True)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)

    args = parser.parse_args(argv)
    build(args.db, args.config)


if __name__ == "__main__":
    main()
