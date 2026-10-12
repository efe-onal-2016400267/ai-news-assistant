import sqlite3
from collections import Counter
from pathlib import Path

from ai_news_assistant.rag.store import RagStore

SNAPSHOT = Path("data/snapshots/news-2026-10-12.db")
HF_SOURCE = "Hugging Face"
HF_CUTOFF = "2026-05-15"

_OTHER_IDS = """
SELECT id FROM articles
WHERE extraction_successful = 1 AND source != ?
"""

_HF_IDS = """
SELECT id FROM articles
WHERE published_at >= ? AND source = ?
"""


def main() -> None:
    conn = sqlite3.connect(SNAPSHOT)

    # Get the id's to be put into the corpus
    ids = [r[0] for r in conn.execute(_OTHER_IDS, (HF_SOURCE,))]
    ids += [
        r[0]
        for r in conn.execute(
            _HF_IDS,
            (
                HF_CUTOFF,
                HF_SOURCE,
            ),
        )
    ]
    conn.close()

    # Create the corpus
    with RagStore(SNAPSHOT) as store:
        n = store.replace_corpus(ids)
        articles = store.corpus_articles()

    print(f"corpus: {n} articles")

    # Print the sources and their article counts
    for source, count in Counter(a.source for a in articles).most_common():
        print(f"    {source:<25} {count}")

    # Print the oldest HuggingFace date for filter check
    hf_dates = [a.published_at for a in articles if a.source == HF_SOURCE and a.published_at]
    if hf_dates:
        print(f"Oldest Hugging Face article kept: {min(hf_dates)}")


if __name__ == "__main__":
    main()
