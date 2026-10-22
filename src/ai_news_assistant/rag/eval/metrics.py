"""Retrieval metrics over saved hits. Pure functions, no I/O.

The retrievers return chunks, so one article can fill several slots. Metrics rank
articles instead: each article at the position of its first chunk, as
`search_articles` shows them to the agent. A `ranking` is a list of article ids,
best first; ranks are 1-based.
"""

from collections.abc import Collection, Sequence

from ai_news_assistant.rag.base import Hit
from ai_news_assistant.rag.eval.dataset import Evidence, GoldMode


def article_ranking(hits: Sequence[Hit]) -> list[str]:
    """Article ids in the order of their first chunk."""
    return list(dict.fromkeys(hit.article_id for hit in hits))


def recall_at_k(ranking: list[str], gold: Collection[str], mode: GoldMode, k: int) -> float | None:
    """`any`: 1.0 if a gold article is in the top k. `all`: fraction of gold found.

    None for empty gold (no_answer questions are not scored).
    """
    _check_k(k)
    if not gold:
        return None
    found = len(set(ranking[:k]) & set(gold))
    if mode == "any":
        return 1.0 if found else 0.0
    return found / len(set(gold))


def first_gold_rank(ranking: list[str], gold: Collection[str]) -> int | None:
    """Best rank of a gold article; None if none was retrieved or gold is empty."""
    return _first_rank(ranking, gold)


def reciprocal_rank(ranking: list[str], gold: Collection[str]) -> float | None:
    """1 / first gold rank, 0.0 if no gold was retrieved, None for empty gold."""
    if not gold:
        return None
    rank = _first_rank(ranking, gold)
    return 0.0 if rank is None else 1 / rank


def stale_in_top_k(ranking: list[str], stale: Collection[str], k: int) -> int:
    """Number of stale articles in the top k."""
    _check_k(k)
    return len(set(ranking[:k]) & set(stale))


def gold_above_stale(
    ranking: list[str], gold: Collection[str], stale: Collection[str]
) -> bool | None:
    """True when the first gold ranks above every stale article (or no stale was retrieved).

    False when no gold was retrieved. None when the question has no stale or no gold ids.
    """
    if not gold or not stale:
        return None
    gold_rank = _first_rank(ranking, gold)
    if gold_rank is None:
        return False
    stale_rank = _first_rank(ranking, stale)
    return stale_rank is None or gold_rank < stale_rank


def hard_negative_above_gold(
    ranking: list[str], gold: Collection[str], negatives: Collection[str]
) -> bool | None:
    """True (the bad case) when a hard negative ranks above the first gold article.

    Also True when a negative was retrieved and no gold was. None when the question
    has no negatives or no gold ids.
    """
    if not gold or not negatives:
        return None
    negative_rank = _first_rank(ranking, negatives)
    if negative_rank is None:
        return False
    gold_rank = _first_rank(ranking, gold)
    return gold_rank is None or negative_rank < gold_rank


def gold_copies_in_top_k(ranking: list[str], gold: Collection[str], k: int) -> int:
    """Number of distinct gold articles in the top k (copies of a duplicate story)."""
    _check_k(k)
    return len(set(ranking[:k]) & set(gold))


def evidence_coverage(hits: Sequence[Hit], evidence: Sequence[Evidence], k: int) -> float | None:
    """Share of evidence quotes found in a retrieved chunk of their article, in the top k.

    Whitespace is normalized on both sides. Limits: a quote cut by a chunk border
    counts as missed, and only retrieved chunks count (other chunks of a top-k
    article are not looked up). None when there is no evidence.
    """
    _check_k(k)
    if not evidence:
        return None
    top = set(article_ranking(hits)[:k])
    texts: dict[str, list[str]] = {}
    for hit in hits:
        if hit.article_id in top:
            texts.setdefault(hit.article_id, []).append(_normalize(hit.text))

    found = sum(
        any(_normalize(e.quote) in text for text in texts.get(e.article_id, [])) for e in evidence
    )
    return found / len(evidence)


def _first_rank(ranking: list[str], ids: Collection[str]) -> int | None:
    wanted = set(ids)
    for rank, article_id in enumerate(ranking, start=1):
        if article_id in wanted:
            return rank
    return None


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _check_k(k: int) -> None:
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
