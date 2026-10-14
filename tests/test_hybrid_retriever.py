from collections.abc import Iterator
from pathlib import Path

import pytest
from conftest import AI_ACT_FIRST_CHUNK, CONFIG_ID

from ai_news_assistant.rag.base import Hit, Retriever
from ai_news_assistant.rag.bm25_retriever import BM25Retriever
from ai_news_assistant.rag.hybrid_retriever import HybridRetriever, rrf_fuse

QUERY = "EU lawmakers approve rules for artificial intelligence"


def hit(chunk_id: str, score: float = 1.0, text: str = "") -> Hit:
    return Hit(chunk_id=chunk_id, article_id=f"article-{chunk_id}", score=score, text=text)


def ids(hits: list[Hit]) -> list[str]:
    return [h.chunk_id for h in hits]


class FakeRetriever:
    """Returns a fixed ranking, cut to k, and records every call."""

    def __init__(self, ranking: list[Hit]) -> None:
        self.ranking = ranking
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, k: int) -> list[Hit]:
        self.calls.append((query, k))
        return self.ranking[:k]


# rrf_fuse


def test_single_ranking_keeps_order_with_rrf_scores() -> None:
    fused = rrf_fuse([[hit("a"), hit("b"), hit("c")]])

    assert ids(fused) == ["a", "b", "c"]
    assert [h.score for h in fused] == pytest.approx([1 / 61, 1 / 62, 1 / 63])


def test_chunk_in_two_rankings_gets_summed_score() -> None:
    fused = rrf_fuse([[hit("a")], [hit("b"), hit("a")]])

    assert fused[0].chunk_id == "a"
    assert fused[0].score == pytest.approx(1 / 61 + 1 / 62)


def test_agreement_beats_a_single_top_rank() -> None:
    fused = rrf_fuse([[hit("a"), hit("b")], [hit("c"), hit("b")]])

    assert ids(fused) == ["b", "a", "c"]


def test_raw_scores_are_ignored() -> None:
    small = rrf_fuse([[hit("a", 0.9), hit("b", 0.8)], [hit("b", 0.7), hit("c", 0.1)]])
    large = rrf_fuse([[hit("a", 25.0), hit("b", 0.01)], [hit("b", 300.0), hit("c", 2.0)]])

    assert ids(small) == ids(large)
    assert [h.score for h in small] == pytest.approx([h.score for h in large])


def test_rrf_k_is_used() -> None:
    fused = rrf_fuse([[hit("a"), hit("b")]], rrf_k=0)

    assert [h.score for h in fused] == pytest.approx([1.0, 0.5])


def test_ties_keep_first_seen_order() -> None:
    fused = rrf_fuse([[hit("a")], [hit("b")]])

    assert ids(fused) == ["a", "b"]
    assert fused[0].score == fused[1].score


def test_fused_hit_keeps_fields_of_first_appearance() -> None:
    first = Hit(chunk_id="a", article_id="art-1", score=0.9, text="first text")
    second = Hit(chunk_id="a", article_id="art-2", score=12.0, text="second text")

    [fused] = rrf_fuse([[first], [second]])

    assert fused.chunk_id == "a"
    assert fused.article_id == "art-1"
    assert fused.text == "first text"
    assert fused.score == pytest.approx(2 / 61)


@pytest.mark.parametrize("rankings", [[], [[], []]])
def test_no_hits_fuse_to_empty(rankings: list[list[Hit]]) -> None:
    assert rrf_fuse(rankings) == []


def test_accepts_a_generator() -> None:
    def rankings() -> Iterator[list[Hit]]:
        yield [hit("a")]
        yield [hit("b"), hit("a")]

    assert ids(rrf_fuse(rankings())) == ["a", "b"]


# HybridRetriever


def test_is_a_retriever() -> None:
    assert isinstance(HybridRetriever([]), Retriever)


@pytest.mark.parametrize(("kwargs", "depth"), [({}, 50), ({"depth": 7}, 7)])
def test_each_retriever_is_asked_for_depth_hits(kwargs: dict[str, int], depth: int) -> None:
    first, second = FakeRetriever([hit("a")]), FakeRetriever([hit("b")])

    HybridRetriever([first, second], **kwargs).search(QUERY, 3)

    assert first.calls == [(QUERY, depth)]
    assert second.calls == [(QUERY, depth)]


def test_result_is_fused_and_cut_to_k() -> None:
    ranking_1 = [hit("a"), hit("b"), hit("c")]
    ranking_2 = [hit("c"), hit("d"), hit("a")]
    retriever = HybridRetriever([FakeRetriever(ranking_1), FakeRetriever(ranking_2)])

    hits = retriever.search(QUERY, 2)

    assert hits == rrf_fuse([ranking_1, ranking_2])[:2]
    assert len(hits) == 2


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_returns_empty_without_searching(k: int) -> None:
    fake = FakeRetriever([hit("a")])

    assert HybridRetriever([fake]).search(QUERY, k) == []
    assert fake.calls == []


def test_one_empty_retriever_keeps_the_other_order() -> None:
    retriever = HybridRetriever([FakeRetriever([]), FakeRetriever([hit("a"), hit("b")])])

    hits = retriever.search(QUERY, 5)

    assert ids(hits) == ["a", "b"]
    assert [h.score for h in hits] == pytest.approx([1 / 61, 1 / 62])


def test_no_retrievers_returns_empty() -> None:
    assert HybridRetriever([]).search(QUERY, 5) == []


def test_rrf_k_is_passed_to_fusion() -> None:
    retriever = HybridRetriever([FakeRetriever([hit("a"), hit("b")])], rrf_k=0)

    assert [h.score for h in retriever.search(QUERY, 2)] == pytest.approx([1.0, 0.5])


def test_works_with_bm25_retriever(rag_db: Path) -> None:
    ai_act = f"{CONFIG_ID}:ai-act:0"
    other = FakeRetriever([hit("elsewhere"), hit(ai_act)])
    retriever = HybridRetriever([BM25Retriever(rag_db, CONFIG_ID), other])

    top = retriever.search(QUERY, 3)[0]

    assert top.chunk_id == ai_act
    assert top.article_id == "ai-act"
    assert top.text == AI_ACT_FIRST_CHUNK
    assert top.score == pytest.approx(1 / 61 + 1 / 62)
