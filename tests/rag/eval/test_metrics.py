import pytest

from ai_news_assistant.rag.base import Hit
from ai_news_assistant.rag.eval.dataset import Evidence
from ai_news_assistant.rag.eval.metrics import (
    article_ranking,
    evidence_coverage,
    first_gold_rank,
    gold_above_stale,
    gold_copies_in_top_k,
    hard_negative_above_gold,
    recall_at_k,
    reciprocal_rank,
    stale_in_top_k,
)


def hits(*article_ids: str, text: str = "chunk text") -> list[Hit]:
    """One chunk per argument, best first; repeated ids are chunks of the same article."""
    return [
        Hit(chunk_id=f"{a}-{i}", article_id=a, score=1.0 / (i + 1), text=text)
        for i, a in enumerate(article_ids)
    ]


def hit(article_id: str, text: str) -> Hit:
    return Hit(chunk_id=f"{article_id}-{text[:8]}", article_id=article_id, score=1.0, text=text)


# article_ranking


def test_article_ranking_collapses_chunks_in_first_seen_order() -> None:
    assert article_ranking(hits("a", "a", "b", "a", "c")) == ["a", "b", "c"]


def test_article_ranking_empty() -> None:
    assert article_ranking([]) == []


# recall_at_k


def test_recall_any_counts_rank_k_not_k_plus_one() -> None:
    ranking = ["x", "y", "g"]

    assert recall_at_k(ranking, {"g"}, "any", k=3) == 1.0
    assert recall_at_k(ranking, {"g"}, "any", k=2) == 0.0


def test_recall_any_is_one_with_several_gold_found() -> None:
    assert recall_at_k(["g1", "g2"], {"g1", "g2", "g3"}, "any", k=5) == 1.0


def test_recall_all_is_fraction_found() -> None:
    assert recall_at_k(["g1", "x", "g2"], {"g1", "g2", "g3"}, "all", k=5) == pytest.approx(2 / 3)


def test_recall_empty_gold_is_none() -> None:
    assert recall_at_k(["x"], set(), "any", k=5) is None


def test_recall_duplicate_chunks_do_not_push_gold_out() -> None:
    # Ranked by chunk, three chunks of "a" would fill the top 2 and miss "g".
    ranking = article_ranking(hits("a", "a", "a", "g"))

    assert recall_at_k(ranking, {"g"}, "any", k=2) == 1.0


@pytest.mark.parametrize("k", [0, -1])
def test_non_positive_k_raises(k: int) -> None:
    with pytest.raises(ValueError, match="k must be positive"):
        recall_at_k(["g"], {"g"}, "any", k=k)
    with pytest.raises(ValueError, match="k must be positive"):
        stale_in_top_k(["g"], {"g"}, k=k)
    with pytest.raises(ValueError, match="k must be positive"):
        gold_copies_in_top_k(["g"], {"g"}, k=k)
    with pytest.raises(ValueError, match="k must be positive"):
        evidence_coverage(hits("g"), [Evidence("g", "chunk")], k=k)


# first_gold_rank, reciprocal_rank


def test_gold_at_rank_three() -> None:
    ranking = ["x", "y", "g", "g2"]

    assert first_gold_rank(ranking, {"g", "g2"}) == 3
    assert reciprocal_rank(ranking, {"g", "g2"}) == pytest.approx(1 / 3)


def test_gold_not_retrieved() -> None:
    assert first_gold_rank(["x", "y"], {"g"}) is None
    assert reciprocal_rank(["x", "y"], {"g"}) == 0.0


def test_rank_with_empty_gold_is_none() -> None:
    assert first_gold_rank(["x"], set()) is None
    assert reciprocal_rank(["x"], set()) is None


# stale


@pytest.mark.parametrize(
    ("ranking", "expected"),
    [
        pytest.param(["s", "g"], False, id="stale-above-gold"),
        pytest.param(["g", "s"], True, id="gold-above-stale"),
        pytest.param(["x", "g"], True, id="stale-not-retrieved"),
        pytest.param(["s", "x"], False, id="gold-not-retrieved"),
        pytest.param(["x"], False, id="neither-retrieved"),
    ],
)
def test_gold_above_stale(ranking: list[str], expected: bool) -> None:
    assert gold_above_stale(ranking, {"g"}, {"s"}) is expected


def test_gold_above_stale_without_stale_or_gold_is_none() -> None:
    assert gold_above_stale(["g"], {"g"}, set()) is None
    assert gold_above_stale(["s"], set(), {"s"}) is None


def test_stale_in_top_k_counts_only_top_k() -> None:
    ranking = ["s1", "g", "s2", "s3"]

    assert stale_in_top_k(ranking, {"s1", "s2", "s3"}, k=3) == 2
    assert stale_in_top_k(ranking, set(), k=3) == 0


# hard negatives


@pytest.mark.parametrize(
    ("ranking", "expected"),
    [
        pytest.param(["n", "g"], True, id="negative-above-gold"),
        pytest.param(["g", "n"], False, id="negative-below-gold"),
        pytest.param(["n", "x"], True, id="negative-without-gold"),
        pytest.param(["x", "y"], False, id="neither-retrieved"),
        pytest.param(["g"], False, id="only-gold"),
    ],
)
def test_hard_negative_above_gold(ranking: list[str], expected: bool) -> None:
    assert hard_negative_above_gold(ranking, {"g"}, {"n"}) is expected


def test_hard_negative_without_negatives_or_gold_is_none() -> None:
    assert hard_negative_above_gold(["n", "g"], {"g"}, set()) is None
    assert hard_negative_above_gold(["n"], set(), {"n"}) is None


# gold_copies_in_top_k


def test_gold_copies_counts_distinct_gold_in_top_k() -> None:
    ranking = article_ranking(hits("g1", "g1", "x", "g2", "g3"))  # [g1, x, g2, g3]

    assert gold_copies_in_top_k(ranking, {"g1", "g2", "g3"}, k=3) == 2
    assert gold_copies_in_top_k(ranking, {"g1", "g2", "g3"}, k=10) == 3


# evidence_coverage


def test_evidence_found() -> None:
    found = [hit("g", "Intro. The model has 41B active parameters. More.")]

    assert evidence_coverage(found, [Evidence("g", "41B active parameters")], k=5) == 1.0


def test_evidence_matches_across_different_whitespace() -> None:
    found = [hit("g", "The model has\n41B   active\tparameters.")]

    assert evidence_coverage(found, [Evidence("g", "41B active  parameters")], k=5) == 1.0


def test_evidence_split_over_two_chunks_is_missed() -> None:
    # Documented limit: a quote cut by a chunk border counts as missed.
    found = [hit("g", "The model has 41B"), hit("g", "active parameters.")]

    assert evidence_coverage(found, [Evidence("g", "41B active parameters")], k=5) == 0.0


def test_evidence_in_article_outside_top_k_is_missed() -> None:
    found = [hit("x", "other"), hit("y", "other"), hit("g", "41B active parameters")]

    assert evidence_coverage(found, [Evidence("g", "41B active parameters")], k=2) == 0.0


def test_evidence_in_another_articles_chunk_does_not_count() -> None:
    found = [hit("g", "unrelated"), hit("copy", "41B active parameters")]

    assert evidence_coverage(found, [Evidence("g", "41B active parameters")], k=5) == 0.0


def test_evidence_half_found() -> None:
    found = [hit("g1", "first quote here"), hit("g2", "nothing useful")]
    evidence = [Evidence("g1", "first quote"), Evidence("g2", "second quote")]

    assert evidence_coverage(found, evidence, k=5) == 0.5


def test_no_evidence_is_none() -> None:
    assert evidence_coverage(hits("g"), [], k=5) is None
