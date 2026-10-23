"""Score saved hits per question and summarize by question type (used by `evaluate.py`)."""

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import mean
from typing import get_args

from ai_news_assistant.rag.base import Hit
from ai_news_assistant.rag.eval.dataset import GoldMode, Question, QuestionFilter, QuestionType
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

RECALL_KS = (5, 10)
EXTRA_K = 5  # k for stale, duplicate copies and evidence
TOP_SHOWN = 3  # articles kept per question to show for misses
TYPE_ORDER: tuple[str, ...] = get_args(QuestionType)


@dataclass(frozen=True)
class QuestionScore:
    question_id: str
    type: QuestionType
    gold_mode: GoldMode
    scored: bool  # False for no_answer: retrieval cannot abstain
    n_articles: int  # Length of the article ranking after the max_articles cut
    recall_5: float | None
    recall_10: float | None
    mrr: float | None
    first_gold_rank: int | None
    evidence_5: float | None
    stale_in_top_5: int | None  # None: no stale ids
    gold_above_stale: bool | None
    hard_negative_above_gold: bool | None  # None: no hard negatives
    gold_copies_in_top_5: int | None  # None unless gold_mode "any" with 2+ gold
    top_articles: tuple[str, ...]
    top_score: float | None  # Score of the first hit, to read no_answer runs
    filter: str | None  # e.g. "source=The Decoder"


@dataclass(frozen=True)
class PairResult:
    """Scores of one (retriever, chunking config) pair."""

    retriever: str
    config_id: str
    questions: tuple[QuestionScore, ...]

    @property
    def label(self) -> str:
        return f"{self.retriever} / {self.config_id}"


@dataclass(frozen=True)
class TypeSummary:
    type: str  # A question type, or "all"
    n: int
    n_scored: int
    recall_5: float | None  # Means over scored questions; None when none is scored
    recall_10: float | None
    mrr: float | None
    found_at_max: int  # Scored questions with a gold article anywhere in the ranking


def score_question(question: Question, hits: Sequence[Hit], max_articles: int) -> QuestionScore:
    """Score one question's hits, ranked by article and cut at `max_articles`."""
    ranking = article_ranking(hits)[:max_articles]
    kept = set(ranking)
    kept_hits = [hit for hit in hits if hit.article_id in kept]
    gold = question.gold_article_ids
    stale = question.stale_article_ids
    k5, k10 = RECALL_KS

    return QuestionScore(
        question_id=question.id,
        type=question.type,
        gold_mode=question.gold_mode,
        scored=bool(gold),
        n_articles=len(ranking),
        recall_5=recall_at_k(ranking, gold, question.gold_mode, k5),
        recall_10=recall_at_k(ranking, gold, question.gold_mode, k10),
        mrr=reciprocal_rank(ranking, gold),
        first_gold_rank=first_gold_rank(ranking, gold),
        evidence_5=evidence_coverage(kept_hits, question.evidence, EXTRA_K),
        stale_in_top_5=stale_in_top_k(ranking, stale, EXTRA_K) if stale else None,
        gold_above_stale=gold_above_stale(ranking, gold, stale),
        hard_negative_above_gold=hard_negative_above_gold(
            ranking, gold, question.hard_negative_article_ids
        ),
        gold_copies_in_top_5=gold_copies_in_top_k(ranking, gold, EXTRA_K)
        if question.gold_mode == "any" and len(gold) >= 2
        else None,
        top_articles=tuple(ranking[:TOP_SHOWN]),
        top_score=hits[0].score if hits else None,
        filter=_filter_text(question.filter),
    )


def summarize(questions: Sequence[QuestionScore], label: str = "all") -> TypeSummary:
    """Means over the scored questions in `questions`."""
    scored = [q for q in questions if q.scored]
    return TypeSummary(
        type=label,
        n=len(questions),
        n_scored=len(scored),
        recall_5=_mean([q.recall_5 for q in scored]),
        recall_10=_mean([q.recall_10 for q in scored]),
        mrr=_mean([q.mrr for q in scored]),
        found_at_max=sum(q.first_gold_rank is not None for q in scored),
    )


def summarize_by_type(pair: PairResult) -> list[TypeSummary]:
    """One summary per question type present, in `QuestionType` order."""
    return [
        summarize(group, label=q_type)
        for q_type in TYPE_ORDER
        if (group := [q for q in pair.questions if q.type == q_type])
    ]


def _mean(values: Sequence[float | None]) -> float | None:
    present = [v for v in values if v is not None]
    return mean(present) if present else None


def _filter_text(qf: QuestionFilter | None) -> str | None:
    if qf is None:
        return None
    parts = []
    if qf.source is not None:
        parts.append(f"source={qf.source}")
    if qf.since is not None:
        parts.append(f"since={qf.since.isoformat()}")
    return ", ".join(parts)
