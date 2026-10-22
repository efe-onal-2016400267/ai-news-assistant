"""Load and check the RAG eval questions (`rag_questions_v*.toml`, design D18)."""

import tomllib
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Literal, cast, get_args

QuestionType = Literal[
    "specific_detail",
    "entity_version",
    "number_or_date",
    "rare_term",
    "paraphrase",
    "multi_article",
    "multi_chunk",
    "duplicate",
    "hard_negative",
    "time_sensitive",
    "filtered",
    "no_answer",
]
GoldMode = Literal["any", "all"]
Split = Literal["dev", "test"]

QUESTION_TYPES = frozenset(get_args(QuestionType))
GOLD_MODES = frozenset(get_args(GoldMode))
SPLITS = frozenset(get_args(Split))

TOP_LEVEL_KEYS = frozenset({"dataset_version", "question"})
REQUIRED_KEYS = frozenset({"id", "type", "split", "question", "gold_article_ids", "gold_mode"})
# legacy_type is the type name from question_list.py; read and dropped
OPTIONAL_KEYS = frozenset(
    {"stale_article_ids", "hard_negative_article_ids", "evidence", "filter", "legacy_type"}
)
EVIDENCE_KEYS = frozenset({"article_id", "quote"})
FILTER_KEYS = frozenset({"source", "since"})


@dataclass(frozen=True)
class Evidence:
    """A verbatim quote from a gold article that supports the answer."""

    article_id: str
    quote: str


@dataclass(frozen=True)
class QuestionFilter:
    """Search filter a question must be asked with. At least one field is set."""

    source: str | None = None  # Exact source name
    since: date | None = None  # Inclusive


@dataclass(frozen=True)
class Question:
    id: str
    type: QuestionType
    split: Split
    question: str
    gold_article_ids: tuple[str, ...]  # Empty only for no_answer
    gold_mode: GoldMode  # any: one gold article is enough; all: recall is the fraction found
    stale_article_ids: tuple[str, ...] = ()
    hard_negative_article_ids: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    filter: QuestionFilter | None = None


@dataclass(frozen=True)
class Dataset:
    version: int
    questions: tuple[Question, ...]


def load_questions(path: Path, split: Split | None = None) -> Dataset:
    """Load and check the questions file, then keep only `split` (None = all).

    The whole file is checked before filtering, so rules across splits hold
    even when only one split is asked for. Raises `ValueError` on any problem.
    """
    with path.open("rb") as f:
        data = tomllib.load(f)

    unknown = set(data) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown top-level key(s): {', '.join(sorted(unknown))}")
    version = data.get("dataset_version")
    if not isinstance(version, int) or isinstance(version, bool) or version < 1:
        raise ValueError(f"{path}: dataset_version must be a positive integer")
    raw_questions = data.get("question")
    if not isinstance(raw_questions, list) or not raw_questions:
        raise ValueError(f"{path}: no [[question]] tables")

    questions = [parse_question(raw) for raw in raw_questions]
    _check_unique_ids(questions)
    _check_split_overlap(questions)

    if split is not None:
        questions = [q for q in questions if q.split == split]
    return Dataset(version=version, questions=tuple(questions))


def parse_question(raw: dict[str, Any]) -> Question:
    """Turn one `[[question]]` table into a `Question`, checking its own rules."""
    qid = raw.get("id")
    if not isinstance(qid, str) or not qid:
        raise ValueError(f"question without a string id (keys: {', '.join(sorted(raw))})")

    missing = REQUIRED_KEYS - set(raw)
    if missing:
        raise ValueError(f"{qid}: missing key(s): {', '.join(sorted(missing))}")
    unknown = set(raw) - REQUIRED_KEYS - OPTIONAL_KEYS
    if unknown:
        raise ValueError(f"{qid}: unknown key(s): {', '.join(sorted(unknown))}")

    text = raw["question"]
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{qid}: question must be a non-empty string")

    q_type = cast(QuestionType, _choice(raw, "type", QUESTION_TYPES, qid))
    gold = _id_list(raw, "gold_article_ids", qid)
    stale = _id_list(raw, "stale_article_ids", qid)
    negatives = _id_list(raw, "hard_negative_article_ids", qid)
    evidence = _evidence(raw.get("evidence", []), qid)

    if q_type == "no_answer":
        if gold or evidence:
            raise ValueError(f"{qid}: no_answer must have no gold articles and no evidence")
    elif not gold:
        raise ValueError(f"{qid}: needs at least one gold article")

    evidence_ids = {e.article_id for e in evidence}
    not_gold = evidence_ids - set(gold)
    if not_gold:
        raise ValueError(f"{qid}: evidence for non-gold article(s): {', '.join(sorted(not_gold))}")
    no_quote = set(gold) - evidence_ids
    if no_quote:
        raise ValueError(f"{qid}: gold article(s) without evidence: {', '.join(sorted(no_quote))}")
    for name, ids in (("stale", stale), ("hard negative", negatives)):
        both = set(gold) & set(ids)
        if both:
            raise ValueError(f"{qid}: gold article(s) also {name}: {', '.join(sorted(both))}")

    return Question(
        id=qid,
        type=q_type,
        split=cast(Split, _choice(raw, "split", SPLITS, qid)),
        question=text,
        gold_article_ids=gold,
        gold_mode=cast(GoldMode, _choice(raw, "gold_mode", GOLD_MODES, qid)),
        stale_article_ids=stale,
        hard_negative_article_ids=negatives,
        evidence=evidence,
        filter=_filter(raw["filter"], qid) if "filter" in raw else None,
    )


def _choice(raw: dict[str, Any], key: str, allowed: frozenset[str], qid: str) -> str:
    value = raw[key]
    if value not in allowed:
        raise ValueError(f"{qid}: {key} {value!r} is not one of {', '.join(sorted(allowed))}")
    return cast(str, value)


def _id_list(raw: dict[str, Any], key: str, qid: str) -> tuple[str, ...]:
    value = raw.get(key, [])
    if not isinstance(value, list) or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{qid}: {key} must be a list of non-empty strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{qid}: {key} has repeated ids")
    return tuple(value)


def _evidence(value: Any, qid: str) -> tuple[Evidence, ...]:
    if not isinstance(value, list):
        raise ValueError(f"{qid}: evidence must be a list of tables")
    items: list[Evidence] = []
    for entry in value:
        if not isinstance(entry, dict) or set(entry) != EVIDENCE_KEYS:
            raise ValueError(f"{qid}: each evidence needs exactly article_id and quote")
        article_id, quote = entry["article_id"], entry["quote"]
        if not isinstance(article_id, str) or not isinstance(quote, str) or not quote.strip():
            raise ValueError(f"{qid}: evidence article_id and quote must be non-empty strings")
        items.append(Evidence(article_id=article_id, quote=quote))
    return tuple(items)


def _filter(value: Any, qid: str) -> QuestionFilter:
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{qid}: filter must set at least one of {', '.join(sorted(FILTER_KEYS))}")
    unknown = set(value) - FILTER_KEYS
    if unknown:
        raise ValueError(f"{qid}: unknown filter key(s): {', '.join(sorted(unknown))}")

    source = value.get("source")
    if source is not None and (not isinstance(source, str) or not source):
        raise ValueError(f"{qid}: filter source must be a non-empty string")

    since = value.get("since")
    # Accept "2026-10-01" and a bare TOML date; a datetime would hide a time of day
    if isinstance(since, str):
        try:
            since = date.fromisoformat(since)
        except ValueError:
            raise ValueError(f"{qid}: filter since {since!r} is not a YYYY-MM-DD date") from None
    elif since is not None and (not isinstance(since, date) or isinstance(since, datetime)):
        raise ValueError(f"{qid}: filter since must be a YYYY-MM-DD date")

    return QuestionFilter(source=source, since=since)


def _check_unique_ids(questions: list[Question]) -> None:
    seen: set[str] = set()
    for q in questions:
        if q.id in seen:
            raise ValueError(f"{q.id}: duplicate question id")
        seen.add(q.id)


def _check_split_overlap(questions: list[Question]) -> None:
    """No article may be labeled (gold, stale or hard negative) in both splits."""
    first_seen: dict[str, Question] = {}
    for q in questions:
        for article_id in (*q.gold_article_ids, *q.stale_article_ids, *q.hard_negative_article_ids):
            first = first_seen.setdefault(article_id, q)
            if first.split != q.split:
                raise ValueError(
                    f"{q.id}: article {article_id} is in {q.split}, "
                    f"but also in {first.split} ({first.id})"
                )
