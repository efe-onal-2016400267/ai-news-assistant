"""Markdown for `evaluate.py`: one run's report, or a comparison of two runs. No I/O."""

from collections.abc import Mapping, Sequence
from statistics import median
from typing import Any

from ai_news_assistant.rag.eval.scoring import (
    PairResult,
    QuestionScore,
    TypeSummary,
    summarize,
    summarize_by_type,
)

NOTES = (
    "n per type is 1 to 3, so per-type means are anecdotes, not measurements.",
    "Ranking is per article (first chunk of each article). "
    "Old chunk-ranked results in eval_results/*.json are not comparable.",
    "Evidence (ev@5): a quote cut by a chunk border counts as missed, "
    "and only retrieved chunks are searched.",
    "no_answer questions are not scored: retrieval always returns something. "
    "They are checked in the agent eval.",
)

Labeled = Sequence[tuple[str, PairResult]]


def render_report(
    meta: Mapping[str, Any],
    run_id: str,
    scored_version: int,
    pairs: Sequence[PairResult],
    titles: Mapping[str, str],
) -> str:
    """Full report for one run folder."""
    cfg = meta["config"]
    questions = pairs[0].questions if pairs else ()
    n_scored = sum(q.scored for q in questions)
    version = f"v{meta['dataset_version']}"
    if scored_version != meta["dataset_version"]:
        version += f" at run time, scored with v{scored_version}"

    lines = [
        f"# RAG eval: {meta['experiment']}",
        "",
        f"- Run: `{run_id}`, created {meta['created_at']}",
        f"- Split: {meta['split']}; dataset {version}",
        f"- Questions: n={len(questions)} ({n_scored} scored, "
        f"{len(questions) - n_scored} no_answer not scored)",
        f"- Embedding model: `{meta['embedding_model']}`; ranking cut at "
        f"{cfg['max_articles']} articles; {cfg['max_articles'] * cfg['chunk_overfetch']} "
        f"chunks asked per retriever; hybrid rrf_k={cfg['hybrid']['rrf_k']}, "
        f"depth={cfg['hybrid']['depth']}",
        "",
        "## Summary (scored questions)",
        "",
        *render_summary([(p.label, p) for p in pairs]),
        "",
        "## First gold rank per question (`-` not found, `n/a` not scored)",
        "",
        *render_matrix([(p.label, p) for p in pairs]),
        "",
    ]
    for pair in pairs:
        lines += [f"## {pair.label}", "", *_type_table(summarize_by_type(pair)), ""]
        lines += [*_question_table(pair.questions, titles), ""]
    lines += ["## Notes", "", *_notes_section(n_scored)]
    return "\n".join(lines) + "\n"


def render_compare(
    label_a: str,
    pairs_a: Sequence[PairResult],
    label_b: str,
    pairs_b: Sequence[PairResult],
    scored_version: int,
) -> str:
    """Two runs scored with the same labels: summaries, rank matrix, R@5 changes."""
    labeled = [(f"A: {p.label}", p) for p in pairs_a] + [(f"B: {p.label}", p) for p in pairs_b]
    n_scored = sum(q.scored for q in pairs_a[0].questions) if pairs_a else 0
    lines = [
        "# RAG eval comparison",
        "",
        f"- A: `{label_a}`",
        f"- B: `{label_b}`",
        f"- Both scored with dataset v{scored_version}",
        "",
        "## Summary (scored questions)",
        "",
        *render_summary(labeled),
        "",
        "## First gold rank per question (`-` not found, `n/a` not scored)",
        "",
        *render_matrix(labeled),
        "",
        "## R@5 changes (same retriever and config in A and B)",
        "",
        *_recall_changes(pairs_a, pairs_b),
        "",
        "## Notes",
        "",
        *_notes_section(n_scored),
    ]
    return "\n".join(lines) + "\n"


def _notes_section(n_scored: int) -> list[str]:
    notes = list(NOTES)
    if n_scored:
        notes.insert(
            1,
            f"With {n_scored} scored questions, one question is {1 / n_scored:.2f} of mean "
            "recall: treat differences of one or two questions as ties.",
        )
    return [f"- {note}" for note in notes]


def render_summary(labeled: Labeled) -> list[str]:
    lines = [
        "| Pair | n scored | R@5 | R@10 | MRR | Found in ranking | Articles min / median |",
        "|---|---|---|---|---|---|---|",
    ]
    for label, pair in labeled:
        s = summarize(pair.questions)
        counts = [q.n_articles for q in pair.questions] or [0]
        lines.append(
            f"| {label} | {s.n_scored} | {_fmt(s.recall_5)} | {_fmt(s.recall_10)} | "
            f"{_fmt(s.mrr)} | {s.found_at_max}/{s.n_scored} | "
            f"{min(counts)} / {median(counts):g} |"
        )
    return lines


def render_matrix(labeled: Labeled) -> list[str]:
    """First gold rank: one row per question, one column per pair."""
    cells: dict[str, dict[str, str]] = {}
    types: dict[str, str] = {}
    for label, pair in labeled:
        for q in pair.questions:
            types.setdefault(q.question_id, q.type)
            cells.setdefault(q.question_id, {})[label] = _rank_cell(q)

    labels = [label for label, _ in labeled]
    lines = [
        "| Question | Type | " + " | ".join(labels) + " |",
        "|---|---|" + "---|" * len(labels),
    ]
    for qid, row in cells.items():
        lines.append(
            f"| {qid} | {types[qid]} | " + " | ".join(row.get(label, "") for label in labels) + " |"
        )
    return lines


def _type_table(summaries: Sequence[TypeSummary]) -> list[str]:
    lines = ["| Type | n | n scored | R@5 | R@10 | MRR |", "|---|---|---|---|---|---|"]
    for s in summaries:
        lines.append(
            f"| {s.type} | {s.n} | {s.n_scored} | {_fmt(s.recall_5)} | "
            f"{_fmt(s.recall_10)} | {_fmt(s.mrr)} |"
        )
    return lines


def _question_table(questions: Sequence[QuestionScore], titles: Mapping[str, str]) -> list[str]:
    lines = [
        "| Question | Type | Rank | R@5 | R@10 | ev@5 | Notes | Top articles (misses at 5) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for q in questions:
        if not q.scored:
            top = q.top_articles[:1]
            shown = "; ".join(_title(a, titles) for a in top)
            score = f", score {q.top_score:.3f}" if q.top_score is not None else ""
            lines.append(
                f"| {q.question_id} | {q.type} | n/a | not scored | not scored | n/a | "
                f"{', '.join(_notes(q))} | {shown}{score} |"
            )
            continue
        missed = q.recall_5 is not None and q.recall_5 < 1.0
        shown = "; ".join(_title(a, titles) for a in q.top_articles) if missed else ""
        lines.append(
            f"| {q.question_id} | {q.type} | {_rank_cell(q)} | {_fmt(q.recall_5)} | "
            f"{_fmt(q.recall_10)} | {_fmt(q.evidence_5)} | {', '.join(_notes(q))} | {shown} |"
        )
    return lines


def _notes(q: QuestionScore) -> list[str]:
    notes = []
    if q.stale_in_top_5 is not None:
        notes.append(f"stale@5={q.stale_in_top_5}, gold>stale={_yes_no(q.gold_above_stale)}")
    if q.hard_negative_above_gold is not None:
        notes.append("hard neg above gold" if q.hard_negative_above_gold else "hard neg below gold")
    if q.gold_copies_in_top_5 is not None:
        notes.append(f"copies@5={q.gold_copies_in_top_5}")
    if q.filter is not None:
        notes.append(f"filter: {q.filter}")
    return notes


def _recall_changes(pairs_a: Sequence[PairResult], pairs_b: Sequence[PairResult]) -> list[str]:
    by_key_b = {(p.retriever, p.config_id): p for p in pairs_b}
    matched = [
        (a, by_key_b[(a.retriever, a.config_id)])
        for a in pairs_a
        if (a.retriever, a.config_id) in by_key_b
    ]
    if not matched:
        return ["No pair is in both runs."]
    lines = []
    for a, b in matched:
        scores_b = {q.question_id: q for q in b.questions}
        for qa in a.questions:
            qb = scores_b.get(qa.question_id)
            if qb is not None and qa.scored and qa.recall_5 != qb.recall_5:
                lines.append(
                    f"- {qa.question_id} ({qa.type}), {a.label}: R@5 {_fmt(qa.recall_5)} -> "
                    f"{_fmt(qb.recall_5)} (rank {_rank_cell(qa)} -> {_rank_cell(qb)})"
                )
    return lines or ["No R@5 changes."]


def _rank_cell(q: QuestionScore) -> str:
    if not q.scored:
        return "n/a"
    return "-" if q.first_gold_rank is None else str(q.first_gold_rank)


def _title(article_id: str, titles: Mapping[str, str]) -> str:
    return titles.get(article_id, article_id[:12]).replace("|", "\\|")


def _fmt(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"


def _yes_no(value: bool | None) -> str:
    return "n/a" if value is None else ("yes" if value else "no")
