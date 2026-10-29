import re
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from ai_news_assistant.agent.eval.dataset import Case
from ai_news_assistant.agent.eval.transcript import RunError, Transcript, TurnResult

# A case can have 4 outcomes:
Outcome = Literal["pass", "fail", "fallback", "invalid"]
FORMAT_ERRORS = frozenset({"StructuredOutputValidationError"})
SOURCES_LINE = re.compile(r"^[\s*_#>-]*sources?[\s*_]*:", re.IGNORECASE | re.MULTILINE)
DIGIT_COMMA = re.compile(r"(?<=\d),(?=\d)")


@dataclass(frozen=True)
class CheckResult:
    layer: str  # L0, L1, ...
    id: str  # id of the case
    outcome: Outcome
    detail: str  # short, never article text


@dataclass(frozen=True)
class ScoredTranscript:
    case_id: str
    run_idx: int
    results: tuple[CheckResult, ...]

    @property
    def outcome(self) -> Outcome:
        """The worst outcome among the results."""
        outcomes = {r.outcome for r in self.results}
        for worst in ("invalid", "fail", "fallback"):
            if worst in outcomes:
                return worst
        return "pass"


def check_format(transcript: Transcript, case: Case) -> CheckResult:
    """L0: Checks if the output format is good.

    - The answer parses into `text`, `citations`, and `grounded`.
        - Follows ai_news_assistant.models.ModelAnswer.
    - Citations are unique.
    - Text has no Sources:
    - Failure should name the cause.
    """

    causes: list[str] = []

    # Check for each turn:
    for i, turn in enumerate(transcript.turns, start=1):
        # 1. Check if the output is structured
        structured = turn.structured
        text = turn.text
        raw_citations = turn.raw_citations
        dropped_markers = turn.dropped_markers
        if not structured:
            causes.append(f"Model did not produce structured output. Turn:{i}.")

        # 2. Check if the text is not empty string or white space
        if text.strip() == "":
            causes.append(f"Response text is empty. Turn:{i}.")

        # 3. Check if "Sources:" like string resides in the text
        if re.findall(SOURCES_LINE, text):
            causes.append(f"Raw output contains sources. It is normally added later. Turn:{i}.")

        # 4. Check no dangling markers
        if dropped_markers:
            causes.append(f"Model produced dangling markers. Turn:{i}.")

        # 5. All citations are unique in the citations list
        if len(set(raw_citations)) != len(raw_citations):
            counts = Counter(raw_citations)
            duplicates = [item for item, n in counts.items() if n > 1]
            causes.append(f"Duplicate citations. Turn:{i}:")
            for dup in duplicates:
                causes.append(f"{dup},")

    if causes:
        return CheckResult(layer="L0", id=case.id, outcome="fail", detail="; ".join(causes))

    return CheckResult(layer="L0", id=case.id, outcome="pass", detail="Output format is correct.")


def check_grounding(transcript: Transcript, case: Case) -> CheckResult:
    """L2: Check if the answer is correctly grounded.

    - Every cited id is in the retrieved ids of the conversation.
    - grounded = True requires at least one citation that mathces the marker.
    - must_cite_gold: at least one gold id is cited.
        - Skipped when an allow_fallback case abstains; L3 grades that as fallback.
    """
    if len(transcript.turns) < 1:
        return CheckResult(
            layer="L2", id=case.id, outcome="fail", detail="No turns in the transcript."
        )

    causes: list[str] = []

    for i, turn in enumerate(transcript.turns, start=1):
        # 1. Check if every cited id is in retrieved ids of the conversation.
        hallucinated_ids = set(turn.raw_citations) - set(turn.retrieved_ids)
        if hallucinated_ids:
            causes.append(
                f"The following ids were hallucinated in turn: {i}: {sorted(hallucinated_ids)}"
            )

        if turn.raw_grounded and not turn.cited_ids:
            causes.append(f"Turn {i} is falsely grounded, no articles cited.")

    gold = case.gold_article_ids
    last = transcript.turns[-1]
    fallback = case.expect.allow_fallback and abstained(last)

    if gold and not fallback and not set(last.cited_ids) & set(gold):
        causes.append(f"Turn {len(transcript.turns)}, no cited article is gold.")

    if causes:
        return CheckResult(
            layer="L2",
            id=case.id,
            outcome="fail",
            detail="; ".join(causes),
        )

    return CheckResult(layer="L2", id=case.id, outcome="pass", detail="Output format is correct.")


def check_correctness(transcript: Transcript, case: Case) -> CheckResult:
    """L3: Check the last answer's text against the case's facts.

    - facts_all: every group has a match; facts_any: one wording matches.
        - Case-insensitive substring, commas stripped from numbers ("40,363" = "40363").
    - facts_none: none of the forbidden strings appears, whatever else holds.
    - Abstention is read from the answer, not from phrases: no citations, grounded = False.
        - must_abstain: abstaining passes.
        - allow_fallback: abstaining instead of the facts is fallback.
    """
    if not transcript.turns:
        return CheckResult(
            layer="L3", id=case.id, outcome="fail", detail="No turns in the transcript."
        )

    last = transcript.turns[-1]
    facts = case.expect.facts
    text = _norm(last.text)
    forbidden = [f for f in facts.none if _norm(f) in text]
    forbidden_cause = f"Forbidden facts found: {forbidden}" if forbidden else ""

    if case.expect.must_abstain:
        causes = [forbidden_cause] if forbidden else []
        if last.cited_ids:
            causes.append(f"Cited {len(last.cited_ids)} source(s) instead of abstaining.")
        if last.grounded:
            causes.append("grounded = true instead of abstaining.")
        if causes:
            return CheckResult(layer="L3", id=case.id, outcome="fail", detail="; ".join(causes))
        return CheckResult(layer="L3", id=case.id, outcome="pass", detail="Abstained.")

    missing = [
        i for i, group in enumerate(facts.all, start=1) if not any(_norm(f) in text for f in group)
    ]
    any_missed = bool(facts.any) and not any(_norm(f) in text for f in facts.any)

    if forbidden:
        return CheckResult(layer="L3", id=case.id, outcome="fail", detail=forbidden_cause)
    if not missing and not any_missed:
        return CheckResult(layer="L3", id=case.id, outcome="pass", detail="Facts found.")
    if case.expect.allow_fallback and abstained(last):
        return CheckResult(
            layer="L3", id=case.id, outcome="fallback", detail="Abstained instead of the facts."
        )

    causes = [f"No wording of facts_all group {i} found." for i in missing]
    if any_missed:
        causes.append("No facts_any wording found.")
    return CheckResult(layer="L3", id=case.id, outcome="fail", detail="; ".join(causes))


def abstained(turn: TurnResult) -> bool:
    """The answer abstains: no sources and not grounded (no phrase matching)."""
    return not turn.cited_ids and not turn.grounded


def _norm(text: str) -> str:
    return DIGIT_COMMA.sub("", text.lower())


def run_check(transcript: Transcript, case: Case) -> ScoredTranscript:
    """Runs all checks and returns a ScoredTranscript."""

    if transcript.error is not None:
        return ScoredTranscript(
            case_id=case.id,
            run_idx=transcript.run_idx,
            results=(error_result(transcript.error),),
        )

    l0 = check_format(transcript, case)
    l2 = check_grounding(transcript, case)
    l3 = check_correctness(transcript, case)

    return ScoredTranscript(case_id=case.id, run_idx=transcript.run_idx, results=(l0, l2, l3))


def error_result(error: RunError) -> CheckResult:
    """A run that ended in an error has no answer to grade, so no other layer runs."""
    detail = f"{error.kind} in turn {error.turn}: {error.error_message}"
    if error.kind in FORMAT_ERRORS:
        # the agent failed to produce a valid answer: that is a real L0 failure
        return CheckResult(layer="L0", id="format", outcome="fail", detail=detail)
    # timeouts, rate limits and the like say nothing about answer quality
    return CheckResult(layer="run", id="error", outcome="invalid", detail=detail)
