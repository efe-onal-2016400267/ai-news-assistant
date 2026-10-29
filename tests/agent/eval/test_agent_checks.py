"""Checks against transcripts with planted faults.

Tests named `test_bug<n>_...` describe the intended behaviour and fail on the current code on
purpose; `<n>` is the number in the bug report. Run one with `-k bug<n>`.
"""

from dataclasses import replace

import pytest

from ai_news_assistant.agent.eval.checks import (
    CheckResult,
    Outcome,
    ScoredTranscript,
    check_format,
    check_grounding,
    run_check,
)
from ai_news_assistant.agent.eval.dataset import Case, Expect, FactChecks
from ai_news_assistant.agent.eval.transcript import RunError, Transcript, TurnResult

ABSTAIN_PHRASES = ("I do not know",)

CLEAN = TurnResult(
    user_query="What happened?",
    tool_calls=(),
    retrieved_ids=("a1", "a2"),
    text="The model shipped [1].",
    cited_ids=("a1",),
    grounded=True,
    structured=True,
    raw_citations=("a1",),
    raw_grounded=True,
    dropped_ids=(),
    dropped_markers=(),
    input_tokens=10,
    output_tokens=5,
    model_calls=2,
    latency_s=1.0,
)


def transcript(*turns: TurnResult, error: RunError | None = None) -> Transcript:
    return Transcript(case_id="c-1", run_idx=0, turns=turns, error=error)


def case(gold: tuple[str, ...] = ("a1",), expect: Expect | None = None) -> Case:
    return Case(
        id="c-1",
        type="single-turn-lookup-and-citation",
        split="dev",
        turns=("What happened?",),
        gold_article_ids=gold,
        expect=expect or Expect(),
    )


ABSTAIN_CASE = case(gold=(), expect=Expect(must_abstain=True))


# clean transcripts


@pytest.mark.parametrize("turns", [(CLEAN,), (CLEAN, CLEAN)])
def test_clean_transcript_passes_every_check(turns: tuple[TurnResult, ...]) -> None:
    scored = run_check(transcript(*turns), case(), ABSTAIN_PHRASES)

    assert [(r.layer, r.id, r.outcome) for r in scored.results] == [
        ("L0", "c-1", "pass"),
        ("L2", "c-1", "pass"),
    ]
    assert (scored.case_id, scored.run_idx, scored.outcome) == ("c-1", 0, "pass")


# L0 check_format: one planted fault each


def test_l0_unstructured_output_fails() -> None:
    result = check_format(transcript(replace(CLEAN, structured=False)), case())

    assert result.outcome == "fail"
    assert "did not produce structured output. Turn:1" in result.detail


@pytest.mark.parametrize("text", ["", "   \n\t "])
def test_l0_empty_text_fails(text: str) -> None:
    result = check_format(transcript(replace(CLEAN, text=text)), case())

    assert result.outcome == "fail"
    assert "Response text is empty. Turn:1" in result.detail


@pytest.mark.parametrize(
    "text",
    [
        "Answer [1].\n\nSources: Guardian",
        "Answer [1].\n\n**Sources:**\n[1]: x",
        "Answer [1].\n## Source:\nx",
        "Answer [1].\n- sources : x",
        "Answer [1].\n> _Sources_: x",
        "sources: at the very start",
    ],
)
def test_l0_model_written_sources_line_fails(text: str) -> None:
    result = check_format(transcript(replace(CLEAN, text=text)), case())

    assert result.outcome == "fail"
    assert "Raw output contains sources" in result.detail


@pytest.mark.parametrize(
    "text",
    [
        "Resources: a list [1].",
        "The sources say: it shipped [1].",
        "It shipped, per open sources: two outlets [1].",
        "Sources say it shipped [1].",  # no colon right after the word
    ],
)
def test_l0_sources_word_in_prose_passes(text: str) -> None:
    assert check_format(transcript(replace(CLEAN, text=text)), case()).outcome == "pass"


def test_l0_dangling_markers_fail() -> None:
    result = check_format(transcript(replace(CLEAN, dropped_markers=(3,))), case())

    assert result.outcome == "fail"
    assert "dangling markers. Turn:1" in result.detail


def test_l0_duplicate_citations_fail_and_name_each_id() -> None:
    bad = replace(CLEAN, raw_citations=("a1", "b", "a1", "b", "c"))

    result = check_format(transcript(bad), case())

    assert result.outcome == "fail"
    assert "Duplicate citations. Turn:1" in result.detail
    assert "a1" in result.detail and "b" in result.detail
    assert "c" not in result.detail.split("Turn:1")[1]  # c appears once, not a duplicate


def test_l0_fault_in_second_turn_names_turn_two() -> None:
    result = check_format(transcript(CLEAN, replace(CLEAN, structured=False)), case())

    assert "Turn:2" in result.detail
    assert "Turn:1" not in result.detail


def test_l0_every_fault_in_a_turn_is_reported() -> None:
    bad = replace(CLEAN, structured=False, text=" ", dropped_markers=(2,))

    result = check_format(transcript(bad), case())

    # 3 faults -> 3 causes -> 2 "; " separators
    assert result.detail.count("; ") == 2
    for cause in ("structured output", "text is empty", "dangling markers"):
        assert cause in result.detail


def test_bug7_l0_detail_has_no_newlines() -> None:
    """BUG 7: causes start with "\\n" and each duplicate id is its own cause."""
    bad = replace(CLEAN, text="", raw_citations=("a1", "a1"))

    detail = check_format(transcript(bad), case()).detail

    assert "\n" not in detail


# L2 check_grounding


def test_l2_unretrieved_citation_fails() -> None:
    bad = replace(CLEAN, raw_citations=("zz",), retrieved_ids=("a1",))

    result = check_grounding(transcript(bad), case(), ABSTAIN_PHRASES)

    assert result.outcome == "fail"
    assert "hallucinated in turn: 1" in result.detail
    assert "zz" in result.detail


def test_l2_grounded_without_citations_fails() -> None:
    bad = replace(CLEAN, raw_grounded=True, cited_ids=())

    result = check_grounding(transcript(bad), case(gold=()), ABSTAIN_PHRASES)

    assert result.outcome == "fail"
    assert "Turn 1 is falsely grounded" in result.detail


def test_l2_last_turn_without_gold_citation_fails() -> None:
    bad = replace(CLEAN, cited_ids=("a2",), raw_citations=("a2",))

    result = check_grounding(transcript(bad), case(gold=("a1",)), ABSTAIN_PHRASES)

    assert result.outcome == "fail"
    assert result.detail == "Turn 1, no cited article is gold."


def test_l2_gold_cited_only_in_an_earlier_turn_fails() -> None:
    last = replace(CLEAN, cited_ids=("a2",), raw_citations=("a2",))

    result = check_grounding(transcript(CLEAN, last), case(gold=("a1",)), ABSTAIN_PHRASES)

    assert result.detail == "Turn 2, no cited article is gold."


def test_l2_unretrieved_id_in_earlier_turn_fails() -> None:
    first = replace(CLEAN, raw_citations=("zz",))

    result = check_grounding(transcript(first, CLEAN), case(), ABSTAIN_PHRASES)

    assert "hallucinated in turn: 1" in result.detail


def test_l2_abstention_without_citations_fallsback() -> None:
    abstain = replace(
        CLEAN, text="I do not know.", cited_ids=(), raw_citations=(), raw_grounded=False
    )

    assert check_grounding(transcript(abstain), ABSTAIN_CASE, ABSTAIN_PHRASES).outcome == "fallback"


def test_l2_all_faults_reported_together() -> None:
    bad = replace(CLEAN, raw_citations=("zz",), cited_ids=(), raw_grounded=True)

    result = check_grounding(transcript(bad), case(gold=("a1",)), ABSTAIN_PHRASES)

    # hallucinated id, falsely grounded, no gold -> 3 causes, 2 separators
    assert result.detail.count("; ") == 2


def test_bug5_l2_hallucinated_ids_are_listed_in_sorted_order() -> None:
    """BUG 5: the detail prints a Python set, whose order depends on the hash seed."""
    bad = replace(CLEAN, raw_citations=("zz2", "zz1"), retrieved_ids=())

    detail = check_grounding(transcript(bad), case(), ABSTAIN_PHRASES).detail

    assert "{" not in detail
    assert detail.index("zz1") < detail.index("zz2")


def test_bug11_l2_no_turns_does_not_raise() -> None:
    """BUG 11: `transcript.turns[-1]` raises IndexError on a transcript with no turns."""
    result = check_grounding(transcript(), case(), ABSTAIN_PHRASES)

    assert result.outcome == "fail"


# run errors


def test_format_error_is_an_l0_fail_and_skips_other_checks() -> None:
    error = RunError(turn=1, kind="StructuredOutputValidationError", error_message="bad json")
    faulty = replace(CLEAN, structured=False, raw_citations=("zz",))  # would fail L0 and L2

    scored = run_check(transcript(faulty, error=error), case(), ABSTAIN_PHRASES)

    assert scored.results == (
        CheckResult(
            layer="L0",
            id="format",
            outcome="fail",
            detail="StructuredOutputValidationError in turn 1: bad json",
        ),
    )
    assert scored.outcome == "fail"


def test_other_run_error_is_invalid() -> None:
    error = RunError(turn=2, kind="TimeoutError", error_message="timed out")

    scored = run_check(transcript(CLEAN, error=error), case(), ABSTAIN_PHRASES)

    assert scored.results == (
        CheckResult(
            layer="run", id="error", outcome="invalid", detail="TimeoutError in turn 2: timed out"
        ),
    )
    assert scored.outcome == "invalid"


def test_error_before_first_turn_does_not_touch_turns() -> None:
    error = RunError(turn=0, kind="KeyError", error_message="'model'")

    scored = run_check(transcript(error=error), case(), ABSTAIN_PHRASES)

    assert scored.outcome == "invalid"


# allow_fallback


def test_bug3_abstaining_on_allow_fallback_case_is_graded_fallback() -> None:
    """BUG 3: L2 fails any answer without a gold citation, so fallback is never reached."""
    fallback_case = case(
        gold=("a1",), expect=Expect(allow_fallback=True, facts=FactChecks(any=("shipped",)))
    )
    abstain = replace(
        CLEAN, text="I do not know.", cited_ids=(), raw_citations=(), raw_grounded=False
    )

    scored = run_check(transcript(abstain), fallback_case, ABSTAIN_PHRASES)

    assert scored.outcome == "fallback"


# ScoredTranscript.outcome


@pytest.mark.parametrize(
    ("outcomes", "worst"),
    [
        (("pass", "pass"), "pass"),
        (("pass", "fallback"), "fallback"),
        (("fallback", "fail", "pass"), "fail"),
        (("fail", "invalid"), "invalid"),
        (("pass", "invalid", "fallback"), "invalid"),
        ((), "pass"),  # no results: nothing failed
    ],
)
def test_outcome_is_the_worst_result(outcomes: tuple[Outcome, ...], worst: Outcome) -> None:
    results = tuple(CheckResult("L0", "c-1", o, "") for o in outcomes)

    assert ScoredTranscript("c-1", 0, results).outcome == worst
