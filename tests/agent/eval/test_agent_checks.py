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
    check_correctness,
    check_format,
    check_grounding,
    check_trajectory,
    run_check,
)
from ai_news_assistant.agent.eval.dataset import Case, Checks, Expect, FactChecks, ToolArgs
from ai_news_assistant.agent.eval.transcript import RunError, ToolCall, Transcript, TurnResult

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
FALLBACK_CASE = case(
    gold=("a1",), expect=Expect(allow_fallback=True, facts=FactChecks(any=("shipped",)))
)
# Abstains by structure only: no abstain phrase in the text, no sources, not grounded
ABSTAINED = replace(
    CLEAN,
    text="Reuters has no coverage of that here.",
    cited_ids=(),
    raw_citations=(),
    grounded=False,
    raw_grounded=False,
)


# clean transcripts


@pytest.mark.parametrize("turns", [(CLEAN,), (CLEAN, CLEAN)])
def test_clean_transcript_passes_every_check(turns: tuple[TurnResult, ...]) -> None:
    scored = run_check(transcript(*turns), case())

    assert [(r.layer, r.id, r.outcome) for r in scored.results] == [
        ("L0", "c-1", "pass"),
        ("L2", "c-1", "pass"),
        ("L3", "c-1", "pass"),
        ("L1", "c-1", "pass"),
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

    result = check_grounding(transcript(bad), case())

    assert result.outcome == "fail"
    assert "hallucinated in turn: 1" in result.detail
    assert "zz" in result.detail


def test_l2_grounded_without_citations_fails() -> None:
    bad = replace(CLEAN, raw_grounded=True, cited_ids=())

    result = check_grounding(transcript(bad), case(gold=()))

    assert result.outcome == "fail"
    assert "Turn 1 is falsely grounded" in result.detail


def test_l2_last_turn_without_gold_citation_fails() -> None:
    bad = replace(CLEAN, cited_ids=("a2",), raw_citations=("a2",))

    result = check_grounding(transcript(bad), case(gold=("a1",)))

    assert result.outcome == "fail"
    assert result.detail == "Turn 1, no cited article is gold."


def test_l2_gold_cited_only_in_an_earlier_turn_fails() -> None:
    last = replace(CLEAN, cited_ids=("a2",), raw_citations=("a2",))

    result = check_grounding(transcript(CLEAN, last), case(gold=("a1",)))

    assert result.detail == "Turn 2, no cited article is gold."


def test_l2_unretrieved_id_in_earlier_turn_fails() -> None:
    first = replace(CLEAN, raw_citations=("zz",))

    result = check_grounding(transcript(first, CLEAN), case())

    assert "hallucinated in turn: 1" in result.detail


def test_l2_must_abstain_abstention_passes() -> None:
    # L2 has nothing to check (no gold, no citations); L3 grades the abstention
    assert check_grounding(transcript(ABSTAINED), ABSTAIN_CASE).outcome == "pass"


def test_l2_allow_fallback_abstention_skips_gold_rule() -> None:
    assert check_grounding(transcript(ABSTAINED), FALLBACK_CASE).outcome == "pass"


def test_l2_plain_case_abstention_still_needs_gold() -> None:
    result = check_grounding(transcript(ABSTAINED), case(gold=("a1",)))

    assert result.detail == "Turn 1, no cited article is gold."


def test_l2_all_faults_reported_together() -> None:
    bad = replace(CLEAN, raw_citations=("zz",), cited_ids=(), raw_grounded=True)

    result = check_grounding(transcript(bad), case(gold=("a1",)))

    # hallucinated id, falsely grounded, no gold -> 3 causes, 2 separators
    assert result.detail.count("; ") == 2


def test_bug5_l2_hallucinated_ids_are_listed_in_sorted_order() -> None:
    """BUG 5: the detail prints a Python set, whose order depends on the hash seed."""
    bad = replace(CLEAN, raw_citations=("zz2", "zz1"), retrieved_ids=())

    detail = check_grounding(transcript(bad), case()).detail

    assert "{" not in detail
    assert detail.index("zz1") < detail.index("zz2")


def test_bug11_l2_no_turns_does_not_raise() -> None:
    """BUG 11: `transcript.turns[-1]` raises IndexError on a transcript with no turns."""
    result = check_grounding(transcript(), case())

    assert result.outcome == "fail"


# run errors


def test_format_error_is_an_l0_fail_and_skips_other_checks() -> None:
    error = RunError(turn=1, kind="StructuredOutputValidationError", error_message="bad json")
    faulty = replace(CLEAN, structured=False, raw_citations=("zz",))  # would fail L0 and L2

    scored = run_check(transcript(faulty, error=error), case())

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

    scored = run_check(transcript(CLEAN, error=error), case())

    assert scored.results == (
        CheckResult(
            layer="run", id="error", outcome="invalid", detail="TimeoutError in turn 2: timed out"
        ),
    )
    assert scored.outcome == "invalid"


def test_error_before_first_turn_does_not_touch_turns() -> None:
    error = RunError(turn=0, kind="KeyError", error_message="'model'")

    scored = run_check(transcript(error=error), case())

    assert scored.outcome == "invalid"


# allow_fallback


def test_abstaining_on_allow_fallback_case_is_graded_fallback() -> None:
    scored = run_check(transcript(ABSTAINED), FALLBACK_CASE)

    assert [(r.layer, r.outcome) for r in scored.results] == [
        ("L0", "pass"),
        ("L2", "pass"),
        ("L3", "fallback"),
        ("L1", "pass"),
    ]
    assert scored.outcome == "fallback"


def test_abstaining_on_must_abstain_case_passes() -> None:
    assert run_check(transcript(ABSTAINED), ABSTAIN_CASE).outcome == "pass"


# L3 check_correctness


def facts_case(
    all_: tuple[tuple[str, ...], ...] = (),
    any_: tuple[str, ...] = (),
    none: tuple[str, ...] = (),
    allow_fallback: bool = False,
) -> Case:
    facts = FactChecks(all=all_, any=any_, none=none)
    return case(expect=Expect(facts=facts, allow_fallback=allow_fallback))


def l3(text: str, c: Case, *earlier: TurnResult, **changes: object) -> CheckResult:
    """L3 on a transcript whose last turn has `text` (and any `changes`)."""
    return check_correctness(transcript(*earlier, replace(CLEAN, text=text, **changes)), c)


def test_l3_every_facts_all_group_matched_passes() -> None:
    c = facts_case(all_=(("Argon", "the Argon model"), ("April", "4/2026")))

    assert l3("argon ships in 4/2026 [1].", c).outcome == "pass"


def test_l3_missing_facts_all_group_fails_and_names_it() -> None:
    c = facts_case(all_=(("Argon",), ("April", "4/2026"), ("free",)))

    result = l3("ARGON is free [1].", c)

    assert (result.outcome, result.detail) == ("fail", "No wording of facts_all group 2 found.")


@pytest.mark.parametrize(
    ("fact", "text"),
    [
        ("40,363", "40363 users signed up [1]."),
        ("40363", "40,363 users signed up [1]."),
        ("1,000,000", "1000,000 users [1]."),  # 1000000 on both sides
    ],
)
def test_l3_commas_in_numbers_are_ignored(fact: str, text: str) -> None:
    assert l3(text, facts_case(any_=(fact,))).outcome == "pass"


def test_l3_comma_outside_numbers_still_counts() -> None:
    c = facts_case(any_=("Argon, a model",))

    assert l3("Argon a model [1].", c).outcome == "fail"
    assert l3("Argon, a model [1].", c).outcome == "pass"


def test_l3_facts_any_one_wording_is_enough() -> None:
    c = facts_case(any_=("HarperCollins", "Simon & Schuster", "Hub City Press"))

    assert l3("Published by simon & schuster [1].", c).outcome == "pass"


def test_l3_facts_any_none_found_fails() -> None:
    result = l3("Published by Penguin [1].", facts_case(any_=("HarperCollins", "Hub City Press")))

    assert (result.outcome, result.detail) == ("fail", "No facts_any wording found.")


def test_l3_forbidden_fact_fails_even_when_facts_match() -> None:
    c = facts_case(any_=("40,363",), none=("Gemini", "OpenAI"))

    result = l3("40,363 users, like gemini [1].", c)

    assert (result.outcome, result.detail) == ("fail", "Forbidden facts found: ['Gemini']")


def test_l3_no_fact_keys_passes() -> None:
    assert l3("Anything [1].", case()).outcome == "pass"


def test_l3_only_the_last_turn_is_graded() -> None:
    first = replace(CLEAN, text="Argon shipped [1].")

    assert l3("It is a model [1].", facts_case(any_=("Argon",)), first).outcome == "fail"


def test_l3_no_turns_fails_without_raising() -> None:
    assert check_correctness(transcript(), case()).outcome == "fail"


# L3 abstention: read from the answer, not from phrases


def test_l3_must_abstain_without_phrase_passes() -> None:
    result = check_correctness(transcript(ABSTAINED), ABSTAIN_CASE)

    assert (result.outcome, result.detail) == ("pass", "Abstained.")


def test_l3_must_abstain_with_phrase_but_citations_fails() -> None:
    answer = replace(ABSTAINED, text="I do not know.", cited_ids=("a1",))

    result = check_correctness(transcript(answer), ABSTAIN_CASE)

    assert (result.outcome, result.detail) == ("fail", "Cited 1 source(s) instead of abstaining.")


def test_l3_must_abstain_grounded_fails() -> None:
    result = check_correctness(transcript(replace(ABSTAINED, grounded=True)), ABSTAIN_CASE)

    assert result.detail == "grounded = true instead of abstaining."


def test_l3_must_abstain_with_invented_specific_fails() -> None:
    c = case(gold=(), expect=Expect(must_abstain=True, facts=FactChecks(none=("Nadella said",))))
    answer = replace(ABSTAINED, text="Not in the archive, but Nadella said models commoditize.")

    result = check_correctness(transcript(answer), c)

    assert (result.outcome, result.detail) == ("fail", "Forbidden facts found: ['Nadella said']")


def test_l3_allow_fallback_facts_present_passes() -> None:
    assert l3("It shipped [1].", FALLBACK_CASE).outcome == "pass"


def test_l3_allow_fallback_abstained_is_fallback() -> None:
    result = check_correctness(transcript(ABSTAINED), FALLBACK_CASE)

    assert (result.outcome, result.detail) == ("fallback", "Abstained instead of the facts.")


def test_l3_allow_fallback_wrong_answer_with_sources_fails() -> None:
    # facts missing, but it cites and claims grounding: a wrong answer, not an abstention
    assert l3("It was cancelled [1].", FALLBACK_CASE).outcome == "fail"


def test_l3_allow_fallback_abstained_with_forbidden_fact_fails() -> None:
    c = facts_case(any_=("shipped",), none=("Gemini",), allow_fallback=True)
    answer = replace(ABSTAINED, text="No coverage, though Gemini may be related.")

    assert check_correctness(transcript(answer), c).outcome == "fail"


def test_l3_plain_case_abstaining_fails() -> None:
    result = check_correctness(transcript(ABSTAINED), facts_case(any_=("shipped",)))

    assert (result.outcome, result.detail) == ("fail", "No facts_any wording found.")


# L1 check_trajectory

L1_PASS = "All tools were called correctly with the correct args."


def call(name: str, **args: object) -> ToolCall:
    return ToolCall(name=name, args=dict(args), result_ids=())


def l1(*calls: ToolCall, earlier: tuple[TurnResult, ...] = (), **expect: object) -> CheckResult:
    """Grade a transcript whose last turn made `calls`, against Expect(**expect)."""
    last = replace(CLEAN, tool_calls=calls)
    return check_trajectory(transcript(*earlier, last), case(expect=Expect(**expect)))


def test_l1_every_expectation_met_passes() -> None:
    result = l1(
        call("search_news", query="q", sources=["The Verge"]),
        call("get_article", article_id="a1"),
        tools=Checks(all=("search_news",), any=("get_article",), none=("get_latest_articles",)),
        tool_args=ToolArgs(tool="search_news", sources=("The Verge",)),
        max_tool_calls=2,
    )

    assert (result.layer, result.outcome, result.detail) == ("L1", "pass", L1_PASS)


def test_l1_no_expectations_and_no_calls_passes() -> None:
    assert (l1().outcome, l1().detail) == ("pass", L1_PASS)


@pytest.mark.parametrize(
    ("tool_args", "args"),
    [
        # agent_cases.toml cases 5-8
        (ToolArgs(tool="search_news", sources=("The Verge",)), {"sources": ["The Verge"]}),
        (ToolArgs(tool="search_news", days=3), {"days": 3}),
        (
            ToolArgs(tool="search_news", sources=("TechCrunch",), days=7),
            {"sources": ["TechCrunch"], "days": 7},
        ),
        (ToolArgs(tool="search_news", absent=frozenset({"sources", "days"})), {}),
    ],
)
def test_l1_filter_cases_with_matching_args_pass(
    tool_args: ToolArgs, args: dict[str, object]
) -> None:
    result = l1(call("search_news", query="q", **args), tool_args=tool_args)

    assert (result.outcome, result.detail) == ("pass", L1_PASS)


# L1 tools_all, tools_any, tools_none


def test_l1_missing_tools_all_fails_and_names_only_the_missing_one() -> None:
    result = l1(call("search_news"), tools=Checks(all=("search_news", "get_article")))

    assert (result.outcome, result.detail) == (
        "fail",
        "Some of the REQUIRED tools were not called: get_article",
    )


def test_l1_no_tools_any_called_fails_and_lists_them() -> None:
    result = l1(call("get_article"), tools=Checks(any=("search_news", "get_latest_articles")))

    assert (result.outcome, result.detail) == (
        "fail",
        "None of the ANY tools are called: search_news, get_latest_articles",
    )


def test_l1_one_tools_any_is_enough() -> None:
    result = l1(
        call("get_latest_articles"), tools=Checks(any=("search_news", "get_latest_articles"))
    )

    assert result.outcome == "pass"


def test_l1_forbidden_tool_fails_and_names_it() -> None:
    result = l1(
        call("search_news"),
        call("get_latest_articles"),
        tools=Checks(none=("get_latest_articles",)),
    )

    assert (result.outcome, result.detail) == (
        "fail",
        "Following tools are used although FORBIDDEN: get_latest_articles",
    )


def test_l1_all_faults_reported_together_in_order() -> None:
    result = l1(
        call("get_latest_articles"),
        call("get_latest_articles"),
        tools=Checks(all=("get_article",), any=("search_news",), none=("get_latest_articles",)),
        tool_args=ToolArgs(tool="search_news", days=3),
        max_tool_calls=1,
    )

    assert (result.outcome, result.detail) == (
        "fail",
        "Following tools are used although FORBIDDEN: get_latest_articles; "
        "None of the ANY tools are called: search_news; "
        "Some of the REQUIRED tools were not called: get_article; "
        "Expected tool to check arguments was not called: search_news; "
        "Too many tool calls, max_tool_calls: 1, called: 2",
    )


def test_l1_only_the_last_turn_is_graded() -> None:
    earlier = replace(CLEAN, tool_calls=(call("search_news"),))
    result = l1(earlier=(earlier,), tools=Checks(all=("search_news",)))

    assert (result.outcome, result.detail) == (
        "fail",
        "Some of the REQUIRED tools were not called: search_news",
    )


# L1 tool_args: first call of the tool only

VERGE_ONLY = ToolArgs(tool="search_news", sources=("The Verge",))


@pytest.mark.parametrize(
    ("args", "got"),
    [
        ({"sources": ["TechCrunch"]}, "['TechCrunch']"),
        ({"sources": ["The Verge", "TechCrunch"]}, "['The Verge', 'TechCrunch']"),
        ({}, "None"),
    ],
)
def test_l1_wrong_sources_fail(args: dict[str, object], got: str) -> None:
    result = l1(call("search_news", query="q", **args), tool_args=VERGE_ONLY)

    assert (result.outcome, result.detail) == (
        "fail",
        f"search_news sources: expected ['The Verge'], got {got}.",
    )


def test_l1_sources_order_does_not_matter() -> None:
    expected = ToolArgs(tool="search_news", sources=("The Verge", "TechCrunch"))
    result = l1(call("search_news", sources=["TechCrunch", "The Verge"]), tool_args=expected)

    assert result.outcome == "pass"


@pytest.mark.parametrize(("args", "got"), [({"days": 7}, "7"), ({}, "None")])
def test_l1_wrong_days_fail(args: dict[str, object], got: str) -> None:
    result = l1(call("search_news", **args), tool_args=ToolArgs(tool="search_news", days=3))

    assert (result.outcome, result.detail) == (
        "fail",
        f"search_news days: expected 3, got {got}.",
    )


def test_l1_absent_arg_passed_fails() -> None:
    expected = ToolArgs(tool="search_news", absent=frozenset({"days"}))
    result = l1(call("search_news", days=3), tool_args=expected)

    assert (result.outcome, result.detail) == (
        "fail",
        "search_news days: expected absent, got 3.",
    )


def test_l1_absent_arg_passed_as_null_passes() -> None:
    expected = ToolArgs(tool="search_news", absent=frozenset({"days"}))

    assert l1(call("search_news", days=None), tool_args=expected).outcome == "pass"


def test_l1_tool_args_tool_not_called_fails() -> None:
    result = l1(call("get_latest_articles"), tool_args=VERGE_ONLY)

    assert (result.outcome, result.detail) == (
        "fail",
        "Expected tool to check arguments was not called: search_news",
    )


def test_l1_first_call_wrong_second_right_fails() -> None:
    result = l1(
        call("search_news", sources=["TechCrunch"]),
        call("search_news", sources=["The Verge"]),
        tool_args=VERGE_ONLY,
    )

    assert result.outcome == "fail"


def test_l1_first_call_right_second_wrong_passes() -> None:
    result = l1(
        call("search_news", sources=["The Verge"]),
        call("search_news", sources=["TechCrunch"]),
        tool_args=VERGE_ONLY,
    )

    assert result.outcome == "pass"


def test_l1_other_tools_args_are_ignored() -> None:
    result = l1(
        call("get_latest_articles", days=30),
        call("search_news", sources=["The Verge"]),
        tool_args=VERGE_ONLY,
    )

    assert result.outcome == "pass"


# L1 max_tool_calls


def test_l1_calls_equal_to_max_pass() -> None:
    assert l1(call("search_news"), call("get_article"), max_tool_calls=2).outcome == "pass"


def test_l1_one_call_over_max_fails() -> None:
    # 3 calls made, 2 allowed
    result = l1(call("search_news"), call("get_article"), call("get_article"), max_tool_calls=2)

    assert (result.outcome, result.detail) == (
        "fail",
        "Too many tool calls, max_tool_calls: 2, called: 3",
    )


def test_l1_max_zero_with_a_call_fails() -> None:
    result = l1(call("search_news"), max_tool_calls=0)

    assert (result.outcome, result.detail) == (
        "fail",
        "Too many tool calls, max_tool_calls: 0, called: 1",
    )


def test_l1_max_zero_without_calls_passes() -> None:
    assert l1(max_tool_calls=0).outcome == "pass"


def test_l1_unset_max_has_no_limit() -> None:
    assert l1(*(call("search_news") for _ in range(5))).outcome == "pass"


def test_l1_no_turns_fails_without_raising() -> None:
    result = check_trajectory(transcript(), case())

    assert (result.layer, result.outcome, result.detail) == (
        "L1",
        "fail",
        "No turns in the transcript.",
    )


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
