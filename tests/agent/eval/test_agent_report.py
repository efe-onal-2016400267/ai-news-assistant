"""Efficiency summary and its markdown, on hand-built transcripts."""

from dataclasses import replace

import pytest

from ai_news_assistant.agent.eval.dataset import Case, CaseDataset, CaseType, Expect
from ai_news_assistant.agent.eval.report import (
    EfficiencyRow,
    render_efficiency,
    render_report,
    summarize_efficiency,
)
from ai_news_assistant.agent.eval.transcript import RunError, ToolCall, Transcript, TurnResult

FILTER = "filter-requests"
FOLLOW_UP = "follow-ups-and-context-management"


def turn(
    tool_calls: int = 1, input_tokens: int = 100, output_tokens: int = 10, latency_s: float = 1.0
) -> TurnResult:
    return TurnResult(
        user_query="q",
        tool_calls=(ToolCall(name="search_news", args={}, result_ids=()),) * tool_calls,
        retrieved_ids=(),
        text="answer",
        cited_ids=(),
        grounded=False,
        structured=True,
        raw_citations=(),
        raw_grounded=False,
        dropped_ids=(),
        dropped_markers=(),
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_calls=1,
        latency_s=latency_s,
    )


def run(
    case_id: str, run_idx: int, *turns: TurnResult, error: RunError | None = None
) -> Transcript:
    return Transcript(case_id=case_id, run_idx=run_idx, turns=turns, error=error)


def case(case_id: str, case_type: CaseType = FILTER) -> Case:
    return Case(
        id=case_id,
        type=case_type,
        split="dev",
        turns=("q",),
        gold_article_ids=(),
        expect=Expect(),
    )


def calls(rows: list[EfficiencyRow]) -> list[tuple[str, int, float, int]]:
    """(group, runs, tool calls median, tool calls max) per row."""
    return [(r.group, r.runs, r.tool_calls_median, r.tool_calls_max) for r in rows]


ERROR = RunError(turn=1, kind="TimeoutError", error_message="timed out")


# summarize_efficiency


def test_runs_are_summed_over_turns_then_median_and_max_over_runs() -> None:
    transcripts = [
        # run 0: two turns, totals 1+2=3 calls, 100+200=300 in, 10+20=30 out, 1.0+1.5=2.5 s
        run("a", 0, turn(1, 100, 10, 1.0), turn(2, 200, 20, 1.5)),
        run("a", 1, turn(1, 100, 10, 0.5)),
        run("a", 2, turn(5, 1000, 50, 4.0)),
    ]

    case_rows, type_rows = summarize_efficiency(transcripts, [case("a")])

    # calls [3, 1, 5], in [300, 100, 1000], out [30, 10, 50], latency [2.5, 0.5, 4.0]
    expected = EfficiencyRow(
        group="a",
        runs=3,
        tool_calls_median=3,
        tool_calls_max=5,
        input_tokens_median=300,
        input_tokens_max=1000,
        output_tokens_median=30,
        output_tokens_max=50,
        latency_median_s=2.5,
        latency_max_s=4.0,
    )
    assert case_rows == [expected]
    assert type_rows == [replace(expected, group=FILTER)]


def test_even_number_of_runs_takes_mean_of_middle_two() -> None:
    transcripts = [run("a", 0, turn(1, 100)), run("a", 1, turn(4, 201))]

    [row], _ = summarize_efficiency(transcripts, [case("a")])

    # calls (1 + 4) / 2 = 2.5, input (100 + 201) / 2 = 150.5
    assert (row.tool_calls_median, row.input_tokens_median) == (2.5, 150.5)


def test_type_row_is_over_all_runs_not_case_medians() -> None:
    transcripts = [
        run("a", 0, turn(1)),
        run("a", 1, turn(1)),
        run("a", 2, turn(1)),
        run("b", 0, turn(9)),
    ]

    case_rows, type_rows = summarize_efficiency(transcripts, [case("a"), case("b")])

    assert calls(case_rows) == [("a", 3, 1, 1), ("b", 1, 9, 9)]
    # median of [1, 1, 1, 9] is 1; the mean of the case medians (1 + 9) / 2 would be 5
    assert calls(type_rows) == [(FILTER, 4, 1, 9)]


def test_types_are_kept_apart() -> None:
    transcripts = [run("a", 0, turn(1)), run("b", 0, turn(2)), run("c", 0, turn(6))]
    cases = [case("a"), case("b", FOLLOW_UP), case("c", FOLLOW_UP)]

    _, type_rows = summarize_efficiency(transcripts, cases)

    assert calls(type_rows) == [(FILTER, 1, 1, 1), (FOLLOW_UP, 2, 4, 6)]


def test_rows_are_sorted_by_case_id_and_by_type() -> None:
    transcripts = [
        run("zeta", 0, turn()),
        run("alpha", 0, turn()),
        run("mid", 0, turn()),
    ]
    cases = [case("zeta", FOLLOW_UP), case("alpha", FOLLOW_UP), case("mid", FILTER)]

    case_rows, type_rows = summarize_efficiency(transcripts, cases)

    assert [r.group for r in case_rows] == ["alpha", "mid", "zeta"]
    assert [r.group for r in type_rows] == [FILTER, FOLLOW_UP]


def test_single_run_median_equals_max() -> None:
    [row], _ = summarize_efficiency([run("a", 0, turn(3, 700, 70, 2.0))], [case("a")])

    assert (row.runs, row.tool_calls_median, row.tool_calls_max) == (1, 3, 3)
    assert (row.input_tokens_median, row.input_tokens_max) == (700, 700)
    assert (row.output_tokens_median, row.output_tokens_max) == (70, 70)
    assert (row.latency_median_s, row.latency_max_s) == (2.0, 2.0)


def test_errored_runs_are_left_out() -> None:
    transcripts = [
        run("a", 0, turn(1, 100)),
        run("a", 1, turn(50, 99_999, 9_999, 240.0), error=ERROR),
        run("a", 2, turn(3, 300)),
    ]

    [row], [type_row] = summarize_efficiency(transcripts, [case("a")])

    assert (row.runs, row.tool_calls_median, row.tool_calls_max) == (2, 2, 3)
    assert (row.input_tokens_max, row.output_tokens_max, row.latency_max_s) == (300, 10, 1.0)
    assert type_row.runs == 2


def test_case_with_only_errored_runs_has_no_row() -> None:
    transcripts = [run("a", 0, turn(1)), run("b", 0, turn(9), error=ERROR)]

    case_rows, type_rows = summarize_efficiency(transcripts, [case("a"), case("b")])

    assert calls(case_rows) == [("a", 1, 1, 1)]
    assert calls(type_rows) == [(FILTER, 1, 1, 1)]


def test_no_transcripts_gives_two_empty_lists() -> None:
    assert summarize_efficiency([], [case("a")]) == ([], [])


def test_case_without_transcripts_has_no_row() -> None:
    case_rows, _ = summarize_efficiency([run("a", 0, turn())], [case("a"), case("b")])

    assert [r.group for r in case_rows] == ["a"]


@pytest.mark.parametrize("error", [None, ERROR])
def test_unknown_case_id_raises(error: RunError | None) -> None:
    with pytest.raises(ValueError, match="case 'zzz' is not in the run's cases file"):
        summarize_efficiency([run("zzz", 0, turn(), error=error)], [case("a")])


# render_efficiency


def row(
    group: str = "a",
    runs: int = 1,
    calls: tuple[float, int] = (1.0, 1),
    input_tokens: tuple[float, int] = (100.0, 100),
    output_tokens: tuple[float, int] = (10.0, 10),
    latency: tuple[float, float] = (1.0, 1.0),
) -> EfficiencyRow:
    return EfficiencyRow(group, runs, *calls, *input_tokens, *output_tokens, *latency)


def test_render_full_output() -> None:
    case_row = row("filter-days-only", 2, (1.0, 1), (10420.0, 10420), (621.0, 632), (5.44, 5.83))
    type_row = row(FILTER, 3, (2.5, 4), (92026.5, 97507), (3830.5, 3998), (25.36, 25.8))

    assert render_efficiency([case_row], [type_row]) == (
        "Per run, summed over its turns; each cell is median / max over runs. "
        "Runs that ended in an error are left out.\n"
        "\n"
        "### By case\n"
        "\n"
        "| Case | Runs | Tool calls | Input tokens | Output tokens | Latency (s) |\n"
        "|---|---|---|---|---|---|\n"
        "| filter-days-only | 2 | 1 / 1 | 10,420 / 10,420 | 621 / 632 | 5.4 / 5.8 |\n"
        "\n"
        "### By case type\n"
        "\n"
        "| Type | Runs | Tool calls | Input tokens | Output tokens | Latency (s) |\n"
        "|---|---|---|---|---|---|\n"
        "| filter-requests | 3 | 2.5 / 4 | 92,026.5 / 97,507 | 3,830.5 / 3,998 | 25.4 / 25.8 |\n"
    )


@pytest.mark.parametrize(
    ("value", "expected"),
    [(10420.0, "10,420"), (92026.5, "92,026.5"), (1.0, "1"), (0.0, "0"), (1234567.0, "1,234,567")],
)
def test_render_count_medians(value: float, expected: str) -> None:
    text = render_efficiency([row(input_tokens=(value, 97507))], [])

    assert f"| {expected} / 97,507 |" in text


@pytest.mark.parametrize(
    ("latency", "expected"), [((5.0, 5.0), "5.0 / 5.0"), ((25.36, 0.04), "25.4 / 0.0")]
)
def test_render_latency_has_one_decimal(latency: tuple[float, float], expected: str) -> None:
    assert f"| {expected} |" in render_efficiency([row(latency=latency)], [])


def test_render_keeps_the_given_row_order() -> None:
    text = render_efficiency([row("zeta"), row("alpha")], [])

    assert text.index("| zeta |") < text.index("| alpha |")


def test_render_empty_lists_say_no_completed_runs() -> None:
    assert render_efficiency([], []) == (
        "Per run, summed over its turns; each cell is median / max over runs. "
        "Runs that ended in an error are left out.\n"
        "\n"
        "### By case\n"
        "\n"
        "No completed runs.\n"
        "\n"
        "### By case type\n"
        "\n"
        "No completed runs.\n"
    )


def test_render_empty_type_list_keeps_case_table() -> None:
    text = render_efficiency([row("a")], [])

    by_case, by_type = text.split("### By case type")
    assert "| a | 1 | 1 / 1 | 100 / 100 | 10 / 10 | 1.0 / 1.0 |" in by_case
    assert by_type == "\n\nNo completed runs.\n"


# render_report


def test_render_report_header_then_efficiency_section() -> None:
    dataset = CaseDataset(version=2, snapshot="news.db", abstain_phrases=(), cases=(case("a"),))
    transcripts = [run("a", 0, turn()), run("a", 1, turn(), error=ERROR)]

    report = render_report("20261030-172111_baseline", transcripts, dataset)

    header, efficiency = report.split("## Efficiency\n\n")
    assert header == (
        "# Agent eval: `20261030-172111_baseline`\n"
        "\n"
        "- Cases file: v2, snapshot `news.db`\n"
        "- Runs: 2 (1 ended in an error)\n"
        "\n"
    )
    assert efficiency == render_efficiency(*summarize_efficiency(transcripts, dataset.cases))
