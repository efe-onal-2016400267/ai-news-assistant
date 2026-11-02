from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from ai_news_assistant.agent.eval.dataset import Case, CaseDataset
from ai_news_assistant.agent.eval.transcript import Transcript


@dataclass(frozen=True)
class EfficiencyRow:
    group: str  # a case id, or a case type for the aggregate row.
    runs: int
    tool_calls_median: float
    tool_calls_max: int
    input_tokens_median: float
    input_tokens_max: int
    output_tokens_median: float
    output_tokens_max: int
    latency_median_s: float
    latency_max_s: float


@dataclass(frozen=True)
class RunTotals:
    """One run's cost, summed over its turns."""

    tool_calls: int
    input_tokens: int
    output_tokens: int
    latency_s: float


def run_totals(transcript: Transcript) -> RunTotals:
    tool_calls = 0
    input_tokens = 0
    output_tokens = 0
    latency_s = 0.0
    for turn in transcript.turns:
        tool_calls += len(turn.tool_calls)
        input_tokens += turn.input_tokens
        output_tokens += turn.output_tokens
        latency_s += turn.latency_s
    return RunTotals(tool_calls, input_tokens, output_tokens, latency_s)


def efficiency_row(group: str, totals: Sequence[RunTotals]) -> EfficiencyRow:
    """Median and max of the run totals in one group."""
    tool_calls = [t.tool_calls for t in totals]
    input_tokens = [t.input_tokens for t in totals]
    output_tokens = [t.output_tokens for t in totals]
    latency = [t.latency_s for t in totals]
    return EfficiencyRow(
        group=group,
        runs=len(totals),
        tool_calls_median=median(tool_calls),
        tool_calls_max=max(tool_calls),
        input_tokens_median=median(input_tokens),
        input_tokens_max=max(input_tokens),
        output_tokens_median=median(output_tokens),
        output_tokens_max=max(output_tokens),
        latency_median_s=median(latency),
        latency_max_s=max(latency),
    )


def summarize_efficiency(
    transcripts: Sequence[Transcript], cases: Sequence[Case]
) -> tuple[list[EfficiencyRow], list[EfficiencyRow]]:
    """Create rows for the efficiency report: (case id rows, case type rows).

    - Each run is summed over its turns once.
    - Case id rows: median and max over that case's runs, sorted by case id.
    - Case type rows: median and max over every run of that type, sorted by type.
    - Runs that ended in an error are left out: their totals stop at the failing turn.
    - Raises if a transcript's case id is not in `cases`.
    """
    type_by_id = {case.id: case.type for case in cases}
    by_case: dict[str, list[RunTotals]] = defaultdict(list)
    by_type: dict[str, list[RunTotals]] = defaultdict(list)

    for transcript in transcripts:
        case_type = type_by_id.get(transcript.case_id)
        if case_type is None:
            raise ValueError(f"case {transcript.case_id!r} is not in the run's cases file")
        if transcript.error is not None:
            continue
        totals = run_totals(transcript)
        by_case[transcript.case_id].append(totals)
        by_type[case_type].append(totals)

    case_rows = [efficiency_row(case_id, by_case[case_id]) for case_id in sorted(by_case)]
    type_rows = [efficiency_row(case_type, by_type[case_type]) for case_type in sorted(by_type)]
    return case_rows, type_rows


def render_report(run_id: str, transcripts: Sequence[Transcript], dataset: CaseDataset) -> str:
    """`report.md` for one run folder."""
    errors = sum(transcript.error is not None for transcript in transcripts)
    lines = [
        f"# Agent eval: `{run_id}`",
        "",
        f"- Cases file: v{dataset.version}, snapshot `{dataset.snapshot}`",
        f"- Runs: {len(transcripts)} ({errors} ended in an error)",
        "",
        "## Efficiency",
        "",
    ]
    efficiency = render_efficiency(*summarize_efficiency(transcripts, dataset.cases))
    return "\n".join(lines) + "\n" + efficiency


def render_efficiency(
    case_rows: Sequence[EfficiencyRow], type_rows: Sequence[EfficiencyRow]
) -> str:
    """Markdown for the efficiency section: one table per case id, one per case type."""
    lines = [
        "Per run, summed over its turns; each cell is median / max over runs. "
        "Runs that ended in an error are left out.",
        "",
        "### By case",
        "",
        *_efficiency_table("Case", case_rows),
        "",
        "### By case type",
        "",
        *_efficiency_table("Type", type_rows),
    ]
    return "\n".join(lines) + "\n"


def _efficiency_table(first_column: str, rows: Sequence[EfficiencyRow]) -> list[str]:
    if not rows:
        return ["No completed runs."]
    lines = [
        f"| {first_column} | Runs | Tool calls | Input tokens | Output tokens | Latency (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.group} | {r.runs} | "
            f"{_count(r.tool_calls_median)} / {r.tool_calls_max:,} | "
            f"{_count(r.input_tokens_median)} / {r.input_tokens_max:,} | "
            f"{_count(r.output_tokens_median)} / {r.output_tokens_max:,} | "
            f"{r.latency_median_s:.1f} / {r.latency_max_s:.1f} |"
        )
    return lines


def _count(value: float) -> str:
    """A median of counts: thousands separators, `.5` kept, `.0` dropped (10420.0 -> 10,420)."""
    return f"{value:,.1f}".removesuffix(".0")
