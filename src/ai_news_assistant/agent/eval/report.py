from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import median

from ai_news_assistant.agent.eval.checks import Outcome, ScoredTranscript
from ai_news_assistant.agent.eval.dataset import Case, CaseDataset
from ai_news_assistant.agent.eval.transcript import Transcript

OUTCOMES: tuple[Outcome, ...] = ("pass", "fallback", "fail", "invalid")


@dataclass(frozen=True)
class OutcomeRow:
    group: str  # a case id, or a layer.
    passed: int
    fallback: int
    failed: int
    invalid: int

    @property
    def total(self) -> int:
        return self.passed + self.fallback + self.failed + self.invalid


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


def summarize_outcomes(
    scored: Sequence[ScoredTranscript],
) -> tuple[list[OutcomeRow], list[OutcomeRow]]:
    """Create rows for the outcomes report: (case id rows, layer rows).

    - Case id rows: one count per run, by that run's worst outcome, sorted by case id.
    - Layer rows: one count per check result, sorted by layer; `run` (errored runs) last.
    """
    by_case: dict[str, Counter[Outcome]] = defaultdict(Counter)
    by_layer: dict[str, Counter[Outcome]] = defaultdict(Counter)

    for s in scored:
        by_case[s.case_id][s.outcome] += 1
        for r in s.results:
            by_layer[r.layer][r.outcome] += 1

    case_rows = [_outcome_row(case_id, by_case[case_id]) for case_id in sorted(by_case)]
    layers = sorted(by_layer, key=lambda layer: (layer == "run", layer))
    layer_rows = [_outcome_row(layer, by_layer[layer]) for layer in layers]
    return case_rows, layer_rows


def _outcome_row(group: str, counts: Counter[Outcome]) -> OutcomeRow:
    return OutcomeRow(
        group=group,
        passed=counts["pass"],
        fallback=counts["fallback"],
        failed=counts["fail"],
        invalid=counts["invalid"],
    )


def render_report(
    run_id: str,
    transcripts: Sequence[Transcript],
    scored: Sequence[ScoredTranscript],
    dataset: CaseDataset,
) -> str:
    """`report.md` for one run folder."""
    errors = sum(transcript.error is not None for transcript in transcripts)
    header = [
        f"# Agent eval: `{run_id}`",
        "",
        f"- Cases file: v{dataset.version}, snapshot `{dataset.snapshot}`",
        f"- Runs: {len(transcripts)} ({errors} ended in an error)",
        "",
    ]
    efficiency = render_efficiency(*summarize_efficiency(transcripts, dataset.cases))
    return (
        "\n".join(header)
        + "\n"
        + "## Outcomes\n\n"
        + render_outcomes(scored)
        + "\n"
        + "## Efficiency\n\n"
        + efficiency
    )


def render_outcomes(scored: Sequence[ScoredTranscript]) -> str:
    """Markdown for the outcomes section: totals, one table per case id and per layer, and
    the detail of every check that did not pass."""
    if not scored:
        return "No runs.\n"

    totals = Counter(s.outcome for s in scored)
    case_rows, layer_rows = summarize_outcomes(scored)
    lines = [
        "Runs by worst outcome: " + ", ".join(f"{totals[o]} {o}" for o in OUTCOMES) + ".",
        "",
        "### By case",
        "",
        *_outcome_table("Case", "Runs", case_rows),
        "",
        "### By layer",
        "",
        "Counts every check result, not runs.",
        "",
        *_outcome_table("Layer", "Checks", layer_rows),
        "",
        "### Not passing",
        "",
        *_not_passing(scored),
    ]
    return "\n".join(lines) + "\n"


def _outcome_table(first_column: str, count_column: str, rows: Sequence[OutcomeRow]) -> list[str]:
    if not rows:
        return ["No completed runs."]
    lines = [
        f"| {first_column} | {count_column} | Pass | Fallback | Fail | Invalid |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r.group} | {r.total} | {r.passed} | {r.fallback} | {r.failed} | {r.invalid} |"
        )
    return lines


def _not_passing(scored: Sequence[ScoredTranscript]) -> list[str]:
    bullets = [
        f"- `{s.case_id}#{s.run_idx}` `{r.layer}`: {r.detail}"
        for s in scored
        for r in s.results
        if r.outcome != "pass"
    ]
    return bullets or ["All runs passed."]


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
