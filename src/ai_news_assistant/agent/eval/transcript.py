"""The transcript is the contract between the runner and every check.
It contains the trace information."""

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class ToolCall:
    """One tool call: arguments as the model sent them, and the article ids it returned."""

    name: str
    args: dict[str, Any]
    result_ids: tuple[str, ...]
    error: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "ToolCall":
        return cls(
            name=d["name"],
            args=d["args"],
            result_ids=tuple(d["result_ids"]),
            error=d.get("error"),
        )


@dataclass(frozen=True)
class TurnResult:
    """Trace information for a single turn."""

    user_query: str
    tool_calls: tuple[ToolCall, ...]
    retrieved_ids: tuple[str, ...]  # Cumulative: retrieved ids in the thread so far, sorted
    text: str  # Answer.text without the sources block
    cited_ids: tuple[str, ...]  # the ids of Answer.sources
    grounded: bool  # grounded information after the check in ask()
    structured: bool
    raw_citations: tuple[str, ...]  # The models citation list before check_citations
    raw_grounded: bool  # Initial grounded output from the model
    dropped_ids: tuple[
        str, ...
    ]  # article_id's that the model cited but were never really retrieved in tool calls
    dropped_markers: tuple[int, ...]  # dropped marker numbers in check_citations
    input_tokens: int  # Summed for each model call, only what the context contains
    output_tokens: int  # Summed over each model call
    model_calls: int
    latency_s: float

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TurnResult":
        return cls(
            user_query=d["user_query"],
            tool_calls=tuple(ToolCall.from_dict(c) for c in d["tool_calls"]),
            retrieved_ids=tuple(d["retrieved_ids"]),
            text=d["text"],
            cited_ids=tuple(d["cited_ids"]),
            grounded=d["grounded"],
            structured=d["structured"],
            raw_citations=tuple(d["raw_citations"]),
            raw_grounded=d["raw_grounded"],
            dropped_ids=tuple(d["dropped_ids"]),
            dropped_markers=tuple(d["dropped_markers"]),
            input_tokens=d["input_tokens"],
            output_tokens=d["output_tokens"],
            model_calls=d["model_calls"],
            latency_s=d["latency_s"],
        )


@dataclass(frozen=True)
class RunError:
    """Run error for reporting."""

    turn: int
    kind: str
    error_message: str

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RunError":
        return cls(
            turn=d["turn"],
            kind=d["kind"],
            error_message=d["error_message"],
        )


@dataclass(frozen=True)
class Transcript:
    """The whole trace information needed for agent evaluation."""

    case_id: str
    run_idx: int  # Each case is run 3 times
    turns: tuple[TurnResult, ...]
    error: RunError | None = None

    @property
    def last(self) -> TurnResult:
        """We check the last turn for some cases."""
        return self.turns[-1]

    def turn(self, n: int) -> TurnResult:
        """Get one specific turn by turn index."""
        return self.turns[n - 1]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Transcript":
        error = d.get("error")
        return cls(
            case_id=d["case_id"],
            run_idx=d["run_idx"],
            turns=tuple(TurnResult.from_dict(t) for t in d["turns"]),
            error=RunError.from_dict(error) if error is not None else None,
        )
