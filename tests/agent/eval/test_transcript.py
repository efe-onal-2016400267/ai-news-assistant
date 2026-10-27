import json
from dataclasses import FrozenInstanceError, replace
from typing import Any

import pytest

from ai_news_assistant.agent.eval.transcript import RunError, ToolCall, Transcript, TurnResult


def tool_call(
    name: str = "search_news",
    args: dict[str, Any] | None = None,
    result_ids: tuple[str, ...] = ("a1", "a2"),
    error: str | None = None,
) -> ToolCall:
    return ToolCall(
        name=name,
        args=args if args is not None else {"query": "q"},
        result_ids=result_ids,
        error=error,
    )


def turn_result(user: str = "u1", **overrides: Any) -> TurnResult:
    base = TurnResult(
        user=user,
        tool_calls=(tool_call(),),
        retrieved_ids=("a1", "a2"),
        text="A [1].",
        cited_ids=("a1",),
        grounded=True,
        structured=True,
        raw_citations=("a1", "made-up"),
        raw_grounded=True,
        dropped_ids=("made-up",),
        dropped_markers=(2,),
        input_tokens=120,
        output_tokens=30,
        model_calls=2,
        latency_s=1.5,
    )
    return replace(base, **overrides)


def transcript(turns: tuple[TurnResult, ...], error: RunError | None = None) -> Transcript:
    return Transcript(case_id="case-1", run_idx=0, turns=turns, error=error)


def json_round_trip(t: Transcript) -> Transcript:
    """The runner writes JSON and the checks read it back: tuples come back as lists."""
    return Transcript.from_dict(json.loads(json.dumps(t.to_dict())))


def three_turns() -> Transcript:
    return transcript((turn_result("u1"), turn_result("u2"), turn_result("u3")))


# round trip


def test_json_round_trip_restores_equal_transcript() -> None:
    nested = {"query": "q", "k": 5, "filter": {"sources": ["Guardian"]}}
    t = transcript(
        (
            turn_result("u1", tool_calls=(tool_call(args=nested), tool_call("get_article"))),
            turn_result("u2", tool_calls=(), grounded=False, structured=False),
        ),
        error=RunError(turn=2, kind="timeout", error_message="took too long"),
    )

    assert json_round_trip(t) == t


def test_round_trip_restores_tuples_not_lists() -> None:
    restored = json_round_trip(transcript((turn_result(),)))

    turn = restored.turns[0]
    assert isinstance(restored.turns, tuple)
    for value in (
        turn.tool_calls,
        turn.retrieved_ids,
        turn.cited_ids,
        turn.raw_citations,
        turn.dropped_ids,
        turn.dropped_markers,
        turn.tool_calls[0].result_ids,
    ):
        assert isinstance(value, tuple)


def test_to_dict_hand_checked_shape() -> None:
    assert transcript((turn_result(),)).to_dict() == {
        "case_id": "case-1",
        "run_idx": 0,
        "turns": (
            {
                "user": "u1",
                "tool_calls": (
                    {
                        "name": "search_news",
                        "args": {"query": "q"},
                        "result_ids": ("a1", "a2"),
                        "error": None,
                    },
                ),
                "retrieved_ids": ("a1", "a2"),
                "text": "A [1].",
                "cited_ids": ("a1",),
                "grounded": True,
                "structured": True,
                "raw_citations": ("a1", "made-up"),
                "raw_grounded": True,
                "dropped_ids": ("made-up",),
                "dropped_markers": (2,),
                "input_tokens": 120,
                "output_tokens": 30,
                "model_calls": 2,
                "latency_s": 1.5,
            },
        ),
        "error": None,
    }


def test_transcript_with_no_turns_round_trips() -> None:
    # The run failed before turn 1 finished.
    t = transcript((), error=RunError(turn=1, kind="exception", error_message="boom"))

    assert json_round_trip(t) == t


def test_to_dict_does_not_share_args_dict() -> None:
    call = tool_call(args={"query": "q"})
    dumped = transcript((turn_result(tool_calls=(call,)),)).to_dict()

    dumped["turns"][0]["tool_calls"][0]["args"]["query"] = "changed"

    assert call.args == {"query": "q"}


# from_dict defaults and errors


def test_from_dict_without_error_key_gives_none() -> None:
    d = transcript((turn_result(),)).to_dict()
    del d["error"]

    assert Transcript.from_dict(d).error is None


def test_from_dict_error_none_gives_none() -> None:
    assert Transcript.from_dict(transcript((turn_result(),)).to_dict()).error is None


def test_tool_call_from_dict_without_error_key_gives_none() -> None:
    d = {"name": "get_article", "args": {"article_id": "a1"}, "result_ids": []}

    call = ToolCall.from_dict(d)

    assert call == ToolCall(name="get_article", args={"article_id": "a1"}, result_ids=())


def test_tool_call_from_dict_keeps_error() -> None:
    d = {"name": "get_article", "args": {}, "result_ids": [], "error": "not found"}

    assert ToolCall.from_dict(d).error == "not found"


def test_from_dict_missing_required_key_raises_key_error() -> None:
    d = json.loads(json.dumps(transcript((turn_result(),)).to_dict()))
    del d["turns"][0]["text"]

    with pytest.raises(KeyError):
        Transcript.from_dict(d)


# accessors


def test_last_returns_final_turn() -> None:
    assert three_turns().last.user == "u3"


def test_turn_is_one_based() -> None:
    t = three_turns()

    assert [t.turn(n).user for n in (1, 2, 3)] == ["u1", "u2", "u3"]


def test_turn_out_of_range_raises_index_error() -> None:
    with pytest.raises(IndexError):
        three_turns().turn(4)


def test_transcript_is_frozen() -> None:
    t = three_turns()

    with pytest.raises(FrozenInstanceError):
        t.case_id = "other"  # type: ignore[misc]
