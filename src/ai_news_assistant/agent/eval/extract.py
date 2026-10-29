import json

from langchain_core.messages import AIMessage, ToolMessage

from ai_news_assistant.agent.answer import AskResult
from ai_news_assistant.agent.citations import retrieved_ids
from ai_news_assistant.agent.eval.transcript import ToolCall, TurnResult

MAX_ERROR_CHARS = 200


def extract_turn(
    user_query: str,
    result: AskResult,
    # From which message we start based on the turn.
    # len(result.messages) after the previous turn; 0 for turn 1
    start: int,
    latency_s: float,
) -> TurnResult:
    """Gets an AskResult, extracts necessary info, and returns a TurnResult."""
    answer = result.answer
    model_answer = result.model_answer
    checked = result.checked
    messages = result.messages[start:]

    # Pass 1 index the results by toolcall id
    tool_results: dict[str, ToolMessage] = {
        m.tool_call_id: m for m in messages if isinstance(m, ToolMessage)
    }

    tool_calls: list[ToolCall] = []
    input_tokens = 0
    output_tokens = 0
    model_calls = 0
    # Pass 2 fill the results
    for message in messages:
        if not isinstance(message, AIMessage):
            continue
        model_calls += 1
        usage = message.usage_metadata
        if usage is not None:
            input_tokens += usage.get("input_tokens", 0)
            output_tokens += usage.get("output_tokens", 0)

        for call in message.tool_calls:
            tool_message = tool_results.get(str(call["id"]))
            tool_calls.append(
                ToolCall(
                    name=call["name"],
                    args=call["args"],
                    result_ids=_result_ids(tool_message),
                    error=_tool_error(tool_message),
                )
            )

    if model_answer is None or checked is None:
        # Fallback path: no structured response, the raw fields are placeholders
        raw_citations: tuple[str, ...] = ()
        raw_grounded = False
        dropped_ids: tuple[str, ...] = ()
        dropped_markers: tuple[int, ...] = ()
        structured = False
    else:
        raw_citations = tuple(model_answer.citations)
        raw_grounded = model_answer.grounded
        dropped_ids = tuple(checked.dropped_ids)
        dropped_markers = tuple(checked.dropped_markers)
        structured = True

    return TurnResult(
        user_query=user_query,
        tool_calls=tuple(tool_calls),
        retrieved_ids=tuple(sorted(retrieved_ids(result.messages))),
        text=answer.text,
        cited_ids=tuple(source.article_id for source in answer.sources),
        grounded=answer.grounded,
        structured=structured,
        raw_citations=raw_citations,
        raw_grounded=raw_grounded,
        dropped_ids=dropped_ids,
        dropped_markers=dropped_markers,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        model_calls=model_calls,
        latency_s=latency_s,
    )


def _result_ids(tool_message: ToolMessage | None) -> tuple[str, ...]:
    if tool_message is None:
        return ()
    return tuple(sorted(retrieved_ids([tool_message])))


def _tool_error(tool_message: ToolMessage | None) -> str | None:
    if tool_message is None:
        return None
    if tool_message.status == "error":
        return tool_message.text[:MAX_ERROR_CHARS]
    try:
        payload = json.loads(tool_message.text)
    except json.JSONDecodeError:
        return None
    if isinstance(payload, dict) and "error" in payload:
        return str(payload["error"])[:MAX_ERROR_CHARS]
    return None
