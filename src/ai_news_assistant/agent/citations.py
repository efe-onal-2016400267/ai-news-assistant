import json
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage

from ai_news_assistant.agent.config import CITATION_TOOLS


def retrieved_ids(messages: list[AnyMessage]) -> set[str]:
    """Check message history and extract the article ids
    returned by tool calls (get_article and search_news)
    to make sure that the citations provided by the model are correct.
    """

    article_ids: set[str] = set()
    for message in messages:
        if not isinstance(message, ToolMessage) or message.name not in CITATION_TOOLS:
            continue
        try:
            payload: Any = json.loads(message.text)
        except json.JSONDecodeError:
            continue
        items = payload if isinstance(payload, list) else [payload]
        for item in items:
            if isinstance(item, dict) and "error" not in item and "article_id" in item:
                article_ids.add(item["article_id"])

    return article_ids
