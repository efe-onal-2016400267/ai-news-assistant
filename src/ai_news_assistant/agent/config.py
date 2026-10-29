from typing import Any

CHAT_MODEL = "anthropic:claude-haiku-5-5"  # "provider:model"; e.g. "openai:gpt-6-luna"
COMMON_MODEL_KWARGS: dict[str, Any] = {"max_retries": 2, "timeout": 240}
PROVIDER_MODEL_KWARGS: dict[str, dict[str, Any]] = {
    "openai": {"reasoning": {"effort": "medium"}},
    "anthropic": {"max_tokens": 4096},
}


def default_model_kwargs(model: str) -> dict[str, Any]:
    provider = model.split(":", 1)[0]
    return {**COMMON_MODEL_KWARGS, **PROVIDER_MODEL_KWARGS.get(provider, {})}


CITATION_TOOLS = ["search_news", "get_article", "get_latest_articles"]
KEEP_TOOL_RESULTS = 3
CONTEXT_TRIGGER_TOKENS = 10000
CLEARED_PLACEHOLDER = (
    "The tool result was removed to save context. Call the tool again if you need it."
)
