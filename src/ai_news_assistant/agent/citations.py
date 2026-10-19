import json
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AnyMessage, ToolMessage

from ai_news_assistant.agent.config import CITATION_TOOLS
from ai_news_assistant.models import ModelAnswer, Source
from ai_news_assistant.rag.store import RagStore

MARKER = re.compile(r"\[(\d+)\]")
# A marker plus the spaces before it, so removing "[2]" from "B [2]." gives "B.".
# Only spaces and tabs: a removed marker never joins two lines.
MARKER_WITH_SPACE = re.compile(r"[ \t]*\[(\d+)\]")


@dataclass(frozen=True)
class CheckedCitations:
    clean_text: str  # Text after dropped ids are cleaned
    valid_citations: dict[int, str]  # marker number: article id
    dropped_ids: list[str]  # dropped ids due to made up citations
    dropped_markers: list[int]  # marker numbers removed from the text, sorted, each once


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


def check_citations(model_answer: ModelAnswer, retrieved: set[str]) -> CheckedCitations:
    """Check the citations in the answer against retrieved_ids at tools.

    Return a CheckedCitations object with cleaned text, valid citations, and dropped ones.
    """

    # Get used marker numbers from the text.
    # First go through the citations enumerating them
    # For each citation check if its in the retrieved_ids
    # If so check if the index was used as a marker
    # If so, keep the citation in valid dict
    # If not, add article_id to dropped.

    used = [int(n) for n in MARKER.findall(model_answer.text)]
    valid: dict[int, str] = {}
    dropped: list[str] = []
    for i, article_id in enumerate(model_answer.citations, start=1):
        if article_id in retrieved:
            if i in used:
                valid[i] = article_id
        else:
            dropped.append(article_id)
    text = MARKER_WITH_SPACE.sub(
        lambda m: m.group(0) if int(m.group(1)) in valid else "", model_answer.text
    )
    # Out of range and unretrieved markers alike.
    dropped_markers = sorted({n for n in used if n not in valid})

    return CheckedCitations(
        clean_text=text,
        valid_citations=valid,
        dropped_ids=dropped,
        dropped_markers=dropped_markers,
    )


def resolve_sources(valid: dict[int, str], store_path: Path) -> list[Source]:
    """Create sources from valid citations.

    Retrieve valid citations articles from the DB.
    Return the list of sources for the valid citations.
    """

    sources: list[Source] = []
    with RagStore(path=store_path) as store:
        for marker, article_id in valid.items():
            article = store.get_article(article_id)
            published_at = (
                datetime.fromisoformat(article.published_at) if article.published_at else None
            )
            sources.append(
                Source(
                    marker=marker,
                    article_id=article_id,
                    title=article.title,
                    source=article.source,
                    published_at=published_at,
                )
            )
    return sorted(sources, key=lambda x: x.marker)
