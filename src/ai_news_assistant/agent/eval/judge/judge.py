"""
LLM judge for faithfulness evaluation.
"""

import html
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from ai_news_assistant.agent.citations import MARKER
from ai_news_assistant.agent.eval.evaluate import load_transcripts
from ai_news_assistant.agent.eval.transcript import TurnResult
from ai_news_assistant.rag.config import DEFAULT_DB
from ai_news_assistant.rag.store import RagStore

RUN_DIR = Path("data/agent_eval/evaluate_the_judge")


# Output schema
class UnsupportedClaim(BaseModel):
    """One unsupported claim."""

    quote: str = Field(description="Exact words copied from the ANSWER.")
    severity: Literal["minor", "major"]
    reason: str


class JudgeOutput(BaseModel):
    """Final output of the judge regarding an answer.

    We will deterministically check the severity of each unsupported claim
    and create the final verdict manually.
    """

    reasoning: str
    unsupported_claims: list[UnsupportedClaim]


# Judge input
@dataclass(frozen=True)
class ArticleView:
    """One cited article to be passed to the judge."""

    id: str
    marker: int
    title: str
    source: str
    published: str
    text: str


@dataclass(frozen=True)
class JudgeInput:
    """The whole judge input."""

    question: str  # User question in the last turn
    answer: str
    articles: tuple[ArticleView, ...]


def format_published(published_at: str | None) -> str:
    if not published_at:
        return "unknown"
    dt = datetime.fromisoformat(published_at)
    if dt.tzinfo is not None:
        dt = dt.astimezone(UTC)
    return dt.strftime("%Y-%m-%d (%A) %H:%M UTC")


def marker_map(turn: TurnResult) -> dict[int, str]:
    """Marker number -> article id, for markers present in the cleaned text."""

    cited = set(turn.cited_ids)
    used = sorted({int(n) for n in MARKER.findall(turn.text)})
    return {
        n: turn.raw_citations[n - 1]
        for n in used
        if 1 <= n <= len(turn.raw_citations) and turn.raw_citations[n - 1] in cited
    }


def render_user_message(inp: JudgeInput) -> str:
    parts = [f"<question>\n{inp.question}\n</question>", f"<answer>\n{inp.answer}\n</answer>"]
    for a in inp.articles:
        text = a.text.replace("</article>", "")
        parts.append(
            f'<article marker="{a.marker}" title="{html.escape(a.title, quote=True)}" '
            f'source="{html.escape(a.source, quote=True)}" published="{a.published}">\n{html.escape(text, quote=True)}\n</article>'
        )
    return "\n\n".join(parts)


def generate_judge_input(turn: TurnResult) -> JudgeInput:
    """Generate judge input for the last turn of a transcript."""

    article_view_list: list[ArticleView] = []
    clean_markers = marker_map(turn)
    with RagStore(DEFAULT_DB) as store:
        for marker, article_id in clean_markers.items():
            article = store.get_article(article_id)
            article_view = ArticleView(
                id=article_id,
                marker=marker,
                title=article.title,
                source=article.source,
                published=format_published(article.published_at),
                text=article.text,
            )
            article_view_list.append(article_view)

    return JudgeInput(
        question=turn.user_query,
        answer=turn.text,
        articles=tuple(article_view_list),
    )


def main() -> None:
    transcripts = load_transcripts(RUN_DIR)

    for transcript in transcripts:
        if transcript.error is None:
            last = transcript.last
            if not marker_map(last):
                continue
            judge_input = generate_judge_input(last)
            print("efe")


if __name__ == "__main__":
    main()
