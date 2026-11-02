"""
LLM judge for faithfulness evaluation.
"""

from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, Field


# Output schema
class UnsupportedClaim(BaseModel):
    """One unsupported claim."""

    quote: str = Field(description="Exact words copied from the ANSWER.")
    severity: Literal["minor", "major"]
    reason: str


class JudgeVerdict(BaseModel):
    """Final verdict of the judge regarding an answer.

    We will deterministically check the severity of each unsupported claim
    and create the final verdict manually.
    """

    reasoning: str
    unsupported_claims: list[UnsupportedClaim]


# Judge input
@dataclass(frozen=True)
class ArticleView:
    """One cited article to be passed to the judge."""

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


def render_user_message(inp: JudgeInput) -> str:
    parts = [f"<question>\n{inp.question}\n</question>", f"<answer>\n{inp.answer}\n</answer>"]
    for a in inp.articles:
        text = a.text.replace("</article>", "")
        parts.append(
            f'<article marker="{a.marker}" title="{a.title}" '
            f'source="{a.source}" published="{a.published}">\n{text}\n</article>'
        )
    return "\n\n".join(parts)
