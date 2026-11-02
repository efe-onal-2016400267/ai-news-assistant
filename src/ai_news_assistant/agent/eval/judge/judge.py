"""
LLM judge for faithfulness evaluation.
"""

import html
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.language_models import LanguageModelInput
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import Runnable
from pydantic import BaseModel, Field

from ai_news_assistant.agent.citations import MARKER
from ai_news_assistant.agent.config import JUDGE_MODEL, default_model_kwargs
from ai_news_assistant.agent.eval.evaluate import load_transcripts
from ai_news_assistant.agent.eval.transcript import Transcript, TurnResult
from ai_news_assistant.rag.config import DEFAULT_DB
from ai_news_assistant.rag.store import RagStore

RUN_DIR = Path("data/agent_eval/evaluate_the_judge")


class Claim(BaseModel):
    quote: str = Field(description="Exact words copied from the ANSWER.")
    reason: str


# Output schema
class UnsupportedClaim(Claim):
    """One unsupported claim."""

    severity: Literal["minor", "major"]


class JudgeOutput(BaseModel):
    """Final output of the judge regarding an answer.

    We will deterministically check the severity of each unsupported claim
    and create the final verdict manually.
    """

    reasoning: str
    unsupported_claims: list[UnsupportedClaim]
    supported_claims: list[Claim] = Field(
        description="Claims in the answer that are backed by the provided source articles."
    )


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


# Judge Config
@dataclass(frozen=True)
class JudgeConfig:
    model: str  # e.g. "anthropic:claude-haiku-4-5", different tier from the agent
    rubric: str = "v1"  # select rubric_v1.md
    temperature_reasoning: float = 0.0  # TODO: Not appliciable for OpenAI's new models


# Final verdict to be stored.
@dataclass(frozen=True)
class JudgeVerdict:
    verdict: Literal["unfaithful", "minor", "faithful"]
    supported_claims: tuple[Claim, ...]
    unsupported_claims: tuple[UnsupportedClaim, ...]
    invalid_claims: tuple[UnsupportedClaim, ...]
    reasoning: str


# Judge Record to be stored along with transcript information
@dataclass(frozen=True)
class JudgeRecord:
    case_id: str
    run_idx: str
    judge_model: str
    rubric: str
    markers: dict[int, str]
    verdict: JudgeVerdict | None = None
    error: str | None = None

    @classmethod
    def from_verdict(
        cls,
        transcript: Transcript,
        verdict: JudgeVerdict,
        *,
        judge_model: str,
        rubric: str,
        markers: dict[int, str],
    ) -> "JudgeRecord":
        return cls(
            case_id=transcript.case_id,
            run_idx=str(transcript.run_idx),
            verdict=verdict,
            judge_model=judge_model,
            rubric=rubric,
            markers=markers,
        )

    @classmethod
    def from_error(
        cls,
        transcript: Transcript,
        error: str,
        *,
        judge_model: str,
        rubric: str,
        markers: dict[int, str],
    ) -> "JudgeRecord":
        return cls(
            case_id=transcript.case_id,
            run_idx=str(transcript.run_idx),
            error=error,
            judge_model=judge_model,
            rubric=rubric,
            markers=markers,
        )

    def to_dict(self) -> dict[str, Any]:
        v = self.verdict
        return {
            "case_id": self.case_id,
            "run_idx": self.run_idx,
            "judge_model": self.judge_model,
            "rubric": self.rubric,
            "markers": {str(k): v for k, v in self.markers.items()},
            "verdict": v.verdict if v else None,
            "supported_claims": [c.model_dump() for c in v.supported_claims] if v else [],
            "unsupported_claims": [c.model_dump() for c in v.unsupported_claims] if v else [],
            "invalid_claims": [c.model_dump() for c in v.invalid_claims] if v else [],
            "error": self.error,
        }


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
        title = html.escape(a.title, quote=True)
        source = html.escape(a.source, quote=True)
        text = a.text.replace("</article>", "")
        parts.append(
            f'<article marker="{a.marker}" title="{title}" '
            f'source="{source}" published="{a.published}">\n'
            f"{text}\n</article>"
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


def judge_verdict(
    inp: JudgeInput, chat: Runnable[LanguageModelInput, dict[str, Any] | BaseModel], rubric: str
) -> JudgeVerdict:
    """Run the judge against an input, return the final verdict."""

    messages = [SystemMessage(rubric), HumanMessage(render_user_message(inp))]
    out = chat.invoke(messages)
    if not isinstance(out, JudgeOutput):
        raise TypeError(f"Judge returned {type(out).__name__}, expected JudgeOutput")

    answer = _norm(inp.answer)
    supported = tuple(c for c in out.supported_claims if _norm(c.quote) in answer)
    unsupported = tuple(c for c in out.unsupported_claims if _norm(c.quote) in answer)
    unverified = tuple(c for c in out.unsupported_claims if _norm(c.quote) not in answer)

    verdict: Literal["unfaithful", "minor", "faithful"] = "faithful"
    severity_set = set(unsup.severity for unsup in unsupported)
    if severity_set:
        verdict = "unfaithful" if "major" in severity_set else "minor"

    return JudgeVerdict(
        verdict=verdict,
        supported_claims=supported,
        unsupported_claims=unsupported,
        invalid_claims=unverified,
        reasoning=out.reasoning,
    )


def load_rubric(version: str) -> str:
    return (Path(__file__).parent / f"rubric_{version}.md").read_text(encoding="utf-8")


def _norm(s: str) -> str:
    """Normalize input str."""
    return " ".join(s.split())


def main() -> None:
    load_dotenv()
    transcripts = load_transcripts(RUN_DIR)
    model = JUDGE_MODEL
    kwargs = {
        **default_model_kwargs(model),
    }
    judge_model = init_chat_model(model, **kwargs).with_structured_output(
        JudgeOutput
    )  # This output will be turned into a verdict manually

    rubric_version = "v1"
    rubric = load_rubric(rubric_version)
    out_path = RUN_DIR / f"judge_{rubric_version}.jsonl"

    n_ok = n_err = 0
    with out_path.open("w", encoding="utf-8") as f:
        for transcript in transcripts:
            if transcript.error is not None:
                continue
            last = transcript.last
            markers = marker_map(last)
            if not markers:
                continue
            try:
                judge_input = generate_judge_input(last)
                verdict = judge_verdict(inp=judge_input, chat=judge_model, rubric=rubric)
                record = JudgeRecord.from_verdict(
                    transcript, verdict, judge_model=model, rubric=rubric_version, markers=markers
                )
                n_ok += 1
            except Exception as exc:
                record = JudgeRecord.from_error(
                    transcript=transcript,
                    error=f"{type(exc).__name__}: {str(exc)[:300]}",
                    judge_model=model,
                    rubric=rubric_version,
                    markers=markers,
                )
                n_err += 1
            f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            f.flush()

    print(f"judged {n_ok}, error {n_err} -> {out_path}")


if __name__ == "__main__":
    main()
