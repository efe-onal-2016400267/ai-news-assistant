"""The judge's input rendering and output models. No model or network."""

import re
from dataclasses import FrozenInstanceError

import pytest
from pydantic import ValidationError

from ai_news_assistant.agent.eval.judge.judge import (
    ArticleView,
    JudgeInput,
    JudgeVerdict,
    UnsupportedClaim,
    render_user_message,
)

CLOSING_ARTICLE_TAG = re.compile(r"</\s*article\s*>", re.IGNORECASE)


def article(
    marker: int = 1,
    text: str = "Body text.",
    title: str = "Title",
    source: str = "Source",
    published: str = "2026-10-09",
) -> ArticleView:
    return ArticleView(marker=marker, title=title, source=source, published=published, text=text)


def judge_input(*articles: ArticleView, question: str = "Q?", answer: str = "A [1].") -> JudgeInput:
    return JudgeInput(question=question, answer=answer, articles=articles)


# render_user_message: happy path


def test_question_and_answer_are_wrapped_in_their_tags_in_order() -> None:
    text = render_user_message(judge_input(article(), question="Q?", answer="A [1]."))

    question_at = text.index("<question>\nQ?\n</question>")
    answer_at = text.index("<answer>\nA [1].\n</answer>")
    assert question_at < answer_at


def test_one_article_renders_exactly() -> None:
    text = render_user_message(judge_input(article()))

    assert text == (
        "<question>\nQ?\n</question>\n"
        "\n"
        "<answer>\nA [1].\n</answer>\n"
        "\n"
        '<article marker="1" title="Title" source="Source" published="2026-10-09">\n'
        "Body text.\n"
        "</article>"
    )


def test_each_article_keeps_its_attributes_and_text() -> None:
    text = render_user_message(
        judge_input(
            article(
                marker=1, title="First", source="Outlet A", published="2026-10-08", text="One."
            ),
            article(
                marker=2, title="Second", source="Outlet B", published="2026-10-09", text="Two."
            ),
        )
    )

    first = '<article marker="1" title="First" source="Outlet A" published="2026-10-08">'
    second = '<article marker="2" title="Second" source="Outlet B" published="2026-10-09">'
    assert f"{first}\nOne.\n</article>" in text
    assert f"{second}\nTwo.\n</article>" in text


def test_articles_keep_the_order_they_were_given() -> None:
    text = render_user_message(judge_input(article(marker=3), article(marker=1)))

    assert text.index('marker="3"') < text.index('marker="1"')


def test_multiline_article_text_is_kept_as_is() -> None:
    body = "First line.\n\nSecond line, with a number 9,000."

    text = render_user_message(judge_input(article(text=body)))

    assert f"\n{body}\n</article>" in text


# render_user_message: edge cases


def test_no_articles_renders_only_question_and_answer() -> None:
    text = render_user_message(judge_input(question="Q?", answer="A."))

    assert text == "<question>\nQ?\n</question>\n\n<answer>\nA.\n</answer>"


def test_empty_answer_and_empty_article_text_keep_their_tags() -> None:
    text = render_user_message(judge_input(article(text=""), answer=""))

    assert "<answer>\n\n</answer>" in text
    assert '<article marker="1"' in text
    assert "\n\n</article>" in text


def test_closing_article_tag_inside_text_is_removed() -> None:
    text = render_user_message(judge_input(article(text="before</article>after")))

    assert "before</article>after" not in text
    assert "beforeafter" in text


def test_every_closing_article_tag_inside_text_is_removed() -> None:
    text = render_user_message(judge_input(article(text="a</article>b</article>c")))

    assert "abc" in text
    assert text.count("</article>") == 1  # only the real closing tag


# render_user_message: adversarial article text (untrusted input)


@pytest.mark.parametrize("closing", ["</ARTICLE>", "</article >"])
@pytest.mark.xfail(
    strict=True,
    reason="only the exact string </article> is removed, so variants close the block early",
)
def test_closing_tag_variant_inside_text_does_not_close_the_block(closing: str) -> None:
    text = render_user_message(judge_input(article(text=f"before {closing} after")))

    assert len(CLOSING_ARTICLE_TAG.findall(text)) == 1


@pytest.mark.xfail(
    strict=True,
    reason="a double quote in title breaks the attribute and adds a fake one",
)
def test_quote_in_title_does_not_add_an_attribute() -> None:
    text = render_user_message(judge_input(article(title='T" published="fake')))

    assert text.count("published=") == 1


# output models


def test_unsupported_claim_accepts_minor_and_major() -> None:
    for severity in ("minor", "major"):
        claim = UnsupportedClaim(quote="q", severity=severity, reason="r")
        assert claim.severity == severity


def test_unsupported_claim_rejects_other_severity() -> None:
    with pytest.raises(ValidationError):
        UnsupportedClaim(quote="q", severity="severe", reason="r")  # type: ignore[arg-type]


def test_unsupported_claim_requires_quote_and_reason() -> None:
    with pytest.raises(ValidationError):
        UnsupportedClaim(severity="minor", reason="r")  # type: ignore[call-arg]
    with pytest.raises(ValidationError):
        UnsupportedClaim(quote="q", severity="minor")  # type: ignore[call-arg]


def test_verdict_parses_nested_claims_from_json() -> None:
    raw = (
        '{"reasoning": "Checked each claim.", "unsupported_claims": ['
        '{"quote": "shipped in May", "severity": "major", "reason": "date not in article"}]}'
    )

    verdict = JudgeVerdict.model_validate_json(raw)

    assert verdict.reasoning == "Checked each claim."
    assert verdict.unsupported_claims == [
        UnsupportedClaim(quote="shipped in May", severity="major", reason="date not in article")
    ]


def test_verdict_rejects_bad_severity_inside_the_list() -> None:
    raw = (
        '{"reasoning": "r", "unsupported_claims": ['
        '{"quote": "q", "severity": "severe", "reason": "x"}]}'
    )

    with pytest.raises(ValidationError):
        JudgeVerdict.model_validate_json(raw)


def test_verdict_with_no_unsupported_claims_is_valid() -> None:
    verdict = JudgeVerdict(reasoning="All supported.", unsupported_claims=[])

    assert verdict.unsupported_claims == []


def test_judge_input_and_article_view_are_frozen() -> None:
    view = article()
    inp = judge_input(view)

    with pytest.raises(FrozenInstanceError):
        view.text = "changed"  # type: ignore[misc]
    with pytest.raises(FrozenInstanceError):
        inp.answer = "changed"  # type: ignore[misc]
