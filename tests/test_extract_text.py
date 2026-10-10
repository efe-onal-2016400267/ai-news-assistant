import json
from pathlib import Path
from typing import Any

import pytest

from ai_news_assistant.extraction.text import MIN_TEXT_CHARS, extract_text_from_html

FIXTURES = Path(__file__).parent / "text_html_fixtures"
EXPECTED: dict[str, dict[str, Any]] = json.loads(
    (FIXTURES / "expected.json").read_text(encoding="utf-8")
)
SENTENCE = "Real reporting about a model release, its pricing and what it means for developers."
ARTICLE = "".join(f"<p>{SENTENCE} Paragraph {n}.</p>" for n in range(6))
MARKER = "SHOULD NOT APPEAR"


def page(body: str, head: str = "") -> str:
    return f"<html><head>{head}</head><body><article>{body}</article></body></html>"


def norm(text: str) -> str:
    return " ".join(text.split())


def fixture_text(name: str) -> str:
    html = (FIXTURES / f"{name.replace('-', '_')}.html").read_text(encoding="utf-8")
    text = extract_text_from_html(html)
    assert text is not None
    return norm(text)


def test_extracts_article_text() -> None:
    text = extract_text_from_html(page(ARTICLE))

    assert text is not None
    assert SENTENCE in text
    assert "Paragraph 5." in text


@pytest.mark.parametrize("html", ["", "   \n\t", "just some words", "\x00\x01<<<>>>", "\ud800"])
def test_degenerate_input_returns_none(html: str) -> None:
    assert extract_text_from_html(html) is None


def test_page_with_xml_declaration_is_extracted() -> None:
    html = '<?xml version="1.0" encoding="iso-8859-1"?>' + page(ARTICLE)

    text = extract_text_from_html(html)

    assert text is not None
    assert SENTENCE in text


def test_short_text_returns_none() -> None:
    short = "A short teaser."
    assert len(short) < MIN_TEXT_CHARS

    assert extract_text_from_html(page(f"<p>{short}</p>")) is None


def test_script_style_and_comments_are_left_out() -> None:
    body = ARTICLE + f"<script>var x = '{MARKER}';</script><!-- {MARKER} -->"
    html = page(body, head=f"<style>/* {MARKER} */ p {{ color: red; }}</style>")

    text = extract_text_from_html(html)

    assert text is not None
    assert MARKER not in text


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_fixture_contains_first_sentence(name: str) -> None:
    assert norm(EXPECTED[name]["first"]) in fixture_text(name)


@pytest.mark.parametrize(
    "name",
    [
        pytest.param(
            name,
            marks=pytest.mark.xfail(strict=True, reason="extract() cuts the article (Step 3)"),
        )
        if name == "arstechnica-ai"
        else name
        for name in sorted(EXPECTED)
    ],
)
def test_fixture_contains_last_sentence(name: str) -> None:
    assert norm(EXPECTED[name]["last"]) in fixture_text(name)
