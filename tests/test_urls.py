import re

import pytest

from ai_news_assistant.ingestion.urls import TRACKING_PARAMS, article_id, normalize_url


def test_lowercases_scheme_and_host_but_not_path() -> None:
    assert normalize_url("HTTPS://Example.COM/News/A") == "https://example.com/News/A"


def test_drops_fragment() -> None:
    assert normalize_url("https://example.com/a#comments") == "https://example.com/a"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://example.com:80/a", "http://example.com/a"),
        ("https://example.com:443/a", "https://example.com/a"),
        ("https://example.com:8443/a", "https://example.com:8443/a"),
    ],
)
def test_drops_only_default_port(url: str, expected: str) -> None:
    assert normalize_url(url) == expected


@pytest.mark.parametrize(
    "param", ["utm_source", "utm_medium", "UTM_Campaign", *sorted(TRACKING_PARAMS)]
)
def test_strips_tracking_param(param: str) -> None:
    assert normalize_url(f"https://example.com/a?{param}=x") == "https://example.com/a"


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("https://techcrunch.com/?p=123", "https://techcrunch.com/?p=123"),
        ("https://example.com/a?ref=main", "https://example.com/a?ref=main"),
        ("https://example.com/a?a=1&utm_source=x&b=2", "https://example.com/a?a=1&b=2"),
        ("https://example.com/a?q=a/b&fbclid=x", "https://example.com/a?q=a/b"),
    ],
)
def test_keeps_real_params_in_order(url: str, expected: str) -> None:
    assert normalize_url(url) == expected


def test_drops_question_mark_when_only_tracking_params() -> None:
    url = "https://example.com/a?utm_source=x&gclid=y"
    assert normalize_url(url) == "https://example.com/a"


def test_keeps_trailing_slash_as_is() -> None:
    with_slash = normalize_url("https://techcrunch.com/2026/10/05/story/")
    without_slash = normalize_url("https://www.theguardian.com/technology/2026/oct/05/story")

    assert with_slash == "https://techcrunch.com/2026/10/05/story/"
    assert without_slash == "https://www.theguardian.com/technology/2026/oct/05/story"
    assert normalize_url("https://example.com/a/") != normalize_url("https://example.com/a")


def test_strips_surrounding_whitespace() -> None:
    assert normalize_url("\n  https://example.com/a  \n") == "https://example.com/a"


@pytest.mark.parametrize(
    "url",
    [
        "HTTPS://Example.COM:443/News/A/?p=1&utm_source=x#top",
        "https://techcrunch.com/?p=3173232",
        "https://example.com/a?q=a%2Fb&fbclid=x",
    ],
)
def test_normalize_is_idempotent(url: str) -> None:
    once = normalize_url(url)
    assert normalize_url(once) == once


def test_article_id_is_sha256_hex() -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", article_id("https://example.com/a"))


def test_article_id_is_stable() -> None:
    assert (
        article_id("https://example.com/news/story")
        == "906074d8fb3ef840e7804b45f5be9eaadc6f2908310da75e8d5d62855c3edf6d"
    )


def test_article_id_ignores_tracking_and_fragment() -> None:
    plain = article_id("https://example.com/news/story")
    noisy = article_id("HTTPS://Example.com:443/news/story?utm_source=rss&fbclid=x#top")
    assert plain == noisy


def test_article_id_differs_between_articles() -> None:
    assert article_id("https://example.com/news/a") != article_id("https://example.com/news/b")
