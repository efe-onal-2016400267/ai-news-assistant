from collections.abc import Callable

import httpx
import pytest

from ai_news_assistant.downloads import USER_AGENT
from ai_news_assistant.extraction.robots import RobotsPolicy, RobotsUnavailableError

PAGE = "https://example.com/news/story"
PRIVATE = "https://example.com/private/page"


class RobotsServer:
    """Serves robots.txt per host from a handler and counts the requests."""

    def __init__(self, respond: Callable[[httpx.Request], httpx.Response]) -> None:
        self.respond = respond
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.respond(request)

    def policy(self, **kwargs: int) -> RobotsPolicy:
        return RobotsPolicy(httpx.Client(transport=httpx.MockTransport(self)), **kwargs)


def serving(text: str, status: int = 200) -> RobotsServer:
    return RobotsServer(lambda request: httpx.Response(status, text=text))


def test_allows_and_disallows_by_rules_for_everyone() -> None:
    policy = serving("User-agent: *\nDisallow: /private/\n").policy()

    assert policy.allowed(PAGE)
    assert not policy.allowed(PRIVATE)


def test_rules_for_our_user_agent_win_over_the_default() -> None:
    robots = "User-agent: ai-news-assistant\nDisallow: /\n\nUser-agent: *\nAllow: /\n"

    assert not serving(robots).policy().allowed(PAGE)


def test_disallow_everything() -> None:
    assert not serving("User-agent: *\nDisallow: /\n").policy().allowed(PAGE)


def test_empty_robots_txt_allows_everything() -> None:
    assert serving("").policy().allowed(PRIVATE)


@pytest.mark.parametrize("status", [401, 403, 404, 410])
def test_client_error_means_no_rules(status: int) -> None:
    policy = serving("User-agent: *\nDisallow: /\n", status=status).policy()

    assert policy.allowed(PAGE)
    assert policy.crawl_delay(PAGE) is None


@pytest.mark.parametrize("status", [500, 503])
def test_server_error_makes_robots_unavailable(status: int) -> None:
    policy = serving("", status=status).policy()

    with pytest.raises(RobotsUnavailableError, match=f"HTTP {status}"):
        policy.allowed(PAGE)


def test_network_error_makes_robots_unavailable() -> None:
    def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    with pytest.raises(RobotsUnavailableError, match="ReadTimeout"):
        RobotsServer(timeout).policy().allowed(PAGE)


def test_oversized_robots_txt_is_unavailable() -> None:
    policy = serving("User-agent: *\n" + "Disallow: /x\n" * 100).policy(max_bytes=100)

    with pytest.raises(RobotsUnavailableError, match="ResponseTooLargeError"):
        policy.allowed(PAGE)


def test_robots_txt_is_downloaded_once_per_host() -> None:
    server = serving("User-agent: *\nDisallow: /private/\n")
    policy = server.policy()

    policy.allowed(PAGE)
    policy.allowed(PRIVATE)
    policy.crawl_delay(PAGE)
    policy.allowed("https://other.example.org/a")

    assert [str(request.url) for request in server.requests] == [
        "https://example.com/robots.txt",
        "https://other.example.org/robots.txt",
    ]


def test_unavailable_host_is_not_asked_again_in_the_same_run() -> None:
    server = serving("", status=503)
    policy = server.policy()

    for _ in range(3):
        with pytest.raises(RobotsUnavailableError):
            policy.allowed(PAGE)

    assert len(server.requests) == 1


def test_sends_our_user_agent() -> None:
    server = serving("")

    server.policy().allowed(PAGE)

    assert server.requests[0].headers["User-Agent"] == USER_AGENT


@pytest.mark.parametrize(
    ("robots", "expected"),
    [
        ("User-agent: *\nCrawl-delay: 5\n", 5.0),
        ("User-agent: ai-news-assistant\nCrawl-delay: 10\n\nUser-agent: *\nCrawl-delay: 1\n", 10.0),
        ("User-agent: *\nDisallow: /private/\n", None),
        ("User-agent: *\nCrawl-delay: 1.5\n", None),  # The stdlib parser ignores fractions
    ],
)
def test_crawl_delay(robots: str, expected: float | None) -> None:
    assert serving(robots).policy().crawl_delay(PAGE) == expected
