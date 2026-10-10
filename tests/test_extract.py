import httpx
import pytest

from ai_news_assistant.extract import HostThrottle, Outcome, classify
from ai_news_assistant.extraction.fetch import NotHTMLError, PageTooLargeError
from ai_news_assistant.extraction.robots import RobotsUnavailableError

URL = "https://example.com/news/story"
REQUEST = httpx.Request("GET", URL)


def status_error(status: int) -> httpx.HTTPStatusError:
    response = httpx.Response(status, request=REQUEST)
    return httpx.HTTPStatusError(f"HTTP {status}", request=REQUEST, response=response)


@pytest.mark.parametrize("status", [400, 401, 403, 404, 410, 451])
def test_client_errors_are_permanent(status: int) -> None:
    assert classify(status_error(status)) is Outcome.FAILED


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_server_errors_timeouts_and_rate_limits_are_retried(status: int) -> None:
    assert classify(status_error(status)) is Outcome.RETRY


@pytest.mark.parametrize(
    "exc",
    [
        httpx.ReadTimeout("timed out", request=REQUEST),
        httpx.ConnectTimeout("timed out", request=REQUEST),
        httpx.ConnectError("refused", request=REQUEST),
        httpx.ReadError("reset", request=REQUEST),
        httpx.RemoteProtocolError("server hung up", request=REQUEST),
        httpx.ProxyError("proxy down", request=REQUEST),
        RobotsUnavailableError("https://example.com", "HTTP 503"),
    ],
)
def test_temporary_errors_are_retried(exc: Exception) -> None:
    assert classify(exc) is Outcome.RETRY


@pytest.mark.parametrize(
    "exc",
    [
        PageTooLargeError(URL, 100),
        NotHTMLError(URL, "application/pdf"),
        httpx.TooManyRedirects("redirect loop", request=REQUEST),
        httpx.DecodingError("bad gzip", request=REQUEST),
        httpx.UnsupportedProtocol("ftp", request=REQUEST),
        httpx.LocalProtocolError("bad header", request=REQUEST),
        httpx.InvalidURL("not a url"),
    ],
)
def test_permanent_errors_are_recorded_as_failed(exc: Exception) -> None:
    assert classify(exc) is Outcome.FAILED


@pytest.mark.parametrize("exc", [ValueError("bug"), KeyError("bug"), RuntimeError("bug")])
def test_unexpected_errors_are_bugs(exc: Exception) -> None:
    assert classify(exc) is Outcome.ERROR


class FakeClock:
    """A clock whose sleep moves time forward and records each wait."""

    def __init__(self) -> None:
        self.now = 1000.0
        self.sleeps: list[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


def throttle(clock: FakeClock, min_delay: float = 2.0, max_delay: float = 60.0) -> HostThrottle:
    return HostThrottle(min_delay, max_delay, clock=clock, sleep=clock.sleep)


def test_first_request_to_a_host_does_not_wait(clock: FakeClock) -> None:
    throttle(clock).wait(URL)

    assert clock.sleeps == []


def test_second_request_to_the_same_host_waits_the_gap(clock: FakeClock) -> None:
    limiter = throttle(clock)

    limiter.wait(URL)
    limiter.wait("https://example.com/other")

    assert clock.sleeps == [2.0]


def test_time_already_passed_is_subtracted(clock: FakeClock) -> None:
    limiter = throttle(clock)

    limiter.wait(URL)
    clock.now += 1.5
    limiter.wait(URL)
    clock.now += 5.0
    limiter.wait(URL)

    assert clock.sleeps == [0.5]


def test_other_hosts_do_not_wait(clock: FakeClock) -> None:
    limiter = throttle(clock)

    limiter.wait(URL)
    limiter.wait("https://other.example.org/a")

    assert clock.sleeps == []


def test_host_matching_ignores_case(clock: FakeClock) -> None:
    limiter = throttle(clock)

    limiter.wait(URL)
    limiter.wait("https://EXAMPLE.com/a")

    assert clock.sleeps == [2.0]


@pytest.mark.parametrize(
    ("crawl_delay", "expected"),
    [(10.0, 10.0), (0.5, 2.0), (None, 2.0), (600.0, 60.0)],
)
def test_crawl_delay_raises_the_gap_up_to_the_cap(
    clock: FakeClock, crawl_delay: float | None, expected: float
) -> None:
    limiter = throttle(clock)

    limiter.wait(URL, crawl_delay)
    limiter.wait(URL, crawl_delay)

    assert clock.sleeps == [expected]
