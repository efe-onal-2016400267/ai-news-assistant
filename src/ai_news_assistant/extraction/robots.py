from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from ai_news_assistant.downloads import (
    DEFAULT_TIMEOUT,
    USER_AGENT,
    ResponseTooLargeError,
    download,
)

MAX_ROBOTS_BYTES = 512 * 1024  # RFC 9309 asks crawlers to parse at least 500 KiB


class RobotsUnavailableError(Exception):
    """A host's robots.txt could not be read, so its pages are not fetched for now."""

    def __init__(self, origin: str, reason: str) -> None:
        super().__init__(f"robots.txt for {origin} unavailable: {reason}")
        self.origin = origin
        self.reason = reason


class RobotsPolicy:
    """Answers whether our User-Agent may fetch a URL, following each host's robots.txt.

    Each robots.txt is downloaded at most once per instance, so create one per run. Status
    handling follows RFC 9309: a 4xx response means there are no rules (everything is
    allowed); a 5xx response, a network error or an oversized file raises
    RobotsUnavailableError, and the same error is raised again for that host for the rest
    of the run.
    """

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = MAX_ROBOTS_BYTES,
    ) -> None:
        self.client = client
        self.timeout = timeout
        self.max_bytes = max_bytes
        self._parsers: dict[str, RobotFileParser | RobotsUnavailableError] = {}

    def allowed(self, url: str) -> bool:
        return self._parser(url).can_fetch(USER_AGENT, url)

    def crawl_delay(self, url: str) -> float | None:
        """The Crawl-delay in seconds the host asks of our User-Agent, if any."""
        delay = self._parser(url).crawl_delay(USER_AGENT)
        if delay is None:
            return None
        try:
            return max(float(delay), 0.0)
        except ValueError:
            return None

    def _parser(self, url: str) -> RobotFileParser:
        parts = urlsplit(url)
        origin = f"{parts.scheme}://{parts.netloc}".lower()
        if origin not in self._parsers:
            self._parsers[origin] = self._load(origin)
        cached = self._parsers[origin]
        if isinstance(cached, RobotsUnavailableError):
            raise cached
        return cached

    def _load(self, origin: str) -> RobotFileParser | RobotsUnavailableError:
        parser = RobotFileParser()
        try:
            robots = download(
                f"{origin}/robots.txt",
                self.client,
                max_bytes=self.max_bytes,
                timeout=self.timeout,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if 400 <= status < 500:
                parser.parse([])  # No rules: everything is allowed
                return parser
            return RobotsUnavailableError(origin, f"HTTP {status}")
        except (httpx.TransportError, ResponseTooLargeError) as exc:
            return RobotsUnavailableError(origin, type(exc).__name__)
        parser.parse(robots.content.decode("utf-8", errors="replace").splitlines())
        return parser
