"""Download one article page per feed into tests/text_html_fixtures.

The page is the first entry of the feed's saved RSS fixture, so feed and page fixtures belong
together. Pages disallowed by robots.txt, or whose robots.txt can't be read, are skipped.
Failures are reported per feed and don't stop the run.

Usage: uv run python scripts/capture_html_fixtures.py [feed-name ...]
With no names, every feed in FEEDS is captured.
"""

import sys
import time
from datetime import date
from pathlib import Path
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from ai_news_assistant.downloads import USER_AGENT, download
from ai_news_assistant.extraction.fetch import fetch_html
from ai_news_assistant.ingestion.feeds import FEEDS
from ai_news_assistant.ingestion.rss import parse_feed

OUT_DIR = Path("tests/text_html_fixtures")
RSS_DIR = Path("tests/rss_fixtures")
MAX_ROBOTS_BYTES = 512 * 1024
PAUSE_SECONDS = 1.0


def fixture_path(name: str) -> Path:
    return OUT_DIR / f"{name.replace('-', '_')}.html"


def first_article_url(name: str) -> str:
    rss = (RSS_DIR / f"{name.replace('-', '_')}.xml").read_bytes()
    articles = parse_feed(rss, FEEDS[name].source)
    if not articles:
        raise ValueError(f"RSS fixture for {name} has no valid entries")
    return str(articles[0].url)


def robots_allows(url: str, client: httpx.Client) -> bool:
    """Check robots.txt with our own User-Agent.

    RobotFileParser.read() is not used: it sends urllib's default User-Agent and treats a
    401/403 on robots.txt as "disallow all". A missing robots.txt (404) allows everything;
    any other error propagates, so the page is skipped as "couldn't check".
    """
    parts = urlsplit(url)
    robots_url = f"{parts.scheme}://{parts.netloc}/robots.txt"
    try:
        robots = download(robots_url, client, max_bytes=MAX_ROBOTS_BYTES)
    except httpx.HTTPStatusError as exc:
        if exc.response.status_code == 404:
            return True
        raise
    parser = RobotFileParser()
    parser.parse(robots.content.decode("utf-8", errors="replace").splitlines())
    return parser.can_fetch(USER_AGENT, url)


def capture(name: str, client: httpx.Client) -> str:
    """Save the page for one feed and return its README table row."""
    url = first_article_url(name)
    if not robots_allows(url, client):
        raise PermissionError(f"robots.txt disallows {url}")
    html = fetch_html(url, client)
    path = fixture_path(name)
    path.write_text(html, encoding="utf-8", newline="")
    size_kb = path.stat().st_size / 1024
    return f"| `{path.name}` | {url} | {date.today().isoformat()} | {size_kb:.0f} KB |"


def main(names: list[str]) -> None:
    unknown = sorted(set(names) - FEEDS.keys())
    if unknown:
        sys.exit(f"Unknown feeds: {', '.join(unknown)}. Known: {', '.join(FEEDS)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[str] = []
    failed: list[str] = []
    with httpx.Client() as client:
        for index, name in enumerate(names or FEEDS):
            if index:
                time.sleep(PAUSE_SECONDS)
            try:
                rows.append(capture(name, client))
                print(f"Saved {name}")
            except Exception as exc:
                failed.append(name)
                print(f"FAILED {name}: {type(exc).__name__}: {exc}")

    print("\nREADME rows:")
    print("| File | URL | Captured | Size |")
    print("| ---- | --- | -------- | ---- |")
    print("\n".join(rows))
    print(f"\nFailed or blocked: {', '.join(failed) or 'none'}")


if __name__ == "__main__":
    main(sys.argv[1:])
