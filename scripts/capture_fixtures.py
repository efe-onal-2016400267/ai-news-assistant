"""Download feeds into tests/fixtures, trimmed to the first KEEP entries.

Usage: uv run python scripts/capture_fixtures.py [feed-name ...]
With no names, every feed in FEEDS is captured.
"""

import re
import sys
from pathlib import Path

from ai_news_assistant.ingestion.feeds import FEEDS
from ai_news_assistant.ingestion.rss import download_feed

OUT_DIR = Path("tests/fixtures")
KEEP = 20
# RSS <item> or Atom <entry>; works on bytes so each file keeps its declared encoding.
ENTRY = re.compile(rb"<(item|entry)\b[^>]*>.*?</\1\s*>", flags=re.DOTALL)


def fixture_path(name: str) -> Path:
    return OUT_DIR / f"{name.replace('-', '_')}.xml"


def keep_first_entries(xml: bytes, keep: int) -> bytes:
    """Keep the first `keep` items or entries, and everything after the last one."""
    entries = list(ENTRY.finditer(xml))
    if len(entries) <= keep:
        return xml
    return xml[: entries[keep].start()].rstrip() + b"\n" + xml[entries[-1].end() :].lstrip()


def main(names: list[str]) -> None:
    unknown = sorted(set(names) - FEEDS.keys())
    if unknown:
        sys.exit(f"Unknown feeds: {', '.join(unknown)}. Known: {', '.join(FEEDS)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name in names or FEEDS:
        xml = download_feed(FEEDS[name].url)
        path = fixture_path(name)
        path.write_bytes(keep_first_entries(xml, KEEP))
        print(f"Saved {name} to {path}")


if __name__ == "__main__":
    main(sys.argv[1:])
