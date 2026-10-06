import time
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any


def struct_time_to_utc(parsed: time.struct_time | None) -> datetime | None:
    """Attach UTC to a feedparser date tuple.

    feedparser already shifts parsed dates to UTC but returns them without an offset,
    so this only tags the value; it does not convert it.
    """
    if parsed is None:
        return None
    year, month, day, hour, minute, second = parsed[:6]
    # A leap second (60) is valid in struct_time but makes datetime raise.
    return datetime(year, month, day, hour, minute, min(second, 59), tzinfo=UTC)


def entry_published_at(entry: Mapping[str, Any]) -> datetime | None:
    """Publication time of a feed entry, falling back to its updated time."""
    parsed = entry.get("published_parsed")
    # Check membership first: FeedParserDict.get("updated_parsed") silently maps a
    # missing key to published_parsed and emits a DeprecationWarning.
    if parsed is None and "updated_parsed" in entry:
        parsed = entry["updated_parsed"]
    return struct_time_to_utc(parsed)
