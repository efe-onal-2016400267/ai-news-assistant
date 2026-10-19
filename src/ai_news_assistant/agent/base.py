from datetime import datetime


def format_date(published_at: str | None) -> str:
    """Convert the datetime string into date string."""
    if published_at is None:
        return "date unknown"
    dt = datetime.fromisoformat(published_at)
    return f"{dt.day} {dt:%b %Y}"
