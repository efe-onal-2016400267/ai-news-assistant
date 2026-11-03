import os
from pathlib import Path

REQUIRED_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
)


def preflight(store_path: Path) -> None:
    """Fail at startup with a clear message instead of failing on the first request."""
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    if not store_path.is_file():
        raise RuntimeError(f"News db not found: {store_path}")
