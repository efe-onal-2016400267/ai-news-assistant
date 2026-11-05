import os
from pathlib import Path
from urllib.parse import quote

REQUIRED_ENV = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "POSTGRES_USER",
    "POSTGRES_PASSWORD",
    "POSTGRES_DB",
)


def preflight(store_path: Path) -> None:
    """Fail at startup with a clear message instead of failing on the first request."""
    missing = [name for name in REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        raise RuntimeError(f"Missing environment variables: {', '.join(missing)}")
    if not store_path.is_file():
        raise RuntimeError(f"News db not found: {store_path}")


def db_uri() -> str:
    user = quote(os.environ["POSTGRES_USER"], safe="")
    pw = quote(os.environ["POSTGRES_PASSWORD"], safe="")
    db = os.environ["POSTGRES_DB"]
    host = os.environ.get("POSTGRES_HOST", "localhost")
    port = os.environ.get("POSTGRES_PORT", "5432")

    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"
