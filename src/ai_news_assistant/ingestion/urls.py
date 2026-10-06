import hashlib
from urllib.parse import unquote_plus, urlsplit, urlunsplit

TRACKING_PARAMS: frozenset[str] = frozenset(
    {"fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid"}
)
DEFAULT_PORTS: dict[str, str] = {"http": "80", "https": "443"}


def _is_tracking(key: str) -> bool:
    key = unquote_plus(key).lower()
    return key.startswith("utm_") or key in TRACKING_PARAMS


def _strip_tracking(query: str) -> str:
    # Filter the raw "key=value" segments instead of decoding and re-encoding them,
    # so the parameters we keep stay byte-for-byte as the publisher wrote them.
    kept = [
        segment
        for segment in query.split("&")
        if segment and not _is_tracking(segment.partition("=")[0])
    ]
    return "&".join(kept)


def normalize_url(url: str) -> str:
    """Return a canonical form of an article URL for dedupe and hashing.

    Lowercases scheme and host, drops the fragment and default port, and strips
    tracking parameters. The path, including any trailing slash, is left as is.
    """
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    default_port = DEFAULT_PORTS.get(scheme)
    if default_port is not None:
        netloc = netloc.removesuffix(f":{default_port}")
    return urlunsplit((scheme, netloc, parts.path, _strip_tracking(parts.query), ""))


def article_id(url: str) -> str:
    """Stable ID for an article: SHA-256 hex digest of its normalized URL."""
    return hashlib.sha256(normalize_url(url).encode("utf-8")).hexdigest()
