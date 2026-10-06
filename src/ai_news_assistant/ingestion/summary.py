import re
from html.parser import HTMLParser

BLOCK_TAGS = frozenset(
    {"p", "li", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "blockquote", "ul", "ol", "tr"}
)
SKIP_TAGS = frozenset({"script", "style", "template", "noscript"})

CONTINUE_READING = re.compile(r"continue reading(\.\.\.|…)?", re.IGNORECASE)
# Entity-escaped markup (&lt;script&gt;) decodes into script-looking text; drop it too.
ESCAPED_SCRIPT = re.compile(r"<script\b[^>]*>.*?</script\s*>", re.IGNORECASE | re.DOTALL)
STRAY_SCRIPT_TAG = re.compile(r"</?script\b[^>]*>", re.IGNORECASE)


class _BlockTextParser(HTMLParser):
    """Collects text as blocks, split at block-level tags, skipping script-like content."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.blocks: list[tuple[str, bool]] = []  # (text, inside a list item)
        self._parts: list[str] = []
        self._li_depth = 0
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth += 1
        elif tag in BLOCK_TAGS:
            self._flush()
            if tag == "li":
                self._li_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in SKIP_TAGS:
            self._skip_depth = max(self._skip_depth - 1, 0)
        elif tag in BLOCK_TAGS:
            self._flush()
            if tag == "li":
                self._li_depth = max(self._li_depth - 1, 0)

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            self._parts.append(data)

    def close(self) -> None:
        super().close()
        self._flush()

    def _flush(self) -> None:
        text = "".join(self._parts)
        self._parts = []
        text = STRAY_SCRIPT_TAG.sub(" ", ESCAPED_SCRIPT.sub(" ", text))
        text = " ".join(text.split())
        if text:
            self.blocks.append((text, self._li_depth > 0))


def _is_boilerplate(text: str, in_list_item: bool) -> bool:
    if CONTINUE_READING.fullmatch(text):
        return True
    # Guardian promo list: "Get our new political email, free app or daily news podcast".
    return in_list_item and text.startswith("Get our")


def clean_summary(raw: str | None) -> str | None:
    """Turn a feed summary into plain text, one blank line between blocks.

    Strips HTML (including script and style content), drops the Guardian
    "Continue reading..." trailer and "Get our ..." promo list items.
    Returns None if nothing is left.
    """
    if raw is None:
        return None
    parser = _BlockTextParser()
    parser.feed(raw)
    parser.close()
    blocks = [text for text, in_li in parser.blocks if not _is_boilerplate(text, in_li)]
    return "\n\n".join(blocks) or None
