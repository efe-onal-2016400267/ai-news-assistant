import re

import lxml.etree
import lxml.html
import trafilatura

MIN_TEXT_CHARS = 200  # Shorter results are cookie notices or teasers, not articles
# Inline styles that hide an element, with any spacing or case.
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
# Some sites hide the whole page until scripts run, so these are never dropped.
_NEVER_DROPPED = {"html", "head", "body"}


def extract_text_from_html(html: str) -> str | None:
    """Return the main article text of a page as plain text, or None if there is none.

    Takes the output of `fetch_html` and never touches the network. The HTML is untrusted:
    bad or empty markup returns None instead of raising, and the returned text is data to
    store, never instructions. Comments sections are left out.

    Hidden elements are dropped first: readers never see them, so they are a place to plant
    instructions, and text hidden until you subscribe is paywalled content we don't take.
    """
    tree = _parse(html)
    if tree is None:
        return None
    _drop_hidden(tree)
    text = (trafilatura.extract(tree, include_comments=False) or "").strip()
    if len(text) < MIN_TEXT_CHARS:
        return None
    return text


def _parse(html: str) -> lxml.html.HtmlElement | None:
    if not html.strip():
        return None
    # Parse UTF-8 bytes with a fixed encoding: a str with an XML declaration raises
    # ValueError, and a <meta charset> must not re-decode text fetch_html already decoded.
    # Lone surrogates can't be encoded, so they are replaced instead of raising.
    parser = lxml.html.HTMLParser(encoding="utf-8")
    try:
        return lxml.html.document_fromstring(html.encode("utf-8", errors="replace"), parser=parser)
    except lxml.etree.ParserError:
        return None


def _drop_hidden(tree: lxml.html.HtmlElement) -> None:
    """Remove hidden elements and their content in place, keeping the text that follows."""
    hidden = [
        element
        for element in tree.iter()
        if isinstance(element.tag, str)  # Skips comments and processing instructions
        and element.tag not in _NEVER_DROPPED
        and _is_hidden(element)
    ]
    for element in hidden:
        if element.getparent() is not None:
            element.drop_tree()


def _is_hidden(element: lxml.html.HtmlElement) -> bool:
    return (
        element.get("hidden") is not None
        or (element.get("aria-hidden") or "").strip().lower() == "true"
        or _HIDDEN_STYLE.search(element.get("style") or "") is not None
    )
