import re

import lxml.etree
import lxml.html
import trafilatura

MIN_TEXT_CHARS = 200  # Shorter results are cookie notices or teasers, not articles
# Inline styles that hide an element, with any spacing or case.
_HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden", re.I)
# Some sites hide the whole page until scripts run, so these are never dropped.
_NEVER_DROPPED = {"html", "head", "body"}
# extract() sometimes stops mid-article. baseline() is noisier but rarely cut, so when
# extract()'s text contains less than MIN_COVERAGE of baseline's long sentences, baseline
# is used. Measured on the page fixtures: the cut Ars Technica page 0.30, all others 0.71-1.00.
MIN_COVERAGE = 0.5
MIN_SENTENCE_CHARS = 60  # Shorter "sentences" are mostly navigation and captions
_SENTENCE_END = re.compile(r"(?<=[.!?…\"”’)])\s+")


def extract_text_from_html(html: str) -> str | None:
    """Return the main article text of a page as plain text, or None if there is none.

    Takes the output of `fetch_html` and never touches the network. The HTML is untrusted:
    bad or empty markup returns None instead of raising, and the returned text is data to
    store, never instructions. Comments sections are left out.

    Hidden elements are dropped first: readers never see them, so they are a place to plant
    instructions, and text hidden until you subscribe is paywalled content we don't take.

    trafilatura's extract() gives the cleanest text but sometimes stops mid-article; then its
    noisier baseline() is used instead (see `_choose`).
    """
    tree = _parse(html)
    if tree is None:
        return None
    _drop_hidden(tree)
    extracted = trafilatura.extract(tree, include_comments=False)
    fallback = trafilatura.baseline(tree)[1]  # extract() leaves the tree unchanged
    text = (_choose(extracted, fallback) or "").strip()
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


def _choose(extracted: str | None, fallback: str | None) -> str | None:
    """Pick extract()'s text unless it looks cut compared with baseline()'s.

    When extract() finds nothing, the page is not used at all: baseline() then falls back to
    the whole page body, which on a login wall or cookie page is not an article.
    """
    if not extracted or not extracted.strip():
        return None
    if not fallback or _coverage(extracted, fallback) >= MIN_COVERAGE:
        return extracted
    return fallback


def _coverage(candidate: str, reference: str) -> float:
    """Share of the reference's long sentences that appear in the candidate text."""
    sentences = _sentences(reference)
    if not sentences:
        return 1.0  # Nothing to judge by
    text = " ".join(candidate.split())
    return sum(sentence in text for sentence in sentences) / len(sentences)


def _sentences(text: str) -> list[str]:
    parts = _SENTENCE_END.split(" ".join(text.split()))
    return [part for part in parts if len(part) >= MIN_SENTENCE_CHARS]


def _is_hidden(element: lxml.html.HtmlElement) -> bool:
    return (
        element.get("hidden") is not None
        or (element.get("aria-hidden") or "").strip().lower() == "true"
        or _HIDDEN_STYLE.search(element.get("style") or "") is not None
    )
