import lxml.etree
import lxml.html
import trafilatura

MIN_TEXT_CHARS = 200  # Shorter results are cookie notices or teasers, not articles


def extract_text_from_html(html: str) -> str | None:
    """Return the main article text of a page as plain text, or None if there is none.

    Takes the output of `fetch_html` and never touches the network. The HTML is untrusted:
    bad or empty markup returns None instead of raising, and the returned text is data to
    store, never instructions. Comments sections are left out.
    """
    tree = _parse(html)
    if tree is None:
        return None
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
