import re
from pathlib import Path

from ai_news_assistant.ingestion.rss import download_feed

FEEDS = {
    "techcrunch_ai": "https://techcrunch.com/category/artificial-intelligence/feed/",
    "guardian_ai": "https://www.theguardian.com/technology/artificialintelligenceai/rss",
}

OUT_DIR = Path("tests/fixtures")
KEEP = 20


def keep_first_items(xml: str, keep: int) -> str:
    """Keep the first keep items from the feed."""
    items = list(re.finditer(r"<item>.*?</item>", xml, flags=re.DOTALL))
    if len(items) <= keep:
        return xml
    return xml[: items[keep].start()].rstrip() + "\n" + xml[items[-1].end() :].lstrip()


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, url in FEEDS.items():
        xml = download_feed(url).decode("utf-8")  # download feed returns bytes, decode with utf-8
        (OUT_DIR / f"{name}.xml").write_text(
            keep_first_items(xml, KEEP), encoding="utf-8"
        )  # Write the first KEEP items to OUTDIR
        print(f"Saved {name}")


if __name__ == "__main__":
    main()
