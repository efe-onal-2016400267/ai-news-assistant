from dataclasses import dataclass


@dataclass(frozen=True)
class Feed:
    """One RSS or Atom feed. `source` is the publisher name stored on its articles."""

    url: str
    source: str


# Feed name -> feed. The name identifies the feed in ingest results, logs and fixture files.
FEEDS: dict[str, Feed] = {
    "guardian-ai": Feed(
        "https://www.theguardian.com/technology/artificialintelligenceai/rss", "The Guardian"
    ),
    "techcrunch-ai": Feed(
        "https://techcrunch.com/category/artificial-intelligence/feed/", "TechCrunch"
    ),
    "verge-ai": Feed(
        "https://www.theverge.com/rss/ai-artificial-intelligence/index.xml", "The Verge"
    ),
    "arstechnica-ai": Feed("https://arstechnica.com/ai/feed/", "Ars Technica"),
    "wired-ai": Feed("https://www.wired.com/feed/tag/ai/latest/rss", "Wired"),
    "mit-tech-review-ai": Feed(
        "https://www.technologyreview.com/topic/artificial-intelligence/feed",
        "MIT Technology Review",
    ),
    "the-decoder": Feed("https://the-decoder.com/feed/", "The Decoder"),
    # "openai-news": Feed("https://openai.com/news/rss.xml", "OpenAI"),
    "deepmind-blog": Feed("https://deepmind.google/blog/rss.xml", "Google DeepMind"),
    "huggingface-blog": Feed("https://huggingface.co/blog/feed.xml", "Hugging Face"),
    "mistral-news": Feed("https://mistral.ai/news/rss", "Mistral AI"),
    "simon-willison": Feed("https://simonwillison.net/atom/everything/", "Simon Willison"),
}
