# Test fixtures

Copies of real RSS and Atom feeds, trimmed to the first 20 entries, used so tests never touch
the network. There is one file per feed in `ingestion/feeds.py`: feed `a-b` is saved as `a_b.xml`.

| File | Source | Format | Captured |
| ---- | ------ | ------ | -------- |
| `guardian_ai.xml` | https://www.theguardian.com/technology/artificialintelligenceai/rss | RSS | 2026-10-05 |
| `techcrunch_ai.xml` | https://techcrunch.com/category/artificial-intelligence/feed/ | RSS | 2026-10-05 |
| `verge_ai.xml` | https://www.theverge.com/rss/ai-artificial-intelligence/index.xml | Atom | 2026-10-08 |
| `arstechnica_ai.xml` | https://arstechnica.com/ai/feed/ | RSS | 2026-10-08 |
| `wired_ai.xml` | https://www.wired.com/feed/tag/ai/latest/rss | RSS | 2026-10-08 |
| `mit_tech_review_ai.xml` | https://www.technologyreview.com/topic/artificial-intelligence/feed | RSS | 2026-10-08 |
| `the_decoder.xml` | https://the-decoder.com/feed/ | RSS | 2026-10-08 |
| `openai_news.xml` | https://openai.com/news/rss.xml | RSS | 2026-10-08 |
| `deepmind_blog.xml` | https://deepmind.google/blog/rss.xml | RSS | 2026-10-08 |
| `huggingface_blog.xml` | https://huggingface.co/blog/feed.xml | RSS | 2026-10-08 |
| `mistral_news.xml` | https://mistral.ai/news/rss | RSS | 2026-10-08 |
| `simon_willison.xml` | https://simonwillison.net/atom/everything/ | Atom | 2026-10-08 |

- Regenerate with `uv run python scripts/capture_fixtures.py [feed-name ...]` (no names: all
  feeds). Then recheck the tests that depend on specific feed content, for example the Guardian
  summary cleaning, the expected tags and the `MAY_LACK_*` sets in `test_parse_feed.py`.
- Guardian and The Decoder summaries are HTML. TechCrunch summaries are plain text. The Decoder
  ends each summary with a WordPress "The article ... appeared first on The Decoder." trailer.
- The Verge titles are double-escaped (`&amp;#8217;`), so they are unescaped once more.
- Missing fields: DeepMind, Hugging Face and Mistral have no tags, and some or all of their
  summaries are missing. OpenAI lacks tags on some entries. Simon Willison lacks tags on one entry.
- `openai_news.xml` repeats one link, which tests the duplicate-link skip on real data.
- Feed titles are not publisher names ("Feed: Artificial Intelligence Latest" for Wired), so each
  feed's source name is configured in `feeds.py`.
