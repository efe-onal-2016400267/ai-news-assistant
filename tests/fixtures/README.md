# Test fixtures

Copies of real RSS feeds, used so tests never touch the network.

| File | Source | Captured |
| ---- | ------ | -------- |
| `guardian_ai.xml` | https://www.theguardian.com/technology/artificialintelligenceai/rss | 2026-10-05 |
| `techcrunch_ai.xml` | https://techcrunch.com/category/artificial-intelligence/feed/ | 2026-10-05 |

- Regenerate with `scripts/capture_fixtures.py`, then recheck the tests that depend on specific feed content (for example the publisher name and the Guardian summary cleaning).
- Guardian summaries are HTML; TechCrunch summaries are plain text. Both cases are covered.