# HTML page fixtures

One real article page per feed, saved as returned by `fetch_html`, so extraction tests never touch
the network. Each page is the first entry of that feed's saved RSS fixture in `tests/rss_fixtures/`,
so feed and page fixtures belong together. Feed `a-b` is saved as `a_b.html`.

| File | URL | Captured | Size |
| ---- | --- | -------- | ---- |
| `guardian_ai.html` | https://www.theguardian.com/technology/2026/oct/05/internet-watch-foundation-huge-rise-ai-child-sexual-abuse-material | 2026-10-10 | 385 KB |
| `techcrunch_ai.html` | https://techcrunch.com/2026/10/05/hot-girl-hotline-is-like-dear-abby-for-the-ai-era/ | 2026-10-10 | 238 KB |
| `verge_ai.html` | https://www.theverge.com/ai-artificial-intelligence/1007276/openai-chatgpt-intelligent-ui-gpt-6 | 2026-10-10 | 396 KB |
| `arstechnica_ai.html` | https://arstechnica.com/ai/2026/10/nvidias-big-bet-on-physical-ai-aims-for-safer-robotaxis-humanoid-robots/ | 2026-10-10 | 154 KB |
| `wired_ai.html` | https://www.wired.com/story/tristan-harris-tech-nonprofit-is-laying-off-most-staff-and-going-founder-led/ | 2026-10-10 | 1407 KB |
| `mit_tech_review_ai.html` | https://www.technologyreview.com/2026/10/08/1145923/ai-breakthroughs-in-robotics-wont-change-your-life-any-time-soon/ | 2026-10-10 | 525 KB |
| `the_decoder.html` | https://the-decoder.com/claude-haiku-5-5-arrives-with-massive-price-cuts-proving-the-ai-pricing-arms-race-is-far-from-over/ | 2026-10-10 | 126 KB |
| `deepmind_blog.html` | https://deepmind.google/blog/embeddinggemma-2-an-open-lightweight-multimodal-embedding-model/ | 2026-10-10 | 978 KB |
| `huggingface_blog.html` | https://huggingface.co/blog/LiquidAI/open-d1 | 2026-10-10 | 190 KB |
| `mistral_news.html` | https://mistral.ai/news/mistral-large-4/ | 2026-10-10 | 293 KB |
| `simon_willison.html` | https://simonwillison.net/2026/Oct/7/ben-affleck/ | 2026-10-10 | 13 KB |

- Regenerate with `uv run python scripts/capture_html_fixtures.py [feed-name ...]` (no names: all
  feeds). It checks robots.txt with our User-Agent and skips a page that is disallowed or fails.
  Then update the table above and `expected.json`.
- All pages are well under `MAX_PAGE_BYTES` (5 MB); the largest is Wired at 1.4 MB. No page
  needed replacement characters when decoded.

## Blocked: OpenAI

`openai-news` has no page fixture. openai.com answers our client with a Cloudflare bot challenge
(`403`, `cf-mitigated: challenge`), whatever the User-Agent, even with trafilatura's own; trafilatura's
urllib3 client gets through, so Cloudflare tells the clients apart by connection fingerprint, not
headers. robots.txt allows the pages. We do not change clients to get past bot protection:
OpenAI articles keep their feed summary, and the pipeline records the 403 as a permanent failure.

## `expected.json`

For each page: its `url` and the `first` and `last` sentence of the article body (not captions,
navigation or teasers), so extraction tests can check that the text is complete and not cut in
the middle. Drafted from the saved HTML and checked against the live page.

Notes for the extraction step, from comparing trafilatura's `extract()` and `baseline()` on
these pages:

- Ars Technica: `extract()` stops mid-article (after "…Goel said."); only `baseline()` reaches
  the last sentence. `baseline()` then continues into the author bio and comment count.
- Wired and Hugging Face: the first sentence is only in `extract()`; `baseline()` starts later.
- TechCrunch: `baseline()` starts with the site navigation.
- `extract()` keeps trailing boilerplate on MIT Technology Review ("Deep Dive", "Stay
  connected"), The Decoder (subscription box), Mistral (pricing card), Simon Willison (recent
  posts) and Wired ("Comments", "Back to top").
- `extract(favor_recall=True)` returned the same text as `extract()` on every page.
