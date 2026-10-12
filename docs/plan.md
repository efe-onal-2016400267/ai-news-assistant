# News Assistant: Project Plan

A production-style build of a news chatbot agent, using Claude Code, uv and GitHub the way a team would, with testing, evaluation and deployment included from the start.

---

## Phase 0: Foundations (day 1)

### uv
- `uv init --package news-assistant` gives a `src/` layout and a proper `pyproject.toml`.
- `uv add` for runtime dependencies, `uv add --dev` for dev tools (pytest, ruff, mypy). Commit `uv.lock`.
- Pin the Python version with `.python-version`, and run everything via `uv run` so nobody uses a stray environment.
- In CI and Docker, use `uv sync --frozen`.

### GitHub
- Protect `main`: PRs only, with required status checks.
- Work in issues, then a branch per issue, then a small PR. Even solo, write the issue and PR descriptions as you would for a team.
- CI workflow that runs ruff, mypy, pytest and the fast evals on every PR.
- Add `.github/pull_request_template.md`, Dependabot, and a pre-commit config.

### Claude Code
- Write a `CLAUDE.md` with the repo layout, commands (`uv run pytest`, `uv run ruff check`), conventions, and what not to touch. Keep it short and update it as the project grows.
- Use plan mode for anything non-trivial and review the plan before it edits.
- Add a few project slash commands under `.claude/commands/`, such as `/add-eval-case` and `/review`.
- Add hooks that run ruff and tests after edits.
- Use the GitHub integration so Claude drafts PRs and reviews them.

---

## Phase 1: Architecture

Keep the layers separate so each one can be tested alone:

1. **Ingestion**: RSS and news APIs, deduplication, and storage with metadata (source, date, URL).
2. **Retrieval**: chunking, embeddings, a vector store plus date and source filters. Recency matters a lot for news.
3. **Agent**: tool-calling loop with tools like `search_news`, `get_article`, `summarize_topic`, and `compare_sources`. Add conversation memory.
4. **API**: FastAPI with streaming, plus a thin UI (Streamlit or a simple chat front end).
5. **Observability**: structured logging and tracing (Langfuse or OpenTelemetry) from the start.

---

## Phase 2: Test and Evaluate

- **Unit tests** for ingestion, chunking and tool functions, with the LLM mocked.
- **Retrieval evals** on a hand-labeled set of 30-50 questions, with recall@k and MRR.
- **Answer evals**: groundedness and citation correctness with an LLM judge, calibrated against a few human-labeled examples.
- **Agent trajectory evals**: did it call the right tools, and did it refuse to answer without sources?
- **Regression gate**: a small eval subset runs in CI, and the full suite runs nightly. Failing a threshold blocks the merge.
- **Adversarial cases**: prompt injection inside article text is a real risk for a news agent, so test for it.

---

## Phase 3: Deploy

- Multi-stage Dockerfile built on uv.
- GitHub Actions builds the image, pushes it to a registry, and deploys to a staging environment, with manual promotion to prod.
- Pick a managed container host (Azure Container Apps, Cloud Run or similar). Keep secrets out of the repo and use environment-based config.
- Add a health endpoint, a rate limit and a cost budget per request.
- Add a scheduled ingestion job with monitoring, so the news stays current.

---

## Suggested Pacing (GitHub Milestones)

| # | Milestone | Outcome |
|---|-----------|---------|
| 1 | Skeleton + CI | Repo, uv, CLAUDE.md, CI green, branch protection |
| 2 | Ingestion + retrieval with evals | Searchable news store, retrieval metrics baseline |
| 3 | Agent + tools | Tool-calling agent, trajectory evals |
| 4 | API + UI | Streaming chat endpoint and front end |
| 5 | Deploy + monitoring | Staging/prod pipeline, tracing, scheduled ingestion |
| 6 | Hardening + write-up | Adversarial tests, README with eval results and architecture diagram |

A README that shows eval results and an architecture diagram will carry weight in interviews and in your first weeks at a new team.

---

## Initial Plan

> Superseded where it differs by [design.md](design.md). For example, the agent layer now uses
> LangChain/LangGraph behind an interface instead of a hand-rolled loop first.

### Components

#### Offline:

1. **Fetchers:** RSS first (feedparser), one adapter per source behind a common interface so adding a news API later is easy.
2. **Extraction:** pull clean article text and metadata (title, source, published date, URL, language) with something like trafilatura.
3. **Dedupe:** canonical URLs plus a near-duplicate detection, since wire stories get republished everywhere.
4. **Story Clustering:** group the articles about the same event into a story. This is what makes a news assistant rather than generic RAG, and it powers timelines and source comparison.
5. **Store and Index:** relational store for articles and stories, plus a vector index over chunks.

#### Online:

6. **Retrieval:** hybrid search (keyword (*BM25*) plus embeddings), date and source filters, a recency boost, and diversity across sources. Optionally a reranker later.
7. **Tools** (read-only, typed inputs and outputs): search_news, get_article, get_headlines, get_story_timeline, compare_sources, and summarize_topic.
8. **Agent loop:** the tool-calling orchestrator with a system prompt that sets the rules: always ground answers in retrieved articles, state dates explicitly, say so when there's no evidence.
9. **Answer layer:** structured output with citations (article IDs), validated after generation so every cited ID was actually retrieved.
10. Conversation memory: history with truncation or summarization, so follow-ups like "what about the other side?" work. User preferences (topics, language) are a later extra.
11. **LLM gateway:** one thin interface over the model provider, which makes mocking in tests and swapping models easy, and gives you one place for cost and token tracking.
12. **Guardrails:** treat article text as untrusted data, never as instructions.
13. **API and UI:** FastAPI with streaming, session handling, and a thin chat front end with thumbs up/down.
14. **Observability and evals:** tracing on every request and the harness from the plan.


### Decisions worth making:
- **Own thin agent loop vs. a framework.** I'd write the loop yourself on the provider SDK first. It's about 100 lines, fully testable, and you'll understand every step. A framework like LangGraph can come later if you need branching workflows.
- **One agent with tools vs. multiple agents.** Start with one agent plus well-designed tools. Multi-agent setups multiply failure modes and make evals harder. Add a lightweight query-understanding step (date range, entities, intent) before it earns more complexity.
- **Pre-ingested news vs. live fetching.** Pre-ingested gives fast, testable, reproducible retrieval. A live fetch_url tool is a possible extension, but it widens the injection surface.
- **Injection defense by design.** Keep every tool read-only. If a malicious article can't trigger any action, the worst case is a bad answer, not a bad side effect. Delimit article text in the prompt and add adversarial eval cases.


### Cut for a first end-to-end slice

One RSS feed, extraction, store, search_news plus get_article, the agent loop with validated citations, and a CLI. Add clustering, timelines, and source comparison once that works and has evals. Each of those then becomes one issue and one PR.


## Next Steps (near future)

Order of work after issue #5 (Article model and RSS fetcher). Each step is its own issue, branch and PR into `main`.

### 1. Finish fetching (issue #5)
- [x] `Fetcher` protocol in `ingestion/base.py`, `parse_feed(content, source)`, `RSSFetcher.fetch()`
- [x] A fetcher downloads a feed and returns typed `Article` objects (title, summary, tags, date, `text=None`). New sources only need to satisfy the protocol.
- Raw XML is optional: keep a few days for debugging or replaying a parser bug, then drop it. The database is the source of truth.

### 2. Store articles (relational DB)
- [x] Storage interface plus SQLite implementation; insert keyed by `Article.id` (hash of the normalized URL)
- [x] A small ArticleStore protocol: save(articles) -> int (returns how many were new), get(id). We will use Postgres in production so the protocol is useful for that change and tests.
- [x] A SQLiteStore that creates its table on startup, inserts with ON CONFLICT DO NOTHING keyed by Article.id, and returns Article objects.
- [x] Tests with an in-memory database
    - [x] Round trip should provide the same article object (same tags list, timezone aware published_at, None summary/text/date.
    - [x] Saving the same article twice inserts once. The future ingest command will be idempotent.
    - [x] Saving an empty list is fine, get of an unknown id returns None.

### 3. Ingest Command
- [x] `ingest(fetchers, store)` takes fetchers and an `ArticleStore` as arguments (no feed URLs or database paths inside it) and returns a small result object: per feed, articles fetched and articles newly saved, or the error.
- [x] A feed that raises is logged with the feed's identity and traceback (`logger.exception`) and does not stop the other feeds.
- [x] One summary line is logged per run (feeds ok/failed, articles fetched, articles new).
- [x] `main()` builds the two `RSSFetcher`s from a feed list, opens a `SQLiteStore`, calls `ingest`, and sets the exit code: non-zero only if every feed failed.
- [x] Configuration: feed list (constant or `feeds.py`), DB path (default `data/news.db`, parent directory created if missing), log file path.
- [x] Logging is configured in `main()` only, so tests don't write log files.
- [x] Runnable with one command (`python -m ai_news_assistant.ingest` or a `[project.scripts]` entry).


### 4. Ingest Job

- [x] **Poll every 30-60 minutes, not daily.** Feeds keep only about the latest 20 items, so a daily poll can miss articles.
- [x] An article with only a feed summary is already valid, so store right after fetching, before extraction.
- [ ] We need to add 10 more Feeds to store enough `Article`s in a day or two.
     - [x] Name the 10 sources
        1. The Verge, AI: `https://www.theverge.com/rss/ai-artificial-intelligence/index.xml`
        2. Ars Technica: `https://arstechnica.com/ai/feed/`
        3. Wired AI: `https://www.wired.com/feed/tag/ai/latest/rss`
        4. MIT Technology Review: `https://www.technologyreview.com/topic/artificial-intelligence/feed`
        5. The Decoder: `https://the-decoder.com/feed/`
        6. OpenAI news: `https://openai.com/news/rss.xml` (`summary, tags` keys are sometimes missing)
        7. Google Deep Mind Blog: `https://deepmind.google/blog/rss.xml` (`bozo: True`) (`summary, tags` keys are sometimes missing)
        8. Hugging Face Blog: `https://huggingface.co/blog/feed.xml` (`summary, tags` keys are sometimes missing)
        9. Mistral AI News: `https://mistral.ai/news/rss` (`bozo: True`) (`summary, tags` keys are sometimes missing)
        10. Simon Willison's Weblog: `https://simonwillison.net/atom/everything/` (`tags` keys are sometimes missing)
     - [x] Manually check what you get
     - [x] Download fixtures (trimmed to 20)
     - [x] Manually check if you can read the content.
     - [x] Add the new feeds to `feeds.py`
     - [x] If needed, extend the following texts: `test_rss_fetcher.py`, `test_sqlite_store.py`, `test_urls.py`, `test_models.py`, `test_parse_feed.py`, `test_summary.py`  

### 5. Extraction (full text)
- [x] Update Store Functionalities for Extractor
    - [x] Add `extracted_at` (if successful) and `extraction_successful` (missing: not tried, 0: unsuccessful, 1: successful) fields to `Article` data model so we do not attempt failing extractions every time.
    - [x] As we change the `Article` data model, we also need to change the DB table with `ALTER TABLE` so we do not loose the records in it.
    - [x] In the `ArticleStore` protocol we need a new function to extract all `Article`s that have their `extraction_successful` field as missing. It will be an empty function in the protocol.
    - [x] In the `ArticleStore` protocol we need a new function to save text on an existing article without creating a new row.
    - [x] Saving text into an article in the DB must not change other article records.
    - [x] Saving text into an article in the DB must change only `extracted_at`, `extraction_successful`, and `text` fields.
- [x] Fetch `text` HTML (own `httpx` download, not `trafilatura.fetch_url`: it returns `None` for every failure, so a timeout can't be told apart from a 404 or paywall, and it can't use our client or `MockTransport`)
    - [x] Step 1: Extract the shared capped downloader (pure refactor, no behavior change)
        - [x] New `src/ai_news_assistant/downloads.py` (not `http.py`, which could shadow the stdlib `http` package) with `USER_AGENT`, `DEFAULT_TIMEOUT`, `ResponseTooLargeError`, and `download(url, client, *, timeout, max_bytes, accept, too_large) -> Download(content, headers)` holding today's `download_feed` logic (stream, `Content-Length` check, running byte count).
        - [x] `rss.py`: `FeedTooLargeError` subclasses `ResponseTooLargeError`; `download_feed` becomes a thin wrapper.
        - [x] `test_download_feed.py` passes untouched; add `test_http_download.py` for `download` directly.
    - [x] Step 2: `fetch_html(url, client) -> str` in a new `extraction/fetch.py`
        - [x] Streamed and capped at `MAX_PAGE_BYTES` (5 MB to start) via `download`, like `download_feed`.
        - [x] Decodes with the `Content-Type` charset, else UTF-8, replacing invalid bytes.
        - [x] Failures are raised, never returned as `None`, with no catching or retries (like D7): `httpx.HTTPStatusError` (4xx/5xx), `httpx.TransportError` (timeouts, connection, redirects), `PageTooLargeError`, `NotHTMLError` (`Content-Type` present but not HTML, e.g. PDF).
        - [x] Tests with `MockTransport` only: success, redirect, size caps, status errors, timeout, content types, charsets.
    - [x] Step 3: HTML fixtures, one real page per source, under `tests/text_html_fixtures/`
        - [x] `scripts/capture_html_fixtures.py`: takes the first entry's link from each RSS fixture, checks robots.txt, saves `<feed_name>.html` raw; reports sizes and blocked sources. Run by me, not Claude (network).
        - [x] `expected.json`: the first and last body sentence per article, so the extraction ToDo can test that text is complete, not cut in the middle.
        - [x] `README.md`: file, URL, capture date, size, notes (blocked, script-rendered, paywalled).
        - [x] Test: every feed has a fixture (or is listed as blocked), and `fetch_html` returns each one as non-empty text under the cap.
    - [x] Step 4: Docs
        - [x] `design.md`: component rows 2 and 3, interfaces in §5 (including the store's extraction fields and methods), new D12 "Page fetching failures", D11 caps also apply to pages.
        - [x] D12 includes the rule for the pipeline ToDo: permanent failures (4xx except 408/429, too large, not HTML) are recorded with `save_extraction(id, None, now)`; transient ones (transport errors, 5xx, 408, 429) record nothing, so the next run retries.
- [x] Text Extraction From HTML: plain function `extract_text_from_html(html: str) -> str | None` in `extraction/text.py`. It takes the output of `fetch_html` and never calls it, so it can be tested standalone with the saved fixtures.
    - [x] Step 1: Basic extractor
        - [x] `uv add lxml` (now imported directly) and `uv add --dev types-lxml` (stubs for mypy strict).
        - [x] Parse once; blank or unparseable input returns `None`. Run `trafilatura.extract(tree, include_comments=False)`; `include_comments=True` leaves "Comments | Back to top" on Wired; `favor_precision` loses MIT's first sentence; `favor_recall` changes nothing.
        - [x] Results shorter than `MIN_TEXT_CHARS` (200) return `None` (cookie notices, teasers).
        - [x] Tests: degenerate input, short page, no script/style/comment text, and each fixture contains its expected first and last sentence (Ars Technica's last one marked xfail until Step 3).
    - [x] Step 2: Drop hidden elements before extracting
        - [x] Remove elements with `hidden`, `aria-hidden="true"`, or inline `display:none` / `visibility:hidden`, never `html`/`head`/`body` themselves. Blocks hidden prompt-injection text, and text hidden until you subscribe is not taken (no paywalled scraping).
        - [x] Adversarial tests: each hidden form is removed; a hidden `<body>` still yields the article; visible injection text is kept as plain data (the defense against it is downstream).
    - [x] Step 3: Fall back to `baseline()` when `extract()` cuts the article
        - [x] Use the extract text if it contains at least half of baseline's sentences of 60+ characters, else use baseline. Measured: Ars Technica 0.30, all other fixtures 0.71 to 1.00. When `extract()` finds nothing, return `None` rather than baseline's whole-page text.
        - [x] Pure helper functions with unit tests; every fixture then contains its first and last sentence.
        - Known limit (in D13): a login wall or navigation-only page can come back from `extract()` itself as a few hundred characters of menu text.
    - [x] Step 4: Docs
        - [x] `design.md`: row 3 and a new D13 "Article text extraction"; update the fixtures README notes.

- [ ] Create the Orchestrator Extract Command: `ai-news-extract`, shaped like `ingest.py`. Gets unextracted articles, collects the HTML for each, cleans it into text and writes the text back to the DB.
    - [ ] Step 1: robots.txt policy in `extraction/robots.py`
        - [ ] One robots.txt download per host per run, with our client and User-Agent. RFC 9309: 2xx parse, 4xx means no rules, 5xx or network error means the host is unreachable this run. Reads `Crawl-delay`.
        - [ ] `scripts/capture_html_fixtures.py` uses it instead of its own copy.
        - [ ] Tests with `MockTransport`: allow and disallow, 404 and 403, 500 and timeout, cache, crawl delay, oversized file.
    - [ ] Step 2: Outcome classification and per-host throttle
        - [ ] `classify(exc)`: permanent (4xx except 408/429, too large, not HTML, robots disallow) is recorded as failed; transient (network errors, 5xx, 408, 429, robots unreachable) records nothing, so the next run retries (D12).
        - [ ] `HostThrottle`: at least 2 s between requests to the same host, or the robots `Crawl-delay` if larger (max 60 s). Clock and sleep injected so tests never sleep.
    - [ ] Step 3: The `extract` command
        - [ ] `extract(store, client, ...)`: for each article from `get_unextracted(limit)`: robots check, throttle, `fetch_html`, `extract_text_from_html`, `save_extraction`. A failing article never stops the others; one summary log line; no article text in logs.
        - [ ] `run()` and `main()`: env `NEWS_DB_PATH`, `NEWS_LOG_FILE` (default `logs/extract.log`), `EXTRACT_LIMIT` (default 100 per run, newest first); script `ai-news-extract`. Exit code 1 only if every article had to be retried.
        - [ ] Offline tests with an in-memory store and the saved page fixtures: saved, no text, failed, retried, robots, idempotent second run, limit, logs.
    - [ ] Step 4: Docs
        - [ ] `design.md` (row 3, scheduler sentence, rate limit and robots decision), `CLAUDE.md` commands, tick this list.

- [ ] Second pipeline stage: find rows with no `text`, download the page with the project's own `httpx` client, run `trafilatura.extract`, update the row
- trafilatura returns plain text by default (as far as I know), so no extra HTML parsing is needed. Check the current docs.
- If extraction returns `None` (paywall, script-rendered page), fall back to the feed summary.
- Extraction can fail or retry without affecting ingestion.
- Check each publisher's terms and robots.txt, rate-limit requests, never scrape paywalled content.
- Tests use saved HTML pages, not live sites.

### 6. Baseline retrieval and eval harness (including chunking experiments)
- [ ] Ingest a few hundred articles
- [ ] Write 30-50 questions myself, each labeled with the **article IDs** that answer them (article-level labels are far cheaper than chunk-level; a chunk hit maps to its parent article)
- [ ] Simplest possible retriever (embeddings in memory, or BM25)
- [ ] Evaluate for only embeddings, only BM25, both.
- [ ] Make chunking a parameter and compare: whole article, fixed 256/512 tokens with overlap, paragraph-based. Metrics: recall@k and MRR.
- Chunking cannot be evaluated without retrieval, so this comes first. News articles are short, so expect the strategies to differ less than you would think; do not spend days here.

### 7. Near-duplicate detection (cut or simplify if time is short)
- [ ] Runs after extraction, since teasers are too short to compare reliably
- Approaches: MinHash with LSH (`datasketch`) on 5-word shingles is the standard fit for republished wire copy; SimHash; embedding similarity (also catches paraphrases but merges different articles on the same topic).
- Compare only against a recent window (a few days), not the whole table.
- Hand-label about 50 pairs (duplicate or not) and choose the threshold from them.
- Store `duplicate_of` on the row instead of deleting, so no information is lost.

### 8. Story clustering (cut or simplify if time is short)
- [ ] `stories` table (id, title, first_seen, last_updated) and a `story_id` column on articles (one story has many articles; not just per-article metadata)
- Online approach: embed title plus lead, assign to the nearest existing story within a time window (about 48 hours) above a similarity threshold, otherwise start a new story. Alternative: periodic batch clustering (agglomerative or HDBSCAN) over recent articles.
- Treat assignments as derived data that can be recomputed, since stories merge and split over time.
- Evaluate with a small hand-labeled set and pairwise precision and recall.

### 9. I should look at LangGraph Deep Agents as well
- Just to see how I could make this agent more advanced.

### Housekeeping
- [x] Update `docs/design.md`: the product is *AI* news, `Article` has `tags`, and the delivery order is now fetch, store, extract, retrieval and evals, then near-dup and clustering.
- [ ] First skills (`add-fetcher`, `add-eval-case`, `review-changes`, `draft-pr-description`) only after the same procedure has been repeated a second time.
- [ ] Stay trunk-based: short-lived branches off `main`, one issue and one PR each. Group the work with milestones and labels (`offline`, `online`), not with long-lived branches.