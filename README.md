# AI News Assistant

A chatbot agent that answers questions about current **AI news** using only articles it has
ingested, and cites a source for every claim. It is built like a small production service: an
offline pipeline that collects and cleans articles, a hybrid retriever, a tool-calling agent
with checked citations, a streaming FastAPI service, a separate MCP tools server, and an eval
harness for both retrieval and the agent.

An example of the CLI output format (shortened, titles and dates are placeholders):

```
you> What did Mistral announce about their new Large model?

Assistant> Mistral announced ... It is optimized for coding and cyberdefense [1] ...
Coverage elsewhere put the training cluster at ... [3]

**Sources:**
[1]: <article title>, Mistral AI, <day Mon year>
[3]: <article title>, TechCrunch, <day Mon year>
```

Citation numbers can skip (`[1]`, `[3]`) when a citation was dropped by the check.

- **Grounded:** the agent answers only from retrieved articles and says "I do not know" when
  the archive has no answer.
- **Checked citations:** after the model answers, code checks that every cited article was
  actually retrieved in the conversation. Unsupported citations are removed before the user
  sees them. Titles, outlets and dates come from the database, never from the model.
- **Defends against untrusted input:** feed and page text are treated as data, never as
  instructions. Hidden HTML is dropped, article text is wrapped in delimiters the article
  cannot fake, and every tool is read-only.
- **Measured:** a labeled retrieval eval set (recall@k, MRR) and an agent eval with
  layered checks (format, tool use, grounding, correctness, cost) plus an LLM judge for
  faithfulness.

> Design decisions, trade-offs and measurements are recorded in [`docs/design.md`](docs/design.md)
> (decisions D1-D26). The original plan and milestones are in [`docs/plan.md`](docs/plan.md).
> Where this README and `design.md` disagree, `design.md` is the reference for intent and the
> code is the reference for behavior.

---

## Contents

- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Getting started](#getting-started)
- [Running the pipeline](#running-the-pipeline)
- [Using the assistant](#using-the-assistant)
- [HTTP API](#http-api)
- [Agent tools](#agent-tools)
- [How it works](#how-it-works)
- [Security: untrusted input](#security-untrusted-input)
- [Configuration reference](#configuration-reference)
- [Evaluation](#evaluation)
- [Development](#development)
- [Known limitations](#known-limitations)

---

## Architecture

```
 OFFLINE (scheduled)                              ONLINE (per request)

 12 RSS/Atom feeds                                 CLI            HTTP client
       |                                            |                  |
       v                                            |                  v
  ai-news-ingest  --> SQLite (articles)             |        FastAPI  (api, :8000)
       |                    |                       |        /chat, /chat/stream
       v                    v                       |           |         |
  ai-news-extract --> full text per article         |           |   Postgres (threads)
                            |                       v           v
                     frozen snapshot  <------  LangChain agent (Claude) + citation check
                            |                       |           |
               build_corpus / build_chunks          |      MCP over HTTP
                    / build_vectors                 |           |
                            |                       v           v
                            +--------------> Tools: search_news, get_article,
                         chunks (SQLite) +           get_latest_articles
                         Chroma vectors                 (tools server, :9000)
                                                        Hybrid retriever (BM25 + vectors)
```

The two halves share one store. The offline pipeline keeps the news fresh; the online agent
only reads.

Online there are two ways to run the agent:

| Entry point | Where the tools run | Conversation memory |
|---|---|---|
| CLI (`agent/cli.py`) and evals | In-process (`build_agent(retriever=...)`) | In memory (`InMemorySaver`) |
| API (`api/app.py`) | Separate MCP server (`mcp_server/app.py`), loaded at startup over HTTP | Postgres (`AsyncPostgresSaver`) |

In the API setup the retrieval stack (embedding model, Chroma, BM25 index) loads only in the
tools process, so the API process stays light (design D26).

---

## Repository layout

```
.
├── src/ai_news_assistant/
│   ├── models.py               # Article, ModelAnswer, Source, Answer (pydantic)
│   ├── downloads.py            # Shared size-capped httpx download + User-Agent
│   ├── ingest.py               # `ai-news-ingest`: fetch every feed into SQLite
│   ├── extract.py              # `ai-news-extract`: fetch pages, extract full text
│   ├── ingestion/
│   │   ├── base.py             # Fetcher protocol
│   │   ├── feeds.py            # The configured feeds (name -> url, publisher)
│   │   ├── rss.py              # RSSFetcher, download_feed, parse_feed
│   │   ├── summary.py          # HTML -> plain text for titles and summaries
│   │   ├── urls.py             # URL normalization and article ids (SHA-256)
│   │   └── dates.py            # feedparser dates -> aware UTC datetimes
│   ├── extraction/
│   │   ├── fetch.py            # fetch_html (capped, raises typed errors)
│   │   ├── robots.py           # robots.txt policy (RFC 9309) with Crawl-delay
│   │   └── text.py             # extract_text_from_html (trafilatura + hidden-element removal)
│   ├── storage/
│   │   ├── base.py             # ArticleStore protocol
│   │   └── sqlite.py           # SQLiteStore with in-place migrations
│   ├── rag/
│   │   ├── config.py           # Snapshot paths, embedding model, chunk settings
│   │   ├── store.py            # RagStore: corpus table, chunks table, listing queries
│   │   ├── chunkers.py         # Word and recursive token chunkers (BGE tokenizer)
│   │   ├── build_chunks.py     # CLI: chunk the corpus for one config
│   │   ├── build_vectors.py    # CLI: embed chunks into a Chroma collection
│   │   ├── base.py             # Hit, SearchFilter, Retriever protocol
│   │   ├── bm25_retriever.py   # BM25 over chunks, with metadata filtering
│   │   ├── vector_retriever.py # Chroma + BGE embeddings, with `where` filters
│   │   ├── hybrid_retriever.py # Reciprocal rank fusion of several retrievers
│   │   ├── article_search.py   # Chunks -> per-article snippets for the agent
│   │   └── eval/               # Retrieval eval: questions, runner, metrics, report
│   ├── agent/
│   │   ├── agent.py            # build_agent / build_retriever (LangChain create_agent)
│   │   ├── tools.py            # search_news, get_article, get_latest_articles
│   │   ├── prompts.py          # SYSTEM_PROMPT
│   │   ├── answer.py           # ask / aask / aask_stream, render
│   │   ├── citations.py        # retrieved_ids, check_citations, resolve_sources
│   │   ├── config.py           # Models, context-editing settings
│   │   ├── cli.py              # Interactive terminal chat
│   │   └── eval/               # Agent eval: cases, runner, checks, report, LLM judge
│   ├── api/
│   │   ├── app.py              # FastAPI app factory: /health, /chat, /chat/stream
│   │   ├── schemas.py          # ChatRequest, ChatResponse
│   │   └── settings.py         # preflight checks, Postgres URI
│   └── mcp_server/
│       └── app.py              # FastMCP server exposing the three tools over HTTP
├── scripts/                    # Corpus and fixture helpers (see below)
├── tests/                      # pytest suite: no network, no real LLM
├── docs/design.md              # Decisions, measurements, known limits
├── docs/plan.md                # Original plan and milestones
├── Dockerfile, compose.yaml    # api + tools + postgres
└── pyproject.toml, uv.lock     # uv-managed project (Python 3.12)
```

`data/` and `logs/` are gitignored: the live database, snapshots, Chroma indexes and eval
results stay local.

---

## Getting started

### Prerequisites

- Python 3.12 and [uv](https://docs.astral.sh/uv/) (the project is managed with uv; do not use pip)
- Docker, for Postgres (API thread memory) or the full compose stack
- An Anthropic API key for the agent and the judge

### Install

```bash
uv sync
```

The first run of anything that touches retrieval downloads the
`BAAI/bge-small-en-v1.5` embedding model and tokenizer from Hugging Face.

### Environment

Configuration and secrets come from environment variables. Create a `.env` in the repo root
(it is gitignored; never commit it). `.env.example` holds the Postgres part:

```dotenv
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...          # checked by the API preflight (see Known limitations)
POSTGRES_USER=changeme
POSTGRES_PASSWORD=changeme
POSTGRES_DB=ai_news
```

The CLI, the API and the eval scripts call `load_dotenv()`, so `.env` is picked up
automatically. See [Configuration reference](#configuration-reference) for every variable.

---

## Running the pipeline

The pipeline has an online part (ingest, extract) that keeps a live database growing, and a
frozen part (snapshot, corpus, chunks, vectors) that retrieval and evals run on (design D10).

### 1. Ingest feeds

```bash
uv run ai-news-ingest
```

Fetches every feed in `ingestion/feeds.py` and inserts new articles into `data/news.db`
(override with `NEWS_DB_PATH`). Logs go to stderr and `logs/ingest.log` (`NEWS_LOG_FILE`).

- Idempotent: articles are keyed by a hash of the normalized URL, so re-runs only add new ones.
- A failing feed is logged and skipped; the exit code is 1 only when every feed failed.
- Feeds keep only about their latest 20 items, so schedule it every 30-60 minutes.

Current sources: The Guardian, TechCrunch, The Verge, Ars Technica, Wired, MIT Technology
Review, The Decoder, Google DeepMind, Hugging Face, Mistral AI and Simon Willison. The OpenAI
feed is configured but commented out: openai.com answers with a bot challenge, which the
project does not try to get around.

### 2. Extract full text

```bash
uv run ai-news-extract
```

Takes up to `EXTRACT_LIMIT` (default 200) articles that were never tried, newest first,
downloads each page, extracts clean text, and records the attempt. It respects robots.txt
and waits at least 2 seconds between requests to the same host (or the host's
`Crawl-delay`, capped at 60 s). Logs go to `logs/extract.log` by default.

Each article ends with one outcome:

| Outcome | Meaning | Recorded? |
|---|---|---|
| `saved` | Text extracted and stored | Yes, success |
| `no_text` | Page fetched, no usable article text | Yes, failure (keeps the feed summary) |
| `failed` | Permanent failure: 4xx (except 408/429), too large, not HTML, disallowed by robots.txt | Yes, failure |
| `retry` | Temporary failure: timeouts, network errors, 5xx, 408, 429, robots.txt unavailable | No, retried next run |
| `error` | Unexpected exception (a bug) | No, logged with traceback |

Exit code 1 means every article had to be retried (network down?).

### Scheduling

Both commands are designed to run from a scheduler (launchd or cron locally). For example,
a crontab entry:

```cron
*/45 * * * * cd /path/to/ai-news-assistant && uv run ai-news-ingest && uv run ai-news-extract
```

### 3. Freeze a snapshot and build the corpus

Retrieval and evals use a frozen copy of the database so metrics do not move as new
articles arrive. The current snapshot is `data/snapshots/news-2026-10-12.db` (paths are set
in `rag/config.py`).

```bash
cp data/news.db data/snapshots/news-2026-10-12.db
uv run python scripts/build_corpus.py
```

`build_corpus.py` fills the `corpus` table with the successfully extracted articles (Hugging
Face posts only from 2026-05-15 on). The snapshot path and the cut-off are constants in the
script.

### 4. Chunk and embed

```bash
uv run python -m ai_news_assistant.rag.build_chunks --config recursive-tok256-32
uv run python -m ai_news_assistant.rag.build_vectors --config recursive-tok256-32
```

Two chunking configs exist, both 256 tokens with 32 overlap using the BGE tokenizer:
`recursive-tok256-32` (the agent default) and `words-tok256-32`. Chunks are stored in the
snapshot's `chunks` table; vectors go to a Chroma collection named after the config in
`data/snapshots/chroma-2026-10-12` (cosine space, normalized embeddings, with `source` and
`published_at` metadata for filtering). Both commands accept `--db`; `build_vectors` also
accepts `--vector-store`.

---

## Using the assistant

### Terminal chat (in-process tools)

```bash
uv run python -m ai_news_assistant.agent.cli
```

Starts an interactive session with in-memory conversation history, so follow-up questions
work. Type `exit` or `quit` to leave.

> `uv run ai-news-assistant` (the `ai_news_assistant:main` script entry) is still a
> placeholder that prints a greeting. Use the module command above for the chat.

### Tools server + API (local)

```bash
# 1. Postgres for conversation threads (reads POSTGRES_* from .env)
docker compose up -d postgres

# 2. MCP tools server on 127.0.0.1:9000 (loads the retriever)
uv run python -m ai_news_assistant.mcp_server.app

# 3. API on :8000, in another terminal
TOOLS_URL=http://127.0.0.1:9000/mcp \
  uv run uvicorn ai_news_assistant.api.app:create_app --factory --port 8000
```

At startup the API runs a preflight (required env vars set, snapshot db exists), opens the
Postgres checkpointer and creates its tables, then loads the tools from the MCP server. If
any of these fails, the API does not start, rather than failing on the first request.

### Docker Compose

```bash
docker compose up --build
```

Three services from one image (`python:3.12-slim` + uv, embedding model baked in at build
time, runs as uid 1000):

| Service | What it runs | Port | Notes |
|---|---|---|---|
| `postgres` | Postgres 17 | `127.0.0.1:5432` | Data in the `pgdata` volume |
| `tools` | `python -m ai_news_assistant.mcp_server.app` | 9000 (compose network only) | Mounts the snapshot (read-only) and `./data/runtime/chroma-2026-10-12` |
| `api` | uvicorn `create_app` | `127.0.0.1:8000` | Reads `.env`; waits for `postgres` and `tools` to be healthy |

The snapshot (`data/snapshots/news-2026-10-12.db`) and the Chroma index
(`data/runtime/chroma-2026-10-12`) must exist on the host. `design.md` notes that the compose
stack has not yet been run end to end.

---

## HTTP API

### `GET /health`

```json
{"status": "ok"}
```

### `POST /chat`

One JSON response with the checked answer.

```bash
curl -s localhost:8000/chat \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is new from Google DeepMind this week?"}'
```

Request:

| Field | Type | Rules |
|---|---|---|
| `question` | string | 1-2000 characters after stripping whitespace |
| `thread_id` | string, optional | `[A-Za-z0-9_-]{1,64}`. Omit to start a new thread; send the returned one to continue it |

Response:

```json
{
  "thread_id": "3f2a...",
  "answer": {
    "text": "Google DeepMind announced ... [1]",
    "sources": [
      {
        "marker": 1,
        "article_id": "c714d68a...",
        "title": "...",
        "source": "Google DeepMind",
        "published_at": "2026-10-09T14:00:00Z"
      }
    ],
    "grounded": true
  }
}
```

`grounded` is true only when the model said its answer was grounded **and** at least one
citation survived the check. Errors return `500` with a generic
`{"detail": "The assistant could not answer."}`; the traceback is logged with the thread id.

### `POST /chat/stream`

Same request body. Returns `text/event-stream` with progress while the agent works, then the
answer:

```
event: status
data: {"tool": "search_news"}

event: status
data: {"tool": "get_article"}

event: answer
data: {"thread_id": "...", "answer": {...same shape as /chat...}}
```

- One `status` event per tool call (`search_news`, `get_article`, `get_latest_articles`).
- One `answer` event at the end. There is no token streaming: the answer is a structured
  object whose citations are checked only once the run ends, so text is never sent before
  the check (design D25).
- If the run fails after streaming has started, an `error` event is sent:
  `{"detail": "FAILED"}`.

---

## Agent tools

All tools are read-only. They are plain Python functions in `agent/tools.py`, wrapped as
LangChain tools for the CLI and evals, and as MCP tools by `mcp_server/app.py`.

| Tool | Arguments | Returns | Limits |
|---|---|---|---|
| `search_news` | `query`, `k=5`, `sources=None`, `days=None` | Up to `k` articles: `article_id`, `title`, `source`, `published_at`, `snippet` (up to 3 best chunks joined by `[...]`) | `k` clamped to 5-10. Unknown source names return an error listing the valid ones. `days` counts back from the newest article in the corpus |
| `get_article` | `article_id` | `article_id`, `title`, `source`, `published_at`, `text` wrapped in `<article_text>` tags, `truncated` | First 12,000 characters. An unknown id returns `{"article_id", "error"}` instead of raising |
| `get_latest_articles` | `days=7` | Listing of `article_id`, `title`, `source`, `published_at`, `date`, newest first | `days` clamped to 1-30, at most 100 items, no text |

The list of valid outlet names is read from the store and written into the `search_news`
description when the tool is built, so it stays correct when the snapshot changes. "Recent"
is measured from the newest article in the snapshot, not from today, because the corpus is
frozen.

---

## How it works

### Ingestion (`ingestion/`, `ingest.py`)

- **Fetchers** implement one method, `fetch() -> list[Article]`. `RSSFetcher` downloads with
  `httpx` and parses with `feedparser`. Fetchers never catch errors, retry or schedule; the
  `ingest` command loops over them and handles failures per feed (D7).
- **Downloads** are streamed and capped at 5 MB, checking `Content-Length` first and then a
  running count of decompressed bytes, so a gzip bomb is cut off too.
- **Parsing:** titles and summaries are reduced to plain text (script and style content
  removed, entity-escaped script tags dropped, Guardian "Continue reading" and WordPress
  "appeared first on" boilerplate removed). Invalid entries are skipped and logged with
  field names only. The configured publisher name takes precedence over the feed's own
  title, so a feed cannot rename itself.
- **Identity:** `Article.id` is the SHA-256 of the normalized URL (lowercased scheme and host,
  default port, fragment and tracking parameters such as `utm_*`, `fbclid` and `gclid`
  removed). The same story linked with different tracking tags is one article (D8).

### Storage (`storage/`)

`SQLiteStore` implements the small `ArticleStore` protocol: `save`, `get`,
`get_unextracted`, `save_extraction`. Dates are stored as UTC ISO 8601 text, tags as a JSON
array. Each article records its extraction state (`NULL` not tried, `0` failed, `1`
succeeded), and the SQL `WHERE` guarantees exactly one recorded attempt per article. New
columns are added with `ALTER TABLE`, so older databases keep their rows. All values go in
as SQL parameters.

### Extraction (`extraction/`, `extract.py`)

- `fetch_html` uses the same capped download, sends an HTML `Accept` header and raises typed
  errors (`HTTPStatusError`, `TransportError`, `PageTooLargeError`, `NotHTMLError`). The
  pipeline decides which failures are permanent (D12).
- `RobotsPolicy` downloads each host's robots.txt once per run and follows RFC 9309: a 4xx
  means no rules; a 5xx or network error means that host is skipped (and retried) this run.
- `extract_text_from_html` (D13):
  1. Removes hidden elements (`hidden`, `aria-hidden="true"`, inline `display:none` or
     `visibility:hidden`). Readers never see them, so they are a place to plant
     instructions, and text hidden until you subscribe is paywalled content.
  2. Runs trafilatura's `extract()` without comments.
  3. If that text covers less than half of `baseline()`'s long sentences, the article was
     cut, and `baseline()` is used instead.
  4. Text under 200 characters counts as no text; the article keeps its feed summary.

### Retrieval (`rag/`)

- **Corpus:** a `corpus` table in the snapshot selects which successfully extracted articles
  are searchable.
- **Chunking:** LangChain text splitters driven by the BGE tokenizer, 256 tokens with 32
  overlap.
- **BM25:** `rank_bm25` over a simple tokenizer that keeps version numbers like `5.5`
  together.
- **Vectors:** `BAAI/bge-small-en-v1.5` embeddings in Chroma, cosine similarity, with the BGE
  query prefix on queries.
- **Hybrid:** reciprocal rank fusion (`rrf_k=60`) over the top 50 chunks from each
  retriever. This is the agent's default.
- **Filters:** a `SearchFilter(sources, since)` is applied *inside* each retriever, before
  the top-k cut, so a filtered search still returns up to `k` results (D17). Chroma gets a
  `where` clause; BM25 zeroes non-matching scores while keeping corpus-wide IDF.
- **Article snippets:** `search_articles` asks for 3x more chunks than articles wanted,
  groups them by article in order of the first hit, and keeps up to 3 chunks per article.

### Agent (`agent/`)

- Built with LangChain `create_agent` (LangGraph underneath). The framework is confined to
  `agent/`; everything else is plain Python, and the project's own models are used at every
  boundary (D1).
- Model: `anthropic:claude-haiku-5-5`, set in `agent/config.py` as `provider:model`.
- The system prompt (`agent/prompts.py`) enforces grounding: answer only from retrieved
  articles, say "I do not know" otherwise, say whether something is announced, available or
  rumored, compare `published_at` for "latest" questions, search again with specific names
  when results are vague, and use the right tool and filters for the question.
- **Structured output:** the model returns a `ModelAnswer` (`text` with `[n]` markers,
  `citations` as article ids, `grounded`).
- **Middleware:** context editing (below) and a limit of 25 calls each for `search_news`
  and `get_article` per run.

### Citations (`agent/citations.py`, `agent/answer.py`)

After every run (D15):

1. `retrieved_ids` collects every article id returned by a successful tool call anywhere in
   the thread (so follow-ups can re-cite earlier articles). A `get_article` error counts
   as nothing, so an invented id is never validated by asking for it.
2. `check_citations` keeps marker `[n]` only if `citations[n-1]` exists and was retrieved.
   Other markers are removed from the text; their ids and numbers are logged.
3. `resolve_sources` looks up title, outlet and date for each surviving citation from the
   store.
4. `render` (CLI) appends a **Sources** block. Article ids are never shown.

If the model returns no structured response, the last message text is used with every
marker removed, no sources, and `grounded=false`.

### Memory and context management

Conversation history lives in the LangGraph checkpointer, keyed by `thread_id`: in memory
for the CLI and evals, Postgres for the API. Both use the same serializer, which allows
`ModelAnswer` explicitly.

Tool results are most of the tokens in a conversation (84-90% in measurements; D16). When
the messages exceed about 10,000 tokens, `ContextEditingMiddleware` replaces every tool
result except the last 3 with a placeholder **in the request sent to the model only**. The
stored thread keeps every result, so citation checks and traces still see them. Settings
are `CONTEXT_TRIGGER_TOKENS`, `KEEP_TOOL_RESULTS` and `CLEARED_PLACEHOLDER` in
`agent/config.py`.

### Observability

Ingestion and extraction log one line per feed or article and a summary per run, to stderr
and a log file. Logs carry ids, hosts, counts and error types, never article text or raw
untrusted values. The agent and API log dropped citations, missing structured output and
failures with the thread id. Tracing is planned, not built.

---

## Security: untrusted input

Every byte from a feed or article page is treated as untrusted (D11, D13, D14):

| Threat | Defense |
|---|---|
| Huge or compressed-bomb responses | 5 MB cap on decompressed bytes for feeds and pages; 512 KiB for robots.txt |
| Malicious charsets or content types | Unknown charsets fall back to UTF-8 with replacement; reported content types are length-capped |
| Markup and scripts in titles and summaries | Reduced to plain text; script content and escaped script tags removed |
| A feed renaming itself | Configured publisher name wins; feed title capped at 300 characters |
| Instructions hidden in page HTML | Hidden elements removed before extraction |
| Prompt injection in article text | Text wrapped in `<article_text>` tags; look-alike tags inside the article become `[removed tag]`; the tool description tells the model not to follow instructions inside |
| Invented citations | Post-generation check against retrieved ids; metadata from the store |
| Injected actions | Every tool is read-only, so the worst case is a bad answer, not a side effect |
| Log injection or leakage | Logs carry ids and counts only; validation errors log field names, not values |
| SQL injection | Parameterized queries only |

Framing lowers the chance that injected instructions are followed; it does not prevent it.
`search_news` snippets and `get_latest_articles` titles are not framed yet.

---

## Configuration reference

### Environment variables

| Variable | Used by | Default | Purpose |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | agent, judge, API preflight | (required) | Model access |
| `OPENAI_API_KEY` | API preflight | (required by preflight) | Needed only if an `openai:` model is configured |
| `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_DB` | API, compose | (required) | Thread storage; user and password are URL-encoded |
| `POSTGRES_HOST`, `POSTGRES_PORT` | API | `localhost`, `5432` | Postgres address (compose sets `postgres`) |
| `TOOLS_URL` | API | (required, no default) | MCP server URL, e.g. `http://127.0.0.1:9000/mcp` |
| `TOOLS_HOST` | MCP server | `127.0.0.1` | Bind address (compose sets `0.0.0.0`); port is fixed at 9000 |
| `NEWS_DB_PATH` | ingest, extract | `data/news.db` | Live database |
| `NEWS_LOG_FILE` | ingest, extract | `logs/ingest.log`, `logs/extract.log` | Log file |
| `EXTRACT_LIMIT` | extract | `200` | Articles per extraction run (0 or more) |

### Code constants

| Setting | Location | Value |
|---|---|---|
| Chat model, judge model | `agent/config.py` | `anthropic:claude-haiku-5-5`, `anthropic:claude-haiku-4-5-20251001` |
| Model kwargs | `agent/config.py` | `timeout=240`, `max_retries=2`; Anthropic `max_tokens=4096` |
| Context trigger, results kept | `agent/config.py` | 10,000 tokens, 3 |
| Snapshot db, Chroma path | `rag/config.py` | `data/snapshots/news-2026-10-12.db`, `data/snapshots/chroma-2026-10-12` |
| Embedding model | `rag/config.py` | `BAAI/bge-small-en-v1.5` |
| Default chunking | `rag/config.py` | `recursive-tok256-32` (256 tokens, 32 overlap) |
| Feeds | `ingestion/feeds.py` | 11 active feeds |
| Tool limits | `agent/tools.py` | `MAX_K=10`, `MAX_DAYS=30`, `MAX_ARTICLE_CHARS=12000` |

---

## Evaluation

Both evals run against the frozen snapshot, are configured in TOML, and keep searching (or
running the agent) separate from scoring, so a label fix or new metric re-scores old runs
without new calls.

### Retrieval eval (`rag/eval/`)

- **Questions:** `rag_questions_v1.toml`, 22 hand-labeled questions (13 dev, 9 test) across 12
  types: specific detail, paraphrase, multi-article, rare term, time-sensitive, filtered,
  no-answer, hard negative and more. Each gold article has a verbatim evidence quote; some
  questions also list stale articles and look-alike hard negatives. No article is labeled in
  both splits (D18).
- **Run settings:** `rag_run.toml` (chunking configs, retrievers, ranking depth, hybrid
  settings). Only the `dev` split is enabled; test is reserved for a final comparison.

```bash
# 1. Search and save raw chunk hits (no scoring)
uv run python -m ai_news_assistant.rag.eval.run_retrieval [--config path/to/rag_run.toml]

# 2. Score a run folder: writes report.md and metrics.json into it
uv run python -m ai_news_assistant.rag.eval.evaluate data/snapshots/eval_results/<run>

# Compare two runs scored with the same labels
uv run python -m ai_news_assistant.rag.eval.evaluate --compare <run_a> <run_b>
```

Metrics rank **articles** (by their first chunk, as the agent sees them): R@5, R@10, MRR,
first gold rank, stale and hard-negative ordering, duplicate copies in the top 5, and
evidence coverage.

**Dev baseline** (12 scored questions, run `20261022-172640_baseline`, D19):

| Retriever | Config | R@5 | R@10 | MRR |
|---|---|---|---|---|
| vector | `words-tok256-32` | 0.76 | 0.77 | 0.74 |
| bm25 | `words-tok256-32` | 0.85 | 0.86 | 0.83 |
| hybrid | `words-tok256-32` | 0.77 | 0.86 | 0.71 |
| vector | `recursive-tok256-32` | 0.76 | 0.87 | 0.74 |
| bm25 | `recursive-tok256-32` | 0.85 | 0.86 | 0.77 |
| hybrid | `recursive-tok256-32` | 0.86 | 0.86 | 0.71 |

With 12 questions, one question is 0.08 of mean recall, so these pairs are effectively tied.
No configuration choice was made from this run.

### Agent eval (`agent/eval/`)

- **Cases:** `agent_cases.toml`, 9 dev cases (context management over 4-turn conversations,
  and outlet/period filter requests), with any/all/none checks on facts, citations and
  tools, expected tool arguments, and graded abstention (D20).
- **Runner:** runs every case `repeats` times in parallel threads, one fresh thread id per
  run, and appends a transcript per run to `records.jsonl` as each finishes, so a crash
  keeps finished runs (D21).

```bash
# Show cases and minimum model calls; no model is called
uv run python -m ai_news_assistant.agent.eval.run_case --dry-run

# Run (all flags optional)
uv run python -m ai_news_assistant.agent.eval.run_case \
  --case filter-days-only --type filter-requests --repeats 1 --workers 3

# Score a run (default: newest finished run); writes report.md into the run folder
uv run python -m ai_news_assistant.agent.eval.evaluate [data/agent_eval/<run>]
```

Each run is graded by layer; the run takes its worst outcome (`invalid` > `fail` >
`fallback` > `pass`):

| Layer | Checks |
|---|---|
| L0 format | Structured output, non-empty text, no model-written "Sources:" line, no dropped markers, no duplicate citations |
| L1 trajectory | Required, optional and forbidden tools; expected `sources`/`days` arguments; `max_tool_calls` |
| L2 grounding | Every cited id was retrieved; `grounded=true` has a surviving source; a gold article is cited |
| L3 correctness | Expected facts present, forbidden strings absent; abstention read from the answer (no sources and `grounded=false`) |
| L6 efficiency | Median and max tool calls, input/output tokens and latency per case and per case type |

**LLM judge (L4, faithfulness):** `agent/eval/judge/judge.py` asks a judge model whether each
claim in the final answer is supported by the articles it cites (rubric
`judge/rubric_v1.md`). The judge must quote each claim; code drops quotes that are not in
the answer and sets the verdict (`faithful`, `minor`, `unfaithful`). It reads
`data/agent_eval/evaluate_the_judge/records.jsonl` and writes `judge_v1.jsonl` next to it.

```bash
uv run python -m ai_news_assistant.agent.eval.judge.judge
```

**Latest results** (D23, D24): a full run of 9 cases x 3 repeats (`20261102-085825_baseline`)
had 14 of 27 runs pass. The judge rated 22 of 27 runs of an earlier full run faithful and 5
unfaithful. The answers behind the failures have not all been read yet, and the judge is not
yet calibrated against human labels, so treat these as a starting baseline.

---

## Development

All commands run through `uv run`.

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy src                 # strict mode
uv run pytest                   # ~1,300 tests
uv run pytest tests/test_ingest.py::test_name   # a single test
```

- Unit tests never call a real LLM or the network: HTTP uses `httpx.MockTransport`, feeds
  and pages use saved fixtures (`tests/rss_fixtures/`, `tests/text_html_fixtures/`), and the
  agent, model and retriever are faked.
- Prompt-injection and malformed-input cases are part of the suite (hidden elements,
  delimiter look-alikes, hostile charsets, oversized bodies).
- CI (`.github/workflows/ci.yml`) runs `uv sync --frozen`, ruff, mypy and pytest on every
  pull request and on pushes to `main`.
- Dependencies are added with `uv add` / `uv add --dev`. On Linux, `torch` is installed from
  the CPU-only PyTorch index.

### Helper scripts (`scripts/`)

| Script | Purpose |
|---|---|
| `build_corpus.py` | Fill the snapshot's `corpus` table |
| `capture_fixtures.py [feed ...]` | Save each feed's XML (first 20 entries) to `tests/rss_fixtures/` |
| `capture_html_fixtures.py [feed ...]` | Save one article page per feed to `tests/text_html_fixtures/`, respecting robots.txt |
| `save_tool_specifics.py OUT.json` | Dump tool names, descriptions and argument schemas to a JSON file |

---

## Known limitations

The full list, with reasons and "revisit when" conditions, is in `docs/design.md`. The main
ones:

- **Citations are checked for retrieval, not support.** The check proves a cited article was
  retrieved, not that it says what the sentence claims; the LLM judge covers that offline.
- **No auth on the API or tools server.** Anyone who knows a `thread_id` can read and extend
  that conversation. Threads are never deleted.
- **No token streaming.** `/chat/stream` sends progress events, then the full answer at once.
- **Fixed snapshot.** The tools and retriever read the paths in `rag/config.py`; nothing
  selects a snapshot at run time.
- **API preflight requires `OPENAI_API_KEY`** even though the default model is Anthropic.
- **Small eval sets.** 12 scored retrieval questions and 9 agent cases; differences of one
  or two questions are ties. The test splits have not been run.
- **Not built yet:** near-duplicate detection, story clustering, `compare_sources` and
  `summarize_topic` tools, recency boost and source diversity in retrieval, tracing, a UI,
  and a deployed scheduled ingestion job.
