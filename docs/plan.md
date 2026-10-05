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