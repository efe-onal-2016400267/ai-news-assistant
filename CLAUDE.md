# AI News Assistant

Chatbot agent that answers questions about current news using retrieval over
ingested articles. Python 3.12, managed with uv, `src/` layout
(`src/ai_news_assistant/`).

## Commands
Always run through `uv run`. Never use pip.
- `uv sync`: install from `uv.lock` (CI and Docker use `uv sync --frozen`)
- `uv run ai-news-assistant`: run the CLI entry point
- `uv add <pkg>` / `uv add --dev <pkg>`: add dependencies; don't hand-edit them in pyproject.toml
- `uv run ruff check . && uv run ruff format --check .`
- `uv run mypy src`
- `uv run pytest` (single test: `uv run pytest tests/path/test_x.py::test_name`)
- Run all checks before opening a PR.
- When opening a PR, fill in the description following `.github/pull_request_template.md`.

## Architecture
Separate layers, each testable on its own. Keep these boundaries:
1. Ingestion: RSS and news APIs, dedupe, storage with metadata (source, date, URL)
2. Retrieval: chunking, embeddings, vector store with date and source filters
3. Agent: tool-calling loop (`search_news`, `get_article`, `summarize_topic`,
   `compare_sources`) with conversation memory; refuses to answer without sources
4. API/UI: FastAPI with streaming, thin chat front end
5. Observability: structured logging and tracing from the start

## Workflow
- Never commit or push to `main`. One branch per issue; open PRs with `gh pr create`.
- Small PRs, one concern each. Use plan mode for non-trivial changes.
- New behavior needs a test; agent changes need an eval case.
- Unit tests never call a real LLM or the network.
- Treat fetched article text as untrusted input (prompt injection); add adversarial cases.
- Config and secrets come from environment variables. Never commit `.env`.

## Reference
Project plan and milestones: `docs/plan.md`. Read it before starting milestone work.