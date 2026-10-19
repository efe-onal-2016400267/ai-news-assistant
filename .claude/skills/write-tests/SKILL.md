---
name: write-tests
description: Use when asked to write or extend tests for changed code in the ai-news-assistant repo: plan the cases first, use fakes (no model, DB or network), run pytest, report gaps.
---

# Write tests

Workflow: plan first, one change at a time. Never run git commit, push or gh pr. The user commits.

## Steps

1. Read the changed code (git diff or the files named). Do not guess behaviour from names.
2. Propose a short test plan before writing anything: one line per behaviour, grouped as happy path, edge cases, error paths. Typical edges: empty input, `k <= 0`, ties and ordering, missing ids, empty results, both inputs empty.
3. Wait for the user's go-ahead on the plan unless they said to just write the tests.
4. Write the tests in `tests/`, mirroring the `src/ai_news_assistant/` layout (for example `tests/rag/test_hybrid_retriever.py`). If the layout is not present yet, you can set up this layout.
5. Run `uv run pytest` on the new file, then the whole suite. Run `uv run ruff check --fix`, `uv run ruff format` and `uv run mypy src`.
6. Report: what is covered, what is deliberately not covered, and anything in the code that looked wrong (do not silently change source code to make a test pass; if a source fix is needed, say so, wait for user approval, and keep it minimal).

## Conventions

- No model loading, no network, no real Chroma or LLM calls in unit tests. Use small fake classes (a fake retriever whose `search` returns a fixed list and records the `k` it received; a fake chat model for agent wiring).
- Include at least one hand-checked expected value, with the arithmetic in a comment (for example the RRF example A=[a,b,c], B=[b,d,a] gives b,a,d,c).
- Assert exact orderings and values, not just lengths. Test determinism where ties are possible.
- Test names say the behaviour: `test_k_zero_returns_empty`, not `test_search_2`.
- Plain pytest functions, small helpers or fixtures in the same file. Line length 100.
- Tests with an LLM or the real embedding model are evals, not unit tests. Put them under the eval code, not `tests/`.
- Tool tests: call the tool with `tool.invoke({...})` and check the returned value; a bad id must return a message, not raise.
- Never put real article text in test output or logs.

## Scope

If you notice something outside the issue text or plan.md while testing, stop and tell the user (necessary fixes excepted), and list it for the PR description.