---
name: update-docs
description: Use when asked to update docs after a change in the ai-news-assistant repo: check the diff against design.md and plan.md, edit only affected sections, and draft PR notes for out-of-scope changes.
---

# Update docs

Workflow: plan first, one change at a time. Never run git commit, push or gh pr. The user commits and opens PRs.

## Steps

1. Read the diff and the relevant issue text or plan.md.
2. List which docs the change affects (design.md, README, .env.example, plan files). Propose the edits as a short list before writing, unless the user said to just do it.
3. Edit only the affected sections. Do not rewrite or reflow untouched text.
4. Check every number, name and path in the text against the code or the saved result files. Do not paste figures from memory.
5. Run `uv run ruff check` and `uv run mypy src` only if code or snippets changed; otherwise skip.
6. Report what changed in each file, in two or three lines.

## What design.md holds

- Decisions with their reasons (a table: decision, choice, reason).
- Formulas and definitions (for example the RRF formula, the fusion level).
- Results tables with the sample size stated. With n=13 questions, differences of about 0.02 to 0.03 are noise: say so, and do not claim a winner.
- The decision rule written before the results, and its outcome.
- Known consequences and open problems (for example: RRF scores cannot drive an abstain threshold).
- Agent notes: tools, prompt rules, eval layers, cost and latency, failure cases.

## What plan.md holds

- Step by step plan of each part of the project (for example, article fetching, RAG, Agent...)
- Each part has it's steps as a checklist, as the tasks are done the user checks the checklist.

## Style

Short sentences, concrete values, no marketing words. Say what was measured and what was not. Keep decisions and results separate.

## PR notes

End with a draft PR description section the user can paste: what changed, and a list of anything outside the issue text (changed signatures, new store queries, renamed modules). Write plain text only; no attribution lines or trailers, because the user writes the commits and PRs.

## Scope

If the code and the docs disagree and it is unclear which is right, stop and ask. If you find something outside the issue text or plan.md, tell the user instead of fixing it.