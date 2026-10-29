"""Agent eval Script 1: Run the cases in the case toml file.

Output: <out_dir>/YYYYmmdd-HHMMSS_<experiment>/run.json.
<out_dir>/YYYYmmdd-HHMMSS_<experiment>/records.jsonl -> one line per record,
appended as each run finishes.

Example:
"run": {
  "experiment": "baseline",
  "started_at": "...",
  "finished_at": "...",
  "config": { "...": "..." },
  "agent": { "model": "...", "prompt_hash": "...", "recursion_limit": 25 },
  "db": "data/snapshots/news-2026-10-12.db"
}

"""

import argparse
import json
import threading
import time
import uuid
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from ai_news_assistant.agent.agent import build_agent, build_retriever
from ai_news_assistant.agent.answer import ask_detailed
from ai_news_assistant.agent.config import CHAT_MODEL, CONTEXT_TRIGGER_TOKENS
from ai_news_assistant.agent.eval.dataset import CASE_TYPES, Case, CaseDataset, load_cases
from ai_news_assistant.agent.eval.extract import extract_turn
from ai_news_assistant.agent.eval.run_config import AgentRunConfig, load_run_config
from ai_news_assistant.agent.eval.transcript import RunError, Transcript, TurnResult
from ai_news_assistant.rag.base import Hit, Retriever, SearchFilter
from ai_news_assistant.rag.config import DEFAULT_DB

RUN_CONFIG_PATH = Path("src/ai_news_assistant/agent/eval/agent_run.toml")


class LockedRetriever:
    """Wraps a retriever so worker threads search one at a time.

    The embedding model crashes the process (native abort on the Apple GPU backend)
    when two threads use it at once. Searches take milliseconds and model calls take
    seconds, so the workers still overlap where it matters.
    """

    def __init__(self, inner: Retriever) -> None:
        self._inner = inner
        self._lock = threading.Lock()

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        with self._lock:
            return self._inner.search(query, k, search_filter)


def run_one(
    case: Case,
    repeat: int,
    cfg: AgentRunConfig,
    retriever: Retriever,
) -> Transcript:
    """Runs all turns of one case. Never raises: failures become records"""
    db = Path(case.setup.fixture_store) if case.setup.fixture_store else cfg.db

    turns: list[TurnResult] = []
    error: RunError | None = None
    current_turn = 0  # 0 means it failed before the first turn

    runnable_config: RunnableConfig = {
        "configurable": {"thread_id": str(uuid.uuid4())},
        "recursion_limit": cfg.recursion_limit,
    }
    try:
        agent = build_agent(
            retriever=retriever,
            trigger_tokens=case.setup.trigger
            if case.setup.trigger is not None
            else CONTEXT_TRIGGER_TOKENS,
            store_path=cfg.db,
        )
        start = 0
        for question in case.turns:
            current_turn += 1
            t0 = time.perf_counter()
            ask_result = ask_detailed(
                agent=agent, question=question, config=runnable_config, store_path=db
            )
            turn_result = extract_turn(
                user_query=question,
                result=ask_result,
                start=start,
                latency_s=time.perf_counter() - t0,
            )
            turns.append(turn_result)
            start = len(ask_result.messages)
    except Exception as e:
        error = RunError(turn=current_turn, kind=type(e).__name__, error_message=str(e)[:200])

    return Transcript(case_id=case.id, run_idx=repeat, turns=tuple(turns), error=error)


def run_all(
    dataset: CaseDataset,
    cfg: AgentRunConfig,
    run_dir: Path,
    retriever: Retriever,
) -> list[Transcript]:
    """Run all cases concurrently."""
    # c: case, i: repetition number
    jobs = [(c, i) for c in dataset.cases for i in range(cfg.repeats)]
    records: list[Transcript] = []
    lock = threading.Lock()

    with (
        (run_dir / "records.jsonl").open("a", encoding="utf-8") as out,
        ThreadPoolExecutor(max_workers=cfg.workers) as pool,
    ):
        futures = {pool.submit(run_one, c, i, cfg, retriever): (c.id, i) for c, i in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                out.write(json.dumps(rec.to_dict(), ensure_ascii=False) + "\n")
                out.flush()
                records.append(rec)
            outcome = "ok" if rec.error is None else rec.error.kind
            print(f"[{n}/{len(jobs)}] {rec.case_id}#{rec.run_idx} {outcome}")
    return sorted(records, key=lambda r: (r.case_id, r.run_idx))


def run_meta(
    cfg: AgentRunConfig,
    dataset: CaseDataset,
    started_at: datetime,
    finished_at: datetime | None = None,
    records: list[Transcript] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": cfg.experiment,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds") if finished_at else None,
        "config": asdict(cfg),
        "dataset": {"version": dataset.version, "snapshot": dataset.snapshot},
        "agent": {"model": CHAT_MODEL, "trigger": CONTEXT_TRIGGER_TOKENS},
        "records": len(records) if records is not None else None,
        "errors": sum(t.error is not None for t in records) if records is not None else None,
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the agent eval cases and save the records.")
    parser.add_argument("--config", type=Path, default=RUN_CONFIG_PATH, help="Run config toml")
    parser.add_argument(
        "--case",
        action="append",
        dest="cases",
        metavar="ID",
        help="Run only this case id (repeat the flag for several)",
    )
    parser.add_argument("--type", action="append", dest="types", choices=sorted(CASE_TYPES))
    parser.add_argument("--repeats", type=_positive_int, help="Overrides repeats in the config")
    parser.add_argument("--workers", type=_positive_int, help="Overrides workers in the config")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would run and the minimum number of model calls; call no model",
    )
    return parser.parse_args(argv)


def select_cases(
    cases: Sequence[Case], case_ids: Sequence[str] | None, types: Sequence[str] | None
) -> tuple[Case, ...]:
    """Keep the cases named by --case and of the types named by --type (no flag = all)."""
    if case_ids:
        unknown = set(case_ids) - {case.id for case in cases}
        if unknown:
            raise ValueError(f"unknown case id(s): {', '.join(sorted(unknown))}")
    selected = tuple(
        case
        for case in cases
        if (not case_ids or case.id in case_ids) and (not types or case.type in types)
    )
    if not selected:
        raise ValueError("no cases selected")
    return selected


def preflight(dataset: CaseDataset, cfg: AgentRunConfig) -> None:
    """Fail before any model call if the run would not measure what the cases describe."""
    if not cfg.db.exists():
        raise ValueError(f"database not found: {cfg.db}")
    if cfg.db.name != dataset.snapshot:
        raise ValueError(f"cases are for snapshot {dataset.snapshot}, config db is {cfg.db.name}")
    if cfg.db.resolve() != DEFAULT_DB.resolve():
        # build_retriever reads DEFAULT_DB, so a different db here would be silently ignored
        raise ValueError(f"the retriever reads {DEFAULT_DB}, but the config db is {cfg.db}")


def _positive_int(text: str) -> int:
    value = int(text)
    if value < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return value


def main(argv: Sequence[str] | None = None) -> None:
    load_dotenv()
    args = parse_args(argv)

    # 0. Experiment config with the command line overrides applied
    run_config = load_run_config(args.config)
    overrides = {
        name: value
        for name, value in (("repeats", args.repeats), ("workers", args.workers))
        if value is not None
    }
    run_config = replace(run_config, **overrides)

    # 1. Cases of the configured split, filtered by --case / --type
    dataset = load_cases(path=run_config.dataset, split=run_config.split)
    try:
        dataset = replace(dataset, cases=select_cases(dataset.cases, args.cases, args.types))
        preflight(dataset, run_config)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc

    min_calls = sum(len(case.turns) for case in dataset.cases) * run_config.repeats
    print(
        f"{len(dataset.cases)} cases x {run_config.repeats} repeats, "
        f"at least {min_calls} model calls (usually 2-5 per turn)"
    )
    if args.dry_run:
        return

    # 2: Run (the retriever loads the embedding model, so it comes after the dry run)
    retriever = LockedRetriever(build_retriever())
    started = datetime.now(UTC)

    run_dir = str(run_config.out_dir) + "/"
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{run_config.experiment}"
    run_path = Path(run_dir + stamp)
    run_path.mkdir(parents=True, exist_ok=False)
    records = run_all(dataset=dataset, cfg=run_config, run_dir=run_path, retriever=retriever)

    finished = datetime.now(UTC)
    run_report = run_meta(
        cfg=run_config,
        dataset=dataset,
        started_at=started,
        finished_at=finished,
        records=records,
    )
    (run_path / "run.json").write_text(json.dumps(run_report, indent=2, default=str))
    errors = run_report["errors"]
    print(f"Wrote {run_dir}: {len(records)} records, {errors} with errors")


if __name__ == "__main__":
    main()
