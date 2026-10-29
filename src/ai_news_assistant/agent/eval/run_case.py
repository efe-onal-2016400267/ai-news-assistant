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

import json
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.runnables import RunnableConfig

from ai_news_assistant.agent.agent import build_agent, build_retriever
from ai_news_assistant.agent.answer import ask_detailed
from ai_news_assistant.agent.config import CHAT_MODEL, CONTEXT_TRIGGER_TOKENS
from ai_news_assistant.agent.eval.dataset import Case, CaseDataset, load_cases
from ai_news_assistant.agent.eval.extract import extract_turn
from ai_news_assistant.agent.eval.run_config import AgentRunConfig, load_run_config
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
) -> dict[str, Any]:
    """Runs all turns of one case. Never raises: failures become records"""
    db = Path(case.setup.fixture_store) if case.setup.fixture_store else cfg.db
    turns = []
    start = 0
    status = "ok"
    error = None
    runnable_config: RunnableConfig = {"configurable": {"thread_id": str(uuid.uuid4())}}
    agent = build_agent(
        retriever=retriever,
        trigger_tokens=case.setup.trigger
        if case.setup.trigger is not None
        else CONTEXT_TRIGGER_TOKENS,
        store_path=cfg.db,
    )

    try:
        for question in case.turns:
            t0 = time.perf_counter()
            ask_result = ask_detailed(
                agent=agent, question=question, config=runnable_config, store_path=DEFAULT_DB
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
        status, error = "error", repr(e)

    return {
        "case_id": case.id,
        "repeat": repeat,
        "db": str(db),
        "status": status,
        "error": error,
        "final_turn": asdict(turns[-1]) if status == "ok" else None,
        "earlier_turns": [asdict(t) for t in turns[:-1]],
    }


def run_all(
    dataset: CaseDataset,
    cfg: AgentRunConfig,
    run_dir: Path,
    retriever: Retriever,
) -> list[dict[str, Any]]:
    """Run all cases concurrently."""
    # c: case, i: repetition number
    jobs = [(c, i) for c in dataset.cases for i in range(cfg.repeats)]
    records, lock = [], threading.Lock()

    with (
        (run_dir / "records.jsonl").open("a") as out,
        ThreadPoolExecutor(max_workers=cfg.workers) as pool,
    ):
        futures = {pool.submit(run_one, c, i, cfg, retriever): (c.id, i) for c, i in jobs}
        for n, fut in enumerate(as_completed(futures), 1):
            rec = fut.result()
            with lock:
                out.write(json.dumps(rec) + "\n")
                out.flush()
                records.append(rec)
            print(f"[{n}/{len(jobs)}] {rec['case_id']}#{rec['repeat']} {rec['status']}")
    return sorted(records, key=lambda r: (r["case_id"], r["repeat"]))


def run_meta(
    cfg: AgentRunConfig,
    dataset: CaseDataset,
    started_at: datetime,
    finished_at: datetime | None = None,
    records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    return {
        "experiment": cfg.experiment,
        "started_at": started_at.isoformat(timespec="seconds"),
        "finished_at": finished_at.isoformat(timespec="seconds") if finished_at else None,
        "config": asdict(cfg),
        "dataset": {"version": dataset.version, "snapshot": dataset.snapshot},
        "agent": {"model": CHAT_MODEL, "trigger": CONTEXT_TRIGGER_TOKENS},
        "records": len(records) if records is not None else None,
        "errors": sum(r["status"] != "ok" for r in records) if records is not None else None,
    }


def main() -> None:
    load_dotenv()

    started = datetime.now(UTC)

    # 0: Load experiment config
    run_config: AgentRunConfig = load_run_config(RUN_CONFIG_PATH)

    # 1: Load the cases in a CaseDataset
    cases_dataset: CaseDataset = load_cases(path=run_config.dataset, split=run_config.split)

    retriever = LockedRetriever(build_retriever())

    run_dir = "data/agent_eval/"
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}_{run_config.experiment}"
    run_path = Path(run_dir + stamp)
    run_path.mkdir(parents=True, exist_ok=False)
    records = run_all(dataset=cases_dataset, cfg=run_config, run_dir=run_path, retriever=retriever)

    finished = datetime.now(UTC)
    run_report = run_meta(
        cfg=run_config,
        dataset=cases_dataset,
        started_at=started,
        finished_at=finished,
        records=records,
    )
    (run_path / "run.json").write_text(json.dumps(run_report, indent=2, default=str))

    print("News assistant. Type 'exit' to quit.")


if __name__ == "__main__":
    main()
