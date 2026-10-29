import json
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AnyMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

from ai_news_assistant.agent.answer import AskResult
from ai_news_assistant.agent.config import CHAT_MODEL, CONTEXT_TRIGGER_TOKENS
from ai_news_assistant.agent.eval import run_case
from ai_news_assistant.agent.eval.dataset import Case, CaseDataset, CaseType, Expect, Setup
from ai_news_assistant.agent.eval.run_case import (
    RUN_CONFIG_PATH,
    LockedRetriever,
    parse_args,
    preflight,
    run_all,
    run_meta,
    run_one,
    select_cases,
)
from ai_news_assistant.agent.eval.run_config import AgentRunConfig
from ai_news_assistant.agent.eval.transcript import RunError, Transcript, TurnResult
from ai_news_assistant.models import Answer
from ai_news_assistant.rag.base import Hit, SearchFilter
from ai_news_assistant.rag.config import DEFAULT_DB

LOOKUP: CaseType = "single-turn-lookup-and-citation"
ABSTAIN: CaseType = "abstention"
SNAPSHOT = DEFAULT_DB.name


def case(
    cid: str = "c-1",
    turns: tuple[str, ...] = ("What happened?",),
    type_: CaseType = LOOKUP,
    setup: Setup | None = None,
) -> Case:
    return Case(
        id=cid,
        type=type_,
        split="dev",
        turns=turns,
        gold_article_ids=("a1",),
        expect=Expect(),
        setup=setup or Setup(),
    )


def dataset(*cases: Case) -> CaseDataset:
    return CaseDataset(
        version=1, snapshot=SNAPSHOT, abstain_phrases=("I do not know",), cases=cases
    )


def cfg(db: Path = DEFAULT_DB, repeats: int = 1, workers: int = 1) -> AgentRunConfig:
    return AgentRunConfig(
        experiment="exp1",
        split="dev",
        dataset=Path("cases.toml"),
        repeats=repeats,
        workers=workers,
        recursion_limit=7,
        out_dir=Path("runs"),
        db=db,
    )


def turn(query: str, start: int = 0) -> TurnResult:
    return TurnResult(
        user_query=query,
        tool_calls=(),
        retrieved_ids=(),
        text=f"answer from {start}",
        cited_ids=(),
        grounded=False,
        structured=True,
        raw_citations=(),
        raw_grounded=False,
        dropped_ids=(),
        dropped_markers=(),
        input_tokens=1,
        output_tokens=1,
        model_calls=1,
        latency_s=0.0,
    )


class FakeRetriever:
    def __init__(self, hits: list[Hit] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[tuple[str, int, SearchFilter | None]] = []

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        self.calls.append((query, k, search_filter))
        return self.hits


@dataclass
class FakeAgent:
    """Replaces build_agent, ask_detailed and extract_turn in run_case.

    Each question in `grow` adds that many messages to the thread; a question in `fail`
    raises RuntimeError with that message. Thread-safe enough for run_all (list.append).
    """

    grow: dict[str, int] = field(default_factory=dict)
    fail: dict[str, str] = field(default_factory=dict)
    build_error: Exception | None = None
    build_kwargs: list[dict[str, Any]] = field(default_factory=list)
    asks: list[tuple[str, RunnableConfig, Path]] = field(default_factory=list)
    starts: list[int] = field(default_factory=list)
    threads: dict[str, list[AnyMessage]] = field(default_factory=dict)

    def install(self, monkeypatch: pytest.MonkeyPatch) -> "FakeAgent":
        monkeypatch.setattr(run_case, "build_agent", self.build_agent)
        monkeypatch.setattr(run_case, "ask_detailed", self.ask_detailed)
        monkeypatch.setattr(run_case, "extract_turn", self.extract_turn)
        return self

    def build_agent(self, **kwargs: Any) -> Any:
        self.build_kwargs.append(kwargs)
        if self.build_error is not None:
            raise self.build_error
        return object()

    def ask_detailed(
        self, agent: Any, question: str, config: RunnableConfig, store_path: Path
    ) -> AskResult:
        self.asks.append((question, config, store_path))
        if question in self.fail:
            raise RuntimeError(self.fail[question])
        thread = self.threads.setdefault(config["configurable"]["thread_id"], [])
        thread.extend(HumanMessage(content=question) for _ in range(self.grow.get(question, 2)))
        answer = Answer(text="x", sources=[], grounded=False)
        return AskResult(answer=answer, model_answer=None, checked=None, messages=list(thread))

    def extract_turn(
        self, user_query: str, result: AskResult, start: int, latency_s: float
    ) -> TurnResult:
        self.starts.append(start)
        return turn(user_query, start)


# run_one: happy path


def test_run_one_two_turns_hand_checked_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAgent(grow={"q1": 3, "q2": 4}).install(monkeypatch)

    rec = run_one(case("c-1", turns=("q1", "q2")), 2, cfg(), FakeRetriever())

    # Turn 1 thread has 3 messages, turn 2 adds 4 -> 7. Turn 2 starts where turn 1 ended.
    assert fake.starts == [0, 3]
    assert rec == Transcript(
        case_id="c-1", run_idx=2, turns=(turn("q1", 0), turn("q2", 3)), error=None
    )


def test_run_one_turns_share_one_thread_and_runs_do_not(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAgent().install(monkeypatch)

    run_one(case(turns=("q1", "q2")), 0, cfg(), FakeRetriever())
    run_one(case(turns=("q1",)), 1, cfg(), FakeRetriever())

    thread_ids = [c["configurable"]["thread_id"] for _, c, _ in fake.asks]
    assert thread_ids[0] == thread_ids[1]
    assert thread_ids[2] != thread_ids[0]
    assert all(c["recursion_limit"] == 7 for _, c, _ in fake.asks)


def test_run_one_trigger_from_case_or_default(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAgent().install(monkeypatch)
    retriever = FakeRetriever()

    run_one(case(setup=Setup(trigger=500)), 0, cfg(), retriever)
    run_one(case(), 0, cfg(), retriever)

    assert [kw["trigger_tokens"] for kw in fake.build_kwargs] == [500, CONTEXT_TRIGGER_TOKENS]
    assert all(kw["retriever"] is retriever for kw in fake.build_kwargs)


def test_run_one_fixture_store_goes_to_ask(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAgent().install(monkeypatch)

    run_one(case(setup=Setup(fixture_store="fixtures/inj.db")), 0, cfg(), FakeRetriever())
    run_one(case(), 0, cfg(), FakeRetriever())

    assert [store for _, _, store in fake.asks] == [Path("fixtures/inj.db"), DEFAULT_DB]


# run_one: errors become records


def test_run_one_error_on_second_turn_keeps_first(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAgent(fail={"q2": "boom"}).install(monkeypatch)

    rec = run_one(case(turns=("q1", "q2", "q3")), 0, cfg(), FakeRetriever())

    assert rec.turns == (turn("q1", 0),)
    assert rec.error == RunError(turn=2, kind="RuntimeError", error_message="boom")


def test_run_one_error_building_agent_is_turn_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAgent(build_error=KeyError("model")).install(monkeypatch)

    rec = run_one(case(), 0, cfg(), FakeRetriever())

    assert rec.turns == ()
    assert rec.error == RunError(turn=0, kind="KeyError", error_message="'model'")


def test_run_one_truncates_error_message(monkeypatch: pytest.MonkeyPatch) -> None:
    FakeAgent(fail={"q1": "x" * 500}).install(monkeypatch)

    rec = run_one(case(turns=("q1",)), 0, cfg(), FakeRetriever())

    assert rec.error is not None
    assert rec.error.error_message == "x" * 200


# run_all


def test_run_all_sorted_records_and_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    FakeAgent().install(monkeypatch)
    (tmp_path / "records.jsonl").write_text('{"earlier": true}\n', encoding="utf-8")
    data = dataset(case("b", turns=("qb",)), case("a", turns=("qa",)))

    records = run_all(data, cfg(repeats=2, workers=3), tmp_path, FakeRetriever())

    assert [(r.case_id, r.run_idx) for r in records] == [("a", 0), ("a", 1), ("b", 0), ("b", 1)]
    lines = (tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert lines[0] == '{"earlier": true}'  # opened for append, not overwritten
    written = [Transcript.from_dict(json.loads(line)) for line in lines[1:]]
    assert sorted(written, key=lambda r: (r.case_id, r.run_idx)) == records


def test_run_all_failing_job_is_written_and_others_finish(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    FakeAgent(fail={"bad": "nope"}).install(monkeypatch)
    data = dataset(case("a", turns=("bad",)), case("b", turns=("ok",)))

    records = run_all(data, cfg(workers=2), tmp_path, FakeRetriever())

    assert [(r.case_id, r.error) for r in records] == [
        ("a", RunError(turn=1, kind="RuntimeError", error_message="nope")),
        ("b", None),
    ]
    assert len((tmp_path / "records.jsonl").read_text(encoding="utf-8").splitlines()) == 2


# run_meta


def test_run_meta_hand_checked() -> None:
    ok = Transcript(case_id="a", run_idx=0, turns=())
    bad = Transcript(case_id="a", run_idx=1, turns=(), error=RunError(1, "E", "m"))
    started = datetime(2026, 10, 29, 9, 0, 0, 123456, tzinfo=UTC)
    finished = datetime(2026, 10, 29, 9, 5, 30, tzinfo=UTC)
    config = cfg()

    meta = run_meta(config, dataset(case()), started, finished, [ok, bad, ok])

    assert meta == {
        "experiment": "exp1",
        "started_at": "2026-10-29T09:00:00+00:00",  # microseconds dropped
        "finished_at": "2026-10-29T09:05:30+00:00",
        "config": {
            "experiment": "exp1",
            "split": "dev",
            "dataset": Path("cases.toml"),
            "repeats": 1,
            "workers": 1,
            "recursion_limit": 7,
            "out_dir": Path("runs"),
            "db": DEFAULT_DB,
        },
        "dataset": {"version": 1, "snapshot": SNAPSHOT},
        "agent": {"model": CHAT_MODEL, "trigger": CONTEXT_TRIGGER_TOKENS},
        "records": 3,
        "errors": 1,  # only `bad` has an error
    }


def test_run_meta_before_finish_has_nones() -> None:
    meta = run_meta(cfg(), dataset(case()), datetime(2026, 1, 1, tzinfo=UTC))

    assert (meta["finished_at"], meta["records"], meta["errors"]) == (None, None, None)


# select_cases

CASES = (
    case("a", type_=LOOKUP),
    case("b", type_=ABSTAIN),
    case("c", type_=LOOKUP),
)


def test_select_no_flags_keeps_all_in_order() -> None:
    assert select_cases(CASES, None, None) == CASES


def test_select_by_ids_keeps_file_order() -> None:
    assert [c.id for c in select_cases(CASES, ["c", "a"], None)] == ["a", "c"]


def test_select_by_type() -> None:
    assert [c.id for c in select_cases(CASES, None, [ABSTAIN])] == ["b"]


def test_select_ids_and_types_intersect() -> None:
    assert [c.id for c in select_cases(CASES, ["a", "b"], [LOOKUP])] == ["a"]


def test_select_unknown_ids_raise_sorted() -> None:
    with pytest.raises(ValueError, match=r"^unknown case id\(s\): x, y$"):
        select_cases(CASES, ["y", "a", "x"], None)


def test_select_empty_intersection_raises() -> None:
    with pytest.raises(ValueError, match="no cases selected"):
        select_cases(CASES, ["b"], [LOOKUP])


# preflight


def test_preflight_missing_db(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="database not found"):
        preflight(dataset(case()), cfg(db=tmp_path / SNAPSHOT))


def test_preflight_snapshot_mismatch(tmp_path: Path) -> None:
    db = tmp_path / "news-2020-01-01.db"
    db.touch()
    with pytest.raises(ValueError, match=f"cases are for snapshot {SNAPSHOT}"):
        preflight(dataset(case()), cfg(db=db))


def test_preflight_db_not_the_retriever_db(tmp_path: Path) -> None:
    db = tmp_path / SNAPSHOT  # right name, wrong place
    db.touch()
    with pytest.raises(ValueError, match="the retriever reads"):
        preflight(dataset(case()), cfg(db=db))


def test_preflight_passes_on_default_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    DEFAULT_DB.parent.mkdir(parents=True)
    DEFAULT_DB.touch()

    preflight(dataset(case()), cfg())


# parse_args


def test_parse_args_defaults() -> None:
    args = parse_args([])

    assert args.config == RUN_CONFIG_PATH
    assert (args.cases, args.types, args.repeats, args.workers) == (None, None, None, None)
    assert args.dry_run is False


def test_parse_args_repeated_flags_collect() -> None:
    args = parse_args(
        ["--case", "a", "--case", "b", "--type", ABSTAIN, "--repeats", "2", "--dry-run"]
    )

    assert (args.cases, args.types, args.repeats, args.dry_run) == (["a", "b"], [ABSTAIN], 2, True)


@pytest.mark.parametrize(
    "argv", [["--repeats", "0"], ["--workers", "-1"], ["--repeats", "x"], ["--type", "bogus"]]
)
def test_parse_args_rejects_bad_values(argv: list[str]) -> None:
    with pytest.raises(SystemExit):
        parse_args(argv)


# LockedRetriever


def test_locked_retriever_forwards() -> None:
    hits = [Hit(chunk_id="a#0", article_id="a", score=1.0, text="t")]
    inner = FakeRetriever(hits)
    search_filter = SearchFilter(sources=frozenset({"BBC"}))

    assert LockedRetriever(inner).search("q", 4, search_filter) == hits
    assert inner.calls == [("q", 4, search_filter)]


def test_locked_retriever_one_search_at_a_time() -> None:
    class Counting:
        def __init__(self) -> None:
            self.inside = 0
            self.max_inside = 0

        def search(
            self, query: str, k: int, search_filter: SearchFilter | None = None
        ) -> list[Hit]:
            self.inside += 1
            self.max_inside = max(self.max_inside, self.inside)
            time.sleep(0.005)
            self.inside -= 1
            return []

    inner = Counting()
    locked = LockedRetriever(inner)
    threads = [threading.Thread(target=locked.search, args=("q", 1)) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert inner.max_inside == 1


# main

CASES_TOML = f"""
dataset_version = 1
snapshot = "{SNAPSHOT}"
abstain_phrases = ["I do not know"]

[[case]]
id = "one"
type = "{LOOKUP}"
split = "dev"
turns = ["q1"]
gold_article_ids = ["a1"]
expect = {{}}

[[case]]
id = "two"
type = "{LOOKUP}"
split = "dev"
turns = ["q2", "q3"]
gold_article_ids = ["a2"]
expect = {{}}
"""

RUN_TOML = """
experiment = "exp1"
split = "dev"
dataset = "cases.toml"
repeats = 3
workers = 3
out_dir = "out"
recursion_limit = 25
"""


@pytest.fixture
def workdir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """cwd with a cases file, a run config and the snapshot db; records build_retriever calls."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "cases.toml").write_text(CASES_TOML, encoding="utf-8")
    (tmp_path / "run.toml").write_text(RUN_TOML, encoding="utf-8")
    DEFAULT_DB.parent.mkdir(parents=True)
    DEFAULT_DB.touch()
    built: list[str] = []

    def build_retriever() -> FakeRetriever:
        built.append("built")
        return FakeRetriever()

    monkeypatch.setattr(run_case, "load_dotenv", lambda: None)
    monkeypatch.setattr(run_case, "build_retriever", build_retriever)
    return built


def test_main_dry_run_counts_and_runs_nothing(
    workdir: list[str], capsys: pytest.CaptureFixture[str]
) -> None:
    run_case.main(["--config", "run.toml", "--dry-run", "--repeats", "2"])

    # (1 + 2) turns x 2 repeats = 6
    assert "2 cases x 2 repeats, at least 6 model calls" in capsys.readouterr().out
    assert workdir == []
    assert not Path("out").exists()


def test_main_writes_records_and_run_json(
    workdir: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    FakeAgent(fail={"q3": "late"}).install(monkeypatch)

    run_case.main(["--config", "run.toml", "--repeats", "1", "--workers", "2"])

    [run_dir] = list(Path("out").iterdir())
    assert run_dir.name.endswith("_exp1")
    lines = (run_dir / "records.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(json.loads(line)["case_id"] for line in lines) == ["one", "two"]
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert (meta["config"]["repeats"], meta["config"]["workers"]) == (1, 2)
    assert (meta["records"], meta["errors"]) == (2, 1)  # "two" fails on its second turn
    assert workdir == ["built"]


def test_main_only_selected_case(workdir: list[str], monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeAgent().install(monkeypatch)

    run_case.main(["--config", "run.toml", "--case", "one", "--repeats", "1"])

    assert [q for q, _, _ in fake.asks] == ["q1"]


def test_main_unknown_case_exits(workdir: list[str]) -> None:
    with pytest.raises(SystemExit, match=r"^error: unknown case id\(s\): nope$"):
        run_case.main(["--config", "run.toml", "--case", "nope"])

    assert not Path("out").exists()
    assert workdir == []
