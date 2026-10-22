import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest
from conftest import CONFIG_ID

from ai_news_assistant.rag import vector_retriever
from ai_news_assistant.rag.base import Hit, Retriever, SearchFilter
from ai_news_assistant.rag.bm25_retriever import BM25Retriever
from ai_news_assistant.rag.eval import run_retrieval
from ai_news_assistant.rag.eval.dataset import Dataset, Evidence, Question, QuestionFilter
from ai_news_assistant.rag.eval.run_config import RunConfig
from ai_news_assistant.rag.eval.run_retrieval import (
    build_retrievers,
    run,
    run_question,
    to_search_filter,
)
from ai_news_assistant.rag.hybrid_retriever import HybridRetriever

SOURCES = ("Food Blog", "Guardian", "TechCrunch", "Verge")
CREATED = datetime(2026, 10, 22, 12, 0, 0)


class FakeRetriever:
    """Returns preset hits and records every search."""

    def __init__(self, hits: list[Hit] | None = None) -> None:
        self.hits = hits or []
        self.calls: list[tuple[str, int, SearchFilter | None]] = []

    def search(self, query: str, k: int, search_filter: SearchFilter | None = None) -> list[Hit]:
        self.calls.append((query, k, search_filter))
        return self.hits[:k]


def question(
    qid: str = "q-1", text: str = "What happened?", qf: QuestionFilter | None = None
) -> Question:
    return Question(
        id=qid,
        type="specific_detail",
        split="dev",
        question=text,
        gold_article_ids=("ai-act",),
        gold_mode="any",
        evidence=(Evidence("ai-act", "AI Act"),),
        filter=qf,
    )


def config(tmp_path: Path, **overrides: Any) -> RunConfig:
    values: dict[str, Any] = {
        "experiment": "exp-1",
        "split": "dev",
        "dataset": tmp_path / "questions.toml",
        "config_ids": ("cfg-a",),
        "retrievers": ("bm25",),
        "out_dir": tmp_path / "results",
        "max_articles": 2,
        "chunk_overfetch": 3,
    }
    return RunConfig(**(values | overrides))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


# to_search_filter


def test_no_filter_is_none() -> None:
    assert to_search_filter(question(), SOURCES) is None


def test_source_filter() -> None:
    result = to_search_filter(question(qf=QuestionFilter(source="Guardian")), SOURCES)

    assert result == SearchFilter(sources=frozenset({"Guardian"}))


def test_since_filter_is_utc_midnight() -> None:
    result = to_search_filter(question(qf=QuestionFilter(since=date(2026, 10, 1))), SOURCES)

    assert result == SearchFilter(since=datetime(2026, 10, 1, tzinfo=UTC))
    assert result.since is not None and result.since.tzinfo is not None


def test_source_and_since_filter() -> None:
    qf = QuestionFilter(source="Verge", since=date(2026, 10, 1))

    result = to_search_filter(question(qf=qf), SOURCES)

    assert result == SearchFilter(
        sources=frozenset({"Verge"}), since=datetime(2026, 10, 1, tzinfo=UTC)
    )


def test_unknown_source_names_the_question() -> None:
    with pytest.raises(ValueError, match="q-7: filter source 'The Guardian'"):
        to_search_filter(question("q-7", qf=QuestionFilter(source="The Guardian")), SOURCES)


# run_question


def test_run_question_passes_k_and_filter_and_keeps_hits() -> None:
    hits = [Hit("c2", "b", 0.9, "second text"), Hit("c1", "a", 0.5, "first text")]
    fake = FakeRetriever(hits)
    search_filter = SearchFilter(sources=frozenset({"Guardian"}))
    q = question(qf=QuestionFilter(source="Guardian"))

    row = run_question(fake, q, k=6, search_filter=search_filter)

    assert fake.calls == [("What happened?", 6, search_filter)]
    assert row["question_id"] == "q-1"
    assert row["filter"] == {"source": "Guardian", "since": None}
    assert row["seconds"] >= 0
    assert row["hits"] == [
        {"chunk_id": "c2", "article_id": "b", "score": 0.9, "text": "second text"},
        {"chunk_id": "c1", "article_id": "a", "score": 0.5, "text": "first text"},
    ]


# run


def test_run_writes_one_file_per_pair_and_run_json(tmp_path: Path) -> None:
    cfg = config(tmp_path, config_ids=("cfg-a", "cfg-b"), retrievers=("vector", "bm25"))
    dataset = Dataset(version=3, questions=(question("q-1"), question("q-2", "Second?")))
    fakes: dict[str, dict[str, Retriever]] = {
        c: {"vector": FakeRetriever(), "bm25": FakeRetriever()} for c in cfg.config_ids
    }

    run_dir = run(cfg, dataset, fakes, SOURCES, created=CREATED)

    assert run_dir == tmp_path / "results" / "20261022-120000_exp-1"
    assert sorted(p.name for p in run_dir.iterdir()) == [
        "bm25__cfg-a.jsonl",
        "bm25__cfg-b.jsonl",
        "run.json",
        "vector__cfg-a.jsonl",
        "vector__cfg-b.jsonl",
    ]
    rows = read_jsonl(run_dir / "bm25__cfg-b.jsonl")
    assert [r["question_id"] for r in rows] == ["q-1", "q-2"]
    meta = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    assert meta["dataset_version"] == 3
    assert meta["split"] == "dev"
    assert meta["question_ids"] == ["q-1", "q-2"]
    assert meta["config"]["max_articles"] == 2
    assert meta["config"]["hybrid"] == {"rrf_k": 60, "depth": 50}
    assert len(meta["files"]) == 4


def test_run_asks_for_max_articles_times_overfetch(tmp_path: Path) -> None:
    fake = FakeRetriever()

    run(config(tmp_path), Dataset(1, (question(),)), {"cfg-a": {"bm25": fake}}, SOURCES)

    assert [k for _, k, _ in fake.calls] == [6]


def test_run_refuses_existing_folder(tmp_path: Path) -> None:
    cfg = config(tmp_path)
    dataset = Dataset(1, (question(),))
    run(cfg, dataset, {"cfg-a": {"bm25": FakeRetriever()}}, SOURCES, created=CREATED)

    with pytest.raises(FileExistsError):
        run(cfg, dataset, {"cfg-a": {"bm25": FakeRetriever()}}, SOURCES, created=CREATED)


def test_run_checks_filters_before_any_search(tmp_path: Path) -> None:
    fake = FakeRetriever()
    dataset = Dataset(1, (question("q-1"), question("q-2", qf=QuestionFilter(source="Nope"))))

    with pytest.raises(ValueError, match="q-2"):
        run(config(tmp_path), dataset, {"cfg-a": {"bm25": fake}}, SOURCES)

    assert fake.calls == []
    assert not (tmp_path / "results").exists()


def test_run_with_real_bm25_applies_source_filter(tmp_path: Path, rag_db: Path) -> None:
    cfg = config(tmp_path, config_ids=(CONFIG_ID,), max_articles=10)
    q = question(text="robotics startup AI Act iPhone", qf=QuestionFilter(source="TechCrunch"))
    retrievers: dict[str, dict[str, Retriever]] = {
        CONFIG_ID: {"bm25": BM25Retriever(rag_db, CONFIG_ID)}
    }

    run_dir = run(cfg, Dataset(1, (q,)), retrievers, SOURCES)

    hits = read_jsonl(run_dir / f"bm25__{CONFIG_ID}.jsonl")[0]["hits"]
    assert hits
    assert {h["article_id"] for h in hits} == {"robotics"}


# main


def write_inputs(tmp_path: Path, db: Path, config_id: str) -> Path:
    (tmp_path / "questions.toml").write_text(
        """dataset_version = 1
[[question]]
id = "q-1"
type = "specific_detail"
split = "dev"
question = "What does the AI Act require?"
gold_article_ids = ["ai-act"]
gold_mode = "any"
[[question.evidence]]
article_id = "ai-act"
quote = "AI Act"
""",
        encoding="utf-8",
    )
    run_toml = tmp_path / "run.toml"
    run_toml.write_text(
        f"""experiment = "main-test"
split = "dev"
dataset = "{tmp_path / "questions.toml"}"
db = "{db}"
out_dir = "{tmp_path / "results"}"
config_ids = ["{config_id}"]
retrievers = ["bm25"]
""",
        encoding="utf-8",
    )
    return run_toml


def test_main_unknown_config_fails_before_building(
    tmp_path: Path, rag_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    built: list[str] = []
    monkeypatch.setattr(
        run_retrieval, "build_retrievers", lambda cfg, config_id: built.append(config_id) or {}
    )

    with pytest.raises(SystemExit):
        run_retrieval.main(["--config", str(write_inputs(tmp_path, rag_db, "no-such-cfg"))])

    assert built == []
    assert not (tmp_path / "results").exists()


def test_main_writes_run_folder(
    tmp_path: Path, rag_db: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeRetriever([Hit("c1", "ai-act", 1.0, "The AI Act.")])
    monkeypatch.setattr(run_retrieval, "build_retrievers", lambda cfg, config_id: {"bm25": fake})

    run_retrieval.main(["--config", str(write_inputs(tmp_path, rag_db, CONFIG_ID))])

    (run_dir,) = (tmp_path / "results").iterdir()
    assert run_dir.name.endswith("_main-test")
    assert read_jsonl(run_dir / f"bm25__{CONFIG_ID}.jsonl")[0]["hits"][0]["article_id"] == "ai-act"
    assert fake.calls == [("What does the AI Act require?", 150, None)]


# build_retrievers (loads the local BGE model; skipped when it is not cached)


def test_build_retrievers_hybrid_reuses_vector_and_bm25(
    tmp_path: Path, rag_db: Path, built_store: Path
) -> None:
    cfg = config(
        tmp_path,
        db=rag_db,
        vector_store=built_store,
        config_ids=(CONFIG_ID,),
        retrievers=("vector", "bm25", "hybrid"),
    )

    retrievers = build_retrievers(cfg, CONFIG_ID)

    assert list(retrievers) == ["vector", "bm25", "hybrid"]
    hybrid = retrievers["hybrid"]
    assert isinstance(hybrid, HybridRetriever)
    assert isinstance(retrievers["vector"], vector_retriever.VectorRetriever)
    assert hybrid.retrievers == [retrievers["vector"], retrievers["bm25"]]
    assert (hybrid.rrf_k, hybrid.depth) == (60, 50)


def test_build_retrievers_only_hybrid_still_builds_its_parts(
    tmp_path: Path, rag_db: Path, built_store: Path
) -> None:
    cfg = config(
        tmp_path,
        db=rag_db,
        vector_store=built_store,
        config_ids=(CONFIG_ID,),
        retrievers=("hybrid",),
    )

    retrievers = build_retrievers(cfg, CONFIG_ID)

    assert list(retrievers) == ["hybrid"]
