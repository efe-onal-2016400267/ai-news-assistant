from pathlib import Path

import pytest

from ai_news_assistant.rag.config import DEFAULT_DB, DEFAULT_VECTOR_STORE, EVAL_RESULT_DIR
from ai_news_assistant.rag.eval import run_config
from ai_news_assistant.rag.eval.run_config import HybridSettings, load_run_config

REAL_FILE = Path(run_config.__file__).with_name("rag_run.toml")

REQUIRED = """
experiment = "exp-1"
split = "dev"
dataset = "questions.toml"
config_ids = ["cfg-a"]
retrievers = ["bm25"]
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_file_loads() -> None:
    cfg = load_run_config(REAL_FILE)

    assert cfg.experiment == "baseline"
    assert cfg.split == "dev"
    assert cfg.dataset == Path("src/ai_news_assistant/rag/eval/rag_questions_v1.toml")
    assert cfg.db == Path("data/snapshots/news-2026-10-12.db")
    assert cfg.vector_store == Path("data/snapshots/chroma-2026-10-12")
    assert cfg.out_dir == Path("data/snapshots/eval_results")
    assert cfg.config_ids == ("words-tok256-32", "recursive-tok256-32")
    assert cfg.retrievers == ("vector", "bm25", "hybrid")
    assert cfg.max_articles == 50
    assert cfg.chunk_overfetch == 3
    assert cfg.chunk_k == 150
    assert cfg.hybrid == HybridSettings(rrf_k=60, depth=50)


def test_defaults_with_only_required_keys(tmp_path: Path) -> None:
    cfg = load_run_config(write(tmp_path, REQUIRED))

    assert cfg.dataset == Path("questions.toml")
    assert cfg.db == DEFAULT_DB
    assert cfg.vector_store == DEFAULT_VECTOR_STORE
    assert cfg.out_dir == EVAL_RESULT_DIR
    assert (cfg.max_articles, cfg.chunk_overfetch) == (50, 3)
    assert cfg.hybrid == HybridSettings()


def test_partial_hybrid_table_keeps_other_default(tmp_path: Path) -> None:
    cfg = load_run_config(write(tmp_path, REQUIRED + "[hybrid]\ndepth = 150\n"))

    assert cfg.hybrid == HybridSettings(rrf_k=60, depth=150)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        pytest.param('split = "dev"', 'split = "test"', "not enabled yet", id="test-split"),
        pytest.param('["bm25"]', '["bm25", "splade"]', "unknown retriever", id="unknown-retriever"),
        pytest.param('["bm25"]', "[]", "non-empty list", id="empty-retrievers"),
        pytest.param('["cfg-a"]', '["cfg-a", "cfg-a"]', "repeated", id="repeated-config-id"),
        pytest.param('"exp-1"', '"exp/1"', "experiment must match", id="experiment-slash"),
        pytest.param('"exp-1"', '"my exp"', "experiment must match", id="experiment-space"),
        pytest.param('dataset = "questions.toml"', "", "missing key", id="missing-dataset"),
    ],
)
def test_invalid_values(tmp_path: Path, old: str, new: str, message: str) -> None:
    assert old in REQUIRED
    path = write(tmp_path, REQUIRED.replace(old, new))

    with pytest.raises(ValueError, match=message):
        load_run_config(path)


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        pytest.param("max_articles = 0\n", "positive integer", id="zero-max-articles"),
        pytest.param("chunk_overfetch = true\n", "positive integer", id="bool-overfetch"),
        pytest.param("top_k = 5\n", "unknown key", id="unknown-key"),
        pytest.param("[hybrid]\nrrf = 60\n", r"unknown \[hybrid\] key", id="unknown-hybrid-key"),
        pytest.param("[hybrid]\ndepth = -1\n", "positive integer", id="negative-depth"),
    ],
)
def test_invalid_extra_keys(tmp_path: Path, extra: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        load_run_config(write(tmp_path, REQUIRED + extra))
