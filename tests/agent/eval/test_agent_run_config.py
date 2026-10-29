from pathlib import Path

import pytest

from ai_news_assistant.agent.eval import run_config
from ai_news_assistant.agent.eval.run_config import AgentRunConfig, load_run_config
from ai_news_assistant.rag.config import DEFAULT_DB

REAL_FILE = Path(run_config.__file__).with_name("agent_run.toml")

REQUIRED = """
experiment = "exp-1"
split = "dev"
dataset = "cases.toml"
repeats = 2
workers = 4
recursion_limit = 10
out_dir = "runs"
"""


def write(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "run.toml"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_file_loads() -> None:
    cfg = load_run_config(REAL_FILE)

    assert cfg == AgentRunConfig(
        experiment="baseline",
        split="dev",
        dataset=Path("src/ai_news_assistant/agent/eval/agent_cases.toml"),
        repeats=3,
        workers=3,
        recursion_limit=25,
        out_dir=Path("data/agent_eval/"),
        db=DEFAULT_DB,  # the db line is commented out
    )


def test_required_keys_only_uses_default_db(tmp_path: Path) -> None:
    cfg = load_run_config(write(tmp_path, REQUIRED))

    assert cfg == AgentRunConfig(
        experiment="exp-1",
        split="dev",
        dataset=Path("cases.toml"),
        repeats=2,
        workers=4,
        recursion_limit=10,
        out_dir=Path("runs"),
        db=DEFAULT_DB,
    )


def test_db_override(tmp_path: Path) -> None:
    cfg = load_run_config(write(tmp_path, REQUIRED + 'db = "other.db"\n'))

    assert cfg.db == Path("other.db")


@pytest.mark.parametrize(
    ("old", "new", "field", "expected"),
    [
        pytest.param("repeats = 2", "repeats = 0", "repeats", 3, id="zero-repeats"),
        pytest.param("workers = 4", "workers = -1", "workers", 3, id="negative-workers"),
        pytest.param(
            "recursion_limit = 10", "recursion_limit = 0", "recursion_limit", 25, id="zero-limit"
        ),
        pytest.param("repeats = 2", "repeats = 1", "repeats", 1, id="one-is-kept"),
    ],
)
def test_non_positive_int_falls_back_to_default(
    tmp_path: Path, old: str, new: str, field: str, expected: int
) -> None:
    assert old in REQUIRED
    cfg = load_run_config(write(tmp_path, REQUIRED.replace(old, new)))

    assert getattr(cfg, field) == expected


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        pytest.param('dataset = "cases.toml"', "", "missing key", id="missing-dataset"),
        pytest.param('"exp-1"', '"exp/1"', "experiment must match", id="experiment-slash"),
        pytest.param('"exp-1"', '"My-Exp"', "experiment must match", id="experiment-upper"),
        pytest.param('"exp-1"', '""', "experiment must match", id="experiment-empty"),
        pytest.param('"exp-1"', "1", "experiment must match", id="experiment-not-str"),
        pytest.param('split = "dev"', 'split = "test"', "not enabled yet", id="test-split"),
        pytest.param('split = "dev"', 'split = "train"', "not enabled yet", id="unknown-split"),
    ],
)
def test_invalid_values(tmp_path: Path, old: str, new: str, message: str) -> None:
    assert old in REQUIRED
    path = write(tmp_path, REQUIRED.replace(old, new))

    with pytest.raises(ValueError, match=message):
        load_run_config(path)


def test_missing_keys_are_listed_sorted(tmp_path: Path) -> None:
    text = REQUIRED.replace('split = "dev"\n', "").replace('experiment = "exp-1"\n', "")

    with pytest.raises(ValueError, match="missing key\\(s\\): experiment, split$"):
        load_run_config(write(tmp_path, text))


def test_unknown_key_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown key\\(s\\): top_k"):
        load_run_config(write(tmp_path, REQUIRED + "top_k = 5\n"))
