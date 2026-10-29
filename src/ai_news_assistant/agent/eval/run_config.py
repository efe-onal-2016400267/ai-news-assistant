"""Load and check agent_run.toml"""

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from ai_news_assistant.agent.eval.dataset import Split
from ai_news_assistant.rag.config import DEFAULT_DB

ENABLED_SPLITS = frozenset({"dev"})
REQUIRED_KEYS = frozenset(
    {"experiment", "split", "dataset", "repeats", "workers", "out_dir", "recursion_limit"}
)
OPTIONAL_KEYS = frozenset({"db", "recursion_limit"})
_EXPERIMENT_RE = re.compile(r"[a-z0-9_-]+")


@dataclass(frozen=True)
class AgentRunConfig:
    experiment: str  # Experiment name to be filled in output file path
    split: Split
    dataset: Path  # path to agent_cases.toml
    repeats: int  # Repeat the same run repeats times
    workers: int  # Execute workers runs in parallel
    recursion_limit: int  # How many recursive calls can the agent make
    out_dir: Path  # The output path where the run results will be written to
    db: Path  # path to articles store


def load_run_config(config_path: Path) -> AgentRunConfig:
    """Loads an agent_run.toml file and returns the AgentRunConfig."""
    with config_path.open("rb") as f:
        data = tomllib.load(f)

    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"{config_path}: missing key(s): {', '.join(sorted(missing))}")

    unknown = set(data) - REQUIRED_KEYS - OPTIONAL_KEYS
    if unknown:
        raise ValueError(f"{config_path}: unknown key(s): {', '.join(sorted(unknown))}")

    experiment = data["experiment"]
    if not isinstance(experiment, str) or not _EXPERIMENT_RE.fullmatch(experiment):
        raise ValueError(f"{config_path}: experiment must match [a-z0-9_-]+, got {experiment!r}")

    split = data["split"]
    if split not in ENABLED_SPLITS:
        raise ValueError(f"{config_path}: split {split!r}: the test split is not enabled yet")

    return AgentRunConfig(
        experiment=experiment,
        split=cast(Split, split),
        dataset=Path(data["dataset"]),
        repeats=_positive_int(data["repeats"], 3),
        workers=_positive_int(data["workers"], 3),
        recursion_limit=_positive_int(data["recursion_limit"], 25),
        out_dir=Path(data["out_dir"]),
        db=Path(data.get("db", DEFAULT_DB)),
    )


def _positive_int(n: int, default: int) -> int:
    if int(n) < 1:
        return default
    return int(n)


if __name__ == "__main__":
    print(load_run_config(Path("src/ai_news_assistant/agent/eval/agent_run.toml")))
