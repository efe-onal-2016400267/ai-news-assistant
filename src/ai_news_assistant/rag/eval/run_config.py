"""Load and check `rag_run.toml`: what one retrieval eval run asks, and with what."""

import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast, get_args

from ai_news_assistant.rag.config import DEFAULT_DB, DEFAULT_VECTOR_STORE, EVAL_RESULT_DIR
from ai_news_assistant.rag.eval.dataset import Split

RetrieverName = Literal["vector", "bm25", "hybrid"]
RETRIEVER_NAMES = frozenset(get_args(RetrieverName))

ENABLED_SPLITS = frozenset({"dev"})  # test is scored once, after decisions are made on dev
REQUIRED_KEYS = frozenset({"experiment", "split", "dataset", "config_ids", "retrievers"})
OPTIONAL_KEYS = frozenset(
    {"db", "vector_store", "out_dir", "max_articles", "chunk_overfetch", "hybrid"}
)
HYBRID_KEYS = frozenset({"rrf_k", "depth"})
DEFAULT_MAX_ARTICLES = 50
DEFAULT_CHUNK_OVERFETCH = 3
_EXPERIMENT_RE = re.compile(r"[a-z0-9_-]+")


@dataclass(frozen=True)
class HybridSettings:
    rrf_k: int = 60
    depth: int = 50  # chunks asked from each retriever before fusion


@dataclass(frozen=True)
class RunConfig:
    experiment: str  # Used in the run folder name
    split: Split
    dataset: Path
    config_ids: tuple[str, ...]
    retrievers: tuple[RetrieverName, ...]
    db: Path = DEFAULT_DB
    vector_store: Path = DEFAULT_VECTOR_STORE
    out_dir: Path = EVAL_RESULT_DIR
    max_articles: int = DEFAULT_MAX_ARTICLES  # Article ranking depth
    chunk_overfetch: int = DEFAULT_CHUNK_OVERFETCH  # Chunks asked per wanted article
    hybrid: HybridSettings = field(default_factory=HybridSettings)

    @property
    def chunk_k(self) -> int:
        """Number of chunks asked from each retriever per question."""
        return self.max_articles * self.chunk_overfetch


def load_run_config(path: Path) -> RunConfig:
    """Read a run config file. Raises `ValueError` on any problem."""
    with path.open("rb") as f:
        data = tomllib.load(f)

    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise ValueError(f"{path}: missing key(s): {', '.join(sorted(missing))}")
    unknown = set(data) - REQUIRED_KEYS - OPTIONAL_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown key(s): {', '.join(sorted(unknown))}")

    experiment = data["experiment"]
    if not isinstance(experiment, str) or not _EXPERIMENT_RE.fullmatch(experiment):
        raise ValueError(f"{path}: experiment must match [a-z0-9_-]+, got {experiment!r}")

    split = data["split"]
    if split not in ENABLED_SPLITS:
        raise ValueError(f"{path}: split {split!r}: the test split is not enabled yet")

    retrievers = _name_list(data, "retrievers", path)
    bad = set(retrievers) - RETRIEVER_NAMES
    if bad:
        raise ValueError(f"{path}: unknown retriever(s): {', '.join(sorted(bad))}")

    return RunConfig(
        experiment=experiment,
        split=cast(Split, split),
        dataset=_path(data, "dataset", path),
        config_ids=_name_list(data, "config_ids", path),
        retrievers=cast(tuple[RetrieverName, ...], retrievers),
        db=_path(data, "db", path, DEFAULT_DB),
        vector_store=_path(data, "vector_store", path, DEFAULT_VECTOR_STORE),
        out_dir=_path(data, "out_dir", path, EVAL_RESULT_DIR),
        max_articles=_positive_int(data, "max_articles", path, DEFAULT_MAX_ARTICLES),
        chunk_overfetch=_positive_int(data, "chunk_overfetch", path, DEFAULT_CHUNK_OVERFETCH),
        hybrid=_hybrid(data.get("hybrid", {}), path),
    )


def _name_list(data: dict[str, Any], key: str, path: Path) -> tuple[str, ...]:
    value = data[key]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v for v in value):
        raise ValueError(f"{path}: {key} must be a non-empty list of strings")
    if len(set(value)) != len(value):
        raise ValueError(f"{path}: {key} has repeated names")
    return tuple(value)


def _path(data: dict[str, Any], key: str, path: Path, default: Path | None = None) -> Path:
    value = data.get(key)
    if value is None and default is not None:
        return default
    if not isinstance(value, str) or not value:
        raise ValueError(f"{path}: {key} must be a non-empty path string")
    return Path(value)


def _positive_int(data: dict[str, Any], key: str, path: Path, default: int) -> int:
    value = data.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{path}: {key} must be a positive integer, got {value!r}")
    return value


def _hybrid(value: Any, path: Path) -> HybridSettings:
    if not isinstance(value, dict):
        raise ValueError(f"{path}: [hybrid] must be a table")
    unknown = set(value) - HYBRID_KEYS
    if unknown:
        raise ValueError(f"{path}: unknown [hybrid] key(s): {', '.join(sorted(unknown))}")
    defaults = HybridSettings()
    return HybridSettings(
        rrf_k=_positive_int(value, "rrf_k", path, defaults.rrf_k),
        depth=_positive_int(value, "depth", path, defaults.depth),
    )
