from pathlib import Path

EMBEDDING_MODEL = "BAAI/bge-small-en-v1.5"
DEFAULT_DB = Path("data/snapshots/news-2026-10-12.db")
DEFAULT_VECTOR_STORE = Path("data/snapshots/chroma-2026-10-12")
EVAL_RESULT_DIR = Path("data/snapshots/eval_results/")
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
