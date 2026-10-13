from pathlib import Path

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DEFAULT_DB = Path("data/snapshots/news-2026-10-12.db")
DEFAULT_VECTOR_STORE = Path("data/snapshots/chroma-2026-10-12")
CHUNK_SIZE = 256
CHUNK_OVERLAP = 32