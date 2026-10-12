from collections.abc import Callable

import pytest

try:
    from ai_news_assistant.rag.chunkers import chunk_recursive, chunk_words, tokenizer
except OSError:
    # chunkers loads the BGE tokenizer at import time; it needs the HF Hub or a local cache.
    pytest.skip(
        "BGE tokenizer unavailable (needs HF Hub download or cache)", allow_module_level=True
    )

Chunker = Callable[[str, int, int], list[str]]

SIZE = 64
OVERLAP = 16

CHUNKERS = [
    pytest.param(chunk_words, id="words"),
    pytest.param(chunk_recursive, id="recursive"),
]

# Unique words, so a word in two chunks can only come from overlap.
NUMBERED_TEXT = " ".join(f"word{i}" for i in range(300))

PARAGRAPHS = [
    " ".join(f"para{i}term{j}" for j in range(6)) + f" ends paragraph {i}." for i in range(10)
]
PARAGRAPH_TEXT = "\n\n".join(PARAGRAPHS)

# The basic tokenizer splits on "/" and ".", so this is many tokens but one whitespace word.
LONG_URL = "https://example.com/" + "/".join(f"segment{i}" for i in range(150))
URL_TEXT = f"Read the full report at {LONG_URL} before the meeting."

UNBROKEN_REASON = "known: an unbroken string longer than size; decision pending"


def n_tokens(text: str) -> int:
    return len(tokenizer.tokenize(text))


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_chunks_within_size_and_non_empty(chunker: Chunker) -> None:
    chunks = chunker(NUMBERED_TEXT, SIZE, OVERLAP)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.strip()
        assert n_tokens(chunk) <= SIZE


@pytest.mark.parametrize("chunker", CHUNKERS)
@pytest.mark.parametrize("text", ["", "   ", "\n\n\t "])
def test_empty_or_whitespace_gives_no_chunks(chunker: Chunker, text: str) -> None:
    assert chunker(text, SIZE, OVERLAP) == []


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_short_text_gives_one_chunk(chunker: Chunker) -> None:
    text = "OpenAI released a new model on Tuesday."
    assert n_tokens(text) < SIZE

    assert chunker(text, SIZE, OVERLAP) == [text]


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_no_chunk_cuts_a_word(chunker: Chunker) -> None:
    words = set(PARAGRAPH_TEXT.split())

    chunks = chunker(PARAGRAPH_TEXT, SIZE, OVERLAP)

    assert len(chunks) > 1
    for chunk in chunks:
        assert set(chunk.split()) <= words


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_long_unbroken_string_respects_size(chunker: Chunker) -> None:
    assert n_tokens(LONG_URL) > SIZE

    chunks = chunker(URL_TEXT, SIZE, OVERLAP)

    assert all(n_tokens(chunk) <= SIZE for chunk in chunks)


def test_recursive_splits_at_paragraph_boundaries() -> None:
    assert all(n_tokens(p) < SIZE for p in PARAGRAPHS)
    assert n_tokens(PARAGRAPH_TEXT) > SIZE
    whole_paragraph_runs = {
        "\n\n".join(PARAGRAPHS[i:j])
        for i in range(len(PARAGRAPHS))
        for j in range(i + 1, len(PARAGRAPHS) + 1)
    }

    chunks = chunk_recursive(PARAGRAPH_TEXT, SIZE, OVERLAP)

    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk in whole_paragraph_runs


@pytest.mark.parametrize("chunker", CHUNKERS)
def test_consecutive_chunks_overlap_roughly(chunker: Chunker) -> None:
    # Overlap is whole words, so it can fall short of OVERLAP by up to about one word.
    margin = 5

    chunks = chunker(NUMBERED_TEXT, SIZE, OVERLAP)

    assert len(chunks) > 1
    for current, following in zip(chunks, chunks[1:], strict=False):
        current_words = current.split()
        following_words = following.split()
        shared = [w for w in following_words if w in set(current_words)]
        k = len(shared)
        assert k > 0
        assert following_words[:k] == current_words[-k:]
        assert OVERLAP - margin <= n_tokens(" ".join(shared)) <= OVERLAP
