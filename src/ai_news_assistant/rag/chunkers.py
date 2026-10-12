from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Word aligned baseline chunker."""
    splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, separator=" ", chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_text(text)


def chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Recursive chunker."""
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_text(text)
