from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
from ai_news_assistant.rag.config import MODEL_NAME

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Word aligned baseline chunker."""
    splitter = CharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, separator=" ", chunk_size=size, chunk_overlap=overlap
    )

    # return splitter.split_text(text)
    chunks: list[str] = []
    for piece in splitter.split_text(text):
        if len(tokenizer.encode(piece)) > size:
            chunks.extend(chunk_recursive(piece, size, overlap))
        else:
            chunks.append(piece)
    return chunks


def chunk_recursive(text: str, size: int, overlap: int) -> list[str]:
    """Recursive chunker."""
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer, chunk_size=size, chunk_overlap=overlap
    )
    return splitter.split_text(text)
