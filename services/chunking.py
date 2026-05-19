import os
import nltk
from nltk.tokenize import sent_tokenize


def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


def chunk_text(text, window_size=5, overlap=2):
    """
    Split text into overlapping sliding windows of sentences.

    Strategy: group sentences into windows of `window_size` with `overlap`
    sentences shared between adjacent chunks. This preserves local context
    that single-sentence chunks would lose.

    Args:
        text: raw document text
        window_size: number of sentences per chunk
        overlap: sentences shared between consecutive chunks

    Returns:
        list of chunk strings
    """
    sentences = [s.strip() for s in sent_tokenize(text) if s.strip()]
    chunks = []
    step = max(1, window_size - overlap)

    for i in range(0, len(sentences), step):
        window = sentences[i : i + window_size]
        if window:
            chunks.append(" ".join(window))

    return chunks


def load_and_chunk_documents(folder, window_size=5, overlap=2):
    """
    Load all .txt files in `folder` and chunk them.

    Returns:
        list of text chunks across all files
    """
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Data folder '{folder}' not found. "
            "Create it and place .txt files inside."
        )

    chunks = []

    for file_name in sorted(os.listdir(folder)):
        if not file_name.endswith(".txt"):
            continue

        file_path = os.path.join(folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()

        file_chunks = chunk_text(text, window_size, overlap)
        chunks.extend(file_chunks)
        print(f"  '{file_name}' → {len(file_chunks)} chunks")

    if not chunks:
        raise ValueError(
            f"No text found in '{folder}'. "
            "Ensure the folder contains non-empty .txt files."
        )

    print(f"Total chunks loaded: {len(chunks)}")
    return chunks
