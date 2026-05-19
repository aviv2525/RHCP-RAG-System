import time
import numpy as np
from huggingface_hub import InferenceClient


EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
BATCH_SIZE = 1  # Keep at 1 to avoid HuggingFace free-tier rate limits


def create_hf_client(api_key):
    return InferenceClient(provider="hf-inference", api_key=api_key)


def _normalize_output(raw, expected_count):
    """
    Coerce HuggingFace feature-extraction output to shape [n, dim].

    The API can return 1-D, 2-D, or 3-D arrays depending on the model
    and whether a single string or a list was passed.
    """
    arr = np.array(raw, dtype="float32")

    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    elif arr.ndim == 2:
        if arr.shape[0] != expected_count and expected_count == 1:
            arr = arr.mean(axis=0, keepdims=True)
    elif arr.ndim == 3:
        arr = arr.mean(axis=1)
    else:
        raise ValueError(f"Unexpected embedding ndim: {arr.ndim}")

    if arr.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count mismatch: expected {expected_count}, got {arr.shape[0]}"
        )

    return arr.astype("float32")


def _embed_with_retries(client, inputs, expected_count, max_retries=5):
    for attempt in range(1, max_retries + 1):
        try:
            raw = client.feature_extraction(inputs, model=EMBEDDING_MODEL)
            return _normalize_output(raw, expected_count)
        except Exception as exc:
            if attempt == max_retries:
                raise
            wait = attempt * 3
            print(f"  Embedding attempt {attempt}/{max_retries} failed ({exc}). Retrying in {wait}s...")
            time.sleep(wait)


def embed_texts(client, texts, batch_size=BATCH_SIZE):
    """
    Embed a list of text chunks in batches.

    Returns:
        np.ndarray of shape [len(texts), embedding_dim]
    """
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for i, start in enumerate(range(0, len(texts), batch_size)):
        batch = texts[start : start + batch_size]
        print(f"  Embedding batch {i + 1}/{total_batches}...")
        embeddings = _embed_with_retries(client, batch, len(batch))
        all_embeddings.append(embeddings)

    result = np.vstack(all_embeddings).astype("float32")
    print(f"Embeddings created: shape {result.shape}")
    return result


def embed_query(client, query):
    """
    Embed a single query string.

    Returns:
        np.ndarray of shape [1, embedding_dim]
    """
    return _embed_with_retries(client, query, 1).astype("float32")
