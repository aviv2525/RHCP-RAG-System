import faiss


def build_index(embeddings):
    """
    Build a FAISS IndexFlatIP (cosine similarity via L2-normalised vectors).

    Normalising before indexing converts inner-product search into cosine
    similarity, so scores range from -1 (opposite) to 1 (identical).

    Args:
        embeddings: np.ndarray of shape [n, dim], float32

    Returns:
        faiss.IndexFlatIP with all vectors added
    """
    embeddings = embeddings.astype("float32")
    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print(f"FAISS index built: {index.ntotal} vectors, dim={embeddings.shape[1]}")
    return index


def search(query_embedding, index, chunks, k=5, min_score=0.25):
    """
    Retrieve the top-k most relevant chunks for a query embedding.

    Args:
        query_embedding: np.ndarray of shape [1, dim], float32
        index: built FAISS index
        chunks: list of text strings aligned with the index
        k: number of results to retrieve
        min_score: cosine similarity threshold — results below this are discarded

    Returns:
        list of dicts with keys 'text' and 'score', ordered by relevance.
        Empty list if nothing exceeds min_score.
    """
    qe = query_embedding.astype("float32")
    faiss.normalize_L2(qe)

    scores, indices = index.search(qe, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx < len(chunks) and score >= min_score:
            results.append({"text": chunks[idx], "score": float(score)})

    return results
