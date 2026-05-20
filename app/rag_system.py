import os
import sys
import time
import json
import hashlib
import faiss
import numpy as np
import nltk
from pathlib import Path


from google import genai
from google.genai import types
from huggingface_hub import InferenceClient
from nltk.tokenize import sent_tokenize
from dotenv import load_dotenv


load_dotenv()

# ==========================================================
# API KEYS
# ==========================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
HF_TOKEN = os.getenv("HF_TOKEN")


# ==========================================================
# CONFIGURATION
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FOLDER = BASE_DIR / "data"

VECTOR_STORE_DIR = BASE_DIR / "vector_store"
FAISS_INDEX_PATH = VECTOR_STORE_DIR / "faiss.index"
CHUNKS_PATH = VECTOR_STORE_DIR / "chunks.npy"
INDEXED_FILES_PATH = VECTOR_STORE_DIR / "indexed_files.json"

HF_EMBEDDING_MODEL = "ibm-granite/granite-embedding-97m-multilingual-r2"
GEMINI_MODEL = "gemini-2.5-flash"

TOP_K = 6
SIMILARITY_THRESHOLD = 0.40
BATCH_SIZE = 1


# ==========================================================
# CLIENTS
# ==========================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=HF_TOKEN
)


# ==========================================================
# NLTK SETUP
# ==========================================================

def setup_nltk():
    nltk.download("punkt", quiet=True)
    nltk.download("punkt_tab", quiet=True)


# ==========================================================
# LOAD DOCUMENTS
# ==========================================================

def _file_hash(file_path):
    h = hashlib.md5()
    with open(file_path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


def load_indexed_files():
    if not INDEXED_FILES_PATH.exists():
        return {}
    with open(INDEXED_FILES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_indexed_files(indexed_files):
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    with open(INDEXED_FILES_PATH, "w", encoding="utf-8") as f:
        json.dump(indexed_files, f, indent=2)


def load_documents(folder=DATA_FOLDER):
    """Load all .txt/.md files and return chunks (used for full rebuild)."""
    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist. Create it and put .txt files inside."
        )

    chunks = []

    for file_name in os.listdir(folder):
        if file_name.endswith((".txt", ".md")):
            file_path = os.path.join(folder, file_name)
            with open(file_path, "r", encoding="utf-8") as file:
                text = file.read()
            for sentence in sent_tokenize(text):
                sentence = sentence.strip()
                if sentence:
                    chunks.append(sentence)

    if not chunks:
        raise ValueError(
            f"No text found. Make sure the '{folder}' folder contains .txt files."
        )

    print(f"Loaded {len(chunks)} text chunks.")
    return chunks


def load_new_documents(folder=DATA_FOLDER, indexed_files=None):
    """Return chunks only from files not yet indexed (by MD5 hash)."""
    if indexed_files is None:
        indexed_files = {}

    if not os.path.exists(folder):
        raise FileNotFoundError(
            f"Folder '{folder}' does not exist. Create it and put .txt files inside."
        )

    new_chunks = []
    new_file_hashes = {}

    for file_name in os.listdir(folder):
        if not file_name.endswith((".txt", ".md")):
            continue
        file_path = os.path.join(folder, file_name)
        file_hash = _file_hash(file_path)

        if indexed_files.get(file_name) == file_hash:
            print(f"Skipping '{file_name}' (already indexed).")
            continue

        print(f"Indexing new/modified file: '{file_name}'")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        for sentence in sent_tokenize(text):
            sentence = sentence.strip()
            if sentence:
                new_chunks.append(sentence)
        new_file_hashes[file_name] = file_hash

    return new_chunks, new_file_hashes


# ==========================================================
# HUGGING FACE CLOUD EMBEDDINGS
# ==========================================================

def normalize_embedding_output(raw_output, expected_count):
    """
    Converts Hugging Face embedding output into a clean 2D numpy array.

    Final shape:
        [number_of_texts, embedding_dimension]
    """

    arr = np.array(raw_output, dtype="float32")

    # Case 1:
    # Single embedding:
    # [embedding_dimension]
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)

    # Case 2:
    # Batch embeddings:
    # [batch_size, embedding_dimension]
    elif arr.ndim == 2:
        if arr.shape[0] == expected_count:
            pass

        # Token embeddings for one input:
        # [tokens, embedding_dimension]
        elif expected_count == 1:
            arr = arr.mean(axis=0, keepdims=True)

        else:
            raise ValueError(
                f"Unexpected 2D embedding shape: {arr.shape}, "
                f"expected_count={expected_count}"
            )

    # Case 3:
    # Token embeddings for batch:
    # [batch_size, tokens, embedding_dimension]
    elif arr.ndim == 3:
        arr = arr.mean(axis=1)

    else:
        raise ValueError(f"Unexpected embedding dimensions: {arr.ndim}")

    if arr.shape[0] != expected_count:
        raise ValueError(
            f"Embedding count mismatch. Expected {expected_count}, got {arr.shape[0]}"
        )

    return arr.astype("float32")


def hf_feature_extraction_with_retries(inputs, expected_count, max_retries=5):
    """
    Calls Hugging Face cloud embedding model with retries.

    inputs can be:
    - string
    - list of strings
    """

    for attempt in range(1, max_retries + 1):
        try:
            result = hf_client.feature_extraction(
                inputs,
                model=HF_EMBEDDING_MODEL
            )

            embeddings = normalize_embedding_output(
                raw_output=result,
                expected_count=expected_count
            )

            return embeddings

        except Exception as e:
            print(
                f"Hugging Face embedding failed. Attempt {attempt}/{max_retries}")
            print("Error:", e)

            if attempt == max_retries:
                raise

            wait_seconds = attempt * 3
            print(f"Retrying in {wait_seconds} seconds...")
            time.sleep(wait_seconds)


def embed_texts_with_huggingface(texts, batch_size=BATCH_SIZE):
    all_embeddings = []
    total_batches = (len(texts) + batch_size - 1) // batch_size

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]

        current_batch = start // batch_size + 1
        print(f"Embedding batch {current_batch}/{total_batches}...")

        embeddings = hf_feature_extraction_with_retries(
            inputs=batch,
            expected_count=len(batch)
        )

        all_embeddings.append(embeddings)

    final_embeddings = np.vstack(all_embeddings).astype("float32")

    print(f"Created document embeddings. Shape: {final_embeddings.shape}")

    return final_embeddings


def embed_query_with_huggingface(query):
    embedding = hf_feature_extraction_with_retries(
        inputs=query,
        expected_count=1
    )

    return embedding.astype("float32")


# ==========================================================
# FAISS VECTOR SEARCH
# ==========================================================

def create_faiss_index(embeddings):
    """
    Creates FAISS index.

    We normalize vectors and use inner product.
    This behaves like cosine similarity.
    """

    faiss.normalize_L2(embeddings)

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print(f"FAISS index created with {index.ntotal} vectors.")

    return index


def retrieve(query, index, chunks, k=TOP_K, threshold=SIMILARITY_THRESHOLD):
    """
    Embeds the user question with Hugging Face and searches FAISS.
    Returns only chunks whose cosine similarity exceeds the threshold.
    """

    query_embedding = embed_query_with_huggingface(query)

    faiss.normalize_L2(query_embedding)

    scores, indexes = index.search(query_embedding, k)

    print("\nFAISS scores:", scores)
    print("FAISS indexes:", indexes)

    retrieved_chunks = []

    for score, idx in zip(scores[0], indexes[0]):
        if idx == -1:
            continue
        if float(score) < threshold:
            print(
                f"  Skipping chunk {idx} — score {score:.3f} below threshold {threshold}")
            continue
        retrieved_chunks.append(chunks[idx])

    return retrieved_chunks


# ==========================================================
# GEMINI LLM
# ==========================================================

def ask_gemini(context, question, history=None):
    """
    Gemini is the LLM.
    Hugging Face is only used for embeddings.
    """
    history_block = ""
    if history:
        lines = []
        for turn in history:
            lines.append(f"User: {turn['question']}")
            lines.append(f"Assistant: {turn['answer']}")
        history_block = "Previous conversation:\n" + "\n".join(lines) + "\n\n"

    prompt = f"""You are an expert assistant on the Red Hot Chili Peppers (RHCP).
Answer the user's question based ONLY on the numbered context excerpts below, retrieved from the RHCP knowledge base.

The knowledge base was built from two sources:
- Wikipedia articles about the band and its members.
- Genius annotations: song lyrics and their meanings, written by music analysts.

Rules:
- Answer using ONLY information explicitly stated in the context excerpts.
- Do not invent facts, names, dates, albums, or events not present in the context.
- If the context contains partial information, answer what you can and clearly note what is missing.
- Only respond with "I don't have enough information in my knowledge base to answer that." if the context contains NO relevant information at all.
- Do not speculate, infer beyond what is stated, or draw on outside knowledge.
- Be concise and accurate.
- Use the previous conversation for context (e.g. resolving pronouns like "he", "they", "it").
- Always reply in the same language the user asked in.

{history_block}Context excerpts:
{context}

Question: {question}

Answer:"""

    response = gemini_client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=500,
            thinking_config=types.ThinkingConfig(
                thinking_budget=0
            )
        )
    )

    return response.text.strip()


# ==========================================================
# VECTOR STORE — SAVE / LOAD
# ==========================================================

def save_vector_store(index, chunks):
    VECTOR_STORE_DIR.mkdir(exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    np.save(CHUNKS_PATH, np.array(chunks, dtype=object))
    print("Vector store saved to disk.")


def load_vector_store():
    if not FAISS_INDEX_PATH.exists() or not CHUNKS_PATH.exists():
        return None, None
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    chunks = np.load(CHUNKS_PATH, allow_pickle=True).tolist()
    print(f"Vector store loaded from disk ({len(chunks)} chunks).")
    return index, chunks


# ==========================================================
# LAZY-LOADED STATE (built once per process)
# ==========================================================

_index = None
_chunks = None


def _init():
    global _index, _chunks
    if _index is not None:
        return
    setup_nltk()
    _index, _chunks = load_vector_store()
    indexed_files = load_indexed_files()

    new_chunks, new_file_hashes = load_new_documents(
        DATA_FOLDER, indexed_files)

    if not new_chunks:
        if _index is None:
            raise ValueError("No files found to index.")
        print("All files already indexed. Nothing to update.")
        return

    print(f"Embedding {len(new_chunks)} new chunks...")
    new_embeddings = embed_texts_with_huggingface(new_chunks)

    if _index is None:
        _index = create_faiss_index(new_embeddings)
        _chunks = new_chunks
    else:
        faiss.normalize_L2(new_embeddings)
        _index.add(new_embeddings)
        _chunks = _chunks + new_chunks

    indexed_files.update(new_file_hashes)
    save_vector_store(_index, _chunks)
    save_indexed_files(indexed_files)


# ==========================================================
# PUBLIC API (called by RHCP_app.py)
# ==========================================================

def rebuild_vector_store():
    """Delete cached index and rebuild from all files in data/."""
    global _index, _chunks
    if FAISS_INDEX_PATH.exists():
        FAISS_INDEX_PATH.unlink()
    if CHUNKS_PATH.exists():
        CHUNKS_PATH.unlink()
    if INDEXED_FILES_PATH.exists():
        INDEXED_FILES_PATH.unlink()
    _index = None
    _chunks = None
    _init()


def ask_question(question, history=None):
    question = question.strip()
    if not question:
        return {"answer": "Please enter a question.", "sources": []}

    _init()

    top_chunks = retrieve(question, _index, _chunks, k=TOP_K)

    if not top_chunks:
        return {
            "answer": (
                "I couldn't find relevant information in the RHCP knowledge base "
                "for that question. Try asking about the band's history, albums, "
                "members, or songs."
            ),
            "sources": [],
        }

    numbered_context = "\n\n".join(
        f"[{i+1}] {chunk}" for i, chunk in enumerate(top_chunks)
    )
    answer = ask_gemini(numbered_context, question, history=history)
    return {"answer": answer, "sources": top_chunks}


# ==========================================================
# MAIN APP
# ==========================================================

def main():
    setup_nltk()

    print("Loading documents...")
    chunks = load_documents(DATA_FOLDER)

    print("\nCreating Hugging Face cloud embeddings...")
    document_embeddings = embed_texts_with_huggingface(chunks)

    print("\nCreating FAISS index...")
    index = create_faiss_index(document_embeddings)

    print("\nRAG system is ready.")
    print("Embeddings: Hugging Face cloud")
    print("Vector search: FAISS local")
    print("LLM: Gemini cloud")
    print("Type 'exit' to quit.")

    while True:
        question = input("\nAsk something: ").strip()

        if question.lower() == "exit":
            print("Goodbye.")
            break

        if not question:
            print("Please enter a real question.")
            continue

        top_chunks = retrieve(
            query=question,
            index=index,
            chunks=chunks,
            k=TOP_K
        )

        context = "\n".join(top_chunks)

        print("\nRetrieved Context:")
        print(context)

        answer = ask_gemini(context, question)

        print("\nGemini Answer:")
        print(answer)


if __name__ == "__main__":
    main()
